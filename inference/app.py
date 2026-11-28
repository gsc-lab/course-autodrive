"""추론기 메인 루프: 입력 → 자율주행 상태 규칙 → (새 사진이 오면) 예측 → 차량 → 기록 → 상태 표시.

주 루프, 감시 스레드, 신호 처리, 종료 순서는 수집기와 같은 것을 쓴다 (common/driveapp.py 의 DriveApp).
여기에는 추론기만의 일이 있다: 모델 열기, 사진마다의 예측, 예측 기록, 추론기의 화면.

예측은 수동 운전 중에도 계속한다. 상태줄에서 "사람은 이렇게 꺾는데 모델은 저렇게 답한다"를 바로 볼 수 있고,
기록(CSV)으로 둘을 나중에 견줄 수 있다. ○ 를 누르면 그 예측이 그대로 조향이 된다.
"""
from __future__ import annotations

import csv
import logging
import os
import shutil
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from common.bgio import BackgroundWriter
from common.config import Config
from common.display import DisplayState
from common.driveapp import DriveApp, _safe_write
from common.inputs.base import STATUS_AUTO, STATUS_ERROR, STATUS_IDLE
from inference.bundle import ModelBundle
from inference.engine import open_backend
from inference.predict import Prediction, Predictor
from inference.state import AutoLogic

log = logging.getLogger("inference")

MARK_FLASH_SEC = 0.3        # □ 를 누르면 도트 매트릭스 전체가 이만큼 켜진다
LOG_FLUSH_SEC = 1.0         # 기록 파일을 디스크로 내보내는 주기
LOG_PENDING_MAX = 3000      # 아직 쓰지 못한 줄이 이만큼 밀리면 새 줄을 버린다 (초당 30줄이면 100초 분량)
LOG_CLOSE_SEC = 5.0         # 끝낼 때 밀린 줄을 마저 쓰도록 기다리는 시간
KEYBOARD_AUTO_SEC = 30.0    # 키보드로 몰 때 safety.deadman_sec 가 0(끔)이어도 자율주행에는 이 한도를 건다
WARMUP_RUNS = 5             # 시작할 때 미리 돌려 보는 횟수


def fit_line(text: str, width: int) -> str:
    """터미널 한 줄(width 칸)에 들어가게 자른다. 한글은 두 칸이고, 터미널마다 폭이 다른 글자(° 등)도 두 칸으로 센다.

    상태줄이 터미널 폭을 넘으면 줄이 바뀌어, 1초에 다섯 줄씩 화면이 밀려 올라가고 경고가 화면 밖으로 사라진다."""
    out, used = [], 0
    for ch in text:
        w = 2 if unicodedata.east_asian_width(ch) in "WFA" else 1
        if used + w > width:
            break
        out.append(ch)
        used += w
    return "".join(out)


class DriveLog:
    """사진마다의 예측을 CSV 로 남긴다: <recording.root>/drives/<시각>_<모델>.csv

    열: 시각, 모드(manual | auto), 쓰기로 한 각도, 이 사진만의 각도와 확률, 클래스별 확률, 차에 낸 각도와 속도,
        사진이 몇 ms 전 것인지, 전처리 + 추론 시간, 그때 있었던 일(note: 자율주행 시작·끝과 이유, □ 표시).
    수동 운전 중의 줄에서는 '차에 낸 각도'가 사람이 꺾은 각도이므로, 모델의 예측과 견줄 수 있다.

    파일에 쓰는 일은 뒤의 스레드가 한다 (common/bgio.py). 주 루프는 줄의 내용만 넘기고 기다리지 않는다:
    2026-11-24 실차에서 자율주행 중 주 루프가 0.68초 멈췄고, SD 카드가 바쁠 때 이 파일의 flush 가 1.19초까지 막혔다.
        failed   쓰다가 난 오류 (디스크가 가득 참 등). 주 루프가 보고 기록을 그만둔다
        dropped  디스크가 밀려 버린 줄 수
    """

    def __init__(self, root: str | Path, run_id: str, classes: list):
        folder = Path(root) / "drives"
        folder.mkdir(parents=True, exist_ok=True)
        safe = "".join("_" if ch in '/\\:*?"<>| ' else ch for ch in run_id)
        self.path = folder / f"{datetime.now():%Y%m%d_%H%M%S}_{safe}.csv"
        self._f = open(self.path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow(["t_iso", "t_mono", "mode", "pred_angle", "raw_angle", "confidence",
                          *[f"p_{c}" for c in classes], "applied_angle", "speed", "frame_age_ms", "infer_ms", "note"])
        self._n_classes = len(classes)
        self._notes: list = []
        self._flushed = time.monotonic()
        self._closed = False
        self.rows = 0                       # 쓰기로 한 줄 수 (버린 줄은 세지 않는다)
        self._io = BackgroundWriter("drive-log", LOG_PENDING_MAX)     # 여기부터 파일은 이 스레드만 만진다

    @property
    def failed(self) -> Optional[BaseException]:
        return self._io.failed

    @property
    def dropped(self) -> int:
        return self._io.dropped

    def note(self, text: str) -> None:
        """다음 줄에 함께 적을 일."""
        self._notes.append(text.replace("\n", " "))

    def _write(self, fields: list, flush: bool) -> None:
        """(쓰는 스레드) 한 줄을 쓰고, 때가 됐으면 디스크로 내보낸다."""
        self._w.writerow(fields)
        if flush:
            self._f.flush()

    def row(self, now: float, mode: str, pred: Prediction, applied_angle: int, speed: int, frame_age: float) -> bool:
        """한 줄을 쓰라고 맡긴다 (기다리지 않는다). 디스크가 밀려 버렸으면 False: 적어 둔 일(note)은 다음 줄에 붙는다."""
        fields = [datetime.now().isoformat(timespec="milliseconds"), f"{now:.3f}", mode, pred.angle, pred.raw_angle,
                  f"{pred.confidence:.4f}", *[f"{p:.4f}" for p in pred.probs], applied_angle, speed,
                  f"{frame_age * 1e3:.0f}", f"{pred.ms:.2f}", "; ".join(self._notes)]
        flush = now - self._flushed >= LOG_FLUSH_SEC
        if not self._io.submit(lambda: self._write(fields, flush)):
            return False
        self._notes.clear()
        self.rows += 1
        if flush:
            self._flushed = now
        return True

    def close(self, timeout: float = LOG_CLOSE_SEC) -> bool:
        """밀린 줄을 마저 쓰고 파일을 닫는다. 시간 안에 끝내지 못했으면(디스크가 멈춤) False."""
        if self._closed:
            return True
        self._closed = True
        if self._notes:                                    # 마지막 사진 뒤에 있었던 일도 남긴다
            fields = [datetime.now().isoformat(timespec="milliseconds"), f"{time.monotonic():.3f}", "", "", "", "",
                      *[""] * self._n_classes, "", "", "", "", "; ".join(self._notes)]
            self._io.submit(lambda: self._write(fields, True))
            self._notes.clear()
        if not self._io.close(timeout):                    # 파일은 쓰는 스레드가 잡고 있다: 닫지 않고 둔다
            log.warning("예측 기록을 끝까지 쓰지 못했습니다 (디스크가 응답하지 않음): %s", self.path)
            return False
        self._f.close()
        return True

    def abandon(self) -> None:
        """기다리지 않고 기록을 그만둔다 (쓰다가 실패했을 때 주 루프에서)."""
        self._closed = True
        self._io.stop_soon()


class Autopilot(DriveApp):
    title = "자율주행"
    log = log

    def __init__(self, cfg: Config, bundle: ModelBundle, mock: bool = False, input_override: Optional[str] = None,
                 skip_checks: bool = False, check_only: bool = False, backend: str = "auto"):
        self.bundle = bundle
        self.backend_kind = backend
        self.predictor: Optional[Predictor] = None
        self.drive_log: Optional[DriveLog] = None
        self.last: Optional[Prediction] = None      # 가장 최근 예측
        self.first: Optional[Prediction] = None     # 시작할 때 한 번 돌려 본 예측 (점검 보고용)
        self._frame_t: Optional[float] = None       # 가장 최근에 예측한 사진의 시각
        self._drop_warned = False                   # '디스크가 느려 기록을 버린다'는 경고는 한 번만
        super().__init__(cfg, mock=mock, input_override=input_override, skip_checks=skip_checks, check_only=check_only)

    # ------------------------------------------------------------------ 시작
    def _before_input(self) -> None:
        """모델을 연다. 패드를 잡기 전에 하므로, 쓸 수 없는 모델이면 아무것도 건드리지 않고 끝난다."""
        self._show(DisplayState(busy=True), time.monotonic())           # 가운데 점: 준비 중
        inf = self.cfg.inference
        backend = open_backend(self.bundle, self.backend_kind, inf.precision, say=lambda m: _safe_write(m + "\r\n"))
        self.predictor = Predictor(self.bundle, backend, inf.smoothing, inf.window, inf.ema_alpha)
        frame, _ = self.camera.read()
        if frame is None:
            raise RuntimeError("카메라에서 사진을 읽지 못했습니다")
        for _ in range(WARMUP_RUNS):                                    # 미리 돌려 본다: 첫 추론은 수십 ms 로 느리다
            self.first = self.predictor(frame)
        self.predictor.reset()
        log.info("모델 %s | 조향 단계 %s | 입력 %s | %s | 흔들림 줄이기 %s",
                 self.bundle.run_id, self.bundle.classes, list(self.bundle.input_shape), backend.note, inf.smoothing)
        s = self.cfg.steering
        outside = [c for c in self.bundle.classes if not (s.min <= c <= s.max)]
        if outside:
            log.warning("모델의 조향 단계 %s 가 이 차의 조향 범위 %d~%d 밖입니다. 범위 끝으로 맞춰서 씁니다 "
                        "(다른 차의 사진으로 학습한 모델이면 이 차에서는 잘 달리지 못할 수 있습니다)", outside, s.min, s.max)

    def _make_logic(self) -> AutoLogic:
        # 자율주행은 손을 놓아도 달린다. 그래서 입력 장치가 끊겼을 때 차가 알 수 있어야만 켠다.
        name, block, timeout = self.input.name, "", 0.0
        if name == "ds4" and getattr(self.input, "link", None) is None:
            block = "패드 연결을 감시할 수 없습니다 (bash scripts/jetson_setup.sh install 을 했는지 확인하세요)"
        if name == "keyboard":
            # 키보드는 SSH 로 온다: Wi-Fi 가 끊겨도 차는 모른다. 진짜 차에서는 켜지 않고 (수동 운전만 된다),
            # 가짜 차량(--mock)에서만 무입력 한도를 걸고 켠다.
            timeout = self.cfg.safety.deadman_sec if self.cfg.safety.deadman_sec > 0 else KEYBOARD_AUTO_SEC
            if self.cfg.hardware.backend != "mock":
                block = "자율주행은 패드로만 켤 수 있습니다 (키보드는 연결이 끊겨도 차가 모릅니다). 패드를 켜고 다시 실행하세요"
        return AutoLogic(self.cfg, requires_deadman=self.input.requires_deadman, block_reason=block, no_input_timeout=timeout)

    def _open_outputs(self) -> None:
        if not self.cfg.inference.log:
            return
        try:
            self.drive_log = DriveLog(self.cfg.recording.root, self.bundle.run_id, self.bundle.classes)
            log.info("예측 기록: %s", self.drive_log.path)
        except OSError as e:                                # 기록은 못 남겨도 달리는 데는 지장이 없다
            log.warning("예측 기록 파일을 만들지 못했습니다 (기록 없이 계속합니다): %s", e)

    def _extra_release_steps(self) -> list:
        steps = []
        if self.predictor is not None:
            steps.append(self.predictor.close)
        if self.drive_log is not None:
            steps.append(self.drive_log.close)
        return steps

    def _target(self) -> Tuple[int, int, Optional[str]]:
        logic = self.logic
        return logic.angle, logic.effective_speed, logic.effective_direction

    def check_report(self) -> str:
        """점검 모드(--check)에서 보여 줄 한 줄."""
        p = self.first
        return (f"모델 {self.bundle.run_id} ({self.predictor.backend.note}). "
                f"지금 카메라 사진의 예측: {p.angle}° (확률 {p.prob:.2f}), 전처리 + 추론 {p.ms:.1f} ms")

    # ------------------------------------------------------------------ 루프
    def _step(self, now: float) -> None:
        logic, st = self.logic, self.logic.state

        self._predict(now)                  # 입력보다 먼저: ○ 를 누른 그 주기에 가장 새 예측이 준비돼 있다
        for ev in self.input.poll():
            logic.apply(ev, now)
        logic.tick(now)

        self._camera_failed(now)            # 카메라가 멈췄으면 여기서 자율주행이 꺼진다 (아래에서 함께 기록된다)
        if logic.pending_mark:
            logic.pending_mark = False
            self._flash_until = now + MARK_FLASH_SEC
            self._note("mark")
        for kind, code, text in logic.changes:
            log.info("%s", text)
            self._note(f"{kind} {code}")      # 기록(CSV)에는 영문 코드만: Excel 로 열어도 깨지지 않고 이유별로 세기 쉽다
            # 화면에는 한 줄씩 따로 남긴다 (상태줄은 그 아래에 다시 그려진다). 상태줄 끝에 붙이면 80칸 터미널에서 잘려,
            # 왜 꺼졌는지와 왜 켜지지 않는지를 읽을 수 없다.
            _safe_write("\r" + text + "\x1b[K\r\n")
        logic.changes.clear()

        self._apply_vehicle()

        if self._status_due(now):
            self.input.set_status(STATUS_ERROR if st.error else (STATUS_AUTO if logic.auto else STATUS_IDLE))
            self._print_status(now)
        self._show(DisplayState(auto=logic.auto, error=bool(st.error), flash=now < self._flash_until), now)

    def _predict(self, now: float) -> None:
        """새 사진이 왔으면 예측한다. 모델이 실패하면 차를 세우고 끝낸다 (그 값으로 몰면 안 된다)."""
        logic, st = self.logic, self.logic.state
        frame, t = self.camera.read()
        if frame is None or t == self._frame_t:
            return
        self._frame_t = t
        try:
            pred = self.predictor(frame)
        except Exception as e:
            if not st.error:
                st.error = f"추론 실패: {e}"
                log.error(st.error)
            logic.apply_stop()
            st.quit = True
            return
        self.last = pred
        logic.set_prediction(pred.angle, t)
        dl = self.drive_log
        if dl is not None:
            # 파일은 뒤의 스레드가 쓴다: 여기서는 줄의 내용만 넘기고 기다리지 않는다. 그 스레드가 쓰다 실패했으면
            # (디스크가 가득 참 등) 기록만 그만두고 주행은 계속한다.
            failure, kept = None, True
            try:
                kept = dl.row(now, logic.mode, pred, logic.angle, logic.effective_speed, now - t)
                failure = dl.failed
            except (OSError, ValueError) as e:
                failure = e
            if failure is not None:
                log.warning("예측 기록을 쓰지 못해 기록을 그만둡니다 (주행은 계속합니다): %s", failure)
                dl.abandon()
                self.drive_log = None
            elif not kept and not self._drop_warned:
                self._drop_warned = True
                log.warning("디스크가 느려 예측 기록의 일부 줄을 버리고 있습니다 (주행은 계속합니다)")

    def _note(self, text: str) -> None:
        if self.drive_log is not None:
            self.drive_log.note(text)

    def _slow_step(self, seconds: float) -> None:
        super()._slow_step(seconds)
        self._note(f"loop_stall ms={seconds * 1000:.0f}")   # 기록에도 남긴다: 이 줄 앞의 빈 구간이 주 루프가 멈춘 것임을

    # ------------------------------------------------------------------ 표시
    def _print_help(self) -> None:
        if self.input.name == "ds4":
            dm = " (L1 을 누르고 있는 동안만 달립니다)" if self.input.requires_deadman else ""
            msg = (f"DS4: ○ 자율주행 시작/정지{dm} | D패드 ↑↓ 속도 단계 | R2·L2·스틱·△ 을 건드리면 바로 수동 운전 | "
                   "□ 기록에 표시 | Options 종료"
                   f"\r\nSSH 가 끊겨도 계속 달립니다. 종료는 Options 버튼, 또는 다른 터미널에서  kill {os.getpid()}")
        else:
            msg = ("키보드: R 자율주행 시작/정지 | A/Z 속도 | ↑↓←→ Space 는 수동 운전 | M 기록에 표시 | Q/ESC 종료"
                   f"\r\n입력 없이 {self.logic.no_input_timeout:g}초가 지나면 자율주행이 꺼집니다 (키보드는 연결이 끊겨도 차가 모릅니다)")
        if self.logic.block_reason:
            msg += "\r\n[주의] 이 상태로는 자율주행을 켤 수 없습니다 (수동 운전만 됩니다): " + self.logic.block_reason
        _safe_write(msg + "\r\n")

    def _status_line(self) -> str:
        logic, st, p = self.logic, self.logic.state, self.last
        mode = "자율" if logic.auto else "수동"
        dead = "" if not self.input.requires_deadman else (" L1" if st.deadman_held else " --")
        # 괄호 안은 이 사진에서 그 각도의 확률. 흔들림 줄이기를 쓰면 각도는 최근 여러 장으로 고른 것이다
        pred = f"예측 {p.angle:03d} ({p.prob:.2f}) {p.ms:.1f}ms" if p is not None else "예측 없음"
        return (f"[{mode}] a={logic.angle:03d} v={logic.effective_speed:03d} {logic.effective_direction or 'stop':8s}{dead}"
                f" | {pred} | 자율주행 속도 {logic.level + 1}/{len(logic.speeds)} ({logic.auto_speed}%)")

    def _print_status(self, now: float) -> None:
        # 터미널 폭에 맞춰 자르고, 줄 끝까지 지운다 (\x1b[K): 앞 줄이 더 길었어도 꼬리가 남지 않는다
        width = max(20, shutil.get_terminal_size((80, 24)).columns - 1)
        _safe_write("\r" + fit_line(self._status_line(), width) + "\x1b[K")

    # ------------------------------------------------------------------ 종료
    def close(self) -> None:
        super().close()
        logic = getattr(self, "logic", None)
        if logic is not None and not self.check_only:
            dl = self.drive_log
            where = "" if dl is None else (f", 기록 {dl.path} ({dl.rows}줄"
                                           + (f", 디스크가 느려 버린 줄 {dl.dropped}" if dl.dropped else "") + ")")
            slow = f", 주 루프 멈춤 {self.slow_steps}회 (가장 긴 것 {self.slowest_step:.2f}초)" if self.slow_steps else ""
            log.info("끝: 자율주행 %d회, 수동 전환 %d회, 예측 %d장%s%s", logic.starts, logic.takeovers,
                     self.predictor.count if self.predictor is not None else 0, slow, where)
