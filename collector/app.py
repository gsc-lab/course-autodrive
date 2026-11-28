"""수집기 메인 루프: 입력 → 상태 머신 → 차량 → (조건부) 프레임 저장 → 상태 표시.

주 루프, 감시 스레드, 신호 처리, 종료 순서 같은 안전장치는 common/driveapp.py 의 DriveApp 에 있고 추론기와 함께 쓴다.
여기에는 수집기만의 일(세션 기록, 사진 저장, 수집 화면)이 있다.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional

from common.config import Config, config_snapshot
from common.display import DisplayState
from common.driveapp import (CAMERA_STALE_SEC, LOOP_HZ, STATUS_HZ, WATCHDOG_SEC, DriveApp,  # noqa: F401  (다시 내보낸다)
                             DroppingStderrHandler, PreflightError, _fd_writable, _safe_write, preflight)
from common.inputs.base import STATUS_ERROR, STATUS_IDLE, STATUS_RECORDING
from common.session import SessionWriter
from collector.state import DriveLogic

log = logging.getLogger("collector")

MARK_FLASH_SEC = 0.3    # □ 를 누르면 도트 매트릭스 전체가 이만큼 켜진다
FRAME_MAX_AGE_SEC = 0.5 # 이보다 오래된 카메라 프레임은 저장하지 않는다 (옛 그림에 지금의 조향 라벨이 붙지 않게)


class Collector(DriveApp):
    title = "수집"
    log = log

    def __init__(self, cfg: Config, mock: bool = False, input_override: Optional[str] = None,
                 skip_checks: bool = False, check_only: bool = False):
        self.session: Optional[SessionWriter] = None
        self._drop_warned = False           # '디스크가 느려 사진을 버린다'는 경고는 한 번만
        super().__init__(cfg, mock=mock, input_override=input_override, skip_checks=skip_checks, check_only=check_only)
        if not check_only:
            log.info("세션 폴더: %s (입력: %s)", self.session.path, self.input.name)

    def _make_logic(self) -> DriveLogic:
        return DriveLogic(self.cfg, requires_deadman=self.input.requires_deadman)

    def _open_outputs(self) -> None:
        cfg = self.cfg
        self.session = SessionWriter(
            cfg.recording.root, config_snapshot(cfg), jpeg_quality=cfg.camera.jpeg_quality,
            extra_meta={"input_device": self.input.name, "camera": self.camera.props(), "mock": self.mock},
        )

    def _extra_release_steps(self) -> list:
        return [self.session.close] if self.session is not None else []

    # ------------------------------------------------------------------ 루프
    def _step(self, now: float) -> None:
        logic, st, s = self.logic, self.logic.state, self.session

        for ev in self.input.poll():
            s.log_event(ev.kind.value, ev.value, ev.t)
            logic.apply(ev, now)
        logic.tick(now)

        if st.pending_mark:
            st.pending_mark = False
            self._flash_until = now + MARK_FLASH_SEC
            s.mark(self.cfg.recording.mark_seconds, note=f"frame={s.last_frame_name or ''}")
        for n in st.notifications:
            self.input.notify(n)
        st.notifications.clear()

        if self._camera_failed(now):
            st.recording = False

        self._apply_vehicle()

        if logic.should_save(now):
            frame, _ = self.camera.read()
            if frame is not None and self.camera.age(now) <= FRAME_MAX_AGE_SEC:   # 멈춘 카메라의 옛 그림은 저장하지 않는다
                # 파일은 뒤의 스레드가 쓴다 (기다리지 않는다). 디스크가 밀려 버렸으면 None: 주행은 그대로 하고 한 번만 알린다
                if s.save_frame(frame, st.angle, logic.effective_speed, datetime.now()) is None and not self._drop_warned:
                    self._drop_warned = True
                    log.warning("디스크가 느려 사진을 저장하지 못하고 있습니다 (주행은 계속합니다). 끝날 때 버린 장수를 알려 줍니다")
                logic.mark_saved(now)

        if self._status_due(now):
            self.input.set_status(STATUS_ERROR if st.error else (STATUS_RECORDING if st.recording else STATUS_IDLE))
            self._print_status()
        self._update_display(now)           # 매 주기 (바뀐 화면만 전송된다). 깜빡임과 □ 번쩍임이 끊기지 않게

    def _update_display(self, now: float) -> None:
        st = self.logic.state
        self._show(DisplayState(recording=st.recording, error=bool(st.error), flash=now < self._flash_until), now)

    # ------------------------------------------------------------------ 표시
    def _print_help(self) -> None:
        if self.input.name == "ds4":
            dm = "L1 을 계속 누른 채 R2 (손 떼면 정지)" if self.input.requires_deadman else "R2 전진 (데드맨 꺼짐)"
            msg = f"DS4: 왼쪽 스틱 조향 | {dm} | L2 후진 | ○ 녹화 | □ 불량 표시 | △ 중앙 | Options 종료"
            msg += f"\r\nSSH 가 끊겨도 계속 달립니다. 종료는 Options 버튼, 또는 다른 터미널에서  kill {os.getpid()}"
        else:
            msg = "키보드: ↑ 전진 ↓ 후진 | Space/T 정지 | ←→ 조향 S 중앙 | A/Z 속도 | R 녹화 | M 불량 표시 | [ ] 간격 | Q/ESC 종료"
        _safe_write(msg + "\r\n")

    def _print_status(self) -> None:
        logic, st, s = self.logic, self.logic.state, self.session
        rec = "REC●" if st.recording else "----"
        dead = "" if not self.input.requires_deadman else (" L1" if st.deadman_held else " --")
        counts = " ".join(f"{a}:{s.counts.get(a, 0)}" for a in logic.steering.classes)
        bat = self.input.battery()
        bat_s = f" bat={bat}%" if bat is not None else ""
        self._write_status(f"[{rec}] a={st.angle:03d} v={logic.effective_speed:03d} {logic.effective_direction or 'stop':8s}{dead}"
                           f" | saved={s.saved} int={st.interval_sec:.1f}s | {counts}{bat_s}")

    # ------------------------------------------------------------------ 종료
    def close(self) -> None:
        super().close()
        if self.session is not None:
            dropped = f", 디스크가 느려 버린 사진 {self.session.dropped}장" if self.session.dropped else ""
            log.info("세션 종료: %s (저장 %d장, 불량 표시 %d회%s)", self.session.path, self.session.saved,
                     self.session.marks, dropped)
