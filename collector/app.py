"""수집기 메인 루프: 입력 → 상태 머신 → 차량 → (조건부) 프레임 저장 → 상태 표시."""
from __future__ import annotations

import logging
import os
import select
import signal
import sys
import threading
import time
from datetime import datetime
from typing import List, Optional

from common.camera import BaseCamera, create_camera
from common.config import Config, config_snapshot
from common.display import Display, DisplayState, create_display
from common.hw import Vehicle, create_vehicle
from common.hw.checks import CheckResult, check_disk, check_i2c_bus, check_input_group, check_module, check_pca9685
from common.inputs import InputDevice, create_input
from common.inputs.base import STATUS_ERROR, STATUS_IDLE, STATUS_RECORDING
from common.session import SessionWriter
from common.steering import SteeringModel
from collector.state import DriveLogic

log = logging.getLogger("collector")

LOOP_HZ = 50
STATUS_HZ = 5
WATCHDOG_SEC = 1.0      # 주 루프가 이보다 오래 멈추면 감시 스레드가 모터를 끈다 (PCA9685 는 스스로 멈추지 않는다)


class PreflightError(RuntimeError):
    pass


def _warn(r: CheckResult) -> CheckResult:
    """실패를 경고로 낮춘다 (대체 수단이 있을 때)."""
    return CheckResult(r.name, None, "[경고] " + r.message) if r.ok is False else r


def preflight(cfg: Config, mock: bool, input_kind: str, ds4_soft: bool, skip: bool) -> List[CheckResult]:
    """시작 전 점검. 실패 항목이 있으면 PreflightError.

    ds4_soft: 패드 관련 점검 실패를 경고로만 남긴다 (키보드 대체가 허용된 경우).
    """
    results: List[CheckResult] = []
    if skip:
        return results
    results.append(check_disk(cfg.recording.root, cfg.recording.min_free_gb))
    if not mock:
        hw = cfg.hardware
        # I2C: 버스 접근 → 서보용 보드(0x40) → 모터용 보드(0x41). 어느 쪽이 빠졌는지 이름으로 구분된다
        for bus in dict.fromkeys((hw.pca9685.bus, hw.pca9685_motor.bus)):
            results.append(check_i2c_bus(bus))
        results.append(check_pca9685(hw.pca9685.bus, hw.pca9685.address, "서보",
                                     "Jetson 핀 1/3/5/6 ↔ 서보용 보드 VCC/SDA/SCL/GND 케이블(I1~I4)을 본다"))
        results.append(check_pca9685(hw.pca9685_motor.bus, hw.pca9685_motor.address, "모터",
                                     "서보용 보드만 보이면 보드 사이 케이블(C1~C4)과 A0 납땜(docs/wiring.pdf 그림 4)을 본다"))
        if input_kind == "ds4":
            for r in (check_module("evdev", "evdev"), check_input_group()):
                results.append(_warn(r) if ds4_soft else r)
    failed = [r for r in results if r.ok is False]
    for r in results:
        log.info("점검 %-14s %s  %s", r.name, {True: "OK", False: "FAIL", None: "--"}[r.ok], r.message)
    if failed:
        raise PreflightError("시작 전 점검 실패: " + "; ".join(f"{r.name}: {r.message}" for r in failed))
    return results


def _safe_write(text: str) -> None:
    """터미널에 상태를 쓰되 절대 멈추지 않는다.

    SSH 가 '끊긴 채 닫히지 않으면'(Wi-Fi 끊김, 노트북 절전) 아무도 터미널을 읽지 않아 출력 버퍼가 몇 초 만에 가득 차고,
    보통의 write 는 거기서 영원히 멈춘다. 그러면 주 루프가 서고 PCA9685 는 마지막 명령을 유지해 차가 계속 달린다
    (2026-11-10 실제 발생). 그래서 쓰기 전에 버퍼에 자리가 있는지 보고, 없으면 그 줄을 버린다.
    """
    try:
        fd = sys.stdout.fileno()
    except (AttributeError, ValueError, OSError):
        return
    try:
        if hasattr(select, "poll"):                       # POSIX
            p = select.poll()
            p.register(fd, select.POLLOUT)
            if not p.poll(0):
                return                                    # 아무도 읽지 않는 터미널: 상태줄을 버린다
        os.write(fd, text.encode(errors="replace"))
    except (OSError, ValueError):                         # EIO/EPIPE: 터미널이 아예 사라짐
        pass


class Collector:
    def __init__(self, cfg: Config, mock: bool = False, input_override: Optional[str] = None,
                 skip_checks: bool = False, check_only: bool = False):
        self.cfg = cfg
        self.mock = mock
        self.check_only = check_only
        if mock:
            cfg.hardware.backend = "mock"
        # 입력 장치: 명시 지정 > (mock 이면 키보드) > 설정의 primary (실패 시 키보드 대체 가능)
        override = input_override or ("keyboard" if mock else None)
        input_kind = override or cfg.input.primary
        ds4_soft = override is None and cfg.input.keyboard_fallback
        preflight(cfg, mock or cfg.hardware.backend == "mock", input_kind, ds4_soft, skip_checks)

        self.camera: Optional[BaseCamera] = None
        self.vehicle: Optional[Vehicle] = None
        self.display: Optional[Display] = None
        self.session: Optional[SessionWriter] = None
        self.input: Optional[InputDevice] = None
        steering = SteeringModel(cfg.steering)
        self._steering_range = (steering.min, steering.max)
        try:
            # 터미널 모드를 바꾸는 입력 장치는 마지막에 연다 — 앞 단계가 실패해도 터미널이 망가지지 않게
            self.camera = create_camera(cfg.camera, mock=mock)
            self.vehicle = create_vehicle(cfg.hardware, steering.center, (steering.min, steering.max))
            self.display = create_display(cfg.display, mock=mock)     # 못 열면 NullDisplay (주행은 계속)
            self.input = create_input(cfg.input, override)
            self.logic = DriveLogic(cfg, requires_deadman=self.input.requires_deadman)
            log.info("카메라 %s | 차량 %s | 표시 %s | 입력 %s", self.camera.props(), cfg.hardware.backend,
                     self.display.name, self.input.name)
            if check_only:
                return
            self.session = SessionWriter(
                cfg.recording.root, config_snapshot(cfg), jpeg_quality=cfg.camera.jpeg_quality,
                extra_meta={"input_device": self.input.name, "camera": self.camera.props(), "mock": mock},
            )
        except Exception:
            self._release_resources()
            raise

        self._applied_angle = steering.center
        self._applied_speed = 0
        self._applied_dir = None
        self._status_last = 0.0
        self._status_width = 0
        self._install_signals()
        log.info("세션 폴더: %s (입력: %s)", self.session.path, self.input.name)

    # ------------------------------------------------------------------ 신호
    def _install_signals(self) -> None:
        def handler(signum, frame):
            log.warning("시그널 %s 수신 → 정지 후 종료", signum)
            self.logic.state.quit = True
        for name in ("SIGTERM", "SIGINT"):
            if hasattr(signal, name):
                try:
                    signal.signal(getattr(signal, name), handler)
                except (ValueError, OSError):
                    pass
        # SIGHUP (SSH 가 닫힘): 패드로 조종할 때는 무시하고 계속 달린다 (2026-11-10 결정). 조종 신호는 블루투스로
        # Jetson 에 직접 들어오고 상태는 도트 매트릭스로 보므로 터미널이 없어도 된다. 종료는 Options 버튼 또는 kill PID.
        # 키보드 모드는 입력 자체가 SSH 로 오므로 닫히면 정지한다.
        if hasattr(signal, "SIGHUP"):
            try:
                signal.signal(signal.SIGHUP, handler if self.input.name == "keyboard" else signal.SIG_IGN)
            except (ValueError, OSError):
                pass
        self.survives_hangup = self.input.name != "keyboard"

    # ------------------------------------------------------------------ 루프
    def run(self) -> int:
        st = self.logic.state
        period = 1.0 / LOOP_HZ
        self._print_help()
        self._last_tick = time.monotonic()
        threading.Thread(target=self._watchdog, name="watchdog", daemon=True).start()
        try:
            while not st.quit:
                t0 = time.monotonic()
                self._step(t0)
                self._last_tick = time.monotonic()
                dt = self._last_tick - t0
                if dt < period:
                    time.sleep(period - dt)
        except Exception:
            log.exception("수집 루프 예외")
            st.error = st.error or "예외 발생"
        finally:
            self.close()
        if st.error:
            log.error("종료 원인: %s", st.error)
            return 2
        return 0

    # ------------------------------------------------------------------ 감시
    def _watchdog(self) -> None:
        """주 루프가 어디선가 멈추면(터미널 쓰기, 카메라, 장치 호출 …) 모터만이라도 끈다."""
        while not self.logic.state.quit:
            time.sleep(0.2)
            self._watchdog_check(time.monotonic())

    def _watchdog_check(self, now: float) -> bool:
        """주 루프가 WATCHDOG_SEC 이상 틱을 못 찍었고 모터가 돌고 있으면 즉시 정지하고 종료를 건다. 정지했으면 True."""
        if now - self._last_tick <= WATCHDOG_SEC or self.vehicle is None or self.vehicle.state.speed == 0:
            return False
        st = self.logic.state
        log.error("주 루프가 %.1f초 멈춤 → 감시 스레드가 비상 정지", now - self._last_tick)
        st.error = st.error or "주 루프 멈춤 → 비상 정지"
        st.quit = True
        try:
            self.vehicle.stop(smooth=False)
        except Exception as e:
            log.error("비상 정지 실패: %s", e)
        return True

    def _step(self, now: float) -> None:
        logic, st, s = self.logic, self.logic.state, self.session

        for ev in self.input.poll():
            s.log_event(ev.kind.value, ev.value, ev.t)
            logic.apply(ev, now)
        logic.tick(now)

        if st.pending_mark:
            st.pending_mark = False
            s.mark(self.cfg.recording.mark_seconds, note=f"frame={s.last_frame_name or ''}")
        for n in st.notifications:
            self.input.notify(n)
        st.notifications.clear()

        if self.camera.error and not st.error:
            st.error = "카메라 프레임 수신 실패"
            log.error(st.error)
            st.recording = False
            logic.apply_stop()
            st.quit = True

        self._apply_vehicle()

        if logic.should_save(now):
            frame, _ = self.camera.read()
            if frame is not None:
                s.save_frame(frame, st.angle, logic.effective_speed, datetime.now())
                logic.mark_saved(now)

        if now - self._status_last >= 1.0 / STATUS_HZ:
            self._status_last = now
            self.input.set_status(STATUS_ERROR if st.error else (STATUS_RECORDING if st.recording else STATUS_IDLE))
            self._update_display(now)
            self._print_status()

    def _update_display(self, now: float) -> None:
        logic, st = self.logic, self.logic.state
        try:
            self.display.update(DisplayState(speed=logic.effective_speed, direction=logic.effective_direction,
                                             angle=st.angle, angle_min=self._steering_range[0],
                                             angle_max=self._steering_range[1], recording=st.recording,
                                             error=bool(st.error)), now)
        except OSError as e:                    # 표시 장치 통신 오류는 주행을 막지 않는다
            log.warning("상태 표시 오류, 표시를 끕니다: %s", e)
            self.display = __import__("common.display", fromlist=["NullDisplay"]).NullDisplay()

    def _apply_vehicle(self) -> None:
        logic, st = self.logic, self.logic.state
        self.vehicle.service()                      # 조향 유격 보정의 두 번째 단계 등
        if st.angle != self._applied_angle:
            self.vehicle.set_steering(st.angle)
            self._applied_angle = st.angle
        speed, direction = logic.effective_speed, logic.effective_direction
        if speed != self._applied_speed or direction != self._applied_dir:
            if speed == 0:
                self.vehicle.stop(smooth=self._applied_speed > 0)
            else:
                self.vehicle.set_throttle(speed, direction)
            self._applied_speed, self._applied_dir = speed, direction

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
        line = (f"[{rec}] a={st.angle:03d} v={logic.effective_speed:03d} {logic.effective_direction or 'stop':8s}{dead}"
                f" | saved={s.saved} int={st.interval_sec:.1f}s | {counts}{bat_s}")
        pad = max(0, self._status_width - len(line))
        self._status_width = len(line)
        _safe_write("\r" + line + " " * pad)

    # ------------------------------------------------------------------ 종료
    def _release_resources(self) -> None:
        """하드웨어 정지가 최우선. 나머지는 실패해도 계속 진행한다."""
        steps = []
        if self.vehicle is not None:
            center = SteeringModel(self.cfg.steering).center
            steps += [lambda: self.vehicle.stop(smooth=True),
                      lambda: self.vehicle.set_steering(center),
                      self.vehicle.close]
        if self.camera is not None:
            steps.append(self.camera.close)
        if self.display is not None:
            steps.append(self.display.close)
        if self.input is not None:
            steps.append(self.input.close)
        if self.session is not None:
            steps.append(self.session.close)
        for step in steps:
            try:
                step()
            except Exception as e:  # 종료 중 오류는 로그만
                log.warning("종료 단계 오류: %s", e)

    def close(self) -> None:
        self._release_resources()
        _safe_write("\r\n")
        if self.session is not None:
            log.info("세션 종료: %s (저장 %d장, 불량 표시 %d회)", self.session.path, self.session.saved, self.session.marks)
