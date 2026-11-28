"""차를 움직이는 프로그램(수집기, 추론기)의 공통 뼈대.

    시작 전 점검 → 장치 열기(카메라, 차량, 표시, 입력) → 주 루프(50 Hz) → 정지하고 종료

안전장치는 이 파일 한 곳에 있고 두 프로그램이 그대로 물려받는다 (2026-11-10, 2026-11-17 에 넣은 것들):
  - 화면 출력은 절대 멈추지 않는다 (_safe_write, DroppingStderrHandler). 끊긴 채 매달린 SSH 에 쓰다 주 루프가 서면
    PCA9685 는 마지막 명령을 유지해 차가 계속 달린다.
  - 감시 스레드: 주 루프가 WATCHDOG_SEC 이상 멈추면 모터부터 끈다.
  - 카메라가 CAMERA_STALE_SEC 동안 새 사진을 주지 못하면 정지한다.
  - SIGTERM / SIGINT 는 정지 후 종료. SIGHUP(SSH 가 닫힘)은 패드로 조종할 때만 무시한다.
  - 종료할 때는 하드웨어 정지가 가장 먼저다.
프로그램마다 다른 것은 한 주기에 할 일(_step)과 화면 표시뿐이다.

주행 로직(self.logic)이 갖춰야 하는 것: state.quit, state.error, state.angle, effective_speed, effective_direction, apply_stop().
"""
from __future__ import annotations

import logging
import os
import select
import signal
import sys
import threading
import time
from typing import List, Optional, Tuple

from common.camera import BaseCamera, create_camera
from common.config import Config
from common.display import Display, DisplayState, NullDisplay, create_display
from common.hw import Vehicle, create_vehicle
from common.hw.checks import CheckResult, check_disk, check_i2c_bus, check_input_group, check_module, check_pca9685
from common.inputs import InputDevice, create_input
from common.steering import SteeringModel

log = logging.getLogger(__name__)

LOOP_HZ = 50
STATUS_HZ = 5
WATCHDOG_SEC = 1.0      # 주 루프가 이보다 오래 멈추면 감시 스레드가 모터를 끈다 (PCA9685 는 스스로 멈추지 않는다)
CAMERA_STALE_SEC = 2.0  # 카메라가 이만큼 새 프레임을 못 주면 고장으로 보고 정지한다
SLOW_STEP_SEC = 0.25    # 한 주기가 이보다 오래 걸리면 경고로 남긴다 (정상은 0.02초. 감속 정지가 0.1~0.2초를 쓴다)
STALL_SAMPLE_SEC = 0.15  # 주 루프가 이만큼 한 주기에 머물러 있으면 감시 스레드가 그 위치를 적어 둔다


def _where(frame, depth: int = 4) -> str:
    """스레드가 지금 있는 곳을 한 줄로: 안쪽 함수부터 '파일:줄 함수 ← …'. 파일을 읽지 않는다 (감시 스레드에서 부른다)."""
    parts = []
    while frame is not None and len(parts) < depth:
        code = frame.f_code
        parts.append(f"{os.path.basename(code.co_filename)}:{frame.f_lineno} {code.co_name}")
        frame = frame.f_back
    return " ← ".join(parts)


class PreflightError(RuntimeError):
    pass


def _warn(r: CheckResult) -> CheckResult:
    """실패를 경고로 낮춘다 (대체 수단이 있을 때)."""
    return CheckResult(r.name, None, "[경고] " + r.message) if r.ok is False else r


def preflight(cfg: Config, mock: bool, input_kind: str, ds4_soft: bool, skip: bool,
              logger: Optional[logging.Logger] = None) -> List[CheckResult]:
    """시작 전 점검. 실패 항목이 있으면 PreflightError.

    ds4_soft: 패드 관련 점검 실패를 경고로만 남긴다 (키보드 대체가 허용된 경우).
    """
    logger = logger or log
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
        logger.info("점검 %-14s %s  %s", r.name, {True: "OK", False: "FAIL", None: "--"}[r.ok], r.message)
    if failed:
        raise PreflightError("시작 전 점검 실패: " + "; ".join(f"{r.name}: {r.message}" for r in failed))
    return results


def _fd_writable(fd: int) -> bool:
    """지금 fd 에 쓰면 멈추지 않는지. 아무도 읽지 않는 터미널(끊긴 채 매달린 SSH)이면 False."""
    if not hasattr(select, "poll"):                       # Windows: 판단할 수 없으니 쓴다
        return True
    p = select.poll()
    p.register(fd, select.POLLOUT)
    return bool(p.poll(0))


class DroppingStderrHandler(logging.Handler):
    """화면(stderr)용 로그 핸들러. 쓸 수 없으면 그 줄을 버린다.

    보통의 StreamHandler 는 끊긴 SSH 에서 영원히 멈춘다. 주 루프가 경고 한 줄을 쓰다 멈추면 차는 마지막 명령으로
    계속 달리고, 감시 스레드도 같은 핸들러의 잠금을 기다리느라 멈춘다. 파일 로그는 그대로 남는다.
    """

    def __init__(self, fd: int = 2, level: int = logging.NOTSET):
        super().__init__(level)
        self.fd = fd
        try:
            self._tty = os.isatty(fd)
        except OSError:
            self._tty = False

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = self.format(record)
            # 터미널이면 줄 맨 앞으로 가서 그 줄을 지우고 쓴다. 같은 줄에 상태줄이 그려져 있어, 그대로 쓰면 경고가
            # 상태줄 뒤에 붙어 나온다 (2026-11-24 실차 시험). 상태줄은 다음 주기에 아랫줄에 다시 그려진다.
            data = (("\r\x1b[K" + text + "\r\n") if self._tty else (text + "\n")).encode(errors="replace")
            if _fd_writable(self.fd):
                os.write(self.fd, data)
        except Exception:                                 # 로그 때문에 주행이 영향을 받으면 안 된다
            pass


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
        if not _fd_writable(fd):
            return                                        # 아무도 읽지 않는 터미널: 상태줄을 버린다
        os.write(fd, text.encode(errors="replace"))
    except (OSError, ValueError):                         # EIO/EPIPE: 터미널이 아예 사라짐
        pass


class DriveApp:
    """수집기와 추론기의 부모. 하위 클래스는 _make_logic() 과 _step() 을 채운다."""

    title = "주행"                      # 로그 문구에 쓰는 프로그램 이름
    log = log                           # 하위 클래스가 자기 로거로 바꾼다

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
        preflight(cfg, mock or cfg.hardware.backend == "mock", input_kind, ds4_soft, skip_checks, self.log)

        self.camera: Optional[BaseCamera] = None
        self.vehicle: Optional[Vehicle] = None
        self.display: Optional[Display] = None
        self.input: Optional[InputDevice] = None
        steering = SteeringModel(cfg.steering)
        self._flash_until = 0.0            # □ 를 누른 뒤 도트 매트릭스를 번쩍이는 마감 시각
        self._last_tick = time.monotonic()
        self._main_thread = threading.get_ident()
        self._stall_where: Optional[str] = None    # 주 루프가 오래 머문 곳 (감시 스레드가 적고, 주 루프가 알린 뒤 지운다)
        self.slow_steps = 0                # SLOW_STEP_SEC 보다 오래 걸린 주기 수와 그 가운데 가장 긴 것(초)
        self.slowest_step = 0.0
        self._expected_pause = 0.0         # 이번 주기에 일부러 쓰는 시간 (감속 정지). '주 루프 멈춤'으로 세지 않는다
        self._stop_signal: Optional[int] = None    # 준비하는 동안 받은 종료 신호 (_install_startup_signals)
        self._install_startup_signals()
        try:
            # 터미널 모드를 바꾸는 입력 장치는 마지막에 연다 — 앞 단계가 실패해도 터미널이 망가지지 않게
            self.camera = create_camera(cfg.camera, mock=mock)
            self._stop_if_asked()
            self.vehicle = create_vehicle(cfg.hardware, steering.center, (steering.min, steering.max))
            self._stop_if_asked()
            self.display = create_display(cfg.display, mock=mock)     # 못 열면 NullDisplay (주행은 계속)
            self._before_input()
            self._stop_if_asked()          # 추론기가 엔진을 만드는 수십 초 동안 온 Ctrl+C / kill 은 여기서 처리된다
            self.input = create_input(cfg.input, override)
            self.logic = self._make_logic()
            self.log.info("카메라 %s | 차량 %s | 표시 %s | 입력 %s", self.camera.props(), cfg.hardware.backend,
                          self.display.name, self.input.name)
            if check_only:
                return
            self._open_outputs()
            self._stop_if_asked()
        except BaseException:              # Ctrl+C 나 kill 로 끊겨도 (KeyboardInterrupt, SystemExit) 연 장치는 놓고 끝낸다
            self._release_resources()
            raise

        self._applied_angle = steering.center
        self._applied_speed = 0
        self._applied_dir = None
        self._status_last = 0.0
        self._status_width = 0
        self._install_signals()
        if self._stop_signal is not None:  # 마지막 확인과 신호 처리기를 바꿔 거는 사이에 온 신호: 루프가 돌자마자 끝난다
            self.logic.state.quit = True

    # ------------------------------------------------------------------ 하위 클래스가 채우는 것
    def _make_logic(self):
        """주행 로직을 만든다 (입력 장치가 열린 뒤에 불린다)."""
        raise NotImplementedError

    def _step(self, now: float) -> None:
        """주 루프의 한 주기."""
        raise NotImplementedError

    def _before_input(self) -> None:
        """카메라, 차량, 표시 장치를 연 뒤, 입력 장치를 열기 전에 할 일 (추론기: 모델 열기)."""

    def _open_outputs(self) -> None:
        """기록 파일을 연다. 점검 모드(check_only)에서는 불리지 않는다."""

    def _extra_release_steps(self) -> list:
        """종료할 때 장치를 닫은 뒤에 더 할 일들 (함수 목록)."""
        return []

    def _print_help(self) -> None:
        pass

    def _target(self) -> Tuple[int, int, Optional[str]]:
        """지금 차에 내야 할 (서보 각도, 속도 듀티 %, 방향)."""
        logic = self.logic
        return logic.state.angle, logic.effective_speed, logic.effective_direction

    # ------------------------------------------------------------------ 신호
    def _install_startup_signals(self) -> None:
        """장치를 여는 동안의 종료 신호 (추론기는 여기서 엔진을 만드느라 수십 초를 쓴다).

        이때는 주행 로직이 아직 없어 '정지 후 종료'를 걸 수 없다. 그대로 두면 kill 한 번에 정리 없이 죽어, 도트 매트릭스에
        '준비 중' 화면이 켜진 채 남고 서보에도 신호가 계속 간다. 그래서 신호를 받으면 **표시만 해 두고**, 준비 단계 사이에서
        확인해(_stop_if_asked) __init__ 의 정리 코드로 간다. 준비가 끝나면 _install_signals 가 '정지 후 종료'로 다시 건다.

        처리기에서 바로 예외를 던지면 안 된다 (Ctrl+C 를 파이썬에 맡겨 두는 것도 같다): 엔진을 만드는 동안 TensorRT 는
        파이썬 함수(로그 받기)를 부르는데, 그 안에서 난 예외는 TensorRT 가 삼켜 버린다. 2026-11-23 보드에서 실제로
        kill 이 무시되고 프로그램이 그대로 시작했다. 대신 하던 단계가 끝날 때까지는 기다려야 한다.
        """
        def handler(signum, frame):
            self._stop_signal = signum
            _safe_write("\r\n종료 요청을 받았습니다. 하던 준비가 끝나는 대로 멈춥니다 …\r\n")
        for name in ("SIGINT", "SIGTERM", "SIGHUP"):
            if hasattr(signal, name):
                try:
                    signal.signal(getattr(signal, name), handler)
                except (ValueError, OSError):
                    pass

    def _stop_if_asked(self) -> None:
        """준비하는 동안 종료 신호를 받았으면 여기서 멈춘다 (Ctrl+C 는 KeyboardInterrupt, 나머지는 SystemExit)."""
        sig = self._stop_signal
        if sig is None:
            return
        if sig == getattr(signal, "SIGINT", None):
            raise KeyboardInterrupt
        raise SystemExit(128 + int(sig))

    def _install_signals(self) -> None:
        def handler(signum, frame):
            self.log.warning("시그널 %s 수신 → 정지 후 종료", signum)
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
        self._last_tick = started = time.monotonic()
        self._main_thread = threading.get_ident()
        threading.Thread(target=self._watchdog, name="watchdog", daemon=True).start()
        try:
            while not st.quit:
                t0 = time.monotonic()
                # 앞 주기(할 일 + 쉬는 시간)가 오래 걸렸는지. 감속 정지에 쓴 시간은 뺀다: 속도 60 % 에서 서는 데만 0.24초다
                if t0 - started > SLOW_STEP_SEC + self._expected_pause:
                    self._slow_step(t0 - started)
                elif self._stall_where is not None:     # 알릴 만큼 길지 않았던 머무름(감속 정지 등)의 흔적은 지운다
                    self._stall_where = None
                self._expected_pause = 0.0
                started = t0
                self._step(t0)
                self._last_tick = time.monotonic()
                dt = self._last_tick - t0
                if dt < period:
                    time.sleep(period - dt)
        except Exception:
            self.log.exception("%s 루프 예외", self.title)
            st.error = st.error or "예외 발생"
            st.quit = True                              # 감시 스레드도 끝낸다
        finally:
            self.close()
        if st.error:
            self.log.error("종료 원인: %s", st.error)
            return 2
        return 0

    # ------------------------------------------------------------------ 감시
    def _watchdog(self) -> None:
        """주 루프가 어디선가 멈추면(터미널 쓰기, 카메라, 장치 호출 …) 모터만이라도 끈다."""
        while not self.logic.state.quit:
            time.sleep(0.2)
            now = time.monotonic()
            self._sample_stall(now)
            self._watchdog_check(now)

    def _sample_stall(self, now: float) -> None:
        """주 루프가 한 주기에 오래 머물러 있으면 지금 어디에 있는지 적어 둔다. 여기서는 아무것도 쓰지 않는다
        (감시 스레드는 막히면 안 된다). 알리는 것은 주 루프가 돌아온 뒤의 _slow_step, 또는 비상 정지의 로그다."""
        if now - self._last_tick <= STALL_SAMPLE_SEC:
            return
        frame = sys._current_frames().get(self._main_thread)
        if frame is not None:
            self._stall_where = _where(frame)

    def _slow_step(self, seconds: float) -> None:
        """한 주기가 SLOW_STEP_SEC 보다 오래 걸렸다. 그동안 차는 직전 명령으로 움직였다.

        감시 한계(WATCHDOG_SEC)보다 짧으면 정지로 이어지지 않아 아무도 모른 채 지나간다. 2026-11-24 실차 시험에서
        자율주행 중에 0.68초가 그렇게 지나갔고, 어디서 걸렸는지 기록이 없어 원인을 찾지 못했다. 그래서 위치와 함께 남긴다.
        """
        where, self._stall_where = self._stall_where, None
        self.slow_steps += 1
        self.slowest_step = max(self.slowest_step, seconds)
        self.log.warning("주 루프가 %.2f초 멈췄습니다 (멈춰 있던 곳: %s)", seconds, where or "잡지 못함")

    def _watchdog_check(self, now: float) -> bool:
        """주 루프가 WATCHDOG_SEC 이상 틱을 못 찍었고 모터가 돌고 있으면 즉시 정지하고 종료를 건다. 정지했으면 True."""
        if now - self._last_tick <= WATCHDOG_SEC or self.vehicle is None or self.vehicle.state.speed == 0:
            return False
        st = self.logic.state
        failure = None
        try:
            self.vehicle.stop(smooth=False)             # 가장 먼저. 아래의 로그 쓰기는 끊긴 터미널에서 막힐 수 있다
        except Exception as e:
            failure = e
        st.error = st.error or "주 루프 멈춤 → 비상 정지"
        st.quit = True
        self.log.error("주 루프가 %.1f초 멈춤 → 감시 스레드가 비상 정지 (멈춰 있던 곳: %s)", now - self._last_tick,
                       self._stall_where or "잡지 못함")
        if failure is not None:
            self.log.error("비상 정지 실패: %s", failure)
        return True

    def _camera_failed(self, now: float) -> bool:
        """카메라가 고장 났거나 멈췄으면 정지하고 종료를 건다. 그렇게 했으면 True."""
        logic, st = self.logic, self.logic.state
        if not (self.camera.error or self.camera.age(now) > CAMERA_STALE_SEC) or st.error:
            return False
        st.error = "카메라 프레임 수신 실패"
        self.log.error(st.error)
        logic.apply_stop()
        st.quit = True
        return True

    def _apply_vehicle(self) -> None:
        angle, speed, direction = self._target()
        self.vehicle.service()                      # 조향 유격 보정의 두 번째 단계 등
        if angle != self._applied_angle:
            self.vehicle.set_steering(angle)
            self._applied_angle = angle
        if speed != self._applied_speed or direction != self._applied_dir:
            if self._applied_speed > 0 and (speed == 0 or direction != self._applied_dir):
                # 달리다 서거나 방향을 바꿀 때 차량 드라이버는 듀티를 단계마다 쉬어 가며 내린다 (그동안 주 루프가 기다린다).
                h = self.cfg.hardware
                self._expected_pause = -(-self._applied_speed // max(1, h.stop_decay_step)) * h.stop_decay_delay
            if speed == 0:
                self.vehicle.stop(smooth=self._applied_speed > 0)
            else:
                self.vehicle.set_throttle(speed, direction)
            self._applied_speed, self._applied_dir = speed, direction

    # ------------------------------------------------------------------ 표시
    def _show(self, state: DisplayState, now: float) -> None:
        try:
            self.display.update(state, now)
        except OSError as e:                    # 표시 장치 통신 오류는 주행을 막지 않는다
            self.log.warning("상태 표시 오류, 표시를 끕니다: %s", e)
            self.display = NullDisplay()

    def _status_due(self, now: float) -> bool:
        """상태줄을 다시 쓸 때가 됐는지 (STATUS_HZ)."""
        if now - self._status_last < 1.0 / STATUS_HZ:
            return False
        self._status_last = now
        return True

    def _write_status(self, line: str) -> None:
        """한 줄 상태를 같은 자리에 덮어쓴다 (앞 줄이 더 길었으면 남은 글자를 지운다)."""
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
        if self.display is not None:              # 패드 끊김·오류로 끝나면 X 를 켜 둔 채 닫는다
            logic = getattr(self, "logic", None)
            failed = bool(logic is not None and logic.state.error)
            steps.append(lambda: self.display.close(keep_error=failed))
        if self.input is not None:
            steps.append(self.input.close)
        steps += self._extra_release_steps()
        for step in steps:
            try:
                step()
            except Exception as e:  # 종료 중 오류는 로그만
                self.log.warning("종료 단계 오류: %s", e)

    def close(self) -> None:
        self._release_resources()
        _safe_write("\r\n")
