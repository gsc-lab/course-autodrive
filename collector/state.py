"""주행·녹화 상태 머신. 입출력이 없어서 PC 에서 단위 테스트할 수 있다.

입력: InputEvent 와 현재 시각(monotonic)
출력: 서보 각도, 유효 속도/방향, 녹화 여부, 저장 시점 판단
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from common.config import Config
from common.events import EventKind, InputEvent
from common.hw.base import BACKWARD, FORWARD
from common.steering import SteeringModel


@dataclass
class DriveState:
    angle: int
    speed: int = 0                        # 요청 속도(듀티 %). 데드맨 전 값
    direction: Optional[str] = None       # FORWARD | BACKWARD | None
    keyboard_speed: int = 40              # 키보드 토글 모드의 설정 속도
    throttle_axis: float = 0.0
    reverse_axis: float = 0.0
    throttle_level: int = -1              # 트리거 주행 중인 속도 단계 인덱스 (-1 = 단계 없음)
    deadman_held: bool = False
    recording: bool = False
    interval_sec: float = 0.5
    quit: bool = False
    error: Optional[str] = None
    last_input_t: float = 0.0
    last_save_t: float = 0.0
    frames_saved: int = 0
    deadman_stops: int = 0
    pending_mark: bool = False            # MARK 이벤트를 앱이 처리하도록 표시
    notifications: list = field(default_factory=list)  # "record_on" | "record_off" | "mark"


class DriveLogic:
    def __init__(self, cfg: Config, requires_deadman: bool):
        self.cfg = cfg
        self.steering = SteeringModel(cfg.steering)
        self.requires_deadman = requires_deadman
        self.state = DriveState(
            angle=self.steering.center,
            keyboard_speed=self._snap_to_level(cfg.throttle.default_speed),
            interval_sec=cfg.recording.interval_sec,
        )

    # ------------------------------------------------------------------ 파생값
    @property
    def effective_speed(self) -> int:
        s = self.state
        if s.direction is None or s.speed <= 0:
            return 0
        if self.requires_deadman and not s.deadman_held:
            return 0
        return s.speed

    @property
    def effective_direction(self) -> Optional[str]:
        return self.state.direction if self.effective_speed > 0 else None

    @property
    def is_driving_forward(self) -> bool:
        return self.effective_direction == FORWARD

    def should_save(self, now: float) -> bool:
        s = self.state
        return s.recording and self.is_driving_forward and (now - s.last_save_t) >= s.interval_sec

    def mark_saved(self, now: float) -> None:
        self.state.last_save_t = now
        self.state.frames_saved += 1

    # ------------------------------------------------------------------ 이벤트
    def apply(self, ev: InputEvent, now: float) -> None:
        s, t, st = self.state, self.cfg.throttle, self.steering
        s.last_input_t = now
        k = ev.kind

        if k == EventKind.STEER_AXIS:
            s.angle = st.from_axis(float(ev.value), s.angle)
        elif k == EventKind.STEER_STEP:
            s.angle = st.step(s.angle, int(ev.value))
        elif k == EventKind.STEER_CENTER:
            s.angle = st.center

        elif k == EventKind.THROTTLE_AXIS:
            s.throttle_axis = max(0.0, min(1.0, float(ev.value)))
            self._recompute_analog()
        elif k == EventKind.REVERSE_AXIS:
            s.reverse_axis = max(0.0, min(1.0, float(ev.value)))
            self._recompute_analog()

        elif k == EventKind.FORWARD_TOGGLE:
            if s.direction == FORWARD:
                self._stop()
            else:
                s.direction, s.speed = FORWARD, min(s.keyboard_speed, t.max_speed)
        elif k == EventKind.REVERSE_TOGGLE:
            if s.direction == BACKWARD:
                self._stop()
            else:
                s.direction, s.speed = BACKWARD, min(s.keyboard_speed, t.reverse_max_speed)
        elif k == EventKind.STOP:
            self._stop()
        elif k == EventKind.SPEED_STEP:
            if t.levels:                                   # 단계가 있으면 한 단계씩 (라벨이 단계 값만 나오도록)
                i = t.levels.index(self._snap_to_level(s.keyboard_speed)) + (1 if int(ev.value) > 0 else -1)
                s.keyboard_speed = t.levels[max(0, min(len(t.levels) - 1, i))]
            else:
                s.keyboard_speed = max(0, min(t.max_speed, s.keyboard_speed + int(ev.value) * t.step))
            if s.direction == FORWARD:
                s.speed = s.keyboard_speed
            elif s.direction == BACKWARD:
                s.speed = min(s.keyboard_speed, t.reverse_max_speed)

        elif k == EventKind.DEADMAN:
            held = bool(ev.value)
            if s.deadman_held and not held:
                s.deadman_stops += 1
            s.deadman_held = held

        elif k == EventKind.RECORD_TOGGLE:
            s.recording = not s.recording
            s.notifications.append("record_on" if s.recording else "record_off")
        elif k == EventKind.MARK:
            s.pending_mark = True
            s.notifications.append("mark")
        elif k == EventKind.INTERVAL_STEP:
            r = self.cfg.recording
            s.interval_sec = round(max(r.interval_min, min(r.interval_max, s.interval_sec + int(ev.value) * 0.1)), 2)

        elif k == EventKind.QUIT:
            s.quit = True
        elif k == EventKind.DEVICE_LOST:
            s.error = "입력 장치 연결 끊김"
            s.deadman_held = False
            self._stop()
            s.quit = True

    def tick(self, now: float) -> None:
        """주기 호출. 키보드 모드(토글 주행)의 무입력 타이머.

        트리거(R2/L2)로 달리는 동안은 적용하지 않는다: 트리거를 놓으면 바로 서고, 일정하게 당기고 있으면
        입력 이벤트가 안 올 수 있어 타이머가 주행을 잘못 끊는다."""
        s = self.state
        if self.requires_deadman or s.throttle_axis > 0 or s.reverse_axis > 0:
            return
        if self.effective_speed > 0 and self.cfg.safety.deadman_sec > 0 \
                and (now - s.last_input_t) > self.cfg.safety.deadman_sec:
            self._stop()
            s.deadman_stops += 1

    def apply_stop(self) -> None:
        """외부(앱)에서 강제 정지. 카메라 오류 등."""
        self.state.throttle_axis = 0.0
        self.state.reverse_axis = 0.0
        self._stop()

    # ------------------------------------------------------------------ 내부
    def _snap_to_level(self, speed: int) -> int:
        """levels 가 있으면 가장 가까운 단계 값으로 (같은 거리면 낮은 쪽). 없으면 그대로."""
        levels = self.cfg.throttle.levels
        if not levels:
            return speed
        return min(levels, key=lambda v: (abs(v - speed), v))

    def _level_from_axis(self, axis: float, current: int) -> int:
        """트리거 값(0~1) → 속도 단계 인덱스, 정지는 -1.

        데드존을 넘은 구간을 단계 수만큼 같은 폭으로 나눈다. 이미 어떤 단계에 있으면 경계를 여유(hysteresis)만큼
        넘어야 바뀌므로, 손이 경계 근처에서 떨려도 속도 라벨이 오르내리지 않는다."""
        t = self.cfg.throttle
        n, dz = len(t.levels), t.trigger_deadzone
        if axis <= 0.0:
            return -1
        width = (1.0 - dz) / n
        margin = t.level_hysteresis * width
        if current >= 0:
            low = dz + current * width
            if (low - margin) <= axis < (low + width + margin):
                return current
        if axis <= dz:
            return -1
        return min(n - 1, int((axis - dz) / width))

    def _recompute_analog(self) -> None:
        s, t = self.state, self.cfg.throttle
        if not t.levels:                                   # 연속 모드: 트리거 깊이에 비례
            if s.throttle_axis > 0 and s.reverse_axis == 0:
                s.direction, s.speed = FORWARD, int(round(s.throttle_axis * t.max_speed))
            elif s.reverse_axis > 0 and s.throttle_axis == 0:
                s.direction, s.speed = BACKWARD, int(round(s.reverse_axis * t.reverse_max_speed))
            else:
                self._stop()
            return
        level = self._level_from_axis(s.throttle_axis, s.throttle_level)
        release = t.trigger_deadzone * (0.5 if s.direction == BACKWARD else 1.0)   # 후진도 같은 식으로 떨림 방지
        reverse_on = s.reverse_axis > release
        if level >= 0 and not reverse_on:
            s.direction, s.speed, s.throttle_level = FORWARD, t.levels[level], level
        elif reverse_on and level < 0:
            s.direction, s.speed, s.throttle_level = BACKWARD, t.reverse_max_speed, -1
        else:                                              # 둘 다 놓았거나 둘 다 당김
            self._stop()

    def _stop(self) -> None:
        self.state.speed = 0
        self.state.direction = None
        self.state.throttle_level = -1
