"""자율주행 상태 규칙. 입출력이 없어서 PC 에서 단위 테스트한다 (tests/test_inference_state.py).

모드는 둘이다.
    수동(MANUAL)   패드로 직접 운전한다. 규칙은 수집기와 똑같다 (collector.state.DriveLogic 을 그대로 쓴다).
    자율(AUTO)     모델이 조향을 정하고, 속도는 정해 둔 단계 하나로 달린다.

○ 로 켜고 끈다. 수집과 달리 자율주행은 손을 놓아도 차가 달리므로, 끄는 길을 여러 개 둔다.
    - ○ 를 다시 누른다
    - R2, L2, 스틱, △, D패드 ←→ 가운데 무엇이든 조작한다 (그 조작이 그대로 수동 운전이 된다)
    - 패드 연결이 끊긴다 (프로그램도 끝난다)
    - 예측이 끊긴다: 가장 최근 예측의 사진이 inference.max_frame_age_sec 보다 오래됐다
    - 키보드로 몰 때: 입력 없이 safety.deadman_sec 가 지난다 (SSH 가 끊겨도 모르기 때문)
켤 수 없는 경우: 패드 연결을 감시할 수 없을 때(끊겨도 차가 모른다), 차가 수동으로 움직이는 중일 때, 예측이 아직 없을 때.

input.ds4.deadman 이 true 인 차는 자율주행도 L1 을 누르고 있는 동안만 달린다 (놓으면 서고, 다시 누르면 이어 간다).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from collector.state import DriveLogic
from common.config import Config
from common.events import EventKind, InputEvent
from common.hw.base import FORWARD

MANUAL, AUTO = "manual", "auto"
TAKEOVER_STICK = 0.25       # 스틱을 이만큼 넘게 밀어야 수동 조작으로 본다 (살짝 쏠린 스틱 때문에 저절로 꺼지지 않게)

# 수동 조작으로 보는 입력: (기록에 적는 이름, 화면에 보이는 이름). 기록(CSV)에는 영문 코드만 적는다: 학생이 Excel 로
# 열어도 글자가 깨지지 않고, 이유별로 세기 쉽다.
_TAKEOVER_NAMES = {
    EventKind.THROTTLE_AXIS: ("R2", "R2"), EventKind.REVERSE_AXIS: ("L2", "L2"), EventKind.STEER_AXIS: ("stick", "스틱"),
    EventKind.STEER_STEP: ("dpad", "D패드"), EventKind.STEER_CENTER: ("triangle", "△"),
    EventKind.FORWARD_TOGGLE: ("key_forward", "전진 키"), EventKind.REVERSE_TOGGLE: ("key_reverse", "후진 키"),
    EventKind.STOP: ("key_stop", "정지 키"),
}


def auto_speeds(cfg: Config) -> List[int]:
    """자율주행이 쓸 수 있는 속도 단계(듀티 %). throttle.levels 가 비어 있으면 step 간격으로 만든다."""
    t = cfg.throttle
    if t.levels:
        return list(t.levels)
    return list(range(t.step, t.max_speed + 1, t.step)) or [t.max_speed]


class AutoLogic:
    def __init__(self, cfg: Config, requires_deadman: bool, block_reason: str = "", no_input_timeout: float = 0.0):
        """block_reason: 비어 있지 않으면 자율주행을 켤 수 없다 (그 이유). no_input_timeout: 0 보다 크면 입력 없이
        그 시간이 지났을 때 자율주행을 끈다 (연결이 끊겨도 알 수 없는 입력 장치, 곧 키보드)."""
        self.cfg = cfg
        self.manual = DriveLogic(cfg, requires_deadman)
        self.state = self.manual.state              # quit, error, deadman_held 는 수동 쪽 상태를 함께 쓴다
        self.steering = self.manual.steering
        self.requires_deadman = requires_deadman
        self.block_reason = block_reason
        self.no_input_timeout = float(no_input_timeout)
        self.max_age = float(cfg.inference.max_frame_age_sec)

        self.mode = MANUAL
        self.speeds = auto_speeds(cfg)
        self.level = max(0, min(len(self.speeds) - 1, int(cfg.inference.speed_level)))
        self.auto_angle = self.steering.center      # 자율주행 중 차에 내는 각도
        self.pred_angle = self.steering.center      # 가장 최근 예측 (수동 중에도 계속 갱신된다)
        self.pred_t = float("-inf")                 # 그 예측에 쓴 사진의 시각
        self.note = ""                              # 가장 최근에 있었던 일 (상태줄)
        # 앱이 기록할 일들: (종류, 기록용 코드, 사람이 읽는 글). 종류는 auto_on | auto_off | refused | level
        self.changes: list = []
        self.pending_mark = False
        self.starts = 0
        self.takeovers = 0

    # ------------------------------------------------------------------ 출력 (차에 낼 값)
    @property
    def auto(self) -> bool:
        return self.mode == AUTO

    @property
    def angle(self) -> int:
        return self.auto_angle if self.auto else self.state.angle

    @property
    def auto_speed(self) -> int:
        return self.speeds[self.level]

    @property
    def effective_speed(self) -> int:
        if not self.auto:
            return self.manual.effective_speed
        if self.requires_deadman and not self.state.deadman_held:
            return 0
        return self.auto_speed

    @property
    def effective_direction(self) -> Optional[str]:
        if not self.auto:
            return self.manual.effective_direction
        return FORWARD if self.effective_speed > 0 else None

    # ------------------------------------------------------------------ 예측
    def set_prediction(self, angle: float, frame_t: float) -> None:
        """새 사진의 예측. 차의 조향 범위(steering.min~max) 안으로 맞춰서 쓴다."""
        self.pred_angle = self.steering.clamp(angle)
        self.pred_t = frame_t
        if self.auto:
            self.auto_angle = self.pred_angle

    # ------------------------------------------------------------------ 입력
    def apply(self, ev: InputEvent, now: float) -> None:
        s, k = self.state, ev.kind
        s.last_input_t = now
        if k == EventKind.RECORD_TOGGLE:                    # ○: 자율주행 켜기 / 끄기
            self._toggle(now)
            return
        if k == EventKind.MARK:                             # □: 지금 순간을 기록에 표시
            self.pending_mark = True
            return
        if k == EventKind.INTERVAL_STEP:
            return
        if k in (EventKind.LEVEL_STEP, EventKind.SPEED_STEP):
            self._step_level(int(ev.value))
            if k == EventKind.LEVEL_STEP:
                return                                      # SPEED_STEP(키보드 A/Z)은 수동 운전의 속도도 바꾼다
        if self.auto and self._is_takeover(ev):
            self.takeovers += 1
            code, name = _TAKEOVER_NAMES[k]
            self._leave(f"takeover:{code}", f"수동 조작 ({name})")
        self.manual.apply(ev, now)
        if self.auto and (s.error or s.quit):               # 패드 끊김, 종료
            self._leave("pad_lost" if k == EventKind.DEVICE_LOST else "quit", s.error or "종료")

    def tick(self, now: float) -> None:
        self.manual.tick(now)
        if not self.auto:
            return
        if now - self.pred_t > self.max_age:
            self._leave("stale_prediction", f"예측이 끊김 (가장 최근 사진이 {now - self.pred_t:.1f}초 전)")
        elif self.no_input_timeout > 0 and now - self.state.last_input_t > self.no_input_timeout:
            self._leave("no_input", f"입력 없이 {self.no_input_timeout:g}초")

    def apply_stop(self) -> None:
        """외부(앱)에서 강제 정지. 카메라 오류, 추론 실패 등."""
        if self.auto:
            self._leave("stop", "정지")
        self.manual.apply_stop()

    # ------------------------------------------------------------------ 내부
    def _is_takeover(self, ev: InputEvent) -> bool:
        k = ev.kind
        if k in (EventKind.THROTTLE_AXIS, EventKind.REVERSE_AXIS):
            return float(ev.value) > self.cfg.throttle.trigger_deadzone
        if k == EventKind.STEER_AXIS:
            return abs(float(ev.value)) >= TAKEOVER_STICK
        return k in (EventKind.STEER_STEP, EventKind.STEER_CENTER, EventKind.FORWARD_TOGGLE,
                     EventKind.REVERSE_TOGGLE, EventKind.STOP)

    def start_blocker(self, now: float) -> Optional[Tuple[str, str]]:
        """지금 자율주행을 켤 수 없는 이유 (기록용 코드, 사람이 읽는 글). 켤 수 있으면 None."""
        s, dz = self.state, self.cfg.throttle.trigger_deadzone
        if self.block_reason:
            return "input_not_watched", self.block_reason
        if s.error:
            return "error", s.error
        if self.manual.effective_speed > 0 or s.direction is not None or s.throttle_axis > dz or s.reverse_axis > dz:
            return "driving", "차를 세우고 R2·L2 를 놓은 뒤 누르세요"
        if now - self.pred_t > self.max_age:
            return "no_prediction", "카메라 사진의 예측이 아직 없습니다"
        return None

    def _toggle(self, now: float) -> None:
        if self.auto:
            self._leave("button", "○ 버튼")
            return
        blocker = self.start_blocker(now)
        if blocker:
            self.note = "자율주행을 켤 수 없음: " + blocker[1]
            self.changes.append(("refused", blocker[0], self.note))
            return
        self.mode = AUTO
        self.auto_angle = self.pred_angle
        self.starts += 1
        self.note = f"자율주행 시작 (속도 {self.auto_speed} %)"
        self.changes.append(("auto_on", f"speed={self.auto_speed}", self.note))

    def _leave(self, code: str, reason: str) -> None:
        self.mode = MANUAL
        self.note = "자율주행 끝: " + reason
        self.changes.append(("auto_off", code, self.note))

    def _step_level(self, direction: int) -> None:
        self.level = max(0, min(len(self.speeds) - 1, self.level + (1 if direction > 0 else -1)))
        self.note = f"자율주행 속도 {self.level + 1}/{len(self.speeds)} 단계 ({self.auto_speed} %)"
        self.changes.append(("level", f"speed={self.auto_speed}", self.note))
