"""자율주행 상태 규칙 (inference/state.py). 입출력이 없는 순수한 규칙이라 어디서나 돈다.

가장 중요한 것은 "끄는 길": 자율주행은 손을 놓아도 차가 달리므로, 꺼져야 할 때 반드시 꺼지는지를 본다.
"""
import pytest

from common.config import DEFAULT_CONFIG_PATH, load_config
from common.events import EventKind as K, InputEvent as E
from common.hw.base import BACKWARD, FORWARD
from inference.state import AUTO, MANUAL, TAKEOVER_STICK, AutoLogic, auto_speeds


def _logic(requires_deadman=False, **kw):
    cfg = load_config(DEFAULT_CONFIG_PATH)
    edit = kw.pop("edit", None)
    if edit:
        edit(cfg)
    return AutoLogic(cfg, requires_deadman=requires_deadman, **kw)


def _auto(logic, angle=120, now=10.0):
    """예측을 하나 주고 ○ 를 눌러 자율주행을 켠다."""
    logic.set_prediction(angle, now)
    logic.apply(E(K.RECORD_TOGGLE), now)
    assert logic.auto, logic.note
    return logic


def _out(logic):
    return logic.angle, logic.effective_speed, logic.effective_direction


# ---------------------------------------------------------------- 켜기와 끄기
def test_starts_in_manual_and_drives_like_the_collector():
    lg = _logic()
    assert lg.mode == MANUAL and _out(lg) == (90, 0, None)
    lg.apply(E(K.THROTTLE_AXIS, 1.0), 0.0)
    lg.apply(E(K.STEER_AXIS, -1.0), 0.0)
    assert _out(lg) == (30, 60, FORWARD)                       # 수집기와 같은 규칙: R2 끝까지 = 가장 빠른 단계
    lg.apply(E(K.THROTTLE_AXIS, 0.0), 0.1)
    assert lg.effective_speed == 0


def test_circle_turns_autonomous_driving_on_and_off():
    lg = _auto(_logic(), angle=120)
    assert _out(lg) == (120, 30, FORWARD)                      # 모델의 각도, 가장 느린 단계로 앞으로
    assert lg.changes[-1][:2] == ("auto_on", "speed=30") and lg.starts == 1
    lg.set_prediction(60, 10.05)
    assert lg.angle == 60                                      # 새 예측이 곧 조향
    lg.apply(E(K.RECORD_TOGGLE), 10.1)
    assert lg.mode == MANUAL and _out(lg) == (90, 0, None)     # 끄면 선다. 조향은 패드의 것(중앙)으로
    assert lg.changes[-1] == ("auto_off", "button", "자율주행 끝: ○ 버튼")       # (종류, 기록용 코드, 사람이 읽는 글)
    assert lg.state.recording is False                         # ○ 는 수집기의 '촬영'으로 넘어가지 않는다


def test_cannot_start_without_a_fresh_prediction_or_while_driving_by_hand():
    lg = _logic()
    lg.apply(E(K.RECORD_TOGGLE), 5.0)
    assert not lg.auto and "예측" in lg.note and lg.changes[-1][:2] == ("refused", "no_prediction")
    lg.set_prediction(90, 5.0)
    lg.apply(E(K.RECORD_TOGGLE), 5.0 + lg.max_age + 0.1)       # 그 예측이 너무 오래됐다 (카메라가 멈췄을 수 있다)
    assert not lg.auto

    lg.set_prediction(90, 6.0)
    lg.apply(E(K.THROTTLE_AXIS, 1.0), 6.0)                     # R2 를 당긴 채로 ○
    lg.apply(E(K.RECORD_TOGGLE), 6.0)
    assert not lg.auto and "R2" in lg.note and lg.effective_speed == 60 and lg.changes[-1][1] == "driving"
    lg.apply(E(K.THROTTLE_AXIS, 0.0), 6.1)
    lg.apply(E(K.RECORD_TOGGLE), 6.1)
    assert lg.auto


def test_a_pad_whose_link_cannot_be_watched_never_starts_autonomous_driving():
    """패드가 조용히 끊겼을 때 알 방법이 없으면(hidraw 감시 불가) 손을 놓고 달리는 모드는 켜지 않는다."""
    lg = _logic(block_reason="패드 연결을 감시할 수 없습니다")
    lg.set_prediction(90, 1.0)
    lg.apply(E(K.RECORD_TOGGLE), 1.0)
    assert not lg.auto and "감시" in lg.note and lg.changes[-1][:2] == ("refused", "input_not_watched")
    lg.apply(E(K.THROTTLE_AXIS, 1.0), 1.1)                     # 수동 운전은 된다
    assert lg.effective_speed == 60


# ---------------------------------------------------------------- 수동 조작이 들어오면 바로 수동
@pytest.mark.parametrize("ev, expect, code", [
    (E(K.THROTTLE_AXIS, 1.0), (90, 60, FORWARD), "takeover:R2"),               # R2: 그대로 수동 전진이 된다
    (E(K.REVERSE_AXIS, 1.0), (90, 30, BACKWARD), "takeover:L2"),               # L2
    (E(K.STEER_AXIS, -1.0), (30, 0, None), "takeover:stick"),                  # 스틱: 꺼지고, R2 를 당기지 않았으니 선다
    (E(K.STEER_STEP, 1), (120, 0, None), "takeover:dpad"),                     # D패드 →
    (E(K.STEER_CENTER), (90, 0, None), "takeover:triangle"),                   # △
    (E(K.STOP), (90, 0, None), "takeover:key_stop"),                           # 키보드 Space
    (E(K.FORWARD_TOGGLE), (90, 45, FORWARD), "takeover:key_forward"),          # 키보드 ↑ (처음 속도 40 → 가까운 단계 45)
])
def test_any_driving_input_takes_over(ev, expect, code):
    lg = _auto(_logic(), angle=150)
    lg.apply(ev, 10.2)
    assert lg.mode == MANUAL and _out(lg) == expect
    assert lg.takeovers == 1 and lg.changes[-1][:2] == ("auto_off", code) and "수동 조작" in lg.note


def test_small_wobbles_do_not_take_over():
    """살짝 쏠린 스틱, 데드존 안의 트리거 떨림으로 자율주행이 저절로 꺼지면 안 된다."""
    lg = _auto(_logic(), angle=150)
    dz = lg.cfg.throttle.trigger_deadzone
    for ev in (E(K.STEER_AXIS, TAKEOVER_STICK - 0.01), E(K.STEER_AXIS, -0.1), E(K.STEER_AXIS, 0.0),
               E(K.THROTTLE_AXIS, dz), E(K.REVERSE_AXIS, dz / 2), E(K.THROTTLE_AXIS, 0.0), E(K.DEADMAN, True)):
        lg.apply(ev, 10.2)
        assert lg.auto and _out(lg) == (150, 30, FORWARD), ev
    lg.apply(E(K.STEER_AXIS, TAKEOVER_STICK), 10.3)
    assert not lg.auto


# ---------------------------------------------------------------- 스스로 꺼지는 경우
def test_autonomous_driving_stops_when_predictions_stop_coming():
    lg = _auto(_logic(), now=10.0)
    lg.tick(10.0 + lg.max_age - 0.01)
    assert lg.auto
    lg.tick(10.0 + lg.max_age + 0.01)                          # 가장 최근 예측의 사진이 너무 오래됐다
    assert lg.mode == MANUAL and lg.effective_speed == 0 and "예측이 끊김" in lg.note
    assert lg.changes[-1][:2] == ("auto_off", "stale_prediction")
    lg.set_prediction(90, 11.0)                                # 예측이 다시 와도 저절로 켜지지는 않는다
    lg.tick(11.0)
    assert not lg.auto


def test_pad_loss_ends_autonomous_driving_and_the_program():
    lg = _auto(_logic())
    lg.apply(E(K.DEVICE_LOST), 10.5)
    assert lg.mode == MANUAL and lg.effective_speed == 0
    assert lg.state.quit and "끊김" in lg.state.error and lg.changes[-1][:2] == ("auto_off", "pad_lost")
    lg.set_prediction(90, 10.6)
    lg.apply(E(K.RECORD_TOGGLE), 10.6)                         # 오류가 난 뒤에는 다시 켜지지 않는다
    assert not lg.auto


def test_quit_and_external_stop_leave_autonomous_driving():
    lg = _auto(_logic())
    lg.apply(E(K.QUIT), 10.5)
    assert lg.mode == MANUAL and lg.state.quit and lg.effective_speed == 0 and lg.changes[-1][1] == "quit"
    lg = _auto(_logic())
    lg.apply_stop()                                            # 카메라 오류, 추론 실패 때 앱이 부른다
    assert lg.mode == MANUAL and lg.effective_speed == 0 and lg.changes[-1][1] == "stop"


def test_what_goes_into_the_log_is_plain_ascii_and_what_people_read_is_korean():
    """기록(CSV)에는 영문 코드만 적는다: 학생이 Excel 로 열어도 깨지지 않고, 이유별로 세기 쉽다."""
    lg = _logic()
    lg.apply(E(K.RECORD_TOGGLE), 1.0)                          # 거절
    _auto(lg)
    lg.apply(E(K.LEVEL_STEP, 1), 10.1)
    lg.apply(E(K.STEER_AXIS, 1.0), 10.2)
    kinds = [c[0] for c in lg.changes]
    assert kinds == ["refused", "auto_on", "level", "auto_off"]
    assert all(code.isascii() and " " not in code for _, code, _ in lg.changes)
    assert all(text and not text.isascii() for _, _, text in lg.changes)


def test_keyboard_autonomous_driving_times_out_without_input():
    """키보드는 SSH 로 오므로 연결이 끊겨도 차가 모른다. 입력 없이 정해 둔 시간이 지나면 끈다."""
    lg = _auto(_logic(no_input_timeout=30.0), now=100.0)
    for t in (110.0, 129.9):
        lg.set_prediction(90, t)
        lg.tick(t)
        assert lg.auto
    lg.apply(E(K.LEVEL_STEP, 1), 120.0)                        # 아무 입력이나 오면 시간을 다시 잰다
    lg.set_prediction(90, 149.9); lg.tick(149.9)
    assert lg.auto
    lg.set_prediction(90, 150.1); lg.tick(150.1)
    assert not lg.auto and "입력 없이" in lg.note and lg.changes[-1][:2] == ("auto_off", "no_input")
    pad = _auto(_logic(), now=100.0)                           # 패드(연결 감시가 있다)에는 이 한도가 없다
    pad.set_prediction(90, 500.0); pad.tick(500.0)
    assert pad.auto


# ---------------------------------------------------------------- 조향과 속도
def test_prediction_is_clamped_to_this_cars_steering_range():
    def narrow(cfg):
        cfg.steering.min, cfg.steering.max, cfg.steering.classes = 40, 140, [40, 65, 90, 115, 140]
    lg = _auto(_logic(edit=narrow), angle=150)                 # 다른 차의 사진으로 학습한 모델: 150° 는 이 차의 범위 밖
    assert lg.angle == 140
    lg.set_prediction(30, 10.1)
    assert lg.angle == 40


def test_speed_level_changes_with_the_dpad_and_stays_inside_the_levels():
    lg = _logic()
    assert lg.speeds == [30, 45, 60] and lg.auto_speed == 30
    lg.apply(E(K.LEVEL_STEP, 1), 1.0)                          # 켜기 전에 미리 골라 둘 수 있다
    _auto(lg)
    assert lg.effective_speed == 45
    for _ in range(5):
        lg.apply(E(K.LEVEL_STEP, 1), 10.2)
    assert lg.auto and lg.effective_speed == 60                # 가장 빠른 단계에서 멈춘다. 꺼지지도 않는다
    for _ in range(5):
        lg.apply(E(K.LEVEL_STEP, -1), 10.3)
    assert lg.effective_speed == 30
    lg.apply(E(K.SPEED_STEP, 1), 10.4)                         # 키보드 A
    assert lg.auto and lg.effective_speed == 45


def test_first_speed_level_comes_from_config_and_levels_may_be_empty():
    def second(cfg):
        cfg.inference.speed_level = 1
    assert _logic(edit=second).auto_speed == 45

    def beyond(cfg):
        cfg.inference.speed_level = 9
    assert _logic(edit=beyond).auto_speed == 60                # 단계 수를 넘으면 가장 빠른 단계

    def continuous(cfg):
        cfg.throttle.levels = []
    lg = _logic(edit=continuous)
    assert lg.speeds == [10, 20, 30, 40, 50, 60] and auto_speeds(lg.cfg) == lg.speeds and lg.auto_speed == 10


def test_with_deadman_autonomous_driving_runs_only_while_l1_is_held():
    lg = _auto(_logic(requires_deadman=True))
    assert lg.auto and _out(lg) == (120, 0, None)              # 켜졌지만 L1 을 누르지 않아 서 있다
    lg.apply(E(K.DEADMAN, True), 10.1)
    assert _out(lg) == (120, 30, FORWARD)
    lg.apply(E(K.DEADMAN, False), 10.2)
    assert lg.auto and lg.effective_speed == 0                 # 놓으면 선다 (모드는 그대로: 다시 누르면 이어 간다)


def test_square_is_a_mark_for_the_log_and_does_not_change_the_mode():
    lg = _auto(_logic())
    lg.apply(E(K.MARK), 10.2)
    assert lg.pending_mark and lg.auto and lg.state.pending_mark is False and lg.state.notifications == []
