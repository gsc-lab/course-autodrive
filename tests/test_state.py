from common.config import DEFAULT_CONFIG_PATH, load_config
from common.events import EventKind as K, InputEvent as E
from common.hw.base import BACKWARD, FORWARD
from collector.state import DriveLogic


def logic(deadman=True, levels=None, **steer):
    cfg = load_config(DEFAULT_CONFIG_PATH)
    if levels is not None:
        cfg.throttle.levels = levels
    for k, v in steer.items():
        setattr(cfg.steering, k, v)
    return DriveLogic(cfg, requires_deadman=deadman)


def test_ds4_requires_deadman_to_move():
    L = logic()
    L.apply(E(K.THROTTLE_AXIS, 1.0), 0)
    assert L.state.speed == 60 and L.effective_speed == 0
    L.apply(E(K.DEADMAN, True), 1)
    assert L.effective_speed == 60 and L.effective_direction == FORWARD
    L.apply(E(K.DEADMAN, False), 2)
    assert L.effective_speed == 0 and L.state.deadman_stops == 1


def test_save_only_when_recording_forward_and_interval():
    L = logic()
    L.apply(E(K.DEADMAN, True), 0)
    L.apply(E(K.THROTTLE_AXIS, 0.5), 0)
    assert not L.should_save(1.0)             # 녹화 꺼짐
    L.apply(E(K.RECORD_TOGGLE), 0)
    assert L.should_save(1.0)
    L.mark_saved(1.0)
    assert not L.should_save(1.3) and L.should_save(1.5)
    L.apply(E(K.REVERSE_AXIS, 0.5), 2)        # 양쪽 트리거 → 정지
    assert L.effective_speed == 0 and not L.should_save(3.0)
    L.apply(E(K.THROTTLE_AXIS, 0.0), 3)       # 후진만
    assert L.effective_direction == BACKWARD and not L.should_save(4.0)


def test_keyboard_toggle_and_speed_steps():
    L = logic(deadman=False, levels=[])                 # 단계 없음: 10 % 씩
    L.apply(E(K.FORWARD_TOGGLE), 0)
    assert L.effective_speed == 40 and L.effective_direction == FORWARD
    L.apply(E(K.SPEED_STEP, 1), 0)
    assert L.effective_speed == 50
    L.apply(E(K.SPEED_STEP, 1), 0); L.apply(E(K.SPEED_STEP, 1), 0)
    assert L.effective_speed == 60                     # max_speed 캡
    L.apply(E(K.FORWARD_TOGGLE), 0)
    assert L.effective_speed == 0
    L.apply(E(K.REVERSE_TOGGLE), 0)
    assert L.effective_direction == BACKWARD and L.effective_speed == 30   # reverse_max_speed 캡
    L.apply(E(K.STOP), 0)
    assert L.effective_speed == 0


def test_keyboard_deadman_timer():
    L = logic(deadman=False)
    L.apply(E(K.FORWARD_TOGGLE), 10.0)
    L.tick(30.0)
    assert L.effective_speed == 45                      # default_speed 40 → 가장 가까운 단계 45
    L.tick(41.0)
    assert L.effective_speed == 0 and L.state.deadman_stops == 1


def test_steering_events():
    L = logic()
    L.apply(E(K.STEER_STEP, 1), 0); assert L.state.angle == 120
    L.apply(E(K.STEER_CENTER), 0); assert L.state.angle == 90
    L.apply(E(K.STEER_AXIS, -1.0), 0); assert L.state.angle == 30


def test_interval_mark_quit_and_device_lost():
    L = logic()
    L.apply(E(K.INTERVAL_STEP, -1), 0); assert L.state.interval_sec == 0.4
    for _ in range(10):
        L.apply(E(K.INTERVAL_STEP, 1), 0)
    assert L.state.interval_sec == 1.0
    L.apply(E(K.MARK), 0); assert L.state.pending_mark and "mark" in L.state.notifications
    L.apply(E(K.DEADMAN, True), 0); L.apply(E(K.THROTTLE_AXIS, 1.0), 0)
    L.apply(E(K.DEVICE_LOST), 0)
    assert L.state.quit and L.effective_speed == 0 and L.state.error


def test_ds4_without_deadman_drives_with_trigger_only():
    """L1 없이 R2 만으로 전진, 놓으면 정지. 기본 설정이 이 모드다."""
    assert load_config(DEFAULT_CONFIG_PATH).input.ds4.deadman is False
    L = logic(deadman=False)
    L.apply(E(K.THROTTLE_AXIS, 0.2), 0)
    assert L.effective_speed == 30 and L.effective_direction == FORWARD
    L.apply(E(K.THROTTLE_AXIS, 1.0), 1)
    assert L.effective_speed == 60
    L.apply(E(K.THROTTLE_AXIS, 0.0), 2)
    assert L.effective_speed == 0 and L.effective_direction is None
    L.apply(E(K.REVERSE_AXIS, 1.0), 3)
    assert L.effective_direction == BACKWARD and L.effective_speed == 30


def test_no_input_timer_does_not_stop_a_held_trigger():
    L = logic(deadman=False)
    L.apply(E(K.THROTTLE_AXIS, 0.8), 10.0)
    L.tick(100.0)                                      # 90초 동안 새 이벤트 없음: 트리거를 일정하게 당기고 있음
    assert L.effective_speed == 60 and L.state.deadman_stops == 0


def test_trigger_is_quantized_into_speed_levels():
    """기본 단계 [30, 45, 60], 데드존 0.1: 0.1~0.4 → 30, 0.4~0.7 → 45, 0.7~1.0 → 60."""
    L = logic(deadman=False)
    seen = set()
    for i in range(0, 101):
        L.apply(E(K.THROTTLE_AXIS, i / 100), i)
        seen.add(L.effective_speed)
    assert seen == {0, 30, 45, 60}                     # 파일 이름의 속도 라벨은 단계 값만 나온다
    L = logic(deadman=False)
    L.apply(E(K.THROTTLE_AXIS, 0.05), 0); assert L.effective_speed == 0      # 데드존
    L.apply(E(K.THROTTLE_AXIS, 0.25), 1); assert L.effective_speed == 30
    L.apply(E(K.THROTTLE_AXIS, 0.55), 2); assert L.effective_speed == 45
    L.apply(E(K.THROTTLE_AXIS, 0.90), 3); assert L.effective_speed == 60


def test_speed_level_hysteresis_and_release():
    L = logic(deadman=False)
    L.apply(E(K.THROTTLE_AXIS, 0.38), 0); assert L.effective_speed == 30
    L.apply(E(K.THROTTLE_AXIS, 0.42), 1); assert L.effective_speed == 30     # 경계 0.40 을 살짝 넘어도 유지
    L.apply(E(K.THROTTLE_AXIS, 0.50), 2); assert L.effective_speed == 45
    L.apply(E(K.THROTTLE_AXIS, 0.38), 3); assert L.effective_speed == 45     # 살짝 내려와도 유지
    L.apply(E(K.THROTTLE_AXIS, 0.30), 4); assert L.effective_speed == 30
    L.apply(E(K.THROTTLE_AXIS, 0.08), 5); assert L.effective_speed == 30     # 데드존 바로 아래: 아직 주행
    L.apply(E(K.THROTTLE_AXIS, 0.02), 6); assert L.effective_speed == 0      # 충분히 놓으면 정지
    L.apply(E(K.THROTTLE_AXIS, 0.08), 7); assert L.effective_speed == 0      # 다시 출발하려면 데드존을 넘어야 한다
    L.apply(E(K.THROTTLE_AXIS, 0.0), 8); assert L.effective_speed == 0 and L.state.throttle_level == -1


def test_reverse_and_both_triggers_with_levels():
    L = logic(deadman=False)
    L.apply(E(K.REVERSE_AXIS, 0.05), 0); assert L.effective_speed == 0       # 데드존
    L.apply(E(K.REVERSE_AXIS, 0.3), 1)
    assert L.effective_direction == BACKWARD and L.effective_speed == 30    # 깊이와 무관하게 reverse_max_speed
    L.apply(E(K.THROTTLE_AXIS, 0.8), 2); assert L.effective_speed == 0       # 둘 다 당기면 정지
    L.apply(E(K.REVERSE_AXIS, 0.0), 3)
    assert L.effective_direction == FORWARD and L.effective_speed == 60


def test_keyboard_steps_through_levels():
    L = logic(deadman=False)
    L.apply(E(K.FORWARD_TOGGLE), 0); assert L.effective_speed == 45          # 40 → 가장 가까운 단계
    L.apply(E(K.SPEED_STEP, 1), 0); assert L.effective_speed == 60
    L.apply(E(K.SPEED_STEP, 1), 0); assert L.effective_speed == 60           # 최고 단계에서 멈춤
    L.apply(E(K.SPEED_STEP, -1), 0); L.apply(E(K.SPEED_STEP, -1), 0); L.apply(E(K.SPEED_STEP, -1), 0)
    assert L.effective_speed == 30


def test_continuous_throttle_when_levels_empty():
    L = logic(deadman=False, levels=[])
    L.apply(E(K.THROTTLE_AXIS, 0.5), 0); assert L.effective_speed == 30      # 0.5 × max_speed 60
    L.apply(E(K.THROTTLE_AXIS, 0.8), 1); assert L.effective_speed == 48


def test_throttle_levels_validation(tmp_path):
    import pytest
    import yaml
    from common.config import ConfigError
    base = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    for bad in ([60, 30], [30, 30, 60], [0, 30], [30, 70], "30", [30.5]):
        base["throttle"]["levels"] = bad
        p = tmp_path / "bad.yaml"; p.write_text(yaml.safe_dump(base), encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(p)
    base["throttle"]["levels"] = [25, 40, 50, 60]
    p = tmp_path / "ok.yaml"; p.write_text(yaml.safe_dump(base), encoding="utf-8")
    assert load_config(p).throttle.levels == [25, 40, 50, 60]
