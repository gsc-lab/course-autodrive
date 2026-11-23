import pytest

from common.config import SteeringConfig
from common.steering import SteeringModel


def discrete(**kw):
    return SteeringModel(SteeringConfig(**kw))


def test_axis_discrete_snaps_to_classes():
    m = discrete()
    assert m.from_axis(0.0, 90) == 90
    assert m.from_axis(-1.0, 90) == 30
    assert m.from_axis(1.0, 90) == 150
    assert m.from_axis(0.5, 90) == 120


def test_hysteresis_prevents_chatter_at_boundary():
    m = discrete(hysteresis=0.15)
    # 90 과 120 의 경계는 105. 히스테리시스 4.5도.
    # 90 에서 출발: 108도 상당(x=0.3)은 아직 90 유지, 110도 상당은 120 으로
    assert m.from_axis(0.3, 90) == 90           # raw 108 < 109.5
    assert m.from_axis(0.34, 90) == 120         # raw 110.4 >= 109.5
    # 120 에서 출발: raw 102 는 아직 120 유지, raw 100 은 90 으로
    assert m.from_axis(0.2, 120) == 120         # raw 102 > 100.5
    assert m.from_axis(0.16, 120) == 90         # raw 99.6 <= 100.5


def test_far_jump_ignores_hysteresis():
    m = discrete()
    assert m.from_axis(-1.0, 150) == 30


def test_step_discrete_and_clamp():
    m = discrete()
    assert m.step(90, 1) == 120
    assert m.step(150, 1) == 150
    assert m.step(30, -1) == 30
    assert m.step(100, 1) == 120   # 클래스 밖 값은 양자화 후 이동


def test_invert_flips_logical_direction():
    m = discrete(invert=True)
    assert m.from_axis(-1.0, 90) == 150
    assert m.step(90, -1) == 120
    assert m.logical_side(150) < 0


def test_continuous_mode():
    m = SteeringModel(SteeringConfig(mode="continuous", step_deg=5))
    assert m.from_axis(0.5, 90) == 120
    assert m.from_axis(0.37, 90) == 112
    assert m.step(90, 1) == 95
    assert m.step(148, 1) == 150
    assert m.quantize(112) == 120 and m.class_index(112) == 3


def test_nearest_class_and_mirror_are_shared_helpers():
    """수집기·가공기·학습기가 같은 함수를 쓴다: 한가운데 각도는 낮은 쪽, 대칭이 아닌 클래스는 뒤집을 수 없다."""
    from common.steering import mirror_class_indices, nearest_class_index
    classes = [30, 60, 90, 120, 150]
    assert [nearest_class_index(a, classes) for a in (0, 44, 45, 46, 112, 200)] == [0, 0, 0, 1, 3, 4]
    assert discrete().quantize(45) == 30 and discrete().quantize(75) == 60
    assert mirror_class_indices(classes, center=90) == [4, 3, 2, 1, 0]
    assert mirror_class_indices([40, 90, 140]) == [2, 1, 0]                 # 직진 각도를 모르면 양 끝의 가운데
    for bad, center in (([30, 60, 90], 90), ([30, 60, 90, 100, 150], 90), (classes, 100)):
        with pytest.raises(ValueError, match="대칭"):
            mirror_class_indices(bad, center=center)


def test_custom_seven_classes():
    m = discrete(classes=[30, 50, 70, 90, 110, 130, 150])
    assert m.step(90, 1) == 110
    assert m.from_axis(0.33, 90) == 110


def test_recommend_steering_from_measured_limits():
    import pytest
    from common.steering import recommend_steering
    r = recommend_steering(52, 131, straight=90)           # 좁은 쪽(왼쪽 38°)에 맞춰 대칭, 3° 안쪽
    assert r == {"center": 90, "min": 55, "max": 125, "classes": [55, 72, 90, 108, 125]}
    r = recommend_steering(60, 120)                        # 정면 표시 안 함 → 가운데
    assert r["center"] == 90 and r["classes"] == [63, 76, 90, 104, 117]
    assert recommend_steering(40, 140, n_classes=7)["classes"] == [43, 59, 74, 90, 106, 121, 137]
    with pytest.raises(ValueError):
        recommend_steering(100, 80)
    with pytest.raises(ValueError):
        recommend_steering(88, 92, straight=90)            # 여유를 빼면 범위가 없다

