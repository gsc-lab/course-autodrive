import pytest

from common.events import EventKind as K
from common.inputs.ds4 import (ABS_HAT0X, ABS_RX, ABS_RY, ABS_RZ, ABS_X, ABS_Z, BTN_C, BTN_EAST, BTN_NORTH,
                               BTN_SOUTH, BTN_START, BTN_TL, BTN_TR2, BTN_WEST, EV_ABS, EV_KEY, DS4Mapper,
                               layout_for_driver, resolve_codes)

ABSINFO = {ABS_X: (0, 255), ABS_Z: (0, 255), ABS_RX: (0, 255), ABS_RY: (0, 255), ABS_RZ: (0, 255), ABS_HAT0X: (-1, 1)}


def mapper(layout, **over):
    return DS4Mapper(resolve_codes(layout, over), ABSINFO, deadzone=0.08)


def test_driver_to_layout():
    assert layout_for_driver("hid-generic") == "generic"
    assert layout_for_driver("unknown") == "generic"
    assert layout_for_driver("sony") == "sony"
    assert layout_for_driver("playstation") == "sony"


def test_sony_layout():
    m = mapper("sony")
    assert m.translate(EV_ABS, ABS_RZ, 255).kind == K.THROTTLE_AXIS
    assert m.translate(EV_ABS, ABS_Z, 128).value == pytest.approx(128 / 255)
    assert m.translate(EV_KEY, BTN_EAST, 1).kind == K.RECORD_TOGGLE
    assert m.translate(EV_KEY, BTN_WEST, 1).kind == K.MARK
    assert m.translate(EV_KEY, BTN_TL, 1).kind == K.DEADMAN
    assert m.translate(EV_KEY, BTN_START, 1).kind == K.QUIT
    # generic 전용 코드는 sony 레이아웃에서 무시
    assert m.translate(EV_KEY, BTN_C, 1) is None
    assert m.translate(EV_ABS, ABS_RY, 200) is None


def test_generic_layout():
    m = mapper("generic")
    assert m.translate(EV_ABS, ABS_RY, 255).kind == K.THROTTLE_AXIS      # R2
    assert m.translate(EV_ABS, ABS_RX, 255).kind == K.REVERSE_AXIS       # L2
    assert m.translate(EV_ABS, ABS_RZ, 200) is None                       # 오른쪽 스틱 Y 는 무시
    assert m.translate(EV_KEY, BTN_C, 1).kind == K.RECORD_TOGGLE          # ○
    assert m.translate(EV_KEY, BTN_SOUTH, 1).kind == K.MARK               # □
    assert m.translate(EV_KEY, BTN_NORTH, 1).kind == K.STEER_CENTER       # △
    assert m.translate(EV_KEY, BTN_TR2, 1).kind == K.QUIT                 # Options
    dm = m.translate(EV_KEY, BTN_WEST, 1)                                 # L1
    assert dm.kind == K.DEADMAN and dm.value is True
    assert m.translate(EV_KEY, BTN_WEST, 0).value is False


def test_stick_deadzone_and_dpad():
    m = mapper("generic")
    assert m.translate(EV_ABS, ABS_X, 128).value == 0.0
    assert m.translate(EV_ABS, ABS_X, 135).value == 0.0                   # 데드존 안
    assert m.translate(EV_ABS, ABS_X, 0).value == -1.0
    assert m.translate(EV_ABS, ABS_X, 255).value == pytest.approx(1.0)
    assert m.translate(EV_ABS, ABS_HAT0X, 1).value == 1
    assert m.translate(EV_ABS, ABS_HAT0X, -1).value == -1
    assert m.translate(EV_ABS, ABS_HAT0X, 0) is None


def test_release_and_autorepeat_ignored():
    m = mapper("sony")
    assert m.translate(EV_KEY, BTN_EAST, 0) is None
    assert m.translate(EV_KEY, BTN_EAST, 2) is None


def test_code_overrides():
    m = mapper("generic", record="0x131")        # 문자열 16진수 허용
    assert m.translate(EV_KEY, 0x131, 1).kind == K.RECORD_TOGGLE
    with pytest.raises(ValueError):
        resolve_codes("generic", {"nope": 1})
    with pytest.raises(ValueError):
        resolve_codes("xbox")
