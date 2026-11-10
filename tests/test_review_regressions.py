"""2026-11-04 코드 리뷰에서 나온 결함의 회귀 테스트."""
import sys

import pytest

from common.config import DEFAULT_CONFIG_PATH, ConfigError, load_config
from common.events import EventKind as K
from common.hw.checks import CheckResult, check_frame_size, check_i2c_bus, check_module, check_pca9685
from common.inputs.base import InputDevice
from common.inputs.keyboard import keys_to_events, parse_terminal_keys


# ---------------------------------------------------------------- 키보드 ESC 시퀀스
def test_arrow_sequences():
    assert parse_terminal_keys("\x1b[A\x1b[B\x1b[C\x1b[D") == ["UP", "DOWN", "RIGHT", "LEFT"]
    assert parse_terminal_keys("\x1bOA") == ["UP"]


def test_other_escape_sequences_do_not_quit():
    # PageUp, Home, End, F5, Delete 는 통째로 버려야 한다 (ESC 로 오인하면 QUIT)
    for seq in ("\x1b[5~", "\x1b[H", "\x1b[F", "\x1b[15~", "\x1b[3~", "\x1b[1;5C"):
        assert parse_terminal_keys(seq) == [], repr(seq)
    assert parse_terminal_keys("\x1b") == ["ESC"]
    assert parse_terminal_keys("r\x1b[5~a") == ["r", "a"]


def test_ctrl_c_and_plain_keys():
    evs = keys_to_events(parse_terminal_keys("\x03"))
    assert [e.kind for e in evs] == [K.QUIT]
    assert [e.kind for e in keys_to_events(["A", "z", " "])] == [K.SPEED_STEP, K.SPEED_STEP, K.STOP]


# ---------------------------------------------------------------- I2C / PCA9685 점검
def test_i2c_checks_report_missing_bus_and_board(tmp_path, monkeypatch):
    if not sys.platform.startswith("linux"):
        r = check_i2c_bus(7)
        assert r.ok is None and "건너뜀" in r.message            # PC 에서는 판단하지 않는다
        assert check_pca9685(7, 0x41, "모터").ok is None
        return
    assert check_i2c_bus(999).ok is False                          # /dev/i2c-999 는 없다
    r = check_pca9685(999, 0x41, "모터", "A0 납땜을 본다")
    assert r.ok is False and "모터" in r.message


def test_pca9685_scan_lists_only_acking_addresses(monkeypatch):
    import common.hw.pca9685 as m

    class Dev:
        def __init__(self, bus, address):
            self.address = address

        def read(self, n):
            if self.address in (0x40, 0x41, 0x70):
                return b"\x00"
            raise OSError(121, "Remote I/O error")

        def close(self):
            pass
    monkeypatch.setattr(m, "I2CDevice", Dev)
    assert m.scan(7) == [0x40, 0x41, 0x70]
    assert m.PCA9685_ALLCALL == 0x70


# ---------------------------------------------------------------- 기타 점검 함수
def test_check_module_catches_non_import_errors(monkeypatch):
    import importlib

    def boom(name):
        raise Exception("Could not determine board model")
    monkeypatch.setattr(importlib, "import_module", boom)
    r = check_module("some.hardware.lib")
    assert r.ok is False and "board model" in r.message


def test_check_frame_size():
    assert check_frame_size((480, 640, 3), 640, 480).ok is True
    r = check_frame_size((720, 1280, 3), 640, 480)
    assert r.ok is False and "1280x720" in r.message


def test_steering_range_must_fit_servo(tmp_path):
    import yaml
    base = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    base["hardware"]["servo"]["max_deg"] = 140
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(base), encoding="utf-8")
    with pytest.raises(ConfigError, match="servo"):
        load_config(p)


# ---------------------------------------------------------------- Collector 초기화·종료 안전성
class _Boom(InputDevice):
    name = "boom"


class _Closable(InputDevice):
    name = "closable"
    closed = False

    def poll(self):
        return []

    def close(self):
        self.closed = True


def test_init_failure_releases_resources(tmp_path, monkeypatch):
    """카메라 생성이 실패하면 이미 연 자원을 닫고 예외를 다시 던진다. 입력 장치는 마지막에 열리므로 터미널은 건드리지 않는다."""
    import collector.app as app_mod
    cfg = load_config(DEFAULT_CONFIG_PATH)
    cfg.recording.root = str(tmp_path / "dataset")
    opened = []

    def bad_camera(cfg_, mock=False):
        raise RuntimeError("카메라 없음")
    monkeypatch.setattr(app_mod, "create_camera", bad_camera)
    monkeypatch.setattr(app_mod, "create_input", lambda cfg_, override: opened.append(1) or _Closable())
    with pytest.raises(RuntimeError, match="카메라 없음"):
        app_mod.Collector(cfg, mock=True, skip_checks=True)
    assert opened == []           # 입력 장치는 열리지 않았다


def test_init_failure_after_input_closes_input(tmp_path, monkeypatch):
    import collector.app as app_mod
    cfg = load_config(DEFAULT_CONFIG_PATH)
    cfg.recording.root = str(tmp_path / "dataset")
    dev = _Closable()
    monkeypatch.setattr(app_mod, "create_input", lambda cfg_, override: dev)

    def bad_session(*a, **kw):
        raise OSError("디스크 쓰기 실패")
    monkeypatch.setattr(app_mod, "SessionWriter", bad_session)
    with pytest.raises(OSError):
        app_mod.Collector(cfg, mock=True, skip_checks=True)
    assert dev.closed


def test_fallback_to_keyboard_when_ds4_missing(monkeypatch):
    """primary=ds4, keyboard_fallback=true 이고 패드가 없으면 키보드로 대체되어야 한다."""
    import common.inputs.factory as fac
    cfg = load_config(DEFAULT_CONFIG_PATH).input
    assert cfg.primary == "ds4" and cfg.keyboard_fallback

    class FakeKeyboard(InputDevice):
        name = "keyboard"

        def poll(self):
            return []
    import common.inputs.keyboard as kb
    monkeypatch.setattr(kb, "KeyboardInput", FakeKeyboard)

    class NoPad(InputDevice):                   # 패드가 실제로 연결된 보드에서도 같은 결과가 나오게 DS4 생성을 실패시킨다
        def __init__(self, cfg_):
            raise RuntimeError("DS4 없음")
    import common.inputs.ds4 as ds4
    monkeypatch.setattr(ds4, "DS4Input", NoPad)
    dev = fac.create_input(cfg, None)
    assert dev.name == "keyboard"
    with pytest.raises(Exception):
        fac.create_input(cfg, "ds4")             # 명시 지정이면 대체하지 않는다


def test_collector_passes_none_override_so_fallback_can_happen(tmp_path, monkeypatch):
    import collector.app as app_mod
    cfg = load_config(DEFAULT_CONFIG_PATH)
    cfg.recording.root = str(tmp_path / "dataset")
    cfg.hardware.backend = "mock"
    seen = {}

    def fake_create_input(cfg_, override):
        seen["override"] = override
        return _Closable()
    from common.camera import MockCamera
    monkeypatch.setattr(app_mod, "create_input", fake_create_input)
    monkeypatch.setattr(app_mod, "create_camera", lambda cfg_, mock=False: MockCamera(cfg_))
    col = app_mod.Collector(cfg, mock=False, skip_checks=True)   # mock=False 이므로 override 가 None 이어야 대체 가능
    assert seen["override"] is None
    col.close()


def test_close_survives_broken_stdout(tmp_path, monkeypatch):
    import collector.app as app_mod
    cfg = load_config(DEFAULT_CONFIG_PATH)
    cfg.recording.root = str(tmp_path / "dataset")
    monkeypatch.setattr(app_mod, "create_input", lambda cfg_, override: _Closable())
    col = app_mod.Collector(cfg, mock=True, skip_checks=True)
    col.logic.apply(__import__("common.events", fromlist=["InputEvent"]).InputEvent(K.FORWARD_TOGGLE), 0)
    col._apply_vehicle()
    assert col.vehicle.state.speed > 0

    class BrokenOut:
        def write(self, s):
            raise OSError("broken pipe")

        def flush(self):
            raise OSError("broken pipe")
    monkeypatch.setattr(sys, "stdout", BrokenOut())
    col._print_status()                 # 예외 없이 지나가야 한다
    col.close()                         # 차량 정지·하드웨어 정리까지 도달해야 한다
    assert col.vehicle.state.speed == 0 and col.vehicle.calls[-1] == ("close",)
