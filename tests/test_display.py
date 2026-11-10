"""상태 표시 화면 생성(순수 함수)과 MAX7219 드라이버를 가짜 SPI 로 검증한다."""
import pytest

from common.config import DEFAULT_CONFIG_PATH, ConfigError, load_config
from common.display import NullDisplay, create_display
from common.display.base import DisplayState
from common.display.glyphs import ARROW_DOWN, ARROW_UP, CROSS, SQUARE, flip_horizontal, render, rotate, speed_row, steering_row


def _st(**kw):
    base = dict(speed=0, direction=None, angle=90, angle_min=30, angle_max=150, recording=False, error=False)
    base.update(kw)
    return DisplayState(**base)


def test_direction_glyphs_and_bars():
    assert render(_st(), True)[:5] == SQUARE
    assert render(_st(speed=45, direction="forward"), True)[:5] == ARROW_UP
    assert render(_st(speed=30, direction="backward"), True)[:5] == ARROW_DOWN
    assert render(_st(), True)[5] == 0                                   # 빈 줄
    assert speed_row(0) == 0 and speed_row(100) == 0xFF and speed_row(50) == 0xF0
    assert steering_row(90, 30, 150) == 0x18                             # 직진 = 가운데 두 칸
    assert steering_row(30, 30, 150) == 0xC0 and steering_row(150, 30, 150) == 0x03
    assert render(_st(speed=60, direction="forward"), True)[7] == speed_row(60)


def test_recording_blinks_corners_and_error_blinks_cross():
    on, off = render(_st(recording=True), True), render(_st(recording=True), False)
    assert on[0] & 0x81 == 0x81 and off[0] & 0x81 == 0
    assert render(_st(error=True), True) == CROSS and render(_st(error=True), False) == [0] * 8


def test_rotate_and_flip():
    rows = [0x80] + [0] * 7                       # 왼쪽 위 점 하나
    assert rotate(rows, 0) == rows
    assert rotate(rows, 90) == [0x01] + [0] * 7    # 시계 90°: 오른쪽 위
    assert rotate(rows, 180) == [0] * 7 + [0x01]   # 오른쪽 아래
    assert rotate(rows, 270) == [0] * 7 + [0x80]   # 왼쪽 아래
    assert rotate(rotate(ARROW_UP + [0] * 3, 90), 270) == ARROW_UP + [0] * 3
    assert flip_horizontal([0b10000001, 0b11000000]) == [0b10000001, 0b00000011]


class FakeSpi:
    def __init__(self):
        self.writes = []
        self.closed = False

    def write(self, data):
        self.writes.append(bytes(data))

    def close(self):
        self.closed = True


def test_max7219_display_writes_only_changed_frames():
    from common.display import max7219 as m
    dev = FakeSpi()
    d = m.Max7219Display("/dev/null", intensity=5, rotate_deg=0, flip=False, blink_hz=2.0, dev=dev)
    init = [w[0] for w in dev.writes]
    assert m.DISPLAYTEST in init and m.SHUTDOWN in init and dev.writes[init.index(m.INTENSITY)][1] == 5
    n0 = len(dev.writes)
    d.update(_st(speed=45, direction="forward"), now=0.0)
    assert len(dev.writes) == n0 + 8 and dev.writes[n0] == bytes([1, ARROW_UP[0]])
    d.update(_st(speed=45, direction="forward"), now=0.1)     # 같은 프레임이면 다시 쓰지 않는다
    assert len(dev.writes) == n0 + 8
    d.update(_st(speed=45, direction="forward", recording=True), now=0.0)   # 깜빡임 on 단계: 모서리 켜짐
    assert dev.writes[-8][1] & 0x81 == 0x81
    d.close()
    assert dev.closed and dev.writes[-1] == bytes([m.SHUTDOWN, 0])


def test_create_display_falls_back_to_null(tmp_path):
    cfg = load_config(DEFAULT_CONFIG_PATH).display
    assert isinstance(create_display(cfg, mock=True), NullDisplay)
    cfg.spi_dev = str(tmp_path / "no-such-spidev")
    assert isinstance(create_display(cfg, mock=False), NullDisplay)   # 장치가 없으면 경고 후 표시 없음
    cfg.enabled = False
    assert isinstance(create_display(cfg), NullDisplay)


def test_display_config_validation(tmp_path):
    import yaml
    base = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    assert base["display"]["enabled"] is True and base["display"]["spi_dev"] == "/dev/spidev0.0"
    for key, bad in (("rotate", 45), ("intensity", 16), ("driver", "ws2812"), ("blink_hz", 0)):
        cfg = yaml.safe_load(yaml.safe_dump(base))
        cfg["display"][key] = bad
        p = tmp_path / f"{key}.yaml"
        p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(p)
