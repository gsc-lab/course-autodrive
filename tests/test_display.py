"""상태 표시 화면 생성(순수 함수)과 MAX7219 드라이버를 가짜 SPI 로 검증한다."""
import pytest

from common.config import DEFAULT_CONFIG_PATH, ConfigError, load_config
from common.display import NullDisplay, create_display
from common.display.base import DisplayState
from common.display.glyphs import BLANK, CROSS, DISC, FULL, RING, flip_horizontal, render, rotate


def _st(**kw):
    return DisplayState(**kw)


def test_idle_shows_ring_and_recording_alternates_disc_and_ring():
    assert render(_st(), True) == RING and render(_st(), False) == RING           # 대기: 고정
    assert render(_st(recording=True), True) == DISC
    assert render(_st(recording=True), False) == RING                             # 꺼지는 순간이 없다
    assert all(d & r == r for d, r in zip(DISC, RING))                            # 테두리는 두 화면에 공통


def test_error_blinks_cross_fast_and_mark_flashes_full():
    assert render(_st(error=True), True, True) == CROSS and render(_st(error=True), True, False) == BLANK
    assert render(_st(recording=True, flash=True), False) == FULL
    assert render(_st(error=True, flash=True, recording=True), True, True) == CROSS   # 오류가 가장 우선


def test_rotate_and_flip():
    rows = [0x80] + [0] * 7                       # 왼쪽 위 점 하나
    assert rotate(rows, 0) == rows
    assert rotate(rows, 90) == [0x01] + [0] * 7    # 시계 90°: 오른쪽 위
    assert rotate(rows, 180) == [0] * 7 + [0x01]   # 오른쪽 아래
    assert rotate(rows, 270) == [0] * 7 + [0x80]   # 왼쪽 아래
    assert rotate(rotate(CROSS, 90), 270) == CROSS and rotate(RING, 90) == RING
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
    d.update(_st(), now=0.0)
    assert len(dev.writes) == n0 + 8 and dev.writes[n0] == bytes([1, RING[0]])
    d.update(_st(), now=0.1)                                  # 같은 프레임이면 다시 쓰지 않는다
    assert len(dev.writes) == n0 + 8
    d.update(_st(recording=True), now=0.0)                    # 깜빡임 on 단계: 꽉 찬 원
    assert [w[1] for w in dev.writes[-8:]] == DISC
    d.update(_st(recording=True), now=0.25)                   # off 단계(2 Hz): 빈 원으로 돌아온다
    assert [w[1] for w in dev.writes[-8:]] == RING
    d.update(_st(error=True), now=0.0); assert [w[1] for w in dev.writes[-8:]] == CROSS
    d.update(_st(error=True), now=0.125); assert [w[1] for w in dev.writes[-8:]] == BLANK   # 4 Hz 깜빡임
    d.close()
    assert dev.closed and dev.writes[-1] == bytes([m.SHUTDOWN, 0])


def test_max7219_keeps_cross_on_screen_when_closed_after_an_error():
    from common.display import max7219 as m
    dev = FakeSpi()
    d = m.Max7219Display("/dev/null", intensity=5, rotate_deg=90, flip=False, blink_hz=2.0, dev=dev)
    d.update(_st(recording=True), now=0.0)
    d.close(keep_error=True)
    assert dev.closed and [w[1] for w in dev.writes[-8:]] == CROSS      # 끄지 않고 X 를 남긴다
    assert bytes([m.SHUTDOWN, 0]) not in dev.writes


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
