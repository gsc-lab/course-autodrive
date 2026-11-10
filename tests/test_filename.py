from datetime import datetime

import pytest

from common.filename import (make_dataset_image_name, make_frame_name, make_session_name, parse_frame_name,
                             try_parse_frame_name, is_session_name)


def test_roundtrip():
    ts = datetime(2026, 11, 4, 14, 30, 15, 250_000)
    name = make_frame_name(ts, 90, 60)
    assert name == "20261104_143015_250_a090_s060.jpg"
    info = parse_frame_name(name)
    assert info.timestamp == ts and info.angle == 90 and info.speed == 60


def test_zero_padding_and_sorting():
    ts = datetime(2026, 1, 1, 0, 0, 0, 5_000)
    assert make_frame_name(ts, 30, 5) == "20260101_000000_005_a030_s005.jpg"


def test_parse_rejects_garbage():
    assert try_parse_frame_name("image.jpg") is None
    with pytest.raises(ValueError):
        parse_frame_name("20261104_143015_a090_s060.jpg")


def test_range_validation():
    ts = datetime.now()
    with pytest.raises(ValueError):
        make_frame_name(ts, 181, 0)
    with pytest.raises(ValueError):
        make_frame_name(ts, 90, 101)


def test_session_and_dataset_names():
    assert make_session_name(datetime(2026, 11, 4, 14, 30)) == "20261104_1430"
    assert is_session_name("20261104_1430") and is_session_name("20261104_1430_2")
    assert not is_session_name("foo")
    assert make_dataset_image_name("20261104_1530", 7) == "20261104_1530_000007.jpg"
