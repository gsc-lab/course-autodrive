"""파일 이름 규약. 수집기가 만들고 가공기가 읽는다. 이 파일 외의 곳에서 형식을 다시 정의하지 않는다.

프레임:   YYYYMMDD_HHMMSS_mmm_a{angle:03d}_s{speed:03d}.jpg   예) 20261104_143015_250_a090_s060.jpg
세션:     YYYYMMDD_HHMM                                        예) 20261104_1430  (중복 시 _2, _3 ...)
데이터셋: dataset_YYYYMMDD_HHMM.zip, 안의 이미지는 YYYYMMDD_HHMM_{index:06d}.jpg
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

FRAME_EXT = "jpg"

FRAME_RE = re.compile(
    r"^(?P<date>\d{8})_(?P<time>\d{6})_(?P<ms>\d{3})_a(?P<angle>\d{3})_s(?P<speed>\d{3})\.(?P<ext>jpe?g|png)$",
    re.IGNORECASE,
)
SESSION_RE = re.compile(r"^(?P<date>\d{8})_(?P<time>\d{4})(?:_(?P<n>\d+))?$")
DATASET_IMAGE_RE = re.compile(r"^(?P<stamp>\d{8}_\d{4})_(?P<index>\d{6})\.(?P<ext>jpe?g|png)$", re.IGNORECASE)


@dataclass(frozen=True)
class FrameInfo:
    name: str
    timestamp: datetime
    angle: int
    speed: int


def make_frame_name(ts: datetime, angle: int, speed: int, ext: str = FRAME_EXT) -> str:
    if not (0 <= angle <= 180):
        raise ValueError(f"angle 범위 오류: {angle}")
    if not (0 <= speed <= 100):
        raise ValueError(f"speed 범위 오류: {speed}")
    ms = ts.microsecond // 1000
    return f"{ts:%Y%m%d_%H%M%S}_{ms:03d}_a{angle:03d}_s{speed:03d}.{ext}"


def parse_frame_name(name: str) -> FrameInfo:
    m = FRAME_RE.match(name)
    if not m:
        raise ValueError(f"프레임 파일 이름 형식이 아닙니다: {name}")
    ts = datetime.strptime(f"{m['date']}_{m['time']}", "%Y%m%d_%H%M%S").replace(microsecond=int(m["ms"]) * 1000)
    return FrameInfo(name=name, timestamp=ts, angle=int(m["angle"]), speed=int(m["speed"]))


def try_parse_frame_name(name: str) -> Optional[FrameInfo]:
    try:
        return parse_frame_name(name)
    except ValueError:
        return None


def make_session_name(dt: datetime) -> str:
    return f"{dt:%Y%m%d_%H%M}"


def is_session_name(name: str) -> bool:
    return SESSION_RE.match(name) is not None


def make_dataset_stamp(dt: datetime) -> str:
    return f"{dt:%Y%m%d_%H%M}"


def make_dataset_image_name(stamp: str, index: int, ext: str = FRAME_EXT) -> str:
    return f"{stamp}_{index:06d}.{ext}"


def make_dataset_zip_name(stamp: str) -> str:
    return f"dataset_{stamp}.zip"
