"""학습·추론 공용 이미지 전처리.

입력: BGR uint8 (H, W, 3)  (OpenCV 기본)
출력: float32 (3, out_h, out_w), RGB, [0, 1]

값이 전달되는 길 (한 방향)
    자르는 범위(위·아래)  가공 프로그램에서 사진을 보며 정한다 → 데이터셋 zip 의 meta.json → 학습기 → model_meta.json
    모델 입력 크기        config.preprocess (out_width, out_height) → 학습기 → model_meta.json
추론기는 반드시 model_meta.json 의 값으로 Preprocessor.from_dict() 를 만들어 쓴다.
config.preprocess 의 crop_* 는 자르는 범위가 적혀 있지 않은 데이터셋(예전 zip)에서만 쓰인다.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Optional, Tuple

import cv2
import numpy as np

from common.config import PreprocessConfig

CROP_STEP = 0.01            # 자르는 선은 사진 높이의 1 % 단위로 움직인다
CROP_MIN_HEIGHT = 0.10      # 남기는 부분은 사진 높이의 10 % 이상


def fit_crop(top: float, bottom: float, moved: str = "top") -> Tuple[float, float]:
    """(위, 아래) 비율을 쓸 수 있는 값으로 맞춘다: 1 % 단위, 0~1, 남는 높이는 CROP_MIN_HEIGHT 이상.

    moved  방금 움직인 쪽("top" | "bottom"). 두 선이 너무 가까워지면 움직인 쪽을 멈춘다 (다른 선을 밀어내지 않는다).
    """
    unit = round(1 / CROP_STEP)
    lo = max(0, min(unit, round(float(top) * unit)))
    hi = max(0, min(unit, round(float(bottom) * unit)))
    gap = round(CROP_MIN_HEIGHT * unit)
    if hi - lo < gap:
        if moved == "bottom":
            hi = min(unit, lo + gap)
            lo = min(lo, hi - gap)
        else:
            lo = max(0, hi - gap)
            hi = max(hi, lo + gap)
    return lo / unit, hi / unit


def crop_to_dict(crop: Tuple[float, float]) -> dict:
    """meta.json, curation.json 에 적는 모양. 키 이름은 Preprocessor 와 같다."""
    return {"crop_top_ratio": float(crop[0]), "crop_bottom_ratio": float(crop[1])}


def crop_from_dict(d) -> Optional[Tuple[float, float]]:
    """crop_to_dict 의 반대. 적혀 있지 않으면 None, 적혀 있는데 쓸 수 없는 값이면 ValueError."""
    if d is None:
        return None
    try:
        top, bottom = float(d["crop_top_ratio"]), float(d["crop_bottom_ratio"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"자르는 범위를 읽을 수 없습니다: {d!r}") from None
    if not (0.0 <= top < bottom <= 1.0):
        raise ValueError(f"자르는 범위가 올바르지 않습니다: 위 {top}, 아래 {bottom} (0 ≤ 위 < 아래 ≤ 1)")
    return top, bottom


def describe_crop(crop: Tuple[float, float]) -> str:
    """사람이 읽는 한 줄. 예: '위쪽 15% · 아래쪽 0% 잘라 냄'"""
    return f"위쪽 {crop[0] * 100:.0f}% · 아래쪽 {(1 - crop[1]) * 100:.0f}% 잘라 냄"


@dataclass(frozen=True)
class Preprocessor:
    crop_top_ratio: float = 0.4
    crop_bottom_ratio: float = 1.0
    out_width: int = 200
    out_height: int = 66

    @classmethod
    def from_config(cls, cfg: PreprocessConfig) -> "Preprocessor":
        return cls(cfg.crop_top_ratio, cfg.crop_bottom_ratio, cfg.out_width, cfg.out_height)

    @classmethod
    def from_dict(cls, d: dict) -> "Preprocessor":
        return cls(
            crop_top_ratio=float(d["crop_top_ratio"]),
            crop_bottom_ratio=float(d["crop_bottom_ratio"]),
            out_width=int(d["out_width"]),
            out_height=int(d["out_height"]),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update({"color": "RGB", "scale": "0-1", "layout": "CHW", "dtype": "float32"})
        return d

    @property
    def input_shape(self) -> tuple:
        return (3, self.out_height, self.out_width)

    @property
    def crop(self) -> Tuple[float, float]:
        return self.crop_top_ratio, self.crop_bottom_ratio

    def with_crop(self, crop: Tuple[float, float]) -> "Preprocessor":
        """자르는 범위만 바꾼 사본 (모델 입력 크기는 그대로)."""
        return replace(self, crop_top_ratio=float(crop[0]), crop_bottom_ratio=float(crop[1]))

    def to_rgb_uint8(self, img_bgr: np.ndarray) -> np.ndarray:
        """자르고 줄인 RGB uint8 (out_h, out_w, 3). 학습기가 사진을 메모리에 작게 올려 둘 때 쓴다."""
        h = img_bgr.shape[0]
        y1 = int(h * self.crop_top_ratio)
        y2 = int(h * self.crop_bottom_ratio)
        cropped = img_bgr[y1:y2, :, :]
        resized = cv2.resize(cropped, (self.out_width, self.out_height), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        rgb = self.to_rgb_uint8(img_bgr).astype(np.float32) / 255.0
        return np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)))
