"""학습·추론 공용 이미지 전처리.

입력: BGR uint8 (H, W, 3)  (OpenCV 기본)
출력: float32 (3, out_h, out_w), RGB, [0, 1]

파라미터는 config.preprocess 에서 오고, 학습 시 model_meta.json 에 복사된다.
추론기는 반드시 model_meta.json 의 값으로 Preprocessor.from_dict() 를 만들어 쓴다.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np

from common.config import PreprocessConfig


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

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        h = img_bgr.shape[0]
        y1 = int(h * self.crop_top_ratio)
        y2 = int(h * self.crop_bottom_ratio)
        cropped = img_bgr[y1:y2, :, :]
        resized = cv2.resize(cropped, (self.out_width, self.out_height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)))
