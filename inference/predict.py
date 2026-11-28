"""사진 한 장 → 조향 각도.

    카메라 사진(BGR) → 전처리(모델이 학습할 때와 똑같이) → 모델 → 클래스별 점수 → 확률 → (흔들림 줄이기) → 각도

흔들림 줄이기(smoothing): 두 단계의 경계에 있는 사진에서는 예측이 사진마다 오락가락해 바퀴가 떤다.
    none  사진마다의 예측을 그대로 쓴다
    vote  최근 N장의 다수결 (같은 표면 가장 최근에 나온 쪽)
    ema   확률을 지수 이동 평균으로 섞은 뒤 가장 큰 것 (alpha 가 클수록 새 사진을 더 믿는다)
줄일수록 덜 떨지만 그만큼 늦게 꺾는다.
"""
from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from inference.bundle import ModelBundle
from inference.engine import Backend


class PredictError(RuntimeError):
    """모델이 쓸 수 없는 값을 냈다 (nan 등). 이 값으로 차를 몰면 안 된다."""


@dataclass
class Prediction:
    index: int              # 쓰기로 한 클래스 번호 (흔들림 줄이기를 거친 것)
    angle: int              # 그 클래스의 각도
    raw_index: int          # 이 사진만 보고 낸 클래스 번호
    raw_angle: int
    confidence: float       # 이 사진에서 raw 클래스의 확률 (가장 큰 확률)
    prob: float             # 이 사진에서 '쓰기로 한 클래스'의 확률. 흔들림 줄이기를 쓰지 않으면 confidence 와 같다
    probs: np.ndarray       # 이 사진의 클래스별 확률
    ms: float               # 전처리 + 추론에 걸린 시간


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64) - float(np.max(logits))
    e = np.exp(z)
    return (e / e.sum()).astype(np.float32)


class Smoother:
    def __init__(self, mode: str = "none", window: int = 3, alpha: float = 0.5):
        self.mode, self.alpha = mode, float(alpha)
        self._recent: deque = deque(maxlen=max(1, int(window)))
        self._mean: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._recent.clear()
        self._mean = None

    def __call__(self, probs: np.ndarray) -> int:
        """이 사진의 확률을 받아, 쓰기로 한 클래스 번호를 돌려준다."""
        raw = int(np.argmax(probs))
        if self.mode == "vote":
            self._recent.append(raw)
            votes = Counter(self._recent)
            top = max(votes.values())
            return next(i for i in reversed(self._recent) if votes[i] == top)      # 같은 표면 가장 최근 것
        if self.mode == "ema":
            self._mean = probs.copy() if self._mean is None else self.alpha * probs + (1.0 - self.alpha) * self._mean
            return int(np.argmax(self._mean))
        return raw


class Predictor:
    def __init__(self, bundle: ModelBundle, backend: Backend, smoothing: str = "none", window: int = 3, alpha: float = 0.5):
        self.bundle, self.backend = bundle, backend
        self.classes = list(bundle.classes)
        self.smoother = Smoother(smoothing, window, alpha)
        self.count = 0

    def reset(self) -> None:
        """지난 예측의 기록을 버리고 처음부터 센다 (시작할 때 미리 돌려 본 것을 지운다)."""
        self.smoother.reset()
        self.count = 0

    def __call__(self, frame_bgr: np.ndarray) -> Prediction:
        t0 = time.perf_counter()
        x = self.bundle.pre(frame_bgr)[None]                    # (1, 3, H, W) float32, RGB 0~1
        logits = np.asarray(self.backend.infer(x), dtype=np.float32).reshape(-1)
        if logits.shape[0] != len(self.classes) or not np.isfinite(logits).all():
            raise PredictError(f"모델의 출력이 올바르지 않습니다: {logits.tolist()}")
        probs = softmax(logits)
        raw = int(np.argmax(probs))
        index = self.smoother(probs)
        self.count += 1
        return Prediction(index=index, angle=self.classes[index], raw_index=raw, raw_angle=self.classes[raw],
                          confidence=float(probs[raw]), prob=float(probs[index]), probs=probs,
                          ms=(time.perf_counter() - t0) * 1e3)

    def close(self) -> None:
        self.backend.close()
