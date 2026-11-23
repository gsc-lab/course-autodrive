"""모델.

새 모델을 써 보려면
    1. 아래에 nn.Module 클래스를 하나 만든다. 생성자는 (num_classes, input_shape) 을 받고,
       forward 는 (B, 3, H, W) 를 받아 (B, num_classes) 의 점수(logit)를 돌려준다.
    2. 파일 맨 아래 MODELS 에 이름을 붙여 등록한다.
    3. config.yaml 의 training.model 에 그 이름을 적는다.
데이터, 손실, 옵티마이저, 내보내기 코드는 고칠 필요가 없다.
"""
from __future__ import annotations

from typing import Callable, Dict, Sequence

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 입력 스케일링 (모델의 맨 앞 층)
# ---------------------------------------------------------------------------
class InputScaler(nn.Module):
    """0~1 로 들어온 사진의 값 범위를 바꾼다. 학습할 값은 없다.

        none         0~255 그대로 쓴다 (스케일링을 하지 않으면 어떻게 되는지 보는 용도)
        minmax       0~1
        standardize  채널마다 (x - 평균) / 표준편차. 평균과 표준편차는 학습 데이터에서 구한다

    이 층을 모델 안에 두는 이유: ONNX 로 내보낼 때 함께 들어가므로, 차에서 추론할 때는 어떤 설정으로 학습한
    모델이든 "0~1 RGB 사진"만 넣으면 된다. 학습과 추론의 스케일링이 어긋날 수 없다.
    """

    def __init__(self, mode: str = "minmax", mean: Sequence[float] = (0.0, 0.0, 0.0),
                 std: Sequence[float] = (1.0, 1.0, 1.0)):
        super().__init__()
        if mode not in ("none", "minmax", "standardize"):
            raise ValueError(f"rescale 은 none | minmax | standardize: {mode!r}")
        self.mode = mode
        self.register_buffer("mean", torch.tensor(list(mean), dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(list(std), dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return x * 255.0
        if self.mode == "standardize":
            return (x - self.mean) / self.std
        return x

    def to_dict(self) -> dict:
        return {"mode": self.mode, "mean": self.mean.flatten().tolist(), "std": self.std.flatten().tolist()}


# ---------------------------------------------------------------------------
# 모델들
# ---------------------------------------------------------------------------
class PilotNet(nn.Module):
    """NVIDIA PilotNet (2016) 을 조향 분류용으로 쓴 것. 합성곱 5층 + 완전 연결 3층, 가중치 약 25만 개.

    입력 (B, 3, 66, 200) → 출력 (B, num_classes)
    """

    def __init__(self, num_classes: int, input_shape: Sequence[int] = (3, 66, 200)):
        super().__init__()
        # 특징 추출: 사진에서 차선의 모양을 찾아낸다. stride 2 로 크기를 줄여 가며 채널(특징의 종류)을 늘린다
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        # 분류: 찾아낸 특징으로 조향 클래스마다 점수를 매긴다
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(_flat_size(self.features, input_shape), 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class SmallCNN(nn.Module):
    """비교용 작은 모델: 합성곱 3층 + 풀링 + 완전 연결 2층. PilotNet 과 결과를 견주어 볼 것."""

    def __init__(self, num_classes: int, input_shape: Sequence[int] = (3, 66, 200)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(_flat_size(self.features, input_shape), 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _flat_size(features: nn.Module, input_shape: Sequence[int]) -> int:
    """합성곱 층들을 지난 뒤 값이 몇 개가 되는지 (첫 Linear 층의 입력 크기). 가짜 사진 한 장을 흘려 보고 센다."""
    with torch.no_grad():
        return int(features(torch.zeros(1, *input_shape)).numel())


# 이름 → 모델을 만드는 함수. config.yaml 의 training.model 이 이 이름을 가리킨다
MODELS: Dict[str, Callable[..., nn.Module]] = {
    "pilotnet": PilotNet,
    "small_cnn": SmallCNN,
}


# ---------------------------------------------------------------------------
# 조립
# ---------------------------------------------------------------------------
class SteeringNet(nn.Module):
    """입력 스케일링 + 모델. 학습, 시험, ONNX 내보내기가 모두 이 한 덩어리를 쓴다.

    입력: (B, 3, H, W) RGB 0~1   출력: (B, num_classes) 클래스별 점수(logit). 가장 큰 점수의 클래스가 답이다.
    """

    def __init__(self, scaler: InputScaler, body: nn.Module):
        super().__init__()
        self.scaler = scaler
        self.body = body

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(self.scaler(x))


def build_model(name: str, num_classes: int, input_shape: Sequence[int], scaler: InputScaler) -> SteeringNet:
    if name not in MODELS:
        raise ValueError(f"모르는 모델 {name!r}. trainer/models.py 의 MODELS 에 있는 이름: {sorted(MODELS)}")
    body = MODELS[name](num_classes=num_classes, input_shape=tuple(input_shape))
    return SteeringNet(scaler, body)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
