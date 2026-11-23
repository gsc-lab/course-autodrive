"""옵티마이저와 학습률 스케줄.

옵티마이저는 손실을 줄이는 쪽으로 가중치를 조금씩 고친다. 한 번에 얼마나 고칠지가 학습률(lr)이다.
    너무 크면  손실이 줄지 않고 튀거나 발산한다.
    너무 작으면  줄기는 하는데 한없이 느리다.

스케줄은 에폭이 지나면서 학습률을 바꾸는 방법이다 (training.optimizer.schedule).
    constant      처음부터 끝까지 lr
    warmup        처음 warmup_epochs 동안 천천히 올려 lr 에 도달, 그 뒤 일정
    decay         lr 에서 시작해 마지막 에폭에 lr × min_lr_ratio 까지 천천히 내린다 (코사인 곡선)
    warmup_decay  천천히 올린 뒤 천천히 내린다
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def lr_factor(epoch: int, total_epochs: int, schedule: str, warmup_epochs: int = 5,
              min_lr_ratio: float = 0.01) -> float:
    """epoch(0부터) 번째 에폭의 학습률이 lr 의 몇 배인지. 순수 함수라 그림을 그리거나 시험하기 쉽다."""
    if schedule == "constant":
        return 1.0
    warm = min(max(0, warmup_epochs), max(0, total_epochs - 1)) if schedule in ("warmup", "warmup_decay") else 0
    if epoch < warm:
        return (epoch + 1) / (warm + 1)                       # 0 에서 시작하지 않게 한 칸 띄운다
    if schedule == "warmup":
        return 1.0
    if schedule in ("decay", "warmup_decay"):
        span = max(1, total_epochs - 1 - warm)
        progress = min(1.0, (epoch - warm) / span)            # 0 → 1
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))   # 1 → 0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    raise ValueError(f"schedule 은 constant | warmup | decay | warmup_decay: {schedule!r}")


def build_optimizer(model: nn.Module, name: str = "adam", lr: float = 1e-3, weight_decay: float = 0.0,
                    momentum: float = 0.9) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"옵티마이저는 adam | sgd: {name!r}")


def build_scheduler(optimizer: torch.optim.Optimizer, total_epochs: int, schedule: str = "constant",
                    warmup_epochs: int = 5, min_lr_ratio: float = 0.01):
    """에폭이 끝날 때마다 scheduler.step() 을 한 번 부른다."""
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: lr_factor(epoch, total_epochs, schedule, warmup_epochs, min_lr_ratio))
