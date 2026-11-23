"""손실 함수: 모델의 답이 정답에서 얼마나 먼지를 숫자 하나로 만든다. 학습은 이 숫자를 줄이는 과정이다.

분류에서는 교차 엔트로피(cross entropy)를 쓴다. 모델이 정답 클래스에 준 확률 p 에 대해 -log(p):
정답에 확률 1 을 주면 0, 확률이 낮을수록 커진다. 클래스가 5개일 때 아무것도 못 배운 모델은 약 log(5) = 1.61 이 나온다.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn


def balanced_class_weights(counts: Sequence[int]) -> torch.Tensor:
    """사진 수의 역수에 비례하는 클래스 가중치. 전체 / (클래스 수 × 그 클래스의 사진 수).

    직진 사진이 80 % 인 데이터에서는 모델이 "언제나 직진"이라고만 답해도 손실이 꽤 낮다.
    적은 클래스의 오답에 더 큰 벌점을 주면 그렇게 빠져나가지 못한다.
    """
    counts_t = torch.tensor(list(counts), dtype=torch.float32)
    present = counts_t > 0
    if not bool(present.any()):
        return torch.ones_like(counts_t)
    weights = torch.zeros_like(counts_t)
    weights[present] = counts_t.sum() / (present.sum() * counts_t[present])
    # 학습 데이터에 한 장도 없는 클래스: 0 으로 두면 그 클래스만 든 검증 배치의 손실이 0/0 = nan 이 된다.
    # 가장 드문 클래스와 같은 가중치를 준다 (학습에는 영향이 없고, 검증에서 그 클래스의 오답이 손실에 잡힌다).
    weights[~present] = weights[present].max()
    return weights


def build_loss(name: str = "cross_entropy", class_weight: str = "none",
               class_counts: Optional[Sequence[int]] = None) -> nn.Module:
    if name != "cross_entropy":
        raise ValueError(f"모르는 손실 함수 {name!r} (지금은 cross_entropy 만)")
    weight = None
    if class_weight == "balanced":
        if class_counts is None:
            raise ValueError("class_weight=balanced 에는 클래스별 사진 수가 필요합니다")
        weight = balanced_class_weights(class_counts)
    elif class_weight != "none":
        raise ValueError(f"class_weight 는 none | balanced: {class_weight!r}")
    return nn.CrossEntropyLoss(weight=weight)
