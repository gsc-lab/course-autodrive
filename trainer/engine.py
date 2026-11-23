"""학습 루프.

한 에폭 = 학습 데이터 전체를 한 번 보는 것. 에폭마다
    1. train_one_epoch : 배치마다 예측 → 손실 → 역전파 → 가중치 갱신
    2. evaluate        : 검증 데이터로 손실과 정확도만 잰다 (가중치는 건드리지 않는다)

학습 손실(train loss)과 검증 손실(val loss)을 나란히 봐야 하는 이유
    학습 손실은 모델이 이미 본 사진에 대한 것이라 학습을 계속하면 거의 언제나 내려간다.
    검증 손실은 모델이 배우는 데 쓰지 않은 사진에 대한 것이다. 처음에는 같이 내려가다가, 모델이 학습 사진을
    외우기 시작하면(과적합) 검증 손실만 다시 올라간다. 그래서 "좋은 모델"은 검증 손실이 가장 낮았던 에폭의 모델이고,
    조기 종료(early stopping)도 검증 손실을 보고 멈춘다.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from trainer.text import rjust


def _batch_weight(loss_fn: nn.Module, labels: torch.Tensor) -> float:
    """이 배치가 에폭 평균 손실에서 차지하는 몫.

    클래스 가중치가 없으면 사진 수다. 가중치가 있으면(loss.class_weight: balanced) CrossEntropyLoss 는 배치 안에서
    "가중치를 곱한 평균"을 내므로, 에폭 평균도 같은 가중치(정답 클래스 가중치의 합)로 모아야 한다. 그래야 같은 모델의
    손실이 배치 크기나 배치에 담긴 사진에 따라 달라지지 않는다.
    """
    weight = getattr(loss_fn, "weight", None)
    return float(len(labels)) if weight is None else float(weight[labels].sum().item())


def train_one_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module,
                    optimizer: torch.optim.Optimizer, device: torch.device) -> tuple:
    """학습 데이터를 한 바퀴 돈다. 돌려주는 값: (평균 손실, 정확도)."""
    model.train()
    loss_sum, weight_sum, correct, seen = 0.0, 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)              # 1. 예측: 클래스별 점수
        loss = loss_fn(logits, labels)      # 2. 손실: 정답과 얼마나 다른가

        optimizer.zero_grad()               # 3. 지난 배치의 기울기를 지운다
        loss.backward()                     # 4. 역전파: 가중치마다 손실에 대한 기울기를 구한다
        optimizer.step()                    # 5. 기울기의 반대 방향으로 가중치를 고친다

        share = _batch_weight(loss_fn, labels)
        loss_sum += loss.item() * share
        weight_sum += share
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += len(labels)
    return loss_sum / weight_sum, correct / seen


@torch.no_grad()                            # 기울기를 구하지 않는다: 가중치를 고치지 않으므로
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> tuple:
    """검증 또는 시험 데이터로 잰다. 돌려주는 값: (평균 손실, 정확도)."""
    model.eval()
    loss_sum, weight_sum, correct, seen = 0.0, 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        share = _batch_weight(loss_fn, labels)
        loss_sum += loss_fn(logits, labels).item() * share
        weight_sum += share
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += len(labels)
    return loss_sum / weight_sum, correct / seen


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple:
    """돌려주는 값: (클래스별 확률 (N, K), 정답 (N,)). 시험 분석에서 쓴다."""
    model.eval()
    probs, targets = [], []
    for images, labels in loader:
        probs.append(torch.softmax(model(images.to(device)), dim=1).cpu().numpy())
        targets.append(np.asarray(labels))
    return np.concatenate(probs), np.concatenate(targets)


@dataclass
class History:
    """에폭마다의 기록. history.csv 와 학습 곡선 그림이 된다."""
    epoch: List[int] = field(default_factory=list)
    lr: List[float] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    seconds: List[float] = field(default_factory=list)
    best_epoch: int = 0                 # 검증 손실이 가장 낮았던 에폭 (1부터). 0 = 쓸 수 있는 에폭이 하나도 없었다
    best_val_loss: float = float("inf")
    stopped_early: bool = False         # 조기 종료로 멈췄다
    diverged: bool = False              # 손실이 nan/inf 가 되어 멈췄다
    interrupted: bool = False           # Ctrl+C 로 멈췄다

    def rows(self) -> list:
        return [dict(epoch=e, lr=lr, train_loss=tl, train_acc=ta, val_loss=vl, val_acc=va, seconds=s)
                for e, lr, tl, ta, vl, va, s in zip(self.epoch, self.lr, self.train_loss, self.train_acc,
                                                    self.val_loss, self.val_acc, self.seconds)]


def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer, scheduler, device: torch.device, epochs: int,
        early_stopping: bool = True, patience: int = 10, min_delta: float = 0.0,
        log: Optional[Callable[[str], None]] = print) -> History:
    """epochs 만큼(또는 조기 종료까지) 학습하고, 검증 손실이 가장 낮았던 에폭의 가중치로 되돌려 놓는다.

    중간에 멈추는 세 경우
        조기 종료   검증 손실이 patience 에폭 동안 나아지지 않음
        발산        손실이 nan/inf (학습률이 너무 크거나 입력 스케일링이 없을 때). 더 돌려도 회복되지 않는다
        Ctrl+C      그때까지 가장 좋았던 가중치로 마무리한다 (끝난 에폭이 하나도 없으면 그대로 중단)
    history.best_epoch 이 0 이면 쓸 수 있는 가중치가 없다는 뜻이다. 부르는 쪽에서 저장하지 않아야 한다.
    """
    log = log or (lambda _msg: None)
    history = History()
    best_state = copy.deepcopy(model.state_dict())
    waited = 0                                              # 검증 손실이 나아지지 않은 채 지난 에폭 수

    log(f"{rjust('에폭', 4)}  {rjust('학습률', 11)}  {'train loss':>10}  {'val loss':>9}  {'train acc':>9}  {'val acc':>8}")
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        try:
            train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        except KeyboardInterrupt:
            if history.best_epoch == 0:
                raise
            history.interrupted = True
            log(f"Ctrl+C: {epoch} 에폭 도중에 멈춥니다. 지금까지의 결과로 마무리합니다.")
            break
        scheduler.step()

        if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
            history.diverged = True
            log(f"{epoch:>4}    {lr:>9.6f}  {train_loss:>10.4f}  {val_loss:>9.4f}")
            log("손실이 nan/inf 가 되어 멈춥니다 (발산). 학습률(lr)을 낮추거나 rescale 을 minmax/standardize 로 두세요.")
            break

        history.epoch.append(epoch); history.lr.append(lr)
        history.train_loss.append(train_loss); history.train_acc.append(train_acc)
        history.val_loss.append(val_loss); history.val_acc.append(val_acc)
        history.seconds.append(time.perf_counter() - t0)

        improved = val_loss < history.best_val_loss - min_delta
        if improved:
            history.best_val_loss, history.best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            waited = 0
        else:
            waited += 1
        log(f"{epoch:>4}    {lr:>9.6f}  {train_loss:>10.4f}  {val_loss:>9.4f}  {train_acc:>9.3f}  {val_acc:>8.3f}"
            + ("   ← 가장 좋음" if improved else ""))

        if early_stopping and waited >= patience:
            history.stopped_early = True
            log(f"검증 손실이 {patience} 에폭 동안 나아지지 않아 멈춥니다 (조기 종료).")
            break

    model.load_state_dict(best_state)
    if history.best_epoch:
        log(f"가장 좋았던 {history.best_epoch} 에폭의 가중치를 씁니다 (val loss {history.best_val_loss:.4f}).")
    return history
