"""데이터 나누기: 학습(train) / 검증(val) / 시험(test).

세 묶음의 역할
    train  가중치를 고치는 데 쓴다.
    val    에폭마다 확인한다. 가장 좋은 모델을 고르고 학습을 멈출 때를 정한다. 가중치를 고치는 데는 쓰지 않는다.
    test   학습이 다 끝난 뒤 한 번만 쓴다. 모델이 처음 보는 사진에서 얼마나 맞히는지 본다.

계층화(stratify): 클래스마다 따로 섞어 나누므로, 세 묶음의 클래스 비율이 전체와 같아진다.
사진이 적은 클래스가 어느 한 묶음에서 통째로 빠지는 일을 막는다.

torch 없이 numpy 만 쓴다 (테스트하기 쉽게).
"""
from __future__ import annotations

from typing import Dict

import numpy as np

SPLIT_NAMES = ("train", "val", "test")


def _counts(n: int, val: float, test: float) -> tuple:
    """n 개를 (train, val, test) 개수로. 비율이 0 보다 크면 가능한 한 1개 이상을 준다."""
    n_val = int(round(n * val))
    n_test = int(round(n * test))
    if val > 0 and n_val == 0 and n >= 3:
        n_val = 1
    if test > 0 and n_test == 0 and n >= 3:
        n_test = 1
    while n_val + n_test >= n and (n_val or n_test):    # 학습 몫이 적어도 1개는 남게
        if n_test >= n_val and n_test > 0:
            n_test -= 1
        else:
            n_val -= 1
    return n - n_val - n_test, n_val, n_test


def _block_ids(n: int, block_size: int, groups=None) -> np.ndarray:
    """사진 → 덩어리 번호. 연속한 block_size 장이 한 덩어리. 주행(groups)이 바뀌는 곳에서는 덩어리도 새로 시작한다."""
    if groups is None:
        return np.arange(n) // block_size
    groups = np.asarray(groups)
    starts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])        # 주행이 바뀌는 자리
    ids = np.empty(n, dtype=np.int64)
    next_id = 0
    for a, b in zip(starts, np.r_[starts[1:], n]):
        ids[a:b] = next_id + np.arange(b - a) // block_size
        next_id = int(ids[b - 1]) + 1
    return ids


def split_indices(labels, train: float = 0.7, val: float = 0.15, test: float = 0.15,
                  stratify: bool = True, block_size: int = 1, seed: int = 42, groups=None) -> Dict[str, np.ndarray]:
    """사진 번호(0..N-1)를 세 묶음으로 나눈다. 돌려주는 값: {"train": 번호 배열, "val": …, "test": …}

    labels      사진마다의 클래스 번호 (수집 순서대로)
    block_size  연속한 사진 N장을 한 덩어리로 묶어 같은 묶음에 넣는다. 덩어리의 클래스는 그 안에서 가장 많은 클래스.
    seed        같은 값이면 언제나 같은 결과
    groups      사진마다의 주행 번호 (없어도 된다). 주면 한 덩어리가 서로 다른 주행에 걸치지 않는다.
    """
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    if n == 0:
        raise ValueError("사진이 없습니다")
    total = train + val + test
    if total <= 0 or train <= 0:
        raise ValueError(f"비율이 올바르지 않습니다: train={train}, val={val}, test={test}")
    val, test = val / total, test / total

    block_size = max(1, int(block_size))
    block_of = _block_ids(n, block_size, groups)             # 사진 → 덩어리 번호
    n_blocks = int(block_of[-1]) + 1
    block_label = np.array([np.bincount(labels[block_of == b]).argmax() for b in range(n_blocks)]) \
        if block_size > 1 else labels

    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(block_label == c) for c in np.unique(block_label)] if stratify \
        else [np.arange(n_blocks)]

    picked = {name: [] for name in SPLIT_NAMES}
    for blocks in groups:
        blocks = rng.permutation(blocks)
        n_train, n_val, _ = _counts(len(blocks), val, test)
        picked["train"].append(blocks[:n_train])
        picked["val"].append(blocks[n_train:n_train + n_val])
        picked["test"].append(blocks[n_train + n_val:])

    out = {}
    for name in SPLIT_NAMES:
        blocks = np.concatenate(picked[name]) if picked[name] else np.array([], dtype=np.int64)
        out[name] = np.sort(np.flatnonzero(np.isin(block_of, blocks)))
    return out


def class_counts(labels, n_classes: int) -> np.ndarray:
    return np.bincount(np.asarray(labels, dtype=np.int64), minlength=n_classes)
