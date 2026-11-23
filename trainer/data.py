"""데이터: 데이터셋 zip 읽기 → Dataset → DataLoader.

PyTorch 가 데이터를 다루는 두 단계
    Dataset     "i 번째 사진과 정답을 달라"에 답한다 (__len__, __getitem__).
    DataLoader  Dataset 에서 사진을 batch_size 장씩 묶어 꺼내 준다. 학습 데이터는 에폭마다 순서를 섞는다 (shuffle).

사진 한 장이 거치는 길
    zip 안의 JPEG (640×480, BGR)
      → common.preprocess.Preprocessor : 데이터셋에 적힌 범위(위·아래)만 남기고 200×66 으로 줄인다, RGB
      → 메모리에 uint8 로 보관 (한 장에 약 40 KB. 1만 장이면 약 400 MB)
      → Dataset.__getitem__ : 0~1 실수로 바꾼다, (좌우 반전)
      → 모델 맨 앞의 InputScaler : rescale 설정대로 값 범위를 바꾼다 (models.py)
"""
from __future__ import annotations

import csv
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common.preprocess import CROP_MIN_HEIGHT, Preprocessor, crop_from_dict, describe_crop
from common.steering import mirror_class_indices, nearest_class_index


class DatasetError(ValueError):
    pass


# ---------------------------------------------------------------------------
# 1. 데이터셋 zip 읽기
# ---------------------------------------------------------------------------
@dataclass
class Samples:
    """메모리에 올린 데이터셋 전체."""
    images: np.ndarray          # (N, 3, H, W) uint8, RGB
    labels: np.ndarray          # (N,) 클래스 번호 0..K-1
    angles: np.ndarray          # (N,) 서보 각도(도). 파일 이름에 적혀 있던 값
    names: List[str]            # "zip 이름/사진 이름"
    classes: List[int]          # 클래스 번호 → 서보 각도. 예: [30, 60, 90, 120, 150]
    datasets: List[dict]        # 읽은 zip 마다의 이름과 meta.json
    center: Optional[float] = None   # 직진일 때의 서보 각도 (좌우 반전의 대칭 축). 옛 zip 에는 없을 수 있다
    groups: Optional[np.ndarray] = None   # (N,) 사진이 속한 주행(zip + 세션)의 번호. 덩어리 나누기(block_size)가 쓴다
    pre: Optional[Preprocessor] = None    # 사진을 자르고 줄이는 데 실제로 쓴 전처리기. 모델과 함께 저장된다
    crop_from_dataset: bool = False       # 자르는 범위가 데이터셋에 적혀 있었다 (아니면 기본값으로 잘랐다)
    sources: Optional[List[str]] = None   # 사진마다의 출처 "세션/원본 사진 이름" (labels.csv 의 source). 없으면 ""

    def __len__(self) -> int:
        return len(self.labels)


def _steering_of(meta: dict) -> tuple:
    """meta.json → (클래스 목록, 직진 각도). 직진 각도가 없는 옛 zip 은 min 과 max 의 가운데로 본다."""
    steering = meta["steering"]
    classes = sorted(int(c) for c in steering["classes"])
    if not classes or len(set(classes)) != len(classes):
        raise ValueError(f"조향 클래스가 올바르지 않습니다: {steering['classes']}")
    if steering.get("center") is not None:
        center = float(steering["center"])
    elif steering.get("min") is not None and steering.get("max") is not None:
        center = (float(steering["min"]) + float(steering["max"])) / 2.0
    else:
        center = None
    return classes, center


def _same_crop(a: tuple, b: tuple) -> bool:
    return abs(a[0] - b[0]) < 1e-6 and abs(a[1] - b[1]) < 1e-6


def load_samples(zip_paths: Sequence[str | Path], pre: Preprocessor, progress=None,
                 follow_dataset_crop: bool = True) -> Samples:
    """데이터셋 zip 을 하나 이상 읽어 메모리에 올린다. 조향 클래스와 자르는 범위는 zip 의 meta.json 을 따른다.

    pre                  모델 입력 크기, 그리고 zip 에 자르는 범위가 적혀 있지 않을 때(예전 zip) 쓸 범위
    follow_dataset_crop  False 면 zip 에 적힌 범위를 보지 않고 pre 그대로 자른다. 이미 학습된 모델을 시험할 때 쓴다:
                         모델은 자기가 학습한 범위로 잘린 사진을 받아야 한다 (차에서도 그렇게 본다)
    """
    if not zip_paths:
        raise DatasetError("데이터셋 zip 을 하나 이상 지정하세요")
    fallback_crop = pre.crop
    used_crop: Optional[tuple] = None           # 첫 zip 이 정한 범위
    used_from: Optional[str] = None             # 그 zip 의 이름. 범위가 적혀 있지 않았으면(기본값으로 읽었으면) None
    first_name = ""
    crop_written = False                        # 범위가 적혀 있던 zip 이 하나라도 있는가
    chunks, labels, angles, names, groups, datasets, sources = [], [], [], [], [], [], []
    classes: Optional[List[int]] = None
    center: Optional[float] = None
    opened = set()                              # 이미 읽은 zip (같은 파일을 두 번 적었는지 본다)
    seen: Dict[str, str] = {}                   # 원본 사진 → 그 사진이 들어 있던 zip
    group_of: Dict[tuple, int] = {}             # (zip, 세션) → 주행 번호
    for path in map(Path, zip_paths):
        if not path.is_file():
            raise DatasetError(f"데이터셋 zip 이 없습니다: {path}")
        if path.resolve() in opened:
            raise DatasetError(f"{path.name} 을 두 번 지정했습니다")
        opened.add(path.resolve())
        try:
            z = zipfile.ZipFile(path)
        except (zipfile.BadZipFile, OSError) as e:
            raise DatasetError(f"{path.name}: zip 파일을 열 수 없습니다 ({e})") from None
        with z:
            try:
                meta = json.loads(z.read("meta.json").decode("utf-8"))
                rows = list(csv.DictReader(io.StringIO(z.read("labels.csv").decode("utf-8"))))
                zip_classes, zip_center = _steering_of(meta)
                # 가공 프로그램에서 정한 범위. 예전 zip 에는 없다. 범위를 따르지 않을 때(모델을 시험할 때)는 읽지도 않는다
                zip_crop = crop_from_dict(meta.get("preprocess")) if follow_dataset_crop else None
            except (KeyError, ValueError, TypeError, AttributeError, zipfile.BadZipFile) as e:
                raise DatasetError(f"{path.name}: 가공 프로그램이 만든 데이터셋 zip 이 아니거나 손상됐습니다 ({e})") from None
            if zip_crop and zip_crop[1] - zip_crop[0] < CROP_MIN_HEIGHT - 1e-9:
                raise DatasetError(f"{path.name}: 자르는 범위가 너무 좁습니다 (위 {zip_crop[0]:.0%} ~ 아래 {zip_crop[1]:.0%}. 사진 높이의 "
                                   f"{CROP_MIN_HEIGHT:.0%} 이상을 남겨야 합니다). 가공 프로그램에서 데이터셋을 다시 만드세요")
            if classes is None:
                classes, center = zip_classes, zip_center
            elif zip_classes != classes:
                raise DatasetError(f"{path.name}: 조향 클래스 {zip_classes} 가 앞의 데이터셋 {classes} 와 다릅니다. "
                                   "클래스가 같은 데이터셋끼리만 함께 학습할 수 있습니다")
            elif zip_center != center:
                raise DatasetError(f"{path.name}: 직진 각도 {zip_center} 가 앞의 데이터셋 {center} 와 다릅니다")
            if follow_dataset_crop:
                crop = zip_crop or fallback_crop
                if used_crop is None:
                    used_crop, used_from, first_name = crop, (path.name if zip_crop else None), path.name
                    pre = pre.with_crop(crop)
                elif not _same_crop(crop, used_crop):
                    if zip_crop is None:
                        why = " (이 zip 에는 범위가 적혀 있지 않아 기본값으로 읽었습니다. 이 zip 을 다시 만드세요)"
                    elif used_from is None:
                        why = f" (앞의 {first_name} 에 범위가 적혀 있지 않아 기본값으로 읽었습니다. 그 zip 을 다시 만드세요)"
                    else:
                        why = ""
                    raise DatasetError(
                        f"{path.name}: 자르는 범위({describe_crop(crop)})가 앞의 데이터셋({describe_crop(used_crop)})과 "
                        "다릅니다. 범위가 같은 데이터셋끼리만 함께 학습할 수 있습니다. 가공 프로그램에서 같은 범위로 다시 "
                        "만드세요" + why)
                crop_written = crop_written or zip_crop is not None
            _reject_repeated_frames(path.name, rows, seen)
            chunk = np.empty((len(rows), 3, pre.out_height, pre.out_width), dtype=np.uint8)
            for i, row in enumerate(rows):
                try:
                    angle = int(row["servo_angle"])
                    data = np.frombuffer(z.read(f"images/{row['image']}"), dtype=np.uint8)
                except (KeyError, ValueError, TypeError, zipfile.BadZipFile, OSError) as e:
                    raise DatasetError(f"{path.name}: labels.csv 의 {i + 1}번째 사진을 읽을 수 없습니다 ({e})") from None
                try:                            # 0바이트 사진(전원이 끊겨 저장되지 못한 것)은 imdecode 가 예외를 던진다
                    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
                except cv2.error:
                    bgr = None
                if bgr is None:
                    raise DatasetError(f"{path.name}: 사진이 손상됐습니다: {row['image']}")
                try:
                    chunk[i] = np.transpose(pre.to_rgb_uint8(bgr), (2, 0, 1))
                except (cv2.error, ValueError):
                    raise DatasetError(f"{path.name}: 사진을 모델 입력 크기로 바꿀 수 없습니다: {row['image']} "
                                       f"({bgr.shape[1]}×{bgr.shape[0]})") from None
                angles.append(angle)
                labels.append(nearest_class_index(angle, classes))
                names.append(f"{path.name}/{row['image']}")
                source = str(row.get("source") or "")                           # "세션/원본 사진 이름"
                sources.append(source)
                session = source.rpartition("/")[0]
                groups.append(group_of.setdefault((path.name, session), len(group_of)))
                if progress and (i % 200 == 0 or i == len(rows) - 1):
                    progress(path.name, i + 1, len(rows))
        chunks.append(chunk)
        datasets.append({"file": path.name, "frames": len(rows), "meta": meta})
    if not labels:
        raise DatasetError("데이터셋에 사진이 없습니다")
    images = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
    return Samples(images, np.array(labels, dtype=np.int64), np.array(angles, dtype=np.int64),
                   names, list(classes), datasets, center, np.array(groups, dtype=np.int64),
                   pre=pre, crop_from_dataset=crop_written, sources=sources)


def _reject_repeated_frames(zip_name: str, rows: List[dict], seen: Dict[str, str]) -> None:
    """같은 원본 사진이 두 번 들어오면 멈춘다 (같은 세션으로 만든 zip 두 개를 함께 준 경우).

    같은 사진이 학습과 시험에 함께 들어가면 모델이 "이미 본 사진"으로 시험을 치르게 되어 시험 정확도가 실제보다 높게 나온다.
    labels.csv 의 source 열(세션/원본 사진 이름)로 알아본다.
    """
    repeated = []
    for row in rows:
        source = str(row.get("source") or "")
        if not source:
            continue
        if source in seen:
            repeated.append(source)
        else:
            seen[source] = zip_name
    if repeated:
        other = seen[repeated[0]]
        where = "이 zip 안에 두 번" if other == zip_name else f"{other} 에도"
        raise DatasetError(f"{zip_name}: 사진 {len(repeated)}장이 {where} 들어 있습니다 (예: {repeated[0]}). 같은 사진이 학습과 "
                           "시험에 함께 들어가면 시험 정확도가 실제보다 높게 나옵니다. 사진이 겹치지 않는 데이터셋만 지정하세요")


# ---------------------------------------------------------------------------
# 2. 좌우 반전 증강
# ---------------------------------------------------------------------------
def mirror_labels(classes: Sequence[int], center: Optional[float] = None) -> List[int]:
    """사진을 좌우로 뒤집었을 때 클래스 번호가 어떻게 바뀌는지. 예: 5 클래스면 [4, 3, 2, 1, 0].

    규칙은 common.steering.mirror_class_indices 에 있다: 클래스가 직진 각도(center)를 가운데 두고 좌우 대칭일 때만
    뒤집을 수 있다. 대칭이 아니면 뒤집은 사진에 붙일 라벨이 없으므로 학습을 멈춘다.
    """
    try:
        return mirror_class_indices(classes, center)
    except ValueError as e:
        raise DatasetError(f"{e}. 좌우 반전 증강을 쓸 수 없으니 config.yaml 의 training.augment.hflip 을 none 으로 "
                           "바꾸세요") from None


# ---------------------------------------------------------------------------
# 3. Dataset
# ---------------------------------------------------------------------------
class SteeringDataset(Dataset):
    """사진과 조향 클래스 번호를 한 쌍씩 내주는 Dataset.

    hflip   "none"    원본만
            "random"  꺼낼 때마다 절반의 확률로 좌우 반전 (에폭마다 다른 모습을 본다. 길이는 그대로)
            "both"    원본과 뒤집은 것을 모두 사용 (길이가 2배. 좌우 클래스의 수가 정확히 같아진다)
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, classes: Sequence[int], hflip: str = "none",
                 center: Optional[float] = None):
        if hflip not in ("none", "random", "both"):
            raise ValueError(f"hflip 은 none | random | both: {hflip!r}")
        self.images = images
        self.labels = labels
        self.hflip = hflip
        self.mirror = mirror_labels(classes, center) if hflip != "none" else None

    def __len__(self) -> int:
        return len(self.labels) * (2 if self.hflip == "both" else 1)

    def __getitem__(self, i: int):
        n = len(self.labels)
        flip = False
        if self.hflip == "both":
            flip, i = i >= n, i % n
        elif self.hflip == "random":
            flip = bool(torch.rand(1).item() < 0.5)
        image = torch.from_numpy(self.images[i]).float() / 255.0      # (3, H, W), 0~1
        label = int(self.labels[i])
        if flip:
            image = torch.flip(image, dims=[2])                       # 가로축(W)을 뒤집는다
            label = self.mirror[label]
        return image, label

    def effective_labels(self) -> np.ndarray:
        """증강까지 넣은 뒤의 라벨 분포를 볼 때 쓴다 (random 은 원본과 같다고 본다)."""
        if self.hflip == "both":
            return np.concatenate([self.labels, np.array([self.mirror[int(v)] for v in self.labels])])
        return self.labels


# ---------------------------------------------------------------------------
# 4. DataLoader
# ---------------------------------------------------------------------------
def make_loaders(samples: Samples, split: Dict[str, np.ndarray], batch_size: int, shuffle: bool,
                 hflip: str, num_workers: int = 0, seed: int = 0) -> Dict[str, DataLoader]:
    """train / val / test 로더. 증강과 섞기는 학습 데이터에만 한다: 검증과 시험은 언제나 같은 사진을 같은 모습으로 봐야
    에폭끼리, 실행끼리 비교할 수 있다."""
    loaders = {}
    for name, idx in split.items():
        if len(idx) == 0:
            continue
        is_train = name == "train"
        ds = SteeringDataset(samples.images[idx], samples.labels[idx], samples.classes,
                             hflip=hflip if is_train else "none", center=samples.center)
        gen = torch.Generator().manual_seed(seed)
        loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle and is_train,
                                   num_workers=num_workers, generator=gen)
    return loaders


def channel_stats(images: np.ndarray, chunk: int = 512) -> tuple:
    """색 채널(R, G, B)마다의 평균과 표준편차 (0~1 기준). standardize 에서 쓰며, 반드시 학습 데이터로만 구한다.

    사진 전체를 한꺼번에 실수로 바꾸면 메모리가 8배로 불어나므로, 몇백 장씩 나눠 합과 제곱합을 쌓는다.
    """
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    count = 0
    for start in range(0, len(images), chunk):
        x = images[start:start + chunk].astype(np.float64) / 255.0
        total += x.sum(axis=(0, 2, 3))
        total_sq += np.square(x).sum(axis=(0, 2, 3))
        count += x.shape[0] * x.shape[2] * x.shape[3]
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean ** 2, 0.0))
    return [float(v) for v in mean], [float(max(v, 1e-6)) for v in std]
