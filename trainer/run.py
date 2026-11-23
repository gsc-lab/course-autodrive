"""학습 한 번의 전체 흐름. 각 단계는 해당 모듈의 함수를 순서대로 부를 뿐이다.

    1. 데이터 읽기        data.load_samples
    2. 데이터 나누기      split.split_indices
    3. DataLoader         data.make_loaders
    4. 모델               models.build_model
    5. 손실, 옵티마이저   losses.build_loss, optim.build_optimizer / build_scheduler
    6. 학습               engine.fit
    7. 저장               model.pth, history.csv, loss_curve.png, split.csv, train_config.json

결과 폴더 models/<실행 이름>/ 은 그 뒤 evaluate.py(시험 분석)와 export.py(ONNX)가 이어서 쓴다.
"""
from __future__ import annotations

import csv
import json
import random
import re
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import torch

from common.config import REPO_ROOT, Config
from common.preprocess import Preprocessor, describe_crop
from trainer import __version__
from trainer.data import Samples, channel_stats, load_samples, make_loaders, mirror_labels
from trainer.engine import fit
from trainer.losses import build_loss
from trainer.models import InputScaler, SteeringNet, build_model, count_parameters
from trainer.optim import build_optimizer, build_scheduler
from trainer.plots import save_training_curves
from trainer.metrics import baseline
from trainer.split import SPLIT_NAMES, class_counts, split_indices
from trainer.text import rjust

CHECKPOINT_SCHEMA_VERSION = 1


class TrainError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# 작은 도구들
# ---------------------------------------------------------------------------
def pick_device(want: str) -> torch.device:
    if want == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if want == "cuda":
        raise TrainError("training.device 가 cuda 인데 이 PC 에서 GPU 를 쓸 수 없습니다. auto 또는 cpu 로 바꾸거나, "
                         "GPU 용 torch 를 설치하세요 (uv sync --extra gui --extra train)")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """무작위가 쓰이는 곳(데이터 나누기, 초기 가중치, 섞기, 증강)을 고정해 같은 설정이면 같은 결과가 나오게 한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # GPU 의 합성곱은 빠른 방법을 그때그때 골라 쓰는데, 그러면 같은 설정으로 두 번 돌려도 결과가 조금씩 달라진다.
    # 설정 하나만 바꿔 비교하는 실습에서는 "바꾼 것 때문에 달라졌다"고 말할 수 있어야 하므로 고정한다 (조금 느려진다).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_run_dir(output_root: str | Path, name: Optional[str], now: Optional[datetime] = None) -> Path:
    """models/<날짜_시각[_이름]>/ 을 만든다. 학습을 시작하기 전에 만들어, 쓸 수 없는 곳이면 바로 알 수 있게 한다."""
    root = Path(output_root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    run_id = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    if name:
        run_id += "_" + re.sub(r"[^0-9A-Za-z가-힣_.-]+", "-", name).strip("-")
    run_dir, n = root / run_id, 1
    try:
        root.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                run_dir.mkdir()
                break
            except FileExistsError:                 # 같은 초에 같은 이름으로 두 번 실행
                n += 1
                run_dir = root / f"{run_id}_{n}"
    except OSError as e:
        raise TrainError(f"결과 폴더를 만들 수 없습니다: {run_dir} ({e}). 다른 곳에 저장하려면 config.yaml 의 training 에 "
                         "output_root: <폴더> 를 적으세요 (기본값 models)") from None
    return run_dir


def describe_split(samples: Samples, split: Dict[str, np.ndarray]) -> str:
    k = len(samples.classes)
    w = max(6, max(len(str(c)) for c in samples.classes) + 2)
    lines = [" " * 8 + rjust("합계", 8) + "".join(rjust(f"{c}°", w) for c in samples.classes)]
    for name in ("all",) + SPLIT_NAMES:
        idx = np.arange(len(samples)) if name == "all" else split[name]
        counts = class_counts(samples.labels[idx], k)
        label = {"all": "전체", "train": "train", "val": "val", "test": "test"}[name]
        lines.append(rjust(label, 8) + rjust(len(idx), 8) + "".join(rjust(int(c), w) for c in counts))
    return "\n".join(lines)


def split_baselines(samples: Samples, split: Dict[str, np.ndarray], hflip: str) -> tuple:
    """기준선: 사진을 보지 않고 한 가지 각도만 답해도 나오는 정확도. 에폭 표의 train acc / val acc 를 이 값과 견주어 본다.

    돌려주는 값: (학습 묶음의 클래스별 사진 수, train 기준선, val 기준선). 기준선은 (가장 많은 클래스의 번호, 그 비율).
    묶음마다 따로 구한다: 가장 많은 각도가 묶음마다 다를 수 있다 (stratify 를 끄거나 block_size 가 클 때).
    train 은 좌우 반전까지 넣은 뒤의 분포로 구한다: 모델이 학습 중에 보는 정답이 그것이고 train acc 도 그 사진들로 잰다.
    (가장 많은 각도가 직진이 아니면, 반전 뒤에는 그 각도의 사진이 절반으로 줄어든다.)
    """
    k = len(samples.classes)
    train_counts = class_counts(samples.labels[split["train"]], k).astype(float)
    if hflip != "none":                                 # 반전까지 넣으면 좌우 클래스의 수가 평균으로 같아진다
        train_counts = (train_counts + train_counts[mirror_labels(samples.classes, samples.center)]) / 2.0
    return train_counts, baseline(train_counts), baseline(class_counts(samples.labels[split["val"]], k))


def write_split_csv(path: Path, samples: Samples, split: Dict[str, np.ndarray]) -> None:
    where = {}
    for name, idx in split.items():
        for i in idx:
            where[int(i)] = name
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # source = 원본 사진의 출처(세션/원본 사진 이름). 나중에 다른 zip 으로 시험할 때, 그 zip 에 학습에 쓴 사진이
        # 섞여 있는지 이 열로 알아본다 (read_seen_sources). zip 을 다시 만들면 사진 이름은 바뀌어도 출처는 그대로다.
        w.writerow(["index", "image", "servo_angle", "class_index", "class_angle", "split", "source"])
        for i in range(len(samples)):
            w.writerow([i, samples.names[i], int(samples.angles[i]), int(samples.labels[i]),
                        samples.classes[int(samples.labels[i])], where[i], samples.sources[i] if samples.sources else ""])


def read_split_csv(path: Path) -> tuple:
    """돌려주는 값: (사진 이름 목록, {"train": 번호 배열, …})"""
    names, split = [], {name: [] for name in SPLIT_NAMES}
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:     # utf-8-sig: Excel 이 붙이는 BOM 이 있어도 읽는다
            for row in csv.DictReader(f):
                names.append(row["image"])
                split[row["split"]].append(int(row["index"]))
    except FileNotFoundError:
        raise TrainError(f"split.csv 가 없습니다: {path}. 이 폴더는 학습이 끝까지 저장된 결과 폴더가 아닙니다") from None
    except (KeyError, ValueError, TypeError, UnicodeDecodeError):
        raise TrainError(f"split.csv 를 읽을 수 없습니다: {path} (손으로 고쳤거나 다른 프로그램에서 다시 저장한 파일입니다). "
                         "다시 학습하거나, --data 로 시험할 데이터셋 zip 을 지정하세요") from None
    return names, {k: np.array(v, dtype=np.int64) for k, v in split.items()}


def read_seen_sources(path: Path) -> Optional[set]:
    """split.csv → 이 모델이 학습(train)과 검증(val)에 쓴 사진의 출처. 출처를 적기 전에 만든 결과 폴더면 None.

    검증 사진도 "쓴 사진"이다: 가중치를 고치지는 않았지만 가장 좋은 에폭을 고르는 데 썼다. 시험(test) 사진은 쓰지 않았다.
    """
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if "source" not in (reader.fieldnames or []):
                return None
            return {row["source"] for row in reader if row.get("source") and row.get("split") in ("train", "val")}
    except (OSError, ValueError, TypeError, UnicodeDecodeError):
        return None


# ---------------------------------------------------------------------------
# 체크포인트 (model.pth)
# ---------------------------------------------------------------------------
def save_checkpoint(path: Path, model: SteeringNet, model_name: str, classes: Sequence[int],
                    input_shape: Sequence[int], pre: Preprocessor) -> None:
    torch.save({
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": model_name,
        "classes": [int(c) for c in classes],
        "input_shape": [int(v) for v in input_shape],
        "scaler": model.scaler.to_dict(),
        "preprocess": pre.to_dict(),
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
    }, path)


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> tuple:
    """model.pth → (모델, 체크포인트 dict). 모델은 eval 상태로 돌려준다."""
    if not Path(path).is_file():
        raise TrainError(f"학습된 모델이 없습니다: {path}")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        scaler = InputScaler(ckpt["scaler"]["mode"], ckpt["scaler"]["mean"], ckpt["scaler"]["std"])
        model = build_model(ckpt["model"], len(ckpt["classes"]), ckpt["input_shape"], scaler)
        model.load_state_dict(ckpt["state_dict"])
        Preprocessor.from_dict(ckpt["preprocess"])      # 시험 분석과 내보내기가 쓴다. 없거나 틀리면 여기서 알린다
    except Exception as e:                  # 덜 복사된 파일, 다른 프로그램의 .pth, MODELS 에서 지운 모델 등
        raise TrainError(f"model.pth 를 읽을 수 없습니다: {path} (손상됐거나 이 학습기로 만든 파일이 아닙니다. "
                         f"{type(e).__name__}: {str(e).splitlines()[0][:120] if str(e) else ''})") from None
    return model.to(device).eval(), ckpt


# ---------------------------------------------------------------------------
# 학습
# ---------------------------------------------------------------------------
def train(zip_paths: Sequence[str | Path], cfg: Config, name: Optional[str] = None,
          log: Callable[[str], None] = print) -> tuple:
    """학습을 한 번 실행한다. 돌려주는 값: (결과 폴더, 메모리에 올린 데이터) — 데이터는 시험 분석이 다시 쓴다."""
    t = cfg.training
    seed_everything(t.seed)
    device = pick_device(t.device)

    # 1. 데이터 읽기 --------------------------------------------------------
    log("[1/7] 데이터 읽기")
    # 모델 입력 크기는 config.yaml 에서, 사진에서 자를 범위는 데이터셋에서 온다 (가공 프로그램에서 사진을 보며 정한 값).
    # 범위가 적혀 있지 않은 예전 데이터셋만 기본값으로 자른다.
    samples = load_samples(zip_paths, Preprocessor.from_config(cfg.preprocess))
    pre = samples.pre
    k = len(samples.classes)
    log(f"      사진 {len(samples)}장, 조향 클래스 {k}개 {samples.classes}, 모델 입력 {pre.out_width}×{pre.out_height}")
    log(f"      자르는 범위: {describe_crop(pre.crop)} "
        + ("(가공 프로그램에서 정한 범위)" if samples.crop_from_dataset else "(데이터셋에 범위가 없어 기본값을 씁니다)"))
    off = int(np.sum(samples.angles != np.array(samples.classes)[samples.labels]))
    if off:
        log(f"      클래스에 없는 각도의 사진 {off}장은 가장 가까운 클래스로 묶었습니다.")
    if len(samples) < 3 * k:
        raise TrainError(f"사진이 너무 적습니다: {len(samples)}장 (클래스 {k}개)")

    # 2. 데이터 나누기 ------------------------------------------------------
    log("[2/7] 데이터 나누기 (train / val / test)")
    sp = t.split
    split = split_indices(samples.labels, sp.train, sp.val, sp.test, stratify=sp.stratify,
                          block_size=sp.block_size, seed=t.seed, groups=samples.groups)
    for line in describe_split(samples, split).splitlines():
        log("      " + line)
    if len(split["val"]) == 0:
        raise TrainError("검증(val) 데이터가 0장입니다. training.split.val 을 키우거나 사진을 더 모으세요")
    train_counts, (top_train, base_train), (top_val, base_val) = split_baselines(samples, split, t.augment.hflip)
    if top_train == top_val:
        log(f"      참고: 가장 많은 {samples.classes[top_train]}° 만 답해도 정확도가 train {base_train:.3f}, val {base_val:.3f} 입니다. "
            "이보다 높아야 배운 것입니다.")
    else:
        log(f"      참고: 한 가지 각도만 답해도 정확도가 train {base_train:.3f} ({samples.classes[top_train]}°), "
            f"val {base_val:.3f} ({samples.classes[top_val]}°) 입니다. 이보다 높아야 배운 것입니다.")
    missing = [f"{c}°" for c, n in zip(samples.classes, class_counts(samples.labels[split["train"]], k)) if n == 0]
    if missing:
        log(f"      주의: 학습 데이터에 {', '.join(missing)} 사진이 한 장도 없습니다. 모델은 이 방향을 배울 수 없습니다.")

    # 3. DataLoader ---------------------------------------------------------
    log("[3/7] DataLoader")
    loaders = make_loaders(samples, split, t.loader.batch_size, t.loader.shuffle, t.augment.hflip,
                           t.loader.num_workers, seed=t.seed)
    flip_note = {"none": "좌우 반전 없음", "random": "좌우 반전: 절반 확률",
                 "both": "좌우 반전: 원본 + 뒤집은 것"}[t.augment.hflip]
    log(f"      배치 {t.loader.batch_size}장, 학습 데이터 섞기 {'켬' if t.loader.shuffle else '끔'}, {flip_note} "
        f"→ 한 에폭에 {len(loaders['train'])} 배치 ({len(loaders['train'].dataset)}장)")

    # 4. 모델 ---------------------------------------------------------------
    log("[4/7] 모델")
    train_images = samples.images[split["train"]]
    mean, std = channel_stats(train_images) if t.rescale == "standardize" else ([0.0] * 3, [1.0] * 3)
    scaler = InputScaler(t.rescale, mean, std)
    try:
        model = build_model(t.model, k, pre.input_shape, scaler).to(device)
    except RuntimeError as e:               # 합성곱을 지나며 사진이 0 보다 작아지는 등, 이 입력 크기로는 만들 수 없는 모델
        raise TrainError(f"모델 {t.model} 을 입력 크기 {pre.out_width}×{pre.out_height} 로 만들 수 없습니다 "
                         f"({str(e).splitlines()[0][:120]}). config.yaml 의 preprocess.out_width / out_height 가 "
                         "이 모델에 너무 작지 않은지 확인하세요") from None
    log(f"      {t.model}: 가중치 {count_parameters(model):,}개, 입력 스케일링 {t.rescale}"
        + (f" (평균 {[round(v, 3) for v in mean]}, 표준편차 {[round(v, 3) for v in std]})"
           if t.rescale == "standardize" else "")
        + f", 장치 {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # 5. 손실 함수와 옵티마이저 ---------------------------------------------
    log("[5/7] 손실 함수, 옵티마이저")
    loss_fn = build_loss(t.loss.name, t.loss.class_weight, train_counts).to(device)
    o = t.optimizer
    optimizer = build_optimizer(model, o.name, o.lr, o.weight_decay, o.momentum)
    scheduler = build_scheduler(optimizer, t.epochs, o.schedule, o.warmup_epochs, o.min_lr_ratio)
    log(f"      {t.loss.name} (클래스 가중치 {t.loss.class_weight}), {o.name} lr={o.lr:g}, 학습률 스케줄 {o.schedule}")
    log(f"      참고: 아무것도 배우지 않은 모델의 손실은 약 ln({k}) = {np.log(k):.3f} 입니다.")

    # 6. 학습 ---------------------------------------------------------------
    es = t.early_stopping
    run_dir = make_run_dir(t.output_root, name)        # 학습 전에 만든다: 쓸 수 없는 곳이면 여기서 바로 멈춘다
    log(f"[6/7] 학습 (최대 {t.epochs} 에폭" + (f", 조기 종료 patience {es.patience}" if es.enabled else "")
        + "). 그만하려면 Ctrl+C: 그때까지 가장 좋았던 모델로 마무리한다")
    try:
        history = fit(model, loaders["train"], loaders["val"], loss_fn, optimizer, scheduler, device, t.epochs,
                      early_stopping=es.enabled, patience=es.patience, min_delta=es.min_delta,
                      log=lambda msg: log("      " + msg))
        if history.best_epoch == 0:
            raise TrainError("쓸 수 있는 모델이 없습니다: 첫 에폭부터 손실이 nan/inf 였습니다. "
                             "training.optimizer.lr 을 낮추거나 training.rescale 을 minmax 또는 standardize 로 두세요")
        _save_run(run_dir, model, history, samples, split, cfg, pre, scaler, device, zip_paths, log)
    except BaseException:
        # 끝까지 저장하지 못한 폴더는 남기지 않는다. model.pth 만 있는 반쪽 폴더는 정상 실행처럼 보이지만
        # 시험 분석도 내보내기도 할 수 없다. (이 폴더는 이번 실행이 방금 만든 것이라 통째로 지워도 된다.)
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    return run_dir, samples


def _save_run(run_dir: Path, model: SteeringNet, history, samples: Samples, split: Dict[str, np.ndarray],
              cfg: Config, pre: Preprocessor, scaler: InputScaler, device: torch.device,
              zip_paths: Sequence[str | Path], log: Callable[[str], None]) -> None:
    # 7. 저장 ---------------------------------------------------------------
    t, o = cfg.training, cfg.training.optimizer
    log(f"[7/7] 저장: {run_dir}")
    save_checkpoint(run_dir / "model.pth", model, t.model, samples.classes, pre.input_shape, pre)
    write_split_csv(run_dir / "split.csv", samples, split)
    with open(run_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "seconds"])
        w.writeheader()
        for row in history.rows():
            w.writerow({key: (round(v, 6) if isinstance(v, float) else v) for key, v in row.items()})
    save_training_curves(history, run_dir / "loss_curve.png", title=f"{run_dir.name}  ({t.model}, lr={o.lr:g}, "
                                                                     f"{o.schedule}, batch {t.loader.batch_size})")
    info = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "tool": {"name": "trainer", "version": __version__, "torch": torch.__version__},
        "device": str(device),
        "datasets": [{"path": str(Path(p).resolve()), **d} for p, d in zip(zip_paths, samples.datasets)],
        "classes": samples.classes,
        "frames": {name: int(len(split[name])) for name in SPLIT_NAMES},
        "training": asdict(t),
        "preprocess": pre.to_dict(),
        "scaler": scaler.to_dict(),
        "parameters": count_parameters(model),
        "result": {"epochs_run": len(history.epoch), "best_epoch": history.best_epoch,
                   "best_val_loss": history.best_val_loss,
                   "best_val_acc": history.val_acc[history.best_epoch - 1],
                   "stopped_early": history.stopped_early, "diverged": history.diverged,
                   "interrupted": history.interrupted,
                   "seconds": round(sum(history.seconds), 1)},
    }
    (run_dir / "train_config.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"      model.pth, history.csv, loss_curve.png, split.csv, train_config.json  "
        f"(학습 시간 {sum(history.seconds):.1f}초)")
