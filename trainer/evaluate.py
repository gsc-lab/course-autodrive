"""시험 분석: 학습에 쓰지 않은 시험(test) 사진으로 모델을 잰다.

    uv run -m trainer test models/<실행 이름>
    uv run -m trainer test models/<실행 이름> --data 다른_날_데이터셋.zip    # 다른 데이터 전체를 시험으로

만드는 파일 (결과 폴더 안)
    test_report.txt        혼동 행렬과 클래스별 정확도 · 정밀도 · 재현율 · F1
    confusion_matrix.png   혼동 행렬 그림
    test_predictions.csv   사진마다의 정답, 예측, 클래스별 확률 (틀린 사진을 찾아볼 때)
    test_metrics.json      같은 숫자를 프로그램이 읽기 좋은 형식으로
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.preprocess import Preprocessor, crop_from_dict, describe_crop
from trainer.data import Samples, SteeringDataset, load_samples
from trainer.engine import predict
from trainer.metrics import confusion_matrix, format_report, per_class_metrics, summarize
from trainer.plots import save_confusion_matrix
from trainer.run import TrainError, load_checkpoint, pick_device, read_seen_sources, read_split_csv


def frames_seen_in_training(run_dir: Path, samples: Samples) -> tuple:
    """--data 로 준 데이터셋에 이 모델이 학습할 때 쓴 사진이 섞여 있는가. 돌려주는 값: (알릴 문장 또는 "", 겹친 장수 또는 None).

    학습에 쓴 zip, 또는 같은 세션으로 다시 만든 zip 으로 시험하면 "이미 본 사진"으로 시험을 치르는 셈이라 정확도가 실제보다
    높게 나온다. 막지는 않고 몇 장인지 알린다 (일부만 겹친 데이터셋으로도 시험할 수 있어야 한다).
    출처를 split.csv 에 적기 전에 만든 결과 폴더는 장수를 셀 수 없으므로, 세션 이름이 겹치는지만 본다 (장수는 None).
    """
    seen = read_seen_sources(run_dir / "split.csv")
    if seen is not None:
        overlap = sum(1 for s in (samples.sources or []) if s and s in seen)
        if not overlap:
            return "", 0
        return (f"주의: 시험 사진 {len(samples)}장 중 {overlap}장은 이 모델을 학습할 때 쓴 사진입니다. "
                "이미 본 사진이라 정확도가 실제보다 높게 나옵니다."), overlap
    try:
        info = json.loads((run_dir / "train_config.json").read_text(encoding="utf-8-sig"))
        trained = {str(s) for d in info["datasets"] for s in (d.get("meta") or {}).get("sessions") or []}
        given = {str(s) for d in samples.datasets for s in (d.get("meta") or {}).get("sessions") or []}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return "", None
    shared = sorted(trained & given)
    if not shared:
        return "", None
    return (f"주의: 이 데이터셋은 학습에 쓴 세션({', '.join(shared)})으로 만든 것입니다. 학습에 쓴 사진이 들어 있으면 "
            "정확도가 실제보다 높게 나옵니다."), None


def analyze(run_dir: str | Path, samples: Optional[Samples] = None, data: Optional[Sequence[str | Path]] = None,
            device: str = "auto", log: Callable[[str], None] = print) -> Optional[dict]:
    """시험 데이터로 예측하고 보고서를 쓴다. 돌려주는 값: 지표 dict (시험 데이터가 없으면 None).

    samples  학습 직후에는 이미 메모리에 있는 데이터를 그대로 넘긴다 (zip 을 다시 읽지 않게)
    data     지정하면 이 데이터셋 zip 의 모든 사진을 시험 데이터로 쓴다. 결과는 test_<zip 이름>/ 아래에 쓴다
    """
    run_dir = Path(run_dir)
    dev = pick_device(device)
    model, ckpt = load_checkpoint(run_dir / "model.pth", dev)
    classes = ckpt["classes"]
    pre = Preprocessor.from_dict(ckpt["preprocess"])
    out_dir = run_dir
    seen_note, seen_count = "", None            # --data 로 준 데이터셋에 학습에 쓴 사진이 섞여 있을 때만 채운다

    # 시험할 때는 언제나 모델이 학습한 범위(pre)로 자른다. 데이터셋에 다른 범위가 적혀 있어도 따르지 않는다:
    # 차에서도 모델은 자기 범위로 잘린 사진을 받으므로, 그렇게 재야 차에서의 성능을 알 수 있다.
    if data:
        samples = load_samples(data, pre, follow_dataset_crop=False)
        if samples.classes != classes:
            raise TrainError(f"데이터셋의 조향 클래스 {samples.classes} 가 모델의 클래스 {classes} 와 다릅니다")
        for d in samples.datasets:
            try:
                other = crop_from_dict(d["meta"].get("preprocess"))
            except ValueError:                              # 읽을 수 없는 범위: 어차피 쓰지 않으므로 견줄 것이 없다
                other = None
            if other and (abs(other[0] - pre.crop[0]) > 1e-6 or abs(other[1] - pre.crop[1]) > 1e-6):
                log(f"      주의: {d['file']} 의 자르는 범위({describe_crop(other)})가 모델이 학습한 범위"
                    f"({describe_crop(pre.crop)})와 다릅니다. 모델의 범위로 잘라 시험합니다.")
        seen_note, seen_count = frames_seen_in_training(run_dir, samples)
        idx = np.arange(len(samples))
        stems = [Path(p).stem for p in data]              # zip 을 여러 개 주면 폴더 이름에 모두 넣는다 (서로 덮어쓰지 않게)
        out_dir = run_dir / ("test_" + "+".join(stems))[:120]
        out_dir.mkdir(exist_ok=True)
    else:
        names, split = read_split_csv(run_dir / "split.csv")
        if samples is None:
            try:
                info = json.loads((run_dir / "train_config.json").read_text(encoding="utf-8-sig"))
                paths = [d["path"] for d in info["datasets"]]
            except (OSError, ValueError, KeyError, TypeError):
                raise TrainError(f"train_config.json 을 읽을 수 없습니다: {run_dir} (학습에 쓴 데이터셋 zip 의 위치를 알 수 없습니다). "
                                 "--data 로 시험할 데이터셋 zip 을 지정하세요") from None
            gone = [p for p in paths if not Path(p).is_file()]
            if gone:
                raise TrainError("학습에 쓴 데이터셋 zip 을 찾을 수 없습니다: " + ", ".join(gone)
                                 + "\n      zip 을 원래 자리에 두거나 --data 로 시험할 zip 을 지정하세요")
            samples = load_samples(paths, pre, follow_dataset_crop=False)
        if samples.names != names:
            raise TrainError("데이터셋의 사진 목록이 학습할 때와 다릅니다 (split.csv 와 맞지 않음). "
                             "학습 뒤에 zip 이 바뀌었다면 다시 학습하세요")
        idx = split["test"]
        if len(idx) == 0:
            log("      시험(test) 데이터가 0장이라 분석을 건너뜁니다 (training.split.test 가 0).")
            return None

    loader = DataLoader(SteeringDataset(samples.images[idx], samples.labels[idx], classes, hflip="none"),
                        batch_size=256, shuffle=False)
    probs, y_true = predict(model, loader, dev)
    y_pred = probs.argmax(axis=1)

    k = len(classes)
    class_names = [f"{c}°" for c in classes]
    cm = confusion_matrix(y_true, y_pred, k)
    summary = summarize(cm)
    summary["baseline_angle"] = classes[summary["baseline_class_index"]]      # 파일에는 각도로도 적는다 (번호만 있으면 헷갈린다)
    report = format_report(cm, class_names)

    title = (f"시험 분석: {run_dir.name}   (시험 사진 {len(idx)}장"
             + (", " + ", ".join(Path(p).name for p in data) if data else "") + ")")
    (out_dir / "test_report.txt").write_text(title + "\n" + (seen_note + "\n" if seen_note else "") + "\n" + report + "\n",
                                             encoding="utf-8")
    save_confusion_matrix(cm, [str(c) for c in classes], out_dir / "confusion_matrix.png",
                          title=f"Confusion matrix, test set ({len(idx)} images)  accuracy {summary['accuracy']:.3f}")
    with open(out_dir / "test_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "image", "servo_angle", "true_angle", "pred_angle", "correct", "class_distance",
                    "confidence"] + [f"p_{c}" for c in classes])
        for row, i in enumerate(idx):
            t, p = int(y_true[row]), int(y_pred[row])
            w.writerow([int(i), samples.names[int(i)], int(samples.angles[int(i)]), classes[t], classes[p],
                        int(t == p), abs(t - p), f"{probs[row, p]:.4f}"] + [f"{v:.4f}" for v in probs[row]])
    metrics = {
        "run_id": run_dir.name,
        "data": [str(Path(p).name) for p in data] if data else "test split",
        "classes": classes,
        "frames_seen_in_training": seen_count,          # --data 일 때: 학습에 쓴 사진과 겹친 장수 (셀 수 없으면 null)
        "summary": summary,
        "per_class": {str(c): m for c, m in zip(classes, per_class_metrics(cm))},
        "confusion_matrix": cm.tolist(),
    }
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    if seen_note:
        log("      " + seen_note)
    for line in report.splitlines():
        log("      " + line)
    log("")
    log(f"      저장: {out_dir}")
    log("      test_report.txt, confusion_matrix.png, test_predictions.csv, test_metrics.json")
    return metrics
