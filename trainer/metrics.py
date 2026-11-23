"""분류 결과 지표: 혼동 행렬과 클래스별 정확도 · 정밀도 · 재현율 · F1.

혼동 행렬 cm[i][j] = "정답이 i 번 클래스인 사진을 모델이 j 번이라고 답한 횟수". 대각선이 맞힌 것이다.

어떤 클래스 c 를 기준으로 사진을 넷으로 나눈다.
    TP  정답도 c, 예측도 c            FN  정답은 c 인데 다른 것으로 예측 (놓침)
    FP  정답은 c 가 아닌데 c 로 예측   TN  정답도 예측도 c 가 아님

    정확도 accuracy  = (TP + TN) / 전체       이 클래스에 대해 옳게 판단한 비율
    정밀도 precision = TP / (TP + FP)         모델이 c 라고 한 것 중 진짜 c 의 비율
    재현율 recall    = TP / (TP + FN)         진짜 c 중 모델이 찾아낸 비율
    F1               = 정밀도와 재현율의 조화 평균

기준선(baseline): 사진을 보지 않고 가장 많은 클래스 하나만 답해도 나오는 정확도. 직진 사진이 70 % 인 데이터에서는 0.70 이다.
전체 정확도는 이 값과 견주어 읽는다: 기준선과 같으면 아무것도 배우지 못한 것이다.

torch 없이 numpy 만 쓴다.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from trainer.text import rjust, width


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)):
        cm[t, p] += 1
    return cm


def _div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def per_class_metrics(cm: np.ndarray) -> List[Dict[str, float]]:
    total = cm.sum()
    rows = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp
        precision = _div(tp, tp + fp)
        recall = _div(tp, tp + fn)
        rows.append({
            "support": int(tp + fn),                 # 이 클래스의 정답 사진 수
            "accuracy": _div(tp + tn, total),
            "precision": precision,
            "recall": recall,
            "f1": _div(2 * precision * recall, precision + recall),
        })
    return rows


def baseline(counts) -> tuple:
    """클래스별 사진 수 → (가장 많은 클래스의 번호, 그 클래스만 답했을 때의 정확도). 사진이 없으면 (0, 0.0)."""
    counts = np.asarray(counts)
    top = int(counts.argmax()) if counts.size else 0
    return top, _div(counts[top], counts.sum()) if counts.size else 0.0


def summarize(cm: np.ndarray) -> Dict[str, float]:
    """전체 지표. within_one 은 정답 또는 바로 옆 클래스로 답한 비율 (조향에서는 한 칸 차이가 덜 치명적이다).
    baseline_accuracy 는 이 사진들에서 가장 많은 클래스(baseline_class_index, 클래스 번호)만 답했을 때의 정확도."""
    total = cm.sum()
    top, base = baseline(cm.sum(axis=1))
    rows = per_class_metrics(cm)
    present = [r for r in rows if r["support"] > 0]
    k = cm.shape[0]
    near = sum(int(cm[i, j]) for i in range(k) for j in range(k) if abs(i - j) <= 1)
    return {
        "samples": int(total),
        "accuracy": _div(np.trace(cm), total),
        "baseline_accuracy": base,
        "baseline_class_index": top,
        "within_one": _div(near, total),
        "macro_precision": float(np.mean([r["precision"] for r in present])) if present else 0.0,
        "macro_recall": float(np.mean([r["recall"] for r in present])) if present else 0.0,
        "macro_f1": float(np.mean([r["f1"] for r in present])) if present else 0.0,
    }


def format_report(cm: np.ndarray, class_names: List[str]) -> str:
    """터미널과 test_report.txt 에 쓰는 표."""
    rows = per_class_metrics(cm)
    s = summarize(cm)
    w = max(6, max(width(n) for n in class_names) + 1)
    lines = ["혼동 행렬 (행 = 정답, 열 = 모델의 예측)", ""]
    lines.append(" " * (w + 3) + "".join(rjust(n, w) for n in class_names))
    for name, row in zip(class_names, cm):
        lines.append(rjust(name, w) + " | " + "".join(rjust(int(v), w) for v in row))
    head = ("클래스", "사진 수", "정확도", "정밀도", "재현율", "F1")
    lines += ["", "클래스별 지표", "", rjust(head[0], w) + "".join(rjust(h, 9) for h in head[1:])]
    for name, r in zip(class_names, rows):
        lines.append(rjust(name, w) + rjust(r["support"], 9)
                     + "".join(rjust(f"{r[key]:.3f}", 9) for key in ("accuracy", "precision", "recall", "f1")))
    lines += ["",
              f"전체 정확도        {s['accuracy']:.3f}   ({int(np.trace(cm))} / {s['samples']})",
              f"기준선             {s['baseline_accuracy']:.3f}   (가장 많은 {class_names[s['baseline_class_index']]} 만 답해도 나오는 정확도)",
              f"한 칸 이내 정확도  {s['within_one']:.3f}   (정답 또는 바로 옆 클래스로 답한 비율)",
              f"매크로 평균        정밀도 {s['macro_precision']:.3f}   재현율 {s['macro_recall']:.3f}   "
              f"F1 {s['macro_f1']:.3f}"]
    return "\n".join(lines)
