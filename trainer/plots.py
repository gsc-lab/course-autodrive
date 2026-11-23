"""그림: 학습 곡선과 혼동 행렬. 화면을 띄우지 않고 PNG 파일로만 저장한다."""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# 그림의 축 이름은 영어로 쓰지만, 제목에는 실행 이름(--name 에 한글 가능)이 들어간다.
# 한글이 있는 글꼴을 찾아 두고, 없으면 제목에서 한글을 뺀다 (네모로 깨지고 경고가 줄줄이 찍히는 것을 막는다).
_KOREAN_FONTS = ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR")


def _use_korean_font() -> bool:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in _KOREAN_FONTS:
        if name in installed:
            matplotlib.rcParams["font.family"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    return False


_HAS_KOREAN_FONT = _use_korean_font()


def _title(text: str) -> str:
    if _HAS_KOREAN_FONT:
        return text
    return "".join(ch if ord(ch) < 0x1100 else "?" for ch in text)


def save_training_curves(history, path: Path, title: str = "") -> None:
    """손실, 정확도, 학습률을 에폭에 따라 그린다. 세로 점선은 검증 손실이 가장 낮았던 에폭."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    e = history.epoch

    ax = axes[0]
    ax.plot(e, history.train_loss, label="train loss", color="#1f77b4")
    ax.plot(e, history.val_loss, label="validation loss", color="#d62728")
    if history.best_epoch:
        ax.axvline(history.best_epoch, color="gray", linestyle="--", linewidth=1)
        right = history.best_epoch > 0.7 * max(e)                 # 오른쪽 끝에 가까우면 글자를 선의 왼쪽에
        ax.text(history.best_epoch, 0.97, f" best: epoch {history.best_epoch} ", transform=ax.get_xaxis_transform(),
                ha="right" if right else "left", va="top", fontsize=9, color="gray")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title("Loss"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(e, history.train_acc, label="train accuracy", color="#1f77b4")
    ax.plot(e, history.val_acc, label="validation accuracy", color="#d62728")
    if history.best_epoch:
        ax.axvline(history.best_epoch, color="gray", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy"); ax.set_title("Accuracy"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(e, history.lr, color="#2ca02c")
    ax.set_xlabel("epoch"); ax.set_ylabel("learning rate"); ax.set_title("Learning rate"); ax.grid(alpha=0.3)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

    if title:
        fig.suptitle(_title(title), fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_confusion_matrix(cm: np.ndarray, class_names: List[str], path: Path, title: str = "") -> None:
    """칸의 숫자는 사진 수, 색은 그 행(정답 클래스) 안에서의 비율. 대각선이 진할수록 좋다."""
    k = len(class_names)
    row_sum = cm.sum(axis=1, keepdims=True)
    ratio = np.divide(cm, row_sum, out=np.zeros(cm.shape, dtype=float), where=row_sum > 0)
    fig, ax = plt.subplots(figsize=(1.1 * k + 2.6, 1.0 * k + 2.2))
    im = ax.imshow(ratio, cmap="Blues", vmin=0, vmax=1)
    for i in range(k):
        for j in range(k):
            if cm[i, j]:
                ax.text(j, i, f"{int(cm[i, j])}\n{ratio[i, j] * 100:.0f}%", ha="center", va="center", fontsize=9,
                        color="white" if ratio[i, j] > 0.55 else "black")
    ax.set_xticks(range(k)); ax.set_xticklabels(class_names)
    ax.set_yticks(range(k)); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted steering angle (deg)")
    ax.set_ylabel("True steering angle (deg)")
    ax.set_title(_title(title) if title else "Confusion matrix (test set)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="share of the true class")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
