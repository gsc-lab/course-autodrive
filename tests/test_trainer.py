"""학습기 가운데 torch 없이 도는 부분: 데이터 나누기, 지표, 설정 검증. (torch 가 필요한 시험은 test_trainer_torch.py)

pytest.importorskip 은 파일 전체를 건너뛰므로 두 파일로 나눴다: 이 파일은 Jetson 과 `uv sync --group dev` 만 한 PC 에서도 돈다.
"""
from pathlib import Path

import numpy as np
import pytest
import yaml

from common.config import DEFAULT_CONFIG_PATH, ConfigError, load_config
from trainer.metrics import confusion_matrix, format_report, per_class_metrics, summarize
from trainer.split import class_counts, split_indices
from trainer.text import width

CLASSES = [30, 60, 90, 120, 150]


# ---------------------------------------------------------------------------
# 데이터 나누기
# ---------------------------------------------------------------------------
def _labels(counts=(200, 50, 600, 30, 60)):
    rng = np.random.default_rng(0)
    return rng.permutation(np.repeat(np.arange(len(counts)), counts))


def test_split_covers_every_frame_once_and_keeps_ratios():
    labels = _labels()
    sp = split_indices(labels, 0.7, 0.15, 0.15, stratify=True, seed=1)
    merged = np.concatenate([sp["train"], sp["val"], sp["test"]])
    assert sorted(merged.tolist()) == list(range(len(labels)))
    assert abs(len(sp["train"]) / len(labels) - 0.7) < 0.02
    assert abs(len(sp["val"]) / len(labels) - 0.15) < 0.02


def test_stratified_split_keeps_class_shares_in_every_part():
    labels = _labels()
    sp = split_indices(labels, 0.7, 0.15, 0.15, stratify=True, seed=1)
    whole = class_counts(labels, 5) / len(labels)
    for name in ("train", "val", "test"):
        part = class_counts(labels[sp[name]], 5) / len(sp[name])
        assert np.abs(part - whole).max() < 0.02, name
        assert class_counts(labels[sp[name]], 5).min() > 0, name      # 적은 클래스도 빠지지 않는다


def test_split_is_reproducible_and_changes_with_seed():
    labels = _labels()
    a = split_indices(labels, seed=7)
    b = split_indices(labels, seed=7)
    c = split_indices(labels, seed=8)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not np.array_equal(a["test"], c["test"])


def test_block_split_keeps_neighbouring_frames_together():
    labels = np.repeat(np.arange(5), 100)
    sp = split_indices(labels, 0.6, 0.2, 0.2, stratify=True, block_size=10, seed=3)
    where = np.empty(len(labels), dtype=object)
    for name, idx in sp.items():
        where[idx] = name
    for start in range(0, len(labels), 10):
        assert len(set(where[start:start + 10])) == 1
    assert len(sp["val"]) == 100 and len(sp["test"]) == 100


def test_blocks_do_not_span_two_drives():
    """덩어리는 "연속해서 찍은 사진"이어야 한다. 주행(zip·세션)이 바뀌는 곳에서 새로 시작하지 않으면, 앞 주행의 끝과
    뒤 주행의 처음이 한 덩어리가 되고 그 뒤의 덩어리 경계가 모두 밀린다."""
    labels = np.zeros(23 + 20, dtype=np.int64)
    groups = np.array([0] * 23 + [1] * 20)                # 앞 주행 23장, 뒤 주행 20장
    sp = split_indices(labels, 0.6, 0.2, 0.2, stratify=False, block_size=10, seed=1, groups=groups)
    where = np.empty(len(labels), dtype=object)
    for name, idx in sp.items():
        where[idx] = name
    for a, b in ((0, 10), (10, 20), (20, 23), (23, 33), (33, 43)):      # 23 에서 끊기고, 뒤 주행은 다시 10장씩
        assert len(set(where[a:b])) == 1, (a, b)
    assert sorted(np.concatenate(list(sp.values())).tolist()) == list(range(43))
    same = split_indices(labels, 0.6, 0.2, 0.2, block_size=1, seed=1, groups=groups)
    plain = split_indices(labels, 0.6, 0.2, 0.2, block_size=1, seed=1)
    assert all(np.array_equal(same[k], plain[k]) for k in same)         # 덩어리를 안 쓰면 결과가 달라지지 않는다


def test_split_without_test_part_and_with_tiny_class():
    labels = np.array([0] * 20 + [1])                    # 한 장뿐인 클래스는 학습으로 간다
    sp = split_indices(labels, 0.8, 0.2, 0.0, stratify=True, seed=0)
    assert len(sp["test"]) == 0 and 20 in sp["train"] and len(sp["val"]) == 4


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------
def test_metrics_match_hand_computed_values():
    y_true = [0, 0, 0, 0, 1, 1, 2, 2, 2, 2]
    y_pred = [0, 0, 0, 1, 1, 2, 2, 2, 2, 0]
    cm = confusion_matrix(y_true, y_pred, 3)
    assert cm.tolist() == [[3, 1, 0], [0, 1, 1], [1, 0, 3]]
    rows = per_class_metrics(cm)
    assert rows[0]["support"] == 4
    assert rows[0]["precision"] == pytest.approx(3 / 4) and rows[0]["recall"] == pytest.approx(3 / 4)
    assert rows[0]["accuracy"] == pytest.approx(8 / 10)       # TP 3 + TN 5
    assert rows[1]["precision"] == pytest.approx(1 / 2) and rows[1]["f1"] == pytest.approx(0.5)
    s = summarize(cm)
    assert s["accuracy"] == pytest.approx(0.7)
    assert s["within_one"] == pytest.approx(0.9)              # 두 칸 떨어진 오답은 (2 → 0) 한 건
    assert "혼동 행렬" in format_report(cm, ["30°", "90°", "150°"])


def test_baseline_is_the_accuracy_of_always_answering_the_most_common_class():
    """직진 사진이 대부분인 데이터에서는 정확도만 보면 속는다: 모든 사진에 직진이라고 답해도 그 비율만큼 맞는다.
    보고서는 그 값(기준선)을 정확도 바로 아래에 함께 적는다."""
    from trainer.metrics import baseline
    assert baseline([6, 84, 26]) == (1, pytest.approx(84 / 116)) and baseline([0, 0, 0]) == (0, 0.0)
    assert baseline([5, 5, 2])[0] == 0                                       # 같으면 앞의 클래스
    always = confusion_matrix([0] * 6 + [1] * 84 + [2] * 26, [1] * 116, 3)   # 모든 사진에 가운데 클래스라고 답한 모델
    s = summarize(always)
    assert s["accuracy"] == s["baseline_accuracy"] == pytest.approx(84 / 116) and s["baseline_class_index"] == 1
    assert "baseline_class" not in s                     # 번호인지 각도인지 헷갈리는 이름은 쓰지 않는다
    learned = confusion_matrix([0] * 6 + [1] * 84 + [2] * 26, [0] * 6 + [1] * 84 + [2] * 20 + [1] * 6, 3)
    s = summarize(learned)
    assert s["accuracy"] == pytest.approx(110 / 116) and s["baseline_accuracy"] == pytest.approx(84 / 116)
    lines = format_report(learned, ["30°", "90°", "150°"]).splitlines()
    at = next(i for i, line in enumerate(lines) if line.startswith("전체 정확도"))
    assert lines[at] == "전체 정확도        0.948   (110 / 116)"
    assert lines[at + 1] == "기준선             0.724   (가장 많은 90° 만 답해도 나오는 정확도)"
    assert lines[at + 2].startswith("한 칸 이내 정확도  ")


def test_metrics_do_not_divide_by_zero_for_a_class_never_predicted():
    cm = confusion_matrix([0, 0, 1], [0, 0, 0], 3)
    rows = per_class_metrics(cm)
    assert rows[1]["precision"] == 0.0 and rows[1]["recall"] == 0.0 and rows[2]["support"] == 0
    assert summarize(cm)["macro_recall"] == pytest.approx(0.5)   # 사진이 없는 클래스는 평균에서 뺀다


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
def _config(tmp_path: Path, **training) -> Path:
    base = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    for key, value in training.items():
        node = base["training"]
        parts = key.split("__")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(base, allow_unicode=True), encoding="utf-8")
    return path


def test_default_training_config_is_valid():
    t = load_config().training
    assert t.task == "classification" and t.split.train + t.split.val + t.split.test == pytest.approx(1.0)


@pytest.mark.parametrize("override", [
    {"split__train": 0.8},                  # 합이 1 이 아님
    {"split__val": 0.0, "split__train": 0.85},
    {"rescale": "zscore"},
    {"augment__hflip": "yes"},
    {"optimizer__schedule": "step"},
    {"optimizer__lr": 0},
    {"task": "regression"},                 # 아직 구현 전
    {"loader__batch_size": 0},
    {"seed": -1},                           # numpy 가 받지 않는 값: 학습을 시작할 때가 아니라 설정을 읽을 때 알려 준다
    {"seed": 2 ** 32},
])
def test_bad_training_settings_are_rejected(tmp_path, override):
    with pytest.raises(ConfigError):
        load_config(_config(tmp_path, **override))


def test_unknown_training_key_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="알 수 없는 키"):
        load_config(_config(tmp_path, optimizer__learning_rate=0.1))


# 아래는 config.yaml 을 "사람이 손으로 고친 글자 그대로" 읽는 시험이다. yaml.safe_dump 로 만든 파일은 off 에 따옴표를
# 붙이고 1e-3 을 0.001 로 바꿔 써서, 학생이 실제로 적는 모양에서만 나는 문제를 놓친다.
def _edited_config(tmp_path: Path, old: str, new: str) -> Path:
    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    assert text.count(old) == 1, old
    path = tmp_path / "config.yaml"
    path.write_text(text.replace(old, new), encoding="utf-8")
    return path


def test_lr_written_as_1e_3_becomes_a_number(tmp_path):
    """PyYAML 은 소수점 없는 1e-3 을 문자열로 읽는다. 그대로 두면 옵티마이저를 만들 때 TypeError 로 죽는다."""
    cfg = load_config(_edited_config(tmp_path, "lr: 0.001 ", "lr: 1e-3 "))
    assert cfg.training.optimizer.lr == 0.001 and isinstance(cfg.training.optimizer.lr, float)
    cfg = load_config(_edited_config(tmp_path, "min_lr_ratio: 0.01 ", "min_lr_ratio: 1e-2 "))
    assert cfg.training.optimizer.min_lr_ratio == 0.01


def test_hflip_off_without_quotes_means_none(tmp_path):
    """YAML 은 따옴표 없는 off 를 false 로 읽는다."""
    assert load_config(_edited_config(tmp_path, "hflip: random ", "hflip: off ")).training.augment.hflip == "none"
    assert load_config(_edited_config(tmp_path, "hflip: random ", "hflip: none ")).training.augment.hflip == "none"
    assert load_config(_edited_config(tmp_path, "hflip: random ", "hflip: both ")).training.augment.hflip == "both"


@pytest.mark.parametrize("old, new", [
    ("epochs: 50 ", "epochs: "),                    # 값을 지움 → None
    ("epochs: 50 ", "epochs: 12.5 "),
    ("epochs: 50 ", "epochs: many "),
    ("lr: 0.001 ", "lr: fast "),
    ("batch_size: 32 ", "batch_size: 32.5 "),
    ("shuffle: true ", "shuffle: maybe "),
    ("stratify: true ", "stratify: 1 "),
    ("patience: 10 ", "patience: "),
    ("train: 0.7 ", "train: "),
    ("lr: 0.001 ", "lr: .nan "),
])
def test_wrong_typed_values_give_a_config_error_not_a_crash(tmp_path, old, new):
    """TypeError 가 새면 `except ConfigError` 를 빠져나가 한 줄 안내 대신 영어 오류 더미로 끝난다.
    (수집기는 2026-11-23 부터 training 절을 검사하지 않는다: tests/test_config_layout.py)"""
    with pytest.raises(ConfigError, match="training"):
        load_config(_edited_config(tmp_path, old, new))


def test_report_columns_line_up_on_screen():
    """한글은 화면에서 두 칸이다. 글자 수가 아니라 화면 폭으로 맞춰야 표의 열이 어긋나지 않는다."""
    cm = confusion_matrix([0, 0, 1, 2, 2, 2], [0, 1, 1, 2, 2, 0], 3)
    lines = format_report(cm, ["30°", "90°", "150°"]).splitlines()
    start = lines.index("클래스별 지표") + 2
    assert len({width(line) for line in lines[start:start + 4]}) == 1          # 머리글 + 세 클래스
    assert width("클래스") == 6 and width("F1") == 2
