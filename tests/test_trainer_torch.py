"""학습기 가운데 torch 가 필요한 부분. torch 가 없으면 이 파일 전체를 건너뛴다 (uv sync --extra train 또는 train-cpu)."""
import csv
import io
import json
import warnings
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from common.config import load_config  # noqa: E402
from test_trainer import CLASSES, _config  # noqa: E402
from trainer.split import split_indices  # noqa: E402


def make_dataset_zip(path: Path, per_class: int = 40, classes=CLASSES, seed: int = 0, center=90,
                     lo: int = 30, hi: int = 150, crop=None) -> Path:
    """가공 프로그램이 만드는 것과 같은 형식의 zip. 밝은 세로 막대의 가로 위치가 조향 클래스를 정한다 (좌우 대칭).

    crop  (위, 아래) 자르는 범위. None 이면 적지 않는다 (범위를 가공 프로그램에서 정하기 전에 만든 zip).
    """
    rng = np.random.default_rng(seed)
    rows, k = [], len(classes)
    order = rng.permutation(np.repeat(np.arange(k), per_class))
    with zipfile.ZipFile(path, "w") as z:
        for i, c in enumerate(order):
            img = rng.integers(0, 60, size=(120, 160, 3), dtype=np.uint8)
            x = int((c + 0.5) * 160 / k) + int(rng.integers(-5, 6))
            img[50:, max(0, x - 8):x + 8] = (40, 160, 255)            # 아래쪽(자르고 남는 부분)에 막대
            ok, buf = cv2.imencode(".jpg", img)
            name = f"20261110_1906_{i:06d}.jpg"
            z.writestr(f"images/{name}", buf.tobytes())
            rows.append([i, name, classes[c], 45, f"{path.stem}/src_{i:06d}.jpg"])     # source = "세션/원본 사진 이름"
        out = io.StringIO()
        w = csv.writer(out, lineterminator="\n")
        w.writerow(["index", "image", "servo_angle", "motor_speed", "source"])
        w.writerows(rows)
        z.writestr("labels.csv", out.getvalue())
        steering = {"mode": "discrete", "classes": list(classes), "min": lo, "max": hi}
        if center is not None:                                        # 옛 zip 에는 center 가 없다
            steering["center"] = center
        meta = {"schema_version": 1, "steering": steering,
                "image": {"width": 160, "height": 120, "format": "jpg"}, "sessions": ["20261110_1906"],
                "frames": len(rows)}
        if crop is not None:
            meta["preprocess"] = {"crop_top_ratio": crop[0], "crop_bottom_ratio": crop[1]}
        z.writestr("meta.json", json.dumps(meta))
    return path


def _train_cfg(tmp_path: Path, **training):
    cfg = load_config(_config(tmp_path, **training))
    t = cfg.training
    t.output_root, t.device, t.epochs = str(tmp_path / "models"), "cpu", 12
    t.early_stopping.enabled = False
    return cfg


def test_lr_schedules_rise_and_fall_as_named():
    from trainer.optim import lr_factor
    n = 20
    f = {s: [lr_factor(e, n, s, warmup_epochs=5, min_lr_ratio=0.01) for e in range(n)]
         for s in ("constant", "warmup", "decay", "warmup_decay")}
    assert set(f["constant"]) == {1.0}
    assert f["warmup"][:6] == sorted(f["warmup"][:6]) and f["warmup"][0] < 0.5 and f["warmup"][5:] == [1.0] * 15
    assert f["decay"][0] == 1.0 and f["decay"] == sorted(f["decay"], reverse=True)
    assert f["decay"][-1] == pytest.approx(0.01)
    wd = f["warmup_decay"]
    assert wd[0] < wd[4] < wd[5] == 1.0 and wd[5:] == sorted(wd[5:], reverse=True) and wd[-1] == pytest.approx(0.01)


def test_mirror_labels_needs_symmetric_classes():
    from trainer.data import DatasetError, mirror_labels
    assert mirror_labels([30, 60, 90, 120, 150]) == [4, 3, 2, 1, 0]
    assert mirror_labels([40, 90, 140]) == [2, 1, 0]
    with pytest.raises(DatasetError, match="대칭"):
        mirror_labels([30, 60, 90, 100, 150])


def test_flip_augmentation_mirrors_image_and_label():
    from trainer.data import SteeringDataset
    images = np.zeros((2, 3, 4, 6), dtype=np.uint8)
    images[0, :, :, 0] = 255                                  # 첫 사진: 왼쪽 끝 열만 밝다
    labels = np.array([0, 2])
    ds = SteeringDataset(images, labels, CLASSES, hflip="both")
    assert len(ds) == 4
    img, lab = ds[0]
    assert lab == 0 and img[0, 0, 0] == 1.0 and img[0, 0, 5] == 0.0
    img, lab = ds[2]                                          # 같은 사진을 뒤집은 것
    assert lab == 4 and img[0, 0, 5] == 1.0 and img[0, 0, 0] == 0.0
    assert ds[3][1] == 2                                      # 가운데 클래스는 그대로
    assert sorted(ds.effective_labels().tolist()) == [0, 2, 2, 4]
    plain = SteeringDataset(images, labels, [30, 60, 90, 100, 150], hflip="none")  # 대칭이 아니어도 none 이면 된다
    assert len(plain) == 2 and plain[0][1] == 0


def test_only_training_data_is_shuffled_and_augmented(tmp_path):
    from common.preprocess import Preprocessor
    from trainer.data import load_samples, make_loaders
    samples = load_samples([make_dataset_zip(tmp_path / "d.zip", per_class=10)], Preprocessor())
    assert samples.images.shape == (50, 3, 66, 200) and samples.classes == CLASSES
    split = split_indices(samples.labels, 0.6, 0.2, 0.2, seed=0)
    loaders = make_loaders(samples, split, batch_size=8, shuffle=True, hflip="both")
    assert len(loaders["train"].dataset) == 2 * len(split["train"])
    assert loaders["val"].dataset.hflip == "none" and loaders["test"].dataset.hflip == "none"
    first = [lab.tolist() for _, lab in loaders["val"]]
    assert first == [lab.tolist() for _, lab in loaders["val"]]          # 검증은 매번 같은 순서


def test_datasets_with_different_classes_cannot_be_mixed(tmp_path):
    from common.preprocess import Preprocessor
    from trainer.data import DatasetError, load_samples
    a = make_dataset_zip(tmp_path / "a.zip", per_class=3)
    b = make_dataset_zip(tmp_path / "b.zip", per_class=3, classes=[40, 90, 140])
    with pytest.raises(DatasetError, match="다릅니다"):
        load_samples([a, b], Preprocessor())


@pytest.mark.parametrize("rescale", ["none", "minmax", "standardize"])
def test_input_scaling_modes(rescale):
    from trainer.models import InputScaler
    x = torch.full((1, 3, 2, 2), 0.5)
    y = InputScaler(rescale, mean=[0.5, 0.25, 0.0], std=[0.5, 0.25, 1.0])(x)
    want = {"none": [127.5] * 3, "minmax": [0.5] * 3, "standardize": [0.0, 1.0, 0.5]}[rescale]
    assert y[0, :, 0, 0].tolist() == pytest.approx(want)


@pytest.mark.parametrize("name", ["pilotnet", "small_cnn"])
def test_models_give_one_score_per_class(name):
    from trainer.models import InputScaler, build_model
    model = build_model(name, 7, (3, 66, 200), InputScaler("minmax"))
    assert model(torch.zeros(4, 3, 66, 200)).shape == (4, 7)
    with pytest.raises(ValueError, match="MODELS"):
        build_model("resnet999", 5, (3, 66, 200), InputScaler("minmax"))


def test_balanced_class_weights_favour_rare_classes():
    from trainer.losses import balanced_class_weights, build_loss
    w = balanced_class_weights([10, 90, 0])
    assert w[0] > w[1] > 0 and w[2] == w[0]            # 학습 데이터에 없는 클래스는 가장 드문 클래스와 같은 가중치
    assert float((w[:2] * torch.tensor([10.0, 90.0])).sum()) == pytest.approx(100.0)
    assert build_loss("cross_entropy", "none").weight is None
    assert build_loss("cross_entropy", "balanced", [10, 90]).weight is not None


def test_early_stopping_restores_the_best_weights():
    """검증 손실이 나빠지기만 하면 patience 뒤에 멈추고, 가장 좋았던 가중치로 되돌린다."""
    from torch.utils.data import DataLoader, TensorDataset
    from trainer.engine import evaluate, fit
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))
    x = torch.randn(32, 4)
    train = DataLoader(TensorDataset(x, (x[:, 0] > 0).long()), batch_size=8)
    val = DataLoader(TensorDataset(x, (x[:, 0] <= 0).long()), batch_size=8)     # 정답이 반대: 배울수록 나빠진다
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: 1.0)
    h = fit(model, train, val, loss_fn, opt, sched, torch.device("cpu"), epochs=50, patience=3, log=None)
    assert h.stopped_early and h.best_epoch == 1 and len(h.epoch) == 4
    assert evaluate(model, val, loss_fn, torch.device("cpu"))[0] == pytest.approx(h.best_val_loss, rel=1e-5)


# ---------------------------------------------------------------------------
# 처음부터 끝까지: 학습 → 시험 분석 → ONNX
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    from trainer.evaluate import analyze
    from trainer.export import export
    from trainer.run import train
    tmp = tmp_path_factory.mktemp("trainer")
    zip_path = make_dataset_zip(tmp / "dataset_20261110_1906.zip")
    cfg = _train_cfg(tmp, rescale="standardize", augment__hflip="random")
    run_dir, samples = train([zip_path], cfg, name="t", log=lambda _m: None)
    metrics = analyze(run_dir, samples=samples, device="cpu", log=lambda _m: None)
    onnx_path = export(run_dir, log=lambda _m: None)
    return {"run_dir": run_dir, "metrics": metrics, "onnx": onnx_path, "zip": zip_path, "tmp": tmp}


def test_run_folder_holds_everything(trained):
    run_dir = trained["run_dir"]
    for name in ("model.pth", "model.onnx", "model_meta.json", "history.csv", "loss_curve.png", "split.csv",
                 "train_config.json", "test_report.txt", "confusion_matrix.png", "test_predictions.csv",
                 "test_metrics.json"):
        assert (run_dir / name).stat().st_size > 0, name
    assert run_dir.name.endswith("_t")
    rows = list(csv.DictReader(open(run_dir / "history.csv", encoding="utf-8")))
    assert len(rows) == 12 and {"train_loss", "val_loss", "lr"} <= set(rows[0])
    parts = [r["split"] for r in csv.DictReader(open(run_dir / "split.csv", encoding="utf-8"))]
    assert len(parts) == 200 and set(parts) == {"train", "val", "test"}


def test_model_learns_a_learnable_dataset(trained):
    s = trained["metrics"]["summary"]
    assert s["samples"] == 30 and s["accuracy"] > 0.9
    assert s["baseline_accuracy"] == pytest.approx(0.2) and s["accuracy"] > s["baseline_accuracy"] + 0.5   # 5 클래스가 같은 수
    assert s["baseline_class_index"] == 0 and s["baseline_angle"] == 30     # 파일에는 번호와 각도를 이름으로 구분해 적는다
    saved = json.loads((trained["run_dir"] / "test_metrics.json").read_text(encoding="utf-8"))["summary"]
    assert saved["baseline_angle"] == 30 and "baseline_class" not in saved
    report = (trained["run_dir"] / "test_report.txt").read_text(encoding="utf-8")
    assert "기준선             0.200   (가장 많은 30°만 답할 때의 정확도)" in report
    preds = list(csv.DictReader(open(trained["run_dir"] / "test_predictions.csv", encoding="utf-8")))
    assert len(preds) == 30 and {"true_angle", "pred_angle", "p_30", "p_150"} <= set(preds[0])
    assert sum(int(r["correct"]) for r in preds) == round(s["accuracy"] * 30)


def test_model_bundle_follows_the_inference_contract(trained):
    import onnx
    import onnxruntime as ort
    from common.preprocess import Preprocessor
    from trainer.run import load_checkpoint
    meta = json.loads((trained["run_dir"] / "model_meta.json").read_text(encoding="utf-8"))
    assert meta["task"] == "classification" and meta["classes"] == CLASSES
    assert meta["input"] == {"name": "input", "shape": [1, 3, 66, 200], "dtype": "float32", "color": "RGB",
                             "range": "0-1", "layout": "NCHW"}
    assert meta["output"]["name"] == "output" and meta["output"]["shape"] == [1, 5]
    assert meta["scaler_in_model"]["mode"] == "standardize"
    assert meta["training"]["sessions"] == ["20261110_1906"] and meta["training"]["test"]["accuracy"] > 0.9
    pre = Preprocessor.from_dict(meta["preprocess"])                    # 추론기가 읽는 방식 그대로
    assert pre.input_shape == (3, 66, 200)

    proto = onnx.load(str(trained["onnx"]))
    assert [i.name for i in proto.graph.input] == ["input"] and [o.name for o in proto.graph.output] == ["output"]
    assert [d.dim_value for d in proto.graph.input[0].type.tensor_type.shape.dim] == [1, 3, 66, 200]   # 배치 1 고정

    # 추론기가 할 일을 그대로 해 본다: 사진 → Preprocessor(0~1) → ONNX → argmax → 각도. 스케일링은 모델 안에 있다
    session = ort.InferenceSession(str(trained["onnx"]), providers=["CPUExecutionProvider"])
    model, _ = load_checkpoint(trained["run_dir"] / "model.pth")
    with zipfile.ZipFile(trained["zip"]) as z:
        rows = list(csv.DictReader(io.StringIO(z.read("labels.csv").decode("utf-8"))))[:25]
        hits = 0
        for row in rows:
            bgr = cv2.imdecode(np.frombuffer(z.read("images/" + row["image"]), np.uint8), cv2.IMREAD_COLOR)
            x = pre(bgr)[None]
            out = session.run(["output"], {"input": x})[0]
            with torch.no_grad():
                ref = model(torch.from_numpy(x)).numpy()
            assert np.abs(out - ref).max() < 1e-3
            hits += int(meta["classes"][int(out.argmax())] == int(row["servo_angle"]))
    assert hits >= 22


def test_test_command_can_use_another_dataset(trained):
    from trainer.evaluate import analyze
    from trainer.run import TrainError
    other = make_dataset_zip(trained["tmp"] / "dataset_other.zip", per_class=6, seed=5)
    m = analyze(trained["run_dir"], data=[other], device="cpu", log=lambda _m: None)
    assert m["summary"]["samples"] == 30 and m["summary"]["accuracy"] > 0.9
    assert (trained["run_dir"] / "test_dataset_other" / "confusion_matrix.png").is_file()
    wrong = make_dataset_zip(trained["tmp"] / "dataset_3class.zip", per_class=3, classes=[40, 90, 140])
    with pytest.raises(TrainError, match="클래스"):
        analyze(trained["run_dir"], data=[wrong], device="cpu", log=lambda _m: None)


def test_crop_range_comes_from_the_dataset_zip(tmp_path):
    """사진에서 자를 범위는 가공 프로그램에서 사진을 보며 정하고 데이터셋 zip 에 적혀 온다 (2026-11-23).
    학습기는 config.yaml 이 아니라 그 값대로 자른다. 범위가 없는 예전 zip 만 기본값으로 자른다."""
    from common.preprocess import Preprocessor
    from trainer.data import DatasetError, load_samples
    pre = Preprocessor()
    old = make_dataset_zip(tmp_path / "old.zip", per_class=4)
    new = make_dataset_zip(tmp_path / "new.zip", per_class=4, seed=1, crop=(0.15, 0.9))
    same = make_dataset_zip(tmp_path / "same.zip", per_class=4, seed=2, crop=(0.4, 1.0))
    s = load_samples([old], pre)
    assert s.pre.crop == (0.4, 1.0) and not s.crop_from_dataset
    s = load_samples([new], pre)
    assert s.pre.crop == (0.15, 0.9) and s.crop_from_dataset and s.pre.input_shape == (3, 66, 200)
    with zipfile.ZipFile(new) as z:                                     # 정말 그 범위로 잘렸는지 같은 사진을 직접 잘라 본다
        first = next(csv.DictReader(io.StringIO(z.read("labels.csv").decode("utf-8"))))["image"]
        bgr = cv2.imdecode(np.frombuffer(z.read("images/" + first), np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(s.images[0], np.transpose(pre.with_crop((0.15, 0.9)).to_rgb_uint8(bgr), (2, 0, 1)))
    assert not np.array_equal(s.images[0], np.transpose(pre.to_rgb_uint8(bgr), (2, 0, 1)))
    mixed = load_samples([old, same], pre)                              # 범위가 같으면 예전 zip 과도 함께 쓸 수 있다
    assert mixed.pre.crop == (0.4, 1.0) and len(mixed) == 40 and mixed.crop_from_dataset    # 적혀 있던 zip 이 하나는 있다
    assert load_samples([same], pre).crop_from_dataset
    with pytest.raises(DatasetError, match="자르는 범위.*위쪽 40%.*위쪽 15%"):
        load_samples([new, same], pre)
    with pytest.raises(DatasetError, match="이 zip 에는 범위가 적혀 있지 않아"):
        load_samples([new, old], pre)
    # 범위가 없는 zip 이 앞에 오면 "앞의 zip 을 다시 만들라"고 해야 한다: 방금 사진을 보며 정한 쪽을 고치게 해서는 안 된다
    with pytest.raises(DatasetError, match="앞의 old.zip 에 범위가 적혀 있지 않아") as err:
        load_samples([old, new], pre)
    assert "이 zip 에는" not in str(err.value)
    forced = load_samples([new], pre.with_crop((0.2, 1.0)), follow_dataset_crop=False)      # 모델을 시험할 때: 모델의 범위
    assert forced.pre.crop == (0.2, 1.0) and not forced.crop_from_dataset
    assert np.array_equal(forced.images[0], np.transpose(pre.with_crop((0.2, 1.0)).to_rgb_uint8(bgr), (2, 0, 1)))
    with zipfile.ZipFile(new) as z:
        meta = json.loads(z.read("meta.json"))
    for bad in ({"crop_top_ratio": 0.9, "crop_bottom_ratio": 0.2}, {"crop_top_ratio": "위"}, "40%"):
        meta["preprocess"] = bad
        broken = _copy_zip(new, tmp_path / "broken.zip", replace=("meta.json", json.dumps(meta).encode("utf-8")))
        with pytest.raises(DatasetError, match="손상"):
            load_samples([broken], pre)
        # 모델을 시험할 때는 zip 의 범위를 쓰지 않으므로, 그 값이 깨져 있어도 시험은 된다
        assert load_samples([broken], pre, follow_dataset_crop=False).pre.crop == (0.4, 1.0)
    meta["preprocess"] = {"crop_top_ratio": 0.5, "crop_bottom_ratio": 0.501}     # 사진 한 줄도 남지 않는 범위 (손으로 고친 meta.json)
    thin = _copy_zip(new, tmp_path / "thin.zip", replace=("meta.json", json.dumps(meta).encode("utf-8")))
    with pytest.raises(DatasetError, match="자르는 범위가 너무 좁습니다"):
        load_samples([thin], pre)


def test_the_model_keeps_the_crop_it_was_trained_with(tmp_path):
    """데이터셋의 범위 → 학습 → model_meta.json 으로 이어져야 추론기가 같은 범위로 자른다. config.yaml 의 값은 쓰이지 않는다.
    다른 범위로 만든 데이터셋으로 시험할 때도 모델의 범위로 자른다 (차에서 모델이 받는 사진이 그렇기 때문이다)."""
    from common.preprocess import Preprocessor
    from trainer.evaluate import analyze
    from trainer.export import export
    from trainer.run import train
    zip_path = make_dataset_zip(tmp_path / "dataset_crop.zip", per_class=12, crop=(0.3, 0.95))
    cfg = _train_cfg(tmp_path)
    cfg.training.epochs = 2
    cfg.preprocess.crop_top_ratio, cfg.preprocess.crop_bottom_ratio = 0.1, 0.5      # 범위가 적힌 zip 에는 영향이 없다
    lines = []
    run_dir, samples = train([zip_path], cfg, name="crop", log=lines.append)
    assert samples.pre.crop == (0.3, 0.95)
    assert any("자르는 범위: 위쪽 30% · 아래쪽 5% 잘라 냄 (가공 프로그램에서 지정)" in line for line in lines)
    # 클래스마다 12장씩이라 어느 클래스만 답해도 0.2 다 (60장을 42 / 9 / 9 로 나눈다: 8.4 → 8, 9 가 섞여 0.19~0.22)
    note = next(line for line in lines if "기준선 (" in line)
    assert "기준선 (가장 많은 30°만 답할 때의 정확도): train 0.2" in note and "val 0.2" in note
    assert note.endswith("이 값을 넘어야 학습된 것이다.")
    export(run_dir, log=lambda _m: None)
    meta = json.loads((run_dir / "model_meta.json").read_text(encoding="utf-8"))
    assert Preprocessor.from_dict(meta["preprocess"]).crop == (0.3, 0.95)           # 추론기가 읽는 방식 그대로
    info = json.loads((run_dir / "train_config.json").read_text(encoding="utf-8"))
    assert info["preprocess"]["crop_top_ratio"] == 0.3 and info["datasets"][0]["meta"]["preprocess"]["crop_bottom_ratio"] == 0.95

    same = analyze(run_dir, device="cpu", log=lambda _m: None)                      # zip 을 다시 읽어도 같은 범위로
    again = analyze(run_dir, samples=samples, device="cpu", log=lambda _m: None)
    assert same["confusion_matrix"] == again["confusion_matrix"]
    other = make_dataset_zip(tmp_path / "dataset_other_crop.zip", per_class=4, seed=3, crop=(0.5, 1.0))
    said = []
    analyze(run_dir, data=[other], device="cpu", log=said.append)
    assert any("dataset_other_crop.zip 의 자르는 범위(위쪽 50% · 아래쪽 0% 잘라 냄)가 모델이 학습한 범위"
               "(위쪽 30% · 아래쪽 5% 잘라 냄)와 다릅니다" in line for line in said)
    said.clear()
    analyze(run_dir, data=[make_dataset_zip(tmp_path / "dataset_same_crop.zip", per_class=4, seed=3, crop=(0.3, 0.95))],
            device="cpu", log=said.append)
    assert not any("주의" in line for line in said)
    # 두 zip 은 사진이 같고 적힌 범위만 다르다. 둘 다 모델의 범위로 잘랐다면 예측이 한 장도 다르지 않아야 한다
    answers = lambda name: [[v for k, v in row.items() if k not in ("index", "image")] for row in csv.DictReader(
        open(run_dir / f"test_{name}" / "test_predictions.csv", encoding="utf-8"))]
    assert answers("dataset_other_crop") == answers("dataset_same_crop") and len(answers("dataset_same_crop")) == 20

    plain = make_dataset_zip(tmp_path / "dataset_plain.zip", per_class=12, seed=6)   # 범위가 없는 예전 zip: config 의 값
    lines.clear()
    run_dir, samples = train([plain], cfg, name="plain", log=lines.append)
    assert samples.pre.crop == (0.1, 0.5)
    assert any("자르는 범위: 위쪽 10% · 아래쪽 50% 잘라 냄 (데이터셋에 값이 없어 기본값 사용)" in line for line in lines)


def test_baseline_note_matches_what_the_epoch_table_measures(tmp_path):
    """[2/7] 의 '기준선' 줄은 에폭 표의 train acc / val acc 와 견주는 값이다. 그 표가 재는 것과 같은 것을 말해야 한다.

    (1) train acc 는 좌우 반전된 사진으로 잰다. 가장 많은 각도가 직진이 아니면 반전 뒤에는 그 각도가 절반으로 줄어든다.
    (2) 가장 많은 각도는 묶음마다 다를 수 있다. 전체에서 가장 많은 각도의 비율을 val 에 그대로 적으면 0 이 나오기도 한다."""
    from trainer.run import train
    # 한쪽으로만 돈 트랙: 60° 가 가장 많다 (30 60 90 120 150 = 4 100 70 20 6)
    z = tmp_path / "dataset_oneway.zip"
    make_dataset_zip(z, per_class=1)                                        # 틀만 빌려 쓰고 라벨을 바꾼다
    rng = np.random.default_rng(0)
    counts = [4, 100, 70, 20, 6]
    with zipfile.ZipFile(z, "w") as out:
        rows = []
        for i, c in enumerate(rng.permutation(np.repeat(np.arange(5), counts))):
            ok, buf = cv2.imencode(".jpg", rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8))
            out.writestr(f"images/f_{i:06d}.jpg", buf.tobytes())
            rows.append(f"{i},f_{i:06d}.jpg,{CLASSES[c]},45,s/src_{i:06d}.jpg")
        out.writestr("labels.csv", "index,image,servo_angle,motor_speed,source\n" + "\n".join(rows) + "\n")
        out.writestr("meta.json", json.dumps({"schema_version": 1, "steering": {"mode": "discrete", "classes": CLASSES,
                                                                              "center": 90, "min": 30, "max": 150},
                                              "image": {"width": 160, "height": 120, "format": "jpg"}, "frames": 200}))

    def note(**training):
        cfg = _train_cfg(tmp_path, **training)
        cfg.training.epochs = 1
        lines = []
        train([z], cfg, name="b", log=lines.append)
        return next(line for line in lines if "기준선 (" in line)

    plain = note(augment__hflip="none")                                     # 반전이 없으면 60° 가 그대로 가장 많다: 70 / 140
    assert "기준선 (가장 많은 60°만 답할 때의 정확도): train 0.500, val 0.500." in plain
    # 학습 묶음은 2 · 70 · 50 · 14 · 4 장. 반전 뒤 60° 는 (70 + 14) / 2 = 42, 90° 는 50 그대로 → 90° 가 가장 많다: 50 / 140
    flipped = note(augment__hflip="both")
    assert "기준선 (가장 많은 각도만 답할 때의 정확도): train 0.357 (90°), val 0.500 (60°)." in flipped
    assert "train 0.357 (90°), val 0.500 (60°)" in note(augment__hflip="random")     # 기본값(절반 확률로 반전)도 같은 분포다

    # 묶음마다 가장 많은 각도가 다른 경우를 손으로 만든다: 전체는 60° 가 가장 많은데 val 에는 60° 가 한 장도 없다
    from trainer.data import Samples
    from trainer.run import split_baselines
    labels = np.array([1] * 300 + [2] * 120 + [3] * 80)                     # 60° 300, 90° 120, 120° 80
    samples = Samples(np.zeros((500, 3, 1, 1), np.uint8), labels, labels, ["x"] * 500, CLASSES, [], center=90)
    split = {"train": np.arange(0, 380), "val": np.arange(380, 480), "test": np.arange(480, 500)}   # val = 90° 40 + 120° 60
    counts, (top_train, base_train), (top_val, base_val) = split_baselines(samples, split, "none")
    assert (top_train, base_train) == (1, pytest.approx(300 / 380))
    assert (top_val, base_val) == (3, pytest.approx(0.6))                   # 60° 의 비율(0)이 아니라 val 에서 가장 많은 120° 의 비율
    counts, (top_train, base_train), _ = split_baselines(samples, split, "both")
    assert counts.tolist() == [0, 150, 80, 150, 0] and base_train == pytest.approx(150 / 380)       # 60° 와 120° 가 반씩 섞인다


def test_split_csv_resaved_by_a_spreadsheet_still_reads_or_says_why(trained, tmp_path):
    """split.csv 를 Excel 에서 'CSV UTF-8' 로 다시 저장하면 맨 앞에 BOM 이 붙는다. 그래도 시험 분석이 되어야 하고,
    정말 읽을 수 없게 고쳐진 파일이면 영어 오류 더미가 아니라 한 줄 안내로 끝나야 한다."""
    import shutil
    from trainer.__main__ import main
    from trainer.evaluate import analyze
    from trainer.run import TrainError, read_split_csv
    run = tmp_path / "run"
    shutil.copytree(trained["run_dir"], run)
    text = (run / "split.csv").read_text(encoding="utf-8")
    (run / "split.csv").write_text(text, encoding="utf-8-sig")              # BOM 을 붙여 다시 저장
    names, split = read_split_csv(run / "split.csv")
    assert len(names) == 200 and len(split["test"]) == 30
    assert analyze(run, device="cpu", log=lambda _m: None)["confusion_matrix"] == trained["metrics"]["confusion_matrix"]
    (run / "split.csv").write_text(text.replace("index,image", "번호,image", 1), encoding="utf-8")   # 열 이름을 고친 파일
    with pytest.raises(TrainError, match="split.csv 를 읽을 수 없습니다"):
        read_split_csv(run / "split.csv")
    assert main(["test", str(run), "--device", "cpu"]) == 1                 # 오류 더미가 아니라 종료 코드 1
    (run / "split.csv").unlink()
    with pytest.raises(TrainError, match="split.csv 가 없습니다"):
        analyze(run, device="cpu", log=lambda _m: None)
    (run / "split.csv").write_text(text, encoding="utf-8")
    (run / "train_config.json").write_text("{ 깨진 파일", encoding="utf-8")
    with pytest.raises(TrainError, match="train_config.json 을 읽을 수 없습니다"):
        analyze(run, device="cpu", log=lambda _m: None)
    assert analyze(run, data=[trained["zip"]], device="cpu", log=lambda _m: None)["summary"]["samples"] == 200   # --data 는 된다


def test_testing_on_frames_the_model_was_trained_on_is_called_out(trained, tmp_path):
    """`trainer test <실행> --data X.zip` 에 학습에 쓴 zip (또는 같은 세션으로 다시 만든 zip) 을 주면 "이미 본 사진"으로
    시험을 치르게 된다. 막지는 않지만, 몇 장이 그런 사진인지 화면과 보고서에 알린다."""
    import shutil
    from trainer.evaluate import analyze
    from trainer.run import read_seen_sources
    run = tmp_path / "run"
    shutil.copytree(trained["run_dir"], run)
    rows = list(csv.DictReader(open(run / "split.csv", encoding="utf-8")))
    assert rows[0]["source"] == "dataset_20261110_1906/src_000000.jpg"        # 학습할 때 사진의 출처를 함께 적어 둔다
    used = {r["source"] for r in rows if r["split"] != "test"}
    assert read_seen_sources(run / "split.csv") == used and len(used) == 170    # 학습 140 + 검증 30. 시험 30장은 쓰지 않았다

    said = []
    m = analyze(run, data=[trained["zip"]], device="cpu", log=said.append)    # 학습에 쓴 zip 을 그대로 준 경우
    note = "주의: 시험 사진 200장 중 170장은 이 모델을 학습할 때 쓴 사진입니다."
    assert any(note in line for line in said) and m["frames_seen_in_training"] == 170
    report = (run / "test_dataset_20261110_1906" / "test_report.txt").read_text(encoding="utf-8").splitlines()
    assert report[0].startswith("시험 분석:") and report[1].startswith(note) and report[2] == ""
    saved = json.loads((run / "test_dataset_20261110_1906" / "test_metrics.json").read_text(encoding="utf-8"))
    assert saved["frames_seen_in_training"] == 170

    # 같은 세션으로 다시 만든 zip: 이름과 사진 번호는 달라도 출처가 같다. 앞의 50장만 든 zip 이면 그중 쓴 것만 센다
    again = tmp_path / "again" / "dataset_20261110_1906.zip"
    again.parent.mkdir()
    make_dataset_zip(again, per_class=10, seed=9)
    expect = sum(1 for r in rows[:50] if r["split"] != "test")
    said.clear()
    m = analyze(run, data=[again], device="cpu", log=said.append)
    assert m["frames_seen_in_training"] == expect and 0 < expect < 50
    assert any(f"시험 사진 50장 중 {expect}장은" in line for line in said)

    said.clear()                                                              # 겹치지 않는 데이터셋: 아무 말도 하지 않는다
    other = make_dataset_zip(tmp_path / "dataset_new_drive.zip", per_class=6, seed=5)
    m = analyze(run, data=[other], device="cpu", log=said.append)
    assert m["frames_seen_in_training"] == 0 and not any("주의" in line for line in said)
    assert not (run / "test_dataset_new_drive" / "test_report.txt").read_text(encoding="utf-8").splitlines()[1]
    assert analyze(run, device="cpu", log=said.append)["frames_seen_in_training"] is None     # 떼어 둔 시험 사진: 해당 없음

    # 출처를 적기 전에 만든 결과 폴더: 장수는 셀 수 없으므로 세션 이름이 같은지만 알린다
    old = tmp_path / "old_run"
    shutil.copytree(trained["run_dir"], old)
    with open(old / "split.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "source"])
        w.writeheader()
        w.writerows([{k: v for k, v in r.items() if k != "source"} for r in rows])
    assert read_seen_sources(old / "split.csv") is None
    said.clear()
    m = analyze(old, data=[other], device="cpu", log=said.append)             # 시험용 zip 은 모두 세션 이름이 같다
    assert m["frames_seen_in_training"] is None
    assert any("학습에 쓴 세션(20261110_1906)으로 만든 것입니다" in line for line in said)
    with zipfile.ZipFile(other) as z:
        meta = json.loads(z.read("meta.json"))
    meta["sessions"] = ["20261124_0900"]
    renamed = _copy_zip(other, tmp_path / "dataset_other_session.zip", replace=("meta.json", json.dumps(meta).encode("utf-8")))
    said.clear()
    analyze(old, data=[renamed], device="cpu", log=said.append)
    assert not any("주의" in line for line in said)
    assert analyze(old, device="cpu", log=lambda _m: None)["confusion_matrix"] == trained["metrics"]["confusion_matrix"]


def test_test_command_rereads_the_training_zip(trained):
    from trainer.evaluate import analyze
    m = analyze(trained["run_dir"], device="cpu", log=lambda _m: None)      # samples 없이: zip 을 다시 읽는다
    assert m["confusion_matrix"] == trained["metrics"]["confusion_matrix"]


def test_same_seed_gives_same_result(trained, tmp_path):
    from trainer.run import train
    cfg = _train_cfg(tmp_path, rescale="standardize", augment__hflip="random")
    cfg.training.epochs = 3
    a, _ = train([trained["zip"]], cfg, name="a", log=lambda _m: None)
    b, _ = train([trained["zip"]], cfg, name="b", log=lambda _m: None)
    read = lambda d: [(r["train_loss"], r["val_loss"]) for r in csv.DictReader(open(d / "history.csv", encoding="utf-8"))]
    assert read(a) == read(b)


def test_asymmetric_classes_stop_with_a_clear_message_when_flip_is_on(tmp_path):
    from trainer.__main__ import main
    z = make_dataset_zip(tmp_path / "d.zip", per_class=6, classes=[30, 60, 90, 100, 150])
    cfg_path = _config(tmp_path, output_root=str(tmp_path / "models"), device="cpu")
    assert main(["train", str(z), "--config", str(cfg_path), "--epochs", "1"]) == 1
    cfg_path = _config(tmp_path, output_root=str(tmp_path / "models"), device="cpu", augment__hflip="none")
    assert main(["train", str(z), "--config", str(cfg_path), "--epochs", "1", "--name", "cli"]) == 0
    run_dir = next((tmp_path / "models").iterdir())
    assert (run_dir / "model.onnx").is_file() and (run_dir / "test_report.txt").is_file()
    assert main(["test", str(run_dir), "--device", "cpu"]) == 0 and main(["export", str(run_dir)]) == 0
    assert main(["train", str(tmp_path / "missing.zip"), "--config", str(cfg_path)]) == 1


def test_skipped_steps_keep_the_test_part_and_say_how_to_run_them_later(tmp_path, capsys):
    """--no-test 는 시험 분석을 미룰 뿐이다: test 사진은 학습에 쓰지 않고 남아 있고, 나중에 하는 명령을 알려 준다."""
    from trainer.__main__ import main
    z = make_dataset_zip(tmp_path / "d.zip", per_class=6)
    cfg_path = _config(tmp_path, output_root=str(tmp_path / "models"), device="cpu")
    assert main(["train", str(z), "--config", str(cfg_path), "--epochs", "1", "--no-test", "--no-export"]) == 0
    run_dir = next((tmp_path / "models").iterdir())
    out = capsys.readouterr().out
    assert f"uv run -m trainer test {run_dir}" in out and f"uv run -m trainer export {run_dir}" in out
    assert not (run_dir / "test_report.txt").exists() and not (run_dir / "model.onnx").exists()
    parts = [r["split"] for r in csv.DictReader(open(run_dir / "split.csv", encoding="utf-8"))]
    assert parts.count("test") > 0 and parts.count("train") < len(parts) * 0.8      # test 는 떼어 둔 채다 (기본 70 %)
    assert main(["test", str(run_dir), "--device", "cpu"]) == 0 and (run_dir / "test_report.txt").is_file()
    assert main(["export", str(run_dir)]) == 0 and (run_dir / "model.onnx").is_file()


# ---------------------------------------------------------------------------
# 2026-11-23 코드 검토에서 나온 것들
# ---------------------------------------------------------------------------
def test_flip_needs_classes_symmetric_about_the_straight_ahead_angle():
    """대칭 축은 목록의 양 끝이 아니라 직진 각도다. 한쪽만 있는 목록(30 60 90, 직진 90)은 60 을 축으로 보면 대칭이라
    통과해 버렸고, 뒤집은 사진의 절반에 틀린 라벨이 붙었다."""
    from trainer.data import DatasetError, SteeringDataset, mirror_labels
    assert mirror_labels([30, 60, 90, 120, 150], center=90) == [4, 3, 2, 1, 0]
    assert mirror_labels([45, 90, 135], center=90.0) == [2, 1, 0]
    with pytest.raises(DatasetError, match="대칭"):
        mirror_labels([30, 60, 90], center=90)
    with pytest.raises(DatasetError, match="대칭"):
        mirror_labels([30, 60, 90, 120, 150], center=100)
    assert mirror_labels([30, 60, 90]) == [2, 1, 0]            # 직진 각도를 모르는 옛 zip: 양 끝의 가운데로 본다
    with pytest.raises(DatasetError):
        SteeringDataset(np.zeros((1, 3, 2, 2), np.uint8), np.array([0]), [30, 60, 90], hflip="random", center=90)


def test_straight_ahead_angle_comes_from_the_dataset(tmp_path):
    from common.preprocess import Preprocessor
    from trainer.data import DatasetError, load_samples
    new = make_dataset_zip(tmp_path / "new.zip", per_class=2, classes=[40, 100, 160], center=100, lo=40, hi=160)
    assert load_samples([new], Preprocessor()).center == 100
    old = make_dataset_zip(tmp_path / "old.zip", per_class=2, center=None)        # center 없음 → min 과 max 의 가운데
    assert load_samples([old], Preprocessor()).center == 90
    other = make_dataset_zip(tmp_path / "other.zip", per_class=2, center=80)
    with pytest.raises(DatasetError, match="직진 각도"):
        load_samples([old, other], Preprocessor())


def test_one_sided_classes_stop_instead_of_training_on_wrong_labels(tmp_path):
    from trainer.__main__ import main
    z = make_dataset_zip(tmp_path / "d.zip", per_class=6, classes=[30, 60, 90], center=90, lo=30, hi=90)
    cfg_path = _config(tmp_path, output_root=str(tmp_path / "models"), device="cpu")          # hflip 기본값 random
    assert main(["train", str(z), "--config", str(cfg_path), "--epochs", "1"]) == 1
    assert not (tmp_path / "models").exists() or not list((tmp_path / "models").iterdir())


def _copy_zip(src: Path, dst: Path, skip=None, replace=None) -> Path:
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for item in zin.infolist():
            if item.filename == skip:
                continue
            data = zin.read(item.filename)
            if replace and item.filename == replace[0]:
                data = replace[1]
            zout.writestr(item.filename, data)
    return dst


def test_broken_dataset_zips_give_a_dataset_error(tmp_path, capsys):
    from common.preprocess import Preprocessor
    from trainer.__main__ import main
    from trainer.data import DatasetError, load_samples
    garbage = tmp_path / "garbage.zip"
    garbage.write_bytes(b"this is not a zip file")
    with pytest.raises(DatasetError, match="열 수 없습니다"):
        load_samples([garbage], Preprocessor())

    good = make_dataset_zip(tmp_path / "good.zip", per_class=2)
    with zipfile.ZipFile(good) as z:
        first = sorted(n for n in z.namelist() if n.startswith("images/"))[0]
    with pytest.raises(DatasetError, match="읽을 수 없습니다"):
        load_samples([_copy_zip(good, tmp_path / "missing_image.zip", skip=first)], Preprocessor())
    with pytest.raises(DatasetError, match="손상"):
        load_samples([_copy_zip(good, tmp_path / "bad_image.zip", replace=(first, b"not a jpeg"))], Preprocessor())
    with pytest.raises(DatasetError, match="손상"):        # 전원이 끊겨 0바이트로 남은 사진: imdecode 가 예외를 던진다
        load_samples([_copy_zip(good, tmp_path / "empty_image.zip", replace=(first, b""))], Preprocessor())
    with pytest.raises(DatasetError, match="데이터셋 zip 이 아니거나"):
        load_samples([_copy_zip(good, tmp_path / "no_meta.zip", skip="meta.json")], Preprocessor())
    with pytest.raises(DatasetError, match="데이터셋 zip 이 아니거나"):
        load_samples([_copy_zip(good, tmp_path / "bad_meta.zip", replace=("meta.json", b"{}"))], Preprocessor())

    cfg_path = _config(tmp_path, output_root=str(tmp_path / "models"), device="cpu")
    assert main(["train", str(garbage), "--config", str(cfg_path)]) == 1          # 트레이스백이 아니라 안내로 끝난다
    assert "오류:" in capsys.readouterr().err


def test_channel_stats_in_chunks_match_the_direct_computation():
    from trainer.data import channel_stats
    images = np.random.default_rng(0).integers(0, 256, size=(37, 3, 5, 7), dtype=np.uint8)
    mean, std = channel_stats(images, chunk=8)
    x = images.astype(np.float64) / 255.0
    assert mean == pytest.approx(x.mean(axis=(0, 2, 3)).tolist())
    assert std == pytest.approx(x.std(axis=(0, 2, 3)).tolist())
    flat = np.full((4, 3, 2, 2), 128, dtype=np.uint8)
    assert min(channel_stats(flat)[1]) > 0                                         # 표준편차 0 으로 나누지 않게


def _toy_problem():
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(0)
    x = torch.randn(32, 4)
    y = (x[:, 0] > 0).long()
    return DataLoader(TensorDataset(x, y), batch_size=8), torch.nn.CrossEntropyLoss()


class _NanModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x) * float("nan")


def test_diverged_training_never_becomes_a_model(tmp_path, monkeypatch):
    """손실이 nan 이면 `val_loss < best` 가 언제나 거짓이라, 예전에는 학습 전의 무작위 가중치가 정상 모델처럼 저장됐다."""
    from trainer.engine import History, fit
    loader, loss_fn = _toy_problem()
    model = _NanModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: 1.0)
    h = fit(model, loader, loader, loss_fn, opt, sched, torch.device("cpu"), epochs=20, patience=3, log=None)
    assert h.diverged and h.best_epoch == 0 and h.epoch == []                      # 20 에폭을 다 돌지 않고 바로 멈춘다

    import trainer.run as run
    monkeypatch.setattr(run, "fit", lambda *a, **k: History(diverged=True))
    z = make_dataset_zip(tmp_path / "d.zip", per_class=6)
    with pytest.raises(run.TrainError, match="nan"):
        run.train([z], _train_cfg(tmp_path), log=lambda _m: None)
    assert list((tmp_path / "models").iterdir()) == []                             # 빈 결과 폴더도 남기지 않는다


class _InterruptedLoader:
    """stop_epoch 번째 에폭의 두 번째 배치에서 Ctrl+C 가 눌린 것처럼."""

    def __init__(self, inner, stop_epoch):
        self.inner, self.stop_epoch, self.epochs = inner, stop_epoch, 0

    def __iter__(self):
        self.epochs += 1
        for i, batch in enumerate(self.inner):
            if self.epochs == self.stop_epoch and i == 1:
                raise KeyboardInterrupt
            yield batch


def test_ctrl_c_keeps_the_best_model_so_far():
    from trainer.engine import fit
    loader, loss_fn = _toy_problem()
    model = torch.nn.Sequential(torch.nn.Linear(4, 2))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: 1.0)
    h = fit(model, _InterruptedLoader(loader, 3), loader, loss_fn, opt, sched, torch.device("cpu"), epochs=50,
            log=None)
    assert h.interrupted and h.epoch == [1, 2] and h.best_epoch == 2
    with pytest.raises(KeyboardInterrupt):                                         # 끝난 에폭이 없으면 그냥 중단
        fit(model, _InterruptedLoader(loader, 1), loader, loss_fn, opt, sched, torch.device("cpu"), epochs=5, log=None)


def test_result_folder_is_made_before_training_and_removed_on_failure(tmp_path, monkeypatch):
    import trainer.run as run
    z = make_dataset_zip(tmp_path / "d.zip", per_class=6)
    seen = []

    def fake_fit(*args, **kwargs):
        seen.append([p.name for p in (tmp_path / "models").iterdir()])
        raise RuntimeError("학습 도중 실패")

    monkeypatch.setattr(run, "fit", fake_fit)
    with pytest.raises(RuntimeError):
        run.train([z], _train_cfg(tmp_path), name="x", log=lambda _m: None)
    assert len(seen) == 1 and len(seen[0]) == 1 and seen[0][0].endswith("_x")      # 학습 전에 이미 있었다
    assert list((tmp_path / "models").iterdir()) == []

    blocker = tmp_path / "file.txt"
    blocker.write_text("폴더가 아니라 파일", encoding="utf-8")
    cfg = _train_cfg(tmp_path)
    cfg.training.output_root = str(blocker / "models")
    with pytest.raises(run.TrainError, match="output_root"):
        run.train([z], cfg, log=lambda _m: None)
    assert len(seen) == 1                                                          # 학습을 시작하기 전에 멈췄다


def test_run_folders_never_collide(tmp_path):
    from datetime import datetime
    from trainer.run import make_run_dir
    now = datetime(2026, 11, 23, 9, 0, 0)
    a = make_run_dir(tmp_path, "lr 0.01/x", now)
    b = make_run_dir(tmp_path, "lr 0.01/x", now)
    assert a.name == "20261123_090000_lr-0.01-x" and b.name == a.name + "_2" and a.is_dir() and b.is_dir()


def test_testing_with_several_zips_names_all_of_them(trained):
    from trainer.evaluate import analyze
    a = make_dataset_zip(trained["tmp"] / "dataset_a.zip", per_class=2, seed=7)
    b = make_dataset_zip(trained["tmp"] / "dataset_b.zip", per_class=2, seed=8)
    m = analyze(trained["run_dir"], data=[a, b], device="cpu", log=lambda _m: None)
    out = trained["run_dir"] / "test_dataset_a+dataset_b"
    assert m["summary"]["samples"] == 20 and (out / "test_report.txt").is_file()
    assert "dataset_a.zip, dataset_b.zip" in (out / "test_report.txt").read_text(encoding="utf-8").splitlines()[0]


def test_korean_run_name_draws_without_font_warnings(tmp_path):
    from trainer.engine import History
    from trainer.plots import save_confusion_matrix, save_training_curves
    h = History(epoch=[1, 2], lr=[0.001, 0.001], train_loss=[1.0, 0.5], train_acc=[0.5, 0.8],
                val_loss=[1.1, 0.7], val_acc=[0.4, 0.7], seconds=[1.0, 1.0], best_epoch=2, best_val_loss=0.7)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        save_training_curves(h, tmp_path / "curve.png", title="20261123_090000_학습률실험")
        save_confusion_matrix(np.array([[3, 1], [0, 4]]), ["30", "150"], tmp_path / "cm.png", title="시험 분석")
    assert (tmp_path / "curve.png").stat().st_size > 0 and (tmp_path / "cm.png").stat().st_size > 0


def test_weighted_loss_stays_finite_for_a_class_missing_from_training():
    """가중치 0 인 클래스만 든 배치의 손실은 0/0 = nan 이 된다 (계층화를 끄면 학습 데이터에 없는 클래스가 생길 수 있다)."""
    from trainer.losses import build_loss
    loss_fn = build_loss("cross_entropy", "balanced", [10, 90, 0])
    assert torch.isfinite(loss_fn(torch.zeros(4, 3), torch.tensor([2, 2, 2, 2])))


def test_cli_overrides_are_validated_like_the_file(tmp_path, capsys):
    from trainer.__main__ import main
    z = make_dataset_zip(tmp_path / "d.zip", per_class=4)
    cfg_path = _config(tmp_path, output_root=str(tmp_path / "models"), device="cpu")
    assert main(["train", str(z), "--config", str(cfg_path), "--lr", "0"]) == 1
    assert main(["train", str(z), "--config", str(cfg_path), "--epochs", "0"]) == 1
    assert "training.epochs" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 2026-11-23 두 번째 검토에서 나온 것들
# ---------------------------------------------------------------------------
def test_the_same_frames_given_twice_are_refused(tmp_path, capsys):
    """같은 사진이 두 번 들어오면 train 과 test 에 함께 들어가 시험 정확도가 부풀려진다."""
    from common.preprocess import Preprocessor
    from trainer.__main__ import main
    from trainer.data import DatasetError, load_samples
    a = make_dataset_zip(tmp_path / "dataset_a.zip", per_class=2)
    with pytest.raises(DatasetError, match="두 번 지정"):
        load_samples([a, tmp_path / "." / "dataset_a.zip"], Preprocessor())
    copy = _copy_zip(a, tmp_path / "dataset_a (1).zip")                  # 내려받은 사본, 또는 같은 세션으로 다시 만든 zip
    with pytest.raises(DatasetError, match=r"사진 10장이 dataset_a.zip 에도"):
        load_samples([a, copy], Preprocessor())
    b = make_dataset_zip(tmp_path / "dataset_b.zip", per_class=2, seed=3)   # 다른 세션: 함께 쓸 수 있다
    samples = load_samples([a, b], Preprocessor())
    assert len(samples) == 20 and samples.groups.tolist() == [0] * 10 + [1] * 10
    cfg_path = _config(tmp_path, output_root=str(tmp_path / "models"), device="cpu")
    assert main(["train", str(a), str(copy), "--config", str(cfg_path), "--epochs", "1"]) == 1
    assert "시험 정확도" in capsys.readouterr().err


def test_weighted_loss_does_not_depend_on_batch_size():
    """클래스 가중치가 있으면 배치마다의 평균을 사진 수로 다시 평균해서는 안 된다 (배치 구성에 따라 값이 달라진다)."""
    from torch.utils.data import DataLoader, TensorDataset
    from trainer.engine import evaluate
    torch.manual_seed(0)
    x = torch.randn(40, 4)
    y = torch.cat([torch.zeros(32, dtype=torch.long), torch.ones(4, dtype=torch.long), torch.full((4,), 2)])
    model = torch.nn.Linear(4, 3)
    weight = torch.tensor([0.3, 5.0, 5.0])
    weighted, plain = torch.nn.CrossEntropyLoss(weight=weight), torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        want = weighted(model(x), y).item()                              # 데이터 전체를 한 번에 본 값
    for batch_size in (3, 8, 40):                                        # 정렬된 데이터라 배치마다 클래스 구성이 크게 다르다
        loader = DataLoader(TensorDataset(x, y), batch_size=batch_size)
        assert evaluate(model, loader, weighted, torch.device("cpu"))[0] == pytest.approx(want, rel=1e-5)
        assert evaluate(model, loader, plain, torch.device("cpu"))[0] == pytest.approx(plain(model(x), y).item(), rel=1e-5)


def test_failed_export_keeps_the_previous_bundle(trained, monkeypatch):
    """검사에서 떨어진 ONNX 가 model.onnx 자리에 남으면 예전 model_meta.json 과 짝이 맞지 않는 모델을 차에 싣게 된다."""
    import trainer.export as ex
    run_dir = trained["run_dir"]
    before = (run_dir / "model.onnx").read_bytes(), (run_dir / "model_meta.json").read_bytes()
    monkeypatch.setattr(ex, "_sha256", lambda _p: (_ for _ in ()).throw(ex.ExportError("검사 실패")))
    with pytest.raises(ex.ExportError):
        ex.export(run_dir, log=lambda _m: None)
    assert ((run_dir / "model.onnx").read_bytes(), (run_dir / "model_meta.json").read_bytes()) == before
    assert not list(run_dir.glob("*.part"))
    monkeypatch.undo()
    ex.export(run_dir, log=lambda _m: None)                              # 다시 하면 정상
    meta = json.loads((run_dir / "model_meta.json").read_text(encoding="utf-8"))
    assert meta["onnx"]["file"] == "model.onnx" and meta["onnx"]["sha256"] == ex._sha256(run_dir / "model.onnx")


def test_run_that_fails_while_saving_leaves_no_half_folder(tmp_path, monkeypatch):
    """model.pth 만 있는 폴더는 정상 실행처럼 보이지만 시험 분석도 내보내기도 할 수 없다."""
    import trainer.run as run
    z = make_dataset_zip(tmp_path / "d.zip", per_class=6)
    cfg = _train_cfg(tmp_path)
    cfg.training.epochs = 1

    def no_disk(*_a, **_k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(run, "save_training_curves", no_disk)           # model.pth, split.csv, history.csv 를 쓴 뒤 실패
    with pytest.raises(OSError):
        run.train([z], cfg, name="x", log=lambda _m: None)
    assert list((tmp_path / "models").iterdir()) == []


def test_damaged_model_file_and_too_small_input_give_messages_not_tracebacks(trained, tmp_path, capsys):
    import shutil
    from trainer.__main__ import main
    from trainer.run import TrainError, load_checkpoint
    broken = tmp_path / "run"
    shutil.copytree(trained["run_dir"], broken)
    data = (broken / "model.pth").read_bytes()
    (broken / "model.pth").write_bytes(data[:len(data) // 2])            # 복사하다 끊긴 파일
    with pytest.raises(TrainError, match="model.pth 를 읽을 수 없습니다"):
        load_checkpoint(broken / "model.pth")
    assert main(["export", str(broken)]) == 1 and "오류:" in capsys.readouterr().err
    (broken / "model.pth").write_bytes(b"not a checkpoint")
    assert main(["test", str(broken), "--device", "cpu"]) == 1 and "오류:" in capsys.readouterr().err

    z = make_dataset_zip(tmp_path / "d.zip", per_class=4)
    text = _config(tmp_path, output_root=str(tmp_path / "models"), device="cpu").read_text(encoding="utf-8")
    assert "out_height: 66" in text
    small = tmp_path / "small.yaml"
    small.write_text(text.replace("out_height: 66", "out_height: 20"), encoding="utf-8")   # PilotNet 에는 너무 작다
    assert main(["train", str(z), "--config", str(small), "--epochs", "1"]) == 1
    assert "preprocess.out_width" in capsys.readouterr().err
    assert not (tmp_path / "models").exists() or not list((tmp_path / "models").iterdir())


def test_trainer_is_not_stopped_by_car_only_settings(tmp_path, capsys):
    """학습기는 training 과 preprocess 만 검사한다 (2026-11-23): PC 의 config.yaml 에서 차가 쓰는 값이 틀려 있어도 학습은 된다.
    거꾸로 학습 값이 틀리면 지금처럼 한 줄 안내로 끝난다."""
    import yaml
    from common.config import ConfigError
    from trainer.__main__ import main
    z = make_dataset_zip(tmp_path / "d.zip", per_class=6)
    cfg_path = _config(tmp_path, output_root=str(tmp_path / "models"), device="cpu")
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    data["steering"]["center"] = 100                    # classes 에 없는 직진 각도: 차에서는 설정 오류
    data["hardware"] = {"dir_forward_high": "yes please"}
    cfg_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(cfg_path, program="collector")
    assert main(["train", str(z), "--config", str(cfg_path), "--epochs", "1", "--no-export"]) == 0
    assert len(list((tmp_path / "models").iterdir())) == 1

    data["training"]["epochs"] = "many"
    cfg_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    assert main(["train", str(z), "--config", str(cfg_path), "--no-export"]) == 1
    assert "training.epochs" in capsys.readouterr().err
    assert len(list((tmp_path / "models").iterdir())) == 1          # 틀린 설정으로는 폴더를 만들지 않는다
