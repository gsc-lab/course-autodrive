"""모델 배포: model.pth → model.onnx + model_meta.json.

    uv run -m trainer export models/<실행 이름>

ONNX 는 학습 도구(PyTorch)와 실행 도구(Jetson 의 TensorRT) 사이의 공용 모델 파일 형식이다.
차에서는 PyTorch 를 쓰지 않는다: 추론기가 model.onnx 로 그 보드에 맞는 TensorRT 엔진을 만들어 쓴다.

약속 (추론기와의 계약, CLAUDE.md)
    입력  이름 "input",  (1, 3, H, W) float32, RGB, 0~1      ← 스케일링(rescale)은 모델 안에 들어 있다
    출력  이름 "output", (1, 클래스 수) float32, 클래스별 점수(logit). 가장 큰 값의 클래스가 조향 각도
    배치 크기는 1 로 고정 (차는 한 번에 사진 한 장만 본다)
model_meta.json 에는 추론기가 사진을 모델 입력으로 바꾸는 데 필요한 것이 모두 들어 있다.
"""
from __future__ import annotations

import hashlib
import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from trainer import __version__
from trainer.run import load_checkpoint

MODEL_META_SCHEMA_VERSION = 1
ONNX_OPSET = 17                 # Jetson 의 TensorRT 10.3 이 읽을 수 있는 버전
INPUT_NAME, OUTPUT_NAME = "input", "output"
N_CHECK = 20                    # PyTorch 와 견주어 볼 무작위 입력의 수


class ExportError(RuntimeError):
    pass


def export(run_dir: str | Path, log: Callable[[str], None] = print) -> Path:
    """ONNX 로 내보내고, PyTorch 와 같은 답을 내는지 확인한 뒤 model_meta.json 을 쓴다. 돌려주는 값: onnx 경로."""
    run_dir = Path(run_dir)
    model, ckpt = load_checkpoint(run_dir / "model.pth", "cpu")
    onnx_path = run_dir / "model.onnx"
    # 검사를 모두 통과한 파일만 model.onnx 가 된다. 그 전까지는 임시 이름으로 쓴다: 검사에서 떨어진 파일이 모델 폴더에
    # 남거나, 다시 내보내다 실패해 예전 model_meta.json 과 짝이 맞지 않는 model.onnx 가 남는 일이 없다.
    part_path = run_dir / "model.onnx.part"
    try:
        meta = _export_and_check(model, ckpt, run_dir, part_path, onnx_path.name)
        os.replace(part_path, onnx_path)
    except BaseException:
        part_path.unlink(missing_ok=True)
        raise
    (run_dir / "model_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    shape_in, shape_out = meta["input"]["shape"], meta["output"]["shape"]
    log(f"      model.onnx ({onnx_path.stat().st_size / 1024:.0f} KB, opset {ONNX_OPSET}), model_meta.json")
    log(f"      입력 \"{INPUT_NAME}\" {shape_in} → 출력 \"{OUTPUT_NAME}\" {shape_out}")
    log(f"      PyTorch 와 비교: 무작위 입력 {N_CHECK}개에서 예측이 모두 일치, 출력의 최대 차이 "
        f"{meta['onnx']['max_abs_diff_vs_pytorch']:.1e}")
    return onnx_path


def _export_and_check(model, ckpt: dict, run_dir: Path, onnx_path: Path, final_name: str) -> dict:
    """onnx_path 에 내보내고 검사한 뒤 model_meta.json 에 쓸 내용을 돌려준다. 검사에서 떨어지면 ExportError."""
    import onnx
    import onnxruntime as ort

    classes, input_shape = ckpt["classes"], ckpt["input_shape"]
    dummy = torch.zeros(1, *input_shape, dtype=torch.float32)

    # 1. 내보내기 -----------------------------------------------------------
    with warnings.catch_warnings():         # 쓰는 내보내기 방식이 오래됐다는 안내는 숨긴다 (작은 CNN 에는 이쪽이 단순하다)
        warnings.simplefilter("ignore", DeprecationWarning)
        torch.onnx.export(model, (dummy,), str(onnx_path), input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
                          opset_version=ONNX_OPSET, do_constant_folding=True, dynamo=False)

    # 2. 파일 검사: 형식이 올바른지, 입출력 이름과 모양이 약속대로인지 ----------
    proto = onnx.load(str(onnx_path))
    onnx.checker.check_model(proto)
    inputs = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim] for i in proto.graph.input}
    outputs = {o.name: [d.dim_value for d in o.type.tensor_type.shape.dim] for o in proto.graph.output}
    want_in, want_out = {INPUT_NAME: [1, *input_shape]}, {OUTPUT_NAME: [1, len(classes)]}
    if inputs != want_in or outputs != want_out:
        raise ExportError(f"ONNX 입출력이 약속과 다릅니다: 입력 {inputs} (기대 {want_in}), 출력 {outputs} (기대 {want_out})")

    # 3. 같은 사진에 PyTorch 와 ONNX 가 같은 답을 내는지 -----------------------
    session = ort.InferenceSession(onnx_path.read_bytes(), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    max_diff, agree, scale = 0.0, 0, 1.0
    for _ in range(N_CHECK):
        x = rng.random((1, *input_shape), dtype=np.float32)
        with torch.no_grad():
            ref = model(torch.from_numpy(x)).numpy()
        got = session.run([OUTPUT_NAME], {INPUT_NAME: x})[0]
        max_diff = max(max_diff, float(np.abs(ref - got).max()))
        scale = max(scale, float(np.abs(ref).max()))          # 출력이 큰 모델(rescale: none)은 차이도 그만큼 크다
        agree += int(ref.argmax() == got.argmax())
    if not np.isfinite(max_diff) or max_diff > 1e-3 * scale or agree != N_CHECK:
        raise ExportError(f"ONNX 의 출력이 PyTorch 와 다릅니다 (최대 차이 {max_diff:.2e}, 같은 답 {agree}/{N_CHECK})")

    # 4. model_meta.json ----------------------------------------------------
    info_path = run_dir / "train_config.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    metrics_path = run_dir / "test_metrics.json"
    test = json.loads(metrics_path.read_text(encoding="utf-8"))["summary"] if metrics_path.is_file() else None
    meta = {
        "schema_version": MODEL_META_SCHEMA_VERSION,
        "task": "classification",
        "model": ckpt["model"],
        "classes": classes,                                   # 출력 i 번째 값 = classes[i] 도(서보 각도)의 점수
        "preprocess": ckpt["preprocess"],                     # common.preprocess.Preprocessor.from_dict 로 그대로 읽는다
        "input": {"name": INPUT_NAME, "shape": [1, *input_shape], "dtype": "float32",
                  "color": "RGB", "range": "0-1", "layout": "NCHW"},
        "output": {"name": OUTPUT_NAME, "shape": [1, len(classes)], "dtype": "float32",
                   "semantics": "logits over classes; steering angle = classes[argmax(output)]"},
        "scaler_in_model": ckpt["scaler"],                    # 참고용. 이미 ONNX 안에 들어 있으니 추론기는 적용하지 않는다
        "onnx": {"file": final_name, "opset": ONNX_OPSET, "sha256": _sha256(onnx_path),
                 "max_abs_diff_vs_pytorch": max_diff},
        "training": {
            "run_id": run_dir.name,
            "datasets": [d.get("file") for d in info.get("datasets", [])],
            "sessions": sorted({s for d in info.get("datasets", []) for s in d.get("meta", {}).get("sessions", [])}),
            "frames": info.get("frames"),
            "settings": info.get("training"),
            "result": info.get("result"),
            "test": test,
        },
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "tool": {"name": "trainer", "version": __version__, "torch": torch.__version__, "onnx": onnx.__version__},
    }
    return meta


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
