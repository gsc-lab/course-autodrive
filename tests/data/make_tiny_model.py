"""tests/data/tiny_model/ 을 만든다: 추론기 테스트용의 아주 작은 모델 번들 (학습한 모델이 아니다).

    uv run python tests/data/make_tiny_model.py        (PC, 학습용 패키지가 있어야 한다: onnx, onnxruntime)

Jetson 에는 onnx 도 torch 도 없으므로 만든 파일을 저장소에 넣어 둔다. 보드에서 pytest 를 돌리면 TensorRT 가 이 model.onnx 로
엔진을 만들고, PC 의 onnxruntime 이 낸 답(reference.json)과 같은지 본다.

모양은 학습기가 내보내는 것과 같다: 입력 "input" (1, 3, H, W) float32 → Conv → Relu → Flatten → Gemm → 출력 "output" (1, 5).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT = Path(__file__).parent / "tiny_model"
H, W, CLASSES = 8, 16, [30, 60, 90, 120, 150]


def main() -> None:
    rng = np.random.default_rng(20261123)
    conv_w = rng.normal(0, 0.5, (4, 3, 3, 3)).astype(np.float32)
    conv_b = rng.normal(0, 0.1, (4,)).astype(np.float32)
    fc_w = rng.normal(0, 0.2, (len(CLASSES), 4 * (H // 2) * (W // 2))).astype(np.float32)
    fc_b = rng.normal(0, 0.1, (len(CLASSES),)).astype(np.float32)
    nodes = [
        helper.make_node("Conv", ["input", "conv_w", "conv_b"], ["c"], kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c"], ["r"]),
        helper.make_node("Flatten", ["r"], ["f"], axis=1),
        helper.make_node("Gemm", ["f", "fc_w", "fc_b"], ["output"], transB=1),      # fc_w 는 (출력, 입력) 모양 (PyTorch 의 Linear 와 같다)
    ]
    graph = helper.make_graph(
        nodes, "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, H, W])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, len(CLASSES)])],
        [numpy_helper.from_array(a, n) for a, n in ((conv_w, "conv_w"), (conv_b, "conv_b"), (fc_w, "fc_w"), (fc_b, "fc_b"))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8                        # 학습기가 내보내는 파일과 같은 판 (TensorRT 10.3 이 읽는다)
    onnx.checker.check_model(model)
    OUT.mkdir(parents=True, exist_ok=True)
    data = model.SerializeToString()
    (OUT / "model.onnx").write_bytes(data)

    session = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
    inputs = rng.random((6, 1, 3, H, W), dtype=np.float32)
    outputs = [session.run(["output"], {"input": x})[0][0] for x in inputs]
    (OUT / "reference.json").write_text(json.dumps({
        "about": "PC 의 onnxruntime 이 낸 답. 입력 6개와 그 출력(logit).",
        "inputs": [np.round(x, 6).tolist() for x in inputs],
        "outputs": [[float(v) for v in o] for o in outputs],
    }), encoding="utf-8")
    # 반올림해 적은 입력으로 다시 계산해 둔다 (파일의 입력과 출력이 정확히 짝이 맞게)
    ref = json.loads((OUT / "reference.json").read_text(encoding="utf-8"))
    ref["outputs"] = [[float(v) for v in session.run(["output"], {"input": np.asarray(x, dtype=np.float32)})[0][0]]
                      for x in ref["inputs"]]
    (OUT / "reference.json").write_text(json.dumps(ref), encoding="utf-8")

    meta = {
        "schema_version": 1, "task": "classification", "model": "tiny (테스트용, 학습하지 않음)", "classes": CLASSES,
        "preprocess": {"crop_top_ratio": 0.4, "crop_bottom_ratio": 1.0, "out_width": W, "out_height": H,
                       "color": "RGB", "scale": "0-1", "layout": "CHW", "dtype": "float32"},
        "input": {"name": "input", "shape": [1, 3, H, W], "dtype": "float32", "color": "RGB", "range": "0-1", "layout": "NCHW"},
        "output": {"name": "output", "shape": [1, len(CLASSES)], "dtype": "float32",
                   "semantics": "logits over classes; steering angle = classes[argmax(output)]"},
        "onnx": {"file": "model.onnx", "opset": 17, "sha256": hashlib.sha256(data).hexdigest()},
        "training": {"run_id": "tiny_model"},
    }
    (OUT / "model_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{OUT}: model.onnx {len(data)} bytes, argmax {[int(np.argmax(o)) for o in ref['outputs']]}")


if __name__ == "__main__":
    main()
