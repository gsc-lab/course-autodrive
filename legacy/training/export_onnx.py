# export_onnx.py

import torch
from model_pilotnet import PilotNet

def export_onnx():
    # ----------------------------------------------------
    # 1. 모델 생성 & 가중치 로드
    # ----------------------------------------------------
    model = PilotNet(
        num_classes=5,
        input_shape=(3, 66, 200)
    )

    model_path = "models/pilotnet_steering.pth"
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # ----------------------------------------------------
    # 2. 더미 입력 (TensorRT에서 사용할 고정 입력)
    # ----------------------------------------------------
    dummy_input = torch.randn(1, 3, 66, 200, dtype=torch.float32)

    # ----------------------------------------------------
    # 3. ONNX Export
    # ----------------------------------------------------
    onnx_path = "models/pilotnet_steering.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,             # 모델 파라미터 포함 저장
        opset_version=11,               # TensorRT 호환 추천
        do_constant_folding=True,       # 최적화
        input_names=["input"],          # TensorRT 바인딩 이름
        output_names=["output"],        # TensorRT 바인딩 이름
        dynamic_axes=None               # 고정 Shape (TensorRT 잘 동작함)
    )

    print(f"[INFO] ONNX model saved to: {onnx_path}")


if __name__ == "__main__":
    export_onnx()
