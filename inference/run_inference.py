# inference/run_inference.py

import time
import cv2
import numpy as np

from preprocessor.RCPreprocessor import RCPreprocessor
from inference.engine_loader import TRTInferenceEngine
import datacollector.hw_control.drive as drive   # RC car servo control module


ANGLE_LIST = [30, 60, 90, 120, 150]


def main():
    drive.activate_jetson_pwm()
    
    # 1) TensorRT 엔진 로드
    engine_path = "models/pilotnet_steering.trt"
    engine = TRTInferenceEngine(engine_path)

    # 2) 전처리기
    preproc = RCPreprocessor(
        out_size=(200, 66),
        crop_top_ratio=0.4,
        crop_bottom_ratio=1.0
    )

    # 3) 카메라 설정
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[INFO] Starting inference loop...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Camera frame read failed")
            continue

        # ------------------------------
        # (1) 전처리
        # ------------------------------
        img_chw = preproc(frame)           # (3,66,200)
        input_batch = img_chw[np.newaxis, ...]  # (1,3,66,200)

        # ------------------------------
        # (2) TensorRT 추론
        # ------------------------------
        logits = engine.infer(input_batch)  # (1,num_classes)
        pred_idx = int(np.argmax(logits, axis=1))
        pred_angle = ANGLE_LIST[pred_idx]

        # ------------------------------
        # (3) RC Car 서보 제어
        # ------------------------------
        drive.set_servo_angle(pred_angle)

        # ------------------------------
        # (4) 디버그 출력
        # ------------------------------
        cv2.putText(frame, f"Angle: {pred_angle}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        cv2.imshow("RC Auto Pilot", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.03)   # 약 30 FPS


if __name__ == "__main__":
    main()
