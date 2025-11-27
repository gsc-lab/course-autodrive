# inference/run_inference.py

import time
import cv2
import numpy as np

from preprocessor.RCPreprocessor import RCPreprocessor
from inference.engine_loader import TRTInferenceEngine
import datacollector.hw_control.drive as drive   
import datacollector.hw_control.input_utils as input_utils


ANGLE_LIST = [30, 60, 90, 120, 150]


def main():
    # 0) Jetson PWM 활성화
    drive.activate_jetson_pwm()

    # 1) TensorRT 엔진 로드
    engine_path = "models/pilotnet_steering.trt"
    engine = TRTInferenceEngine(engine_path)

    # 2) 전처리기 (학습과 동일)
    preproc = RCPreprocessor(
        out_size=(200, 66),
        crop_top_ratio=0.4,
        crop_bottom_ratio=1.0
    )

    # 3) 카메라 설정
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[ERROR] Failed to open camera (index 0)")
        return

    print("[INFO] Starting inference loop... (q / ESC to quit)")
    print(
        "\n=== KEY CONTROL (during auto-pilot) ===\n"
        "  ↑ / ↓  : Forward / Backward\n"
        "  ← / →  : Steering Left / Right (1-step override)\n"
        "  S      : Center steering (90°)\n"
        "  A / Z  : Speed + / -\n"
        "  T      : Smooth stop\n"
        "  q / ESC: Exit\n"
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Camera frame read failed")
                continue

            # ------------------------------------------------------
            # 0) 키 입력 처리 (모터 속도 & 긴급 조향)
            # ------------------------------------------------------
            key = input_utils.get_key_nonblock()
            steering_overridden = False  # 이 프레임에서 수동 조향했는지 여부

            if key:
                # 방향 제어
                if key == "UP":
                    # run_drive_control 과 동일 로직 사용
                    if drive.current_direction == "backward":
                        drive.smooth_stop()
                    else:
                        drive.control_motor("forward")

                elif key == "DOWN":
                    if drive.current_direction == "forward":
                        drive.smooth_stop()
                    else:
                        drive.control_motor("backward")

                # 조향 수동 오버라이드
                elif key == "LEFT":
                    drive.SERVO_INDEX = max(0, drive.SERVO_INDEX - 1)
                    drive.set_servo_angle(drive.SERVO_STEPS[drive.SERVO_INDEX])
                    steering_overridden = True

                elif key == "RIGHT":
                    drive.SERVO_INDEX = min(
                        len(drive.SERVO_STEPS) - 1,
                        drive.SERVO_INDEX + 1
                    )
                    drive.set_servo_angle(drive.SERVO_STEPS[drive.SERVO_INDEX])
                    steering_overridden = True

                elif key in ("s", "S"):
                    # 중앙 복귀
                    drive.SERVO_INDEX = drive.SERVO_STEPS.index(90)
                    drive.set_servo_angle(90)
                    steering_overridden = True

                # 속도 조절
                elif key in ("a", "A"):
                    drive.motor_speed = min(100, drive.motor_speed + drive.MOTOR_STEP)
                    print("[MOTOR] speed:", drive.motor_speed)
                    if drive.current_direction is not None:
                        drive.motor_pwm.ChangeDutyCycle(drive.motor_speed)

                elif key in ("z", "Z"):
                    drive.motor_speed = max(0, drive.motor_speed - drive.MOTOR_STEP)
                    print("[MOTOR] speed:", drive.motor_speed)
                    if drive.current_direction is not None:
                        drive.motor_pwm.ChangeDutyCycle(drive.motor_speed)

                # 비상 정지
                elif key in ("t", "T"):
                    drive.smooth_stop()

                # 종료
                elif key in ("ESC", "CTRL_C"):
                    break

            # ------------------------------------------------------
            # 1) 전처리
            # ------------------------------------------------------
            img_chw = preproc(frame)                  # (3,66,200)
            input_batch = img_chw[np.newaxis, ...]    # (1,3,66,200)

            # ------------------------------------------------------
            # 2) TensorRT 추론
            # ------------------------------------------------------
            logits = engine.infer(input_batch)        # (1,num_classes)
            pred_idx = int(np.argmax(logits, axis=1))
            pred_angle = ANGLE_LIST[pred_idx]

            # ------------------------------------------------------
            # 3) RC Car 서보 제어
            #    - 긴급 조향 키를 누른 프레임은 수동 값 유지
            # ------------------------------------------------------
            if not steering_overridden:
                drive.set_servo_angle(pred_angle)

            # ------------------------------------------------------
            # 4) 디버그 출력
            # ------------------------------------------------------
            cv2.putText(
                frame,
                f"Angle: {pred_angle}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

            cv2.imshow("RC Auto Pilot", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break

            # 약 30 FPS
            time.sleep(0.03)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C).")

    finally:
        # 자원 정리
        cap.release()
        cv2.destroyAllWindows()
        drive.smooth_stop()
        print("[INFO] Inference stopped, resources cleaned up.")


if __name__ == "__main__":
    main()
