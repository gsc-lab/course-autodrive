# camera/camera_capture.py
# =============================================================================
# Description : OpenCV를 사용하여 웹캠에서 프레임을 읽고,
#               일정 간격(save_interval)마다 이미지 파일 + CSV 라벨을 저장하는 모듈.
#               - 데이터 수집(img-collector.py)의 카메라 스레드에서 호출됨.
#
# 입력:
#   output_dir   : 이미지 저장 폴더
#   csv_file     : 라벨(CSV) 저장 파일 경로
#   image_width  : 캡처할 이미지 너비(px)
#   image_height : 캡처할 이미지 높이(px)
#   save_interval: 이미지 저장 간격(초)
#   stop_flag    : [True/False] 종료 신호 공유 리스트
#   state_getter : (servo_angle, motor_speed) 반환 함수 (drive 모듈로부터 제공)
#
# Author : Youngchul Jung
# =============================================================================

import cv2
import csv
import os
import time
from datetime import datetime


def camera_capture_loop(
    output_dir,
    csv_file,
    image_width,
    image_height,
    save_interval,
    stop_flag,
    state_getter,
):
    """
    웹캠으로부터 프레임을 실시간으로 읽고,
    주기적으로 이미지 파일 및 (조향각, 모터속도 등) 라벨 값을 CSV로 저장하는 루프.

    매개변수:
        output_dir   : 이미지가 저장될 폴더
        csv_file     : 라벨을 기록할 CSV 경로
        image_width  : 캡처해올 이미지의 가로 해상도
        image_height : 캡처해올 이미지의 세로 해상도
        save_interval: 몇 초 간격으로 1장을 저장할지
        stop_flag    : stop_flag[0] == True 이면 루프 종료
        state_getter : 현재 주행 상태(servo_angle, motor_speed) 조회 콜백 함수
    """
    
    # -------------------------------------------------------------------------
    # 1) 저장 폴더 및 CSV 파일 초기화
    # -------------------------------------------------------------------------
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # CSV 파일이 없다면 헤더 생성
    if not os.path.exists(csv_file):
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "image_path", "servo_angle", "dc_motor_speed"])

    # -------------------------------------------------------------------------
    # 2) 웹캠 초기화
    # -------------------------------------------------------------------------
    cap = cv2.VideoCapture(0)  # 0번 카메라(기본 웹캠)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)

    if not cap.isOpened():
        print("Error: Cannot access the camera.")
        return

    last_save = time.time()  # 마지막 저장 시점

    # -------------------------------------------------------------------------
    # 3) 메인 캡처 루프
    # -------------------------------------------------------------------------
    while not stop_flag[0]:
        ret, frame = cap.read()  # 한 프레임 캡처

        # 프레임 읽기 실패 시 종료
        if not ret:
            print("Error: Unable to retrieve frame from camera.")
            break

        now = time.time()

        # ---------------------------------------------------------------------
        # 3-1) save_interval 초마다 이미지 + 라벨 저장
        # ---------------------------------------------------------------------
        if now - last_save >= save_interval:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

            # 주행 상태(서보 각도, 모터 속도)를 외부 모듈(drive)에서 조회
            servo_angle, motor_speed = state_getter()

            # 파일 이름에 timestamp + angle + speed 포함 → 라벨링 자동화 용이
            filename = f"{timestamp}_angle{servo_angle}_speed{motor_speed}.png"
            image_path = os.path.join(output_dir, filename)

            # 이미지 파일 저장
            cv2.imwrite(image_path, frame)

            # CSV 라벨 기록
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, filename, servo_angle, motor_speed])

            #print(f"[SAVE] {filename} | angle={servo_angle} | speed={motor_speed}")
            last_save = now  # 저장 시점 업데이트

        # ---------------------------------------------------------------------
        # 3-2) 모니터에 현재 프레임 출력
        # ---------------------------------------------------------------------
        cv2.imshow("Webcam Feed", frame)

        # 'q' 키 입력 시 즉시 종료 (카메라 스레드 종료 → 전체 종료)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_flag[0] = True
            break

    # -------------------------------------------------------------------------
    # 4) 종료 처리 – 카메라/윈도우 정리
    # -------------------------------------------------------------------------
    cap.release()
    cv2.destroyAllWindows()
