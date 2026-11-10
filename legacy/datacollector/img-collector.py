# img-collector.py (data-collector 루트)
# =============================================================================
# Description : 차량 주행(DC/Servo)과 웹캠 영상 촬영을 동시에 수행하는
#               데이터 수집 통합 실행 스크립트.
#               - drive.py : 키보드 입력 기반 주행 제어
#               - camera_capture.py : OpenCV 기반 이미지 캡처 및 저장
# 
# 구성:
#   - 주행 스레드 : 모터/서보 조작 (drive.run_drive_control)
#   - 카메라 스레드 : 프레임 캡처 + 라벨(각도/속도) 저장
#   - stop_flag    : 두 스레드가 공유하는 종료 신호
#
# Author : Youngchul Jung
# =============================================================================

import threading

from camera.camera_capture import camera_capture_loop   # 영상 캡처 모듈
import hw_control.drive as drive                        # 주행 제어 모듈


# -----------------------------------------------------------------------------
# 공통 상태 / 설정값
# -----------------------------------------------------------------------------
# stop_flag을 리스트로 둔 이유:
#   - 파이썬 스레드에서는 불변 타입(bool)은 참조 공유가 어렵기 때문에
#     리스트로 감싸서 한 객체를 공유하도록 설계
stop_flag = [False]

# 데이터 저장 디렉토리 및 파일 경로
OUTPUT_DIR = "dataset"
CSV_FILE = "dataset/data_labels.csv"

# 촬영 해상도 및 프레임 저장 주기
IMAGE_W, IMAGE_H = 640, 480
SAVE_INTERVAL = 0.5  # 초 단위 (0.5초 = 초당 2프레임 저장)


# -----------------------------------------------------------------------------
# 현재 주행 상태 조회 함수
#   camera_capture_loop은 drive 모듈을 직접 접근하지 않고,
#   이 콜백을 통해 각도/속도를 조회함 → 모듈 간 결합도를 낮춤
# -----------------------------------------------------------------------------
def get_state():
    """
    camera_capture_loop에서 라벨 저장 시 사용하는 콜백 함수.
    drive 모듈 내부의 현재 서보 각도, 모터 속도 값을 반환한다.
    반환: (servo_angle, motor_speed)
    """
    return drive.get_current_state()


# -----------------------------------------------------------------------------
# 메인 실행부
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        # ---------------------------------------------------------------------
        # 1) 주행 제어 스레드
        #    - 키보드 입력을 받아 모터/서보를 제어
        #    - stop_flag[0]이 True로 바뀌면 자동 종료
        # ---------------------------------------------------------------------
        drive_thread = threading.Thread(
            target=drive.run_drive_control,  # drive.py 내부 메인 루프
            args=(stop_flag,),               # 종료 신호 공유
            daemon=True,                     # 메인 종료 시 자동 종료
        )

        # ---------------------------------------------------------------------
        # 2) 카메라 캡처 스레드
        #    - 웹캠에서 프레임 읽기
        #    - SAVE_INTERVAL 간격으로 이미지/라벨 저장
        # ---------------------------------------------------------------------
        camera_thread = threading.Thread(
            target=camera_capture_loop,
            args=(
                OUTPUT_DIR,      # 저장 폴더
                CSV_FILE,        # CSV 라벨 파일
                IMAGE_W, IMAGE_H,
                SAVE_INTERVAL,
                stop_flag,       # 종료 플래그 공유
                get_state,       # 라벨(각도/속도) 조회 콜백
            ),
            daemon=True,
        )

        # 스레드 시작
        drive_thread.start()
        camera_thread.start()

        # 두 스레드 종료까지 대기
        drive_thread.join()
        camera_thread.join()

    except KeyboardInterrupt:
        # Ctrl+C 를 누르면 두 스레드 종료 요청
        print("Interrupted by user.")
        stop_flag[0] = True
