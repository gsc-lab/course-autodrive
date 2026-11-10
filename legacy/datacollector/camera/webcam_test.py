import cv2  # OpenCV 라이브러리 임포트

# ------------------------------------------------------------
# 1. 웹 카메라 장치 열기
# ------------------------------------------------------------
# 일반적으로 기본 카메라는 인덱스 0번 장치
cap = cv2.VideoCapture(0)

# 카메라가 정상적으로 열리지 않은 경우(연결 문제, 장치 없음 등)
if not cap.isOpened():
    print("Error: Cannot access the camera.")
    exit()

# ------------------------------------------------------------
# 2. 웹캠 해상도 설정
# ------------------------------------------------------------
desired_width = 640   # 원하는 가로 해상도
desired_height = 480  # 원하는 세로 해상도

# OpenCV 속성 설정 (카메라가 원하는 해상도를 지원하지 않으면 다른 값으로 설정될 수 있음)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

# 실제로 적용된 해상도를 확인 (카메라 마다 지원 해상도가 다름)
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Resolution set to: {int(actual_width)}x{int(actual_height)}")

# ------------------------------------------------------------
# 3. 프레임 캡처 루프 시작
# ------------------------------------------------------------
while True:
    # 카메라로부터 현재 프레임 하나 읽기
    ret, frame = cap.read()

    # 프레임 읽기 실패 시 오류 처리
    if not ret:
        print("Error: Unable to retrieve frame from camera.")
        break

    # 화면 창에 프레임 보여주기
    cv2.imshow("Webcam Feed", frame)

    # 키 입력 대기: 1ms 동안 키 입력을 체크
    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# ------------------------------------------------------------
# 4. 종료 처리 (리소스 해제)
# ------------------------------------------------------------
cap.release()            # 카메라 장치 해제
cv2.destroyAllWindows()  # OpenCV가 생성한 모든 창 닫기
