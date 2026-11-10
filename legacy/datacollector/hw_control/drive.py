# drive.py
# -*- coding: utf-8 -*-
"""
===============================================================================
File Name     : drive.py
Description   : Jetson Nano + L298N(DC Motor) + MG996R Servo Control
                - DC Motor : L298N (PWM pin 33)
                - Servo    : MG996R (PWM pin 32)
                - Keyboard input: input_utils.get_key_nonblock()

Author        : Youngchul Jung
Date Created  : 2025-11-13
===============================================================================
"""
import time
import Jetson.GPIO as GPIO
import subprocess
import getpass

# get_key_nonblock() 경로별 폴백 처리
try:
    # 패키지로 설치/실행되는 경우 (예: img-collector 패키지 내부)
    from datacollector.hw_control.input_utils import get_key_nonblock
except ImportError:
    try:
        # 로컬 소스 트리에서 hw_control 패키지로 실행하는 경우
        #   python3 -m hw_control.drive
        from hw_control.input_utils import get_key_nonblock
    except ImportError:
        # 같은 디렉토리에서 직접 실행하는 경우
        #   python3 drive.py
        from input_utils import get_key_nonblock


# ================= PWM ENABLE ==================
def activate_jetson_pwm(auto_install_busybox=True):
    """
    Jetson Nano에서 특정 핀을 PWM 기능으로 사용하기 위해
    레지스터(devmem)를 직접 설정하는 함수
    - busybox가 없으면 (옵션에 따라) 자동 설치
    - sudo 권한이 필요하므로 실행 시 비밀번호를 한 번 입력받음
    """
    sudo_pw = getpass.getpass("Enter sudo password: ")

    def run_sudo(cmd):
        # 입력받은 sudo 비밀번호를 사용해 root 권한 명령 실행
        full_cmd = f"echo {sudo_pw} | sudo -S {cmd}"
        subprocess.run(full_cmd, shell=True, check=True)

    # busybox(devmem 포함) 자동 설치 옵션
    if auto_install_busybox:
        try:
            subprocess.run(
                "busybox --help",
                shell=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            # busybox가 설치되어 있지 않으면 설치
            run_sudo("apt update && apt install -y busybox")

    # Jetson 특정 레지스터를 devmem으로 직접 설정하여
    # 해당 핀들을 PWM 모드로 매핑
    cmds = [
        "busybox devmem 0x700031fc 32 0x45",
        "busybox devmem 0x6000d504 32 0x2",
        "busybox devmem 0x70003248 32 0x46",
        "busybox devmem 0x6000d100 32 0x00",
    ]
    for c in cmds:
        run_sudo(c)


# ================= CONSTANTS ====================
# Jetson BOARD 핀 번호
MOTOR_PWM_PIN         = 33   # DC 모터 PWM 핀
MOTOR_DIRECTION_PIN1  = 29   # L298N IN2
MOTOR_DIRECTION_PIN2  = 31   # L298N IN1
SERVO_PWM_PIN         = 32   # 서보 PWM 핀

# PWM 설정 값
MOTOR_PWM_FREQUENCY   = 1000  # DC 모터 PWM 주파수 (Hz)
SERVO_PWM_FREQUENCY   = 50    # 서보 PWM 주파수 (Hz)

# 서보 제어 범위 (듀티 비율)
SERVO_MIN_DC = 5.0            # 0도 근처 듀티(대략 값, 서보에 따라 조정 가능)
SERVO_MAX_DC = 10.0           # 180도 근처 듀티
SERVO_STEPS  = [30, 60, 90, 120, 150]  # 사용 가능한 5단계 조향 각도
SERVO_INDEX  = 2              # 초기 인덱스(90도, 중앙)

# 모터 속도 및 제어 관련 상수
motor_speed     = 60          # 기본 모터 속도 (0~100)
MOTOR_STEP      = 10          # A/Z 키로 속도 변경 단위
KEY_DELAY       = 0.04        # 키 입력 폴링 주기 (초)

# 정지 시 감속 설정
STOP_DECAY_STEP  = 3          # 속도를 몇씩 줄일지
STOP_DECAY_DELAY = 0.015      # 감속 단계 사이 딜레이

# 모터 방향 상태 (None: 정지, "forward": 전진, "backward": 후진)
current_direction = None


# ================= GPIO INIT ====================
# BOARD 모드 사용 (실제 보드 핀 번호 기준)
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# 핀 모드 설정
GPIO.setup(MOTOR_PWM_PIN, GPIO.OUT)
GPIO.setup(MOTOR_DIRECTION_PIN1, GPIO.OUT)
GPIO.setup(MOTOR_DIRECTION_PIN2, GPIO.OUT)
GPIO.setup(SERVO_PWM_PIN, GPIO.OUT)

# PWM 객체 생성
motor_pwm = GPIO.PWM(MOTOR_PWM_PIN, MOTOR_PWM_FREQUENCY)
servo_pwm = GPIO.PWM(SERVO_PWM_PIN, SERVO_PWM_FREQUENCY)

# 초기 PWM 시작 (모터는 정지, 서보는 대략 중앙 7.5% 근처)
motor_pwm.start(0)
servo_pwm.start(7.5)


# ================= SERVO ========================
def angle_to_duty(angle):
    """
    서보 각도(0~180도)를 PWM 듀티(%)로 변환
    SERVO_MIN_DC ~ SERVO_MAX_DC 범위를 선형 매핑
    """
    # 보호: 입력 각도를 0~180 범위로 클램핑
    angle = max(0, min(180, angle))
    return SERVO_MIN_DC + (angle / 180.0) * (SERVO_MAX_DC - SERVO_MIN_DC)


def set_servo_angle(angle):
    """
    서보를 특정 각도(도 단위)로 회전시키는 함수
    - angle_to_duty()로 듀티 계산 후 PWM 듀티 변경
    """
    duty = angle_to_duty(angle)
    servo_pwm.ChangeDutyCycle(duty)
    # 서보가 움직일 시간을 조금 부여 (필요 시 튜닝 가능)
    time.sleep(0.005)


# ================= MOTOR ========================
def control_motor(direction):
    """
    DC 모터를 지정한 방향으로 구동하는 함수.
    direction:
        - "forward"  : 전진
        - "backward" : 후진
    현재 설정된 motor_speed (0~100)를 듀티로 사용.
    """
    global current_direction

    if direction == "forward":
        # IN1=LOW, IN2=HIGH → 전진
        GPIO.output(MOTOR_DIRECTION_PIN1, GPIO.LOW)
        GPIO.output(MOTOR_DIRECTION_PIN2, GPIO.HIGH)

    elif direction == "backward":
        # IN1=HIGH, IN2=LOW → 후진
        GPIO.output(MOTOR_DIRECTION_PIN1, GPIO.HIGH)
        GPIO.output(MOTOR_DIRECTION_PIN2, GPIO.LOW)

    # 설정된 속도로 PWM 출력
    motor_pwm.ChangeDutyCycle(motor_speed)
    # 현재 방향 상태 기록
    current_direction = direction


def smooth_stop():
    """
    모터를 갑자기 0으로 끄지 않고,
    현재 설정된 motor_speed에서 0까지 서서히 줄이면서 정지.
    - 기계적 충격을 줄이기 위한 감속 정지 함수
    """
    global current_direction

    # motor_speed에서 0까지 STOP_DECAY_STEP만큼 감소시키며 듀티 변경
    for sp in range(motor_speed, -1, -STOP_DECAY_STEP):
        motor_pwm.ChangeDutyCycle(sp)
        time.sleep(STOP_DECAY_DELAY)

    # 완전 정지 상태
    motor_pwm.ChangeDutyCycle(0)
    GPIO.output(MOTOR_DIRECTION_PIN1, GPIO.LOW)
    GPIO.output(MOTOR_DIRECTION_PIN2, GPIO.LOW)

    # 방향 상태 초기화
    current_direction = None

def get_current_state():
    """
    img-collector에서 현재 서보 각도, 모터 속도를 라벨로 쓰기 위해 조회하는 함수.
    반환:
        (servo_angle_deg, motor_speed_percent)
    """
    global SERVO_INDEX, SERVO_STEPS, motor_speed
    return SERVO_STEPS[SERVO_INDEX], motor_speed

def run_drive_control(stop_flag=None):
    """
    키보드 입력을 받아 모터/서보를 제어하는 메인 루프.
    - stop_flag가 [True]가 되면 루프 종료 (img-collector와 연동용)
    - stop_flag가 None이면 ESC/CTRL_C 기준으로만 종료 (단독 실행용)
    """
    global current_direction, SERVO_INDEX, motor_speed

    activate_jetson_pwm()
    set_servo_angle(SERVO_STEPS[SERVO_INDEX])

    print(
        "\n=== CONTROL MODE ===\n"
        "  ↑ : Forward\n"
        "  ↓ : Backward\n"
        "  ← / → : Steering Left / Right\n"
        "  S : Center (90°)\n"
        "  A/Z : Speed + / -\n"
        "  ESC / Ctrl+C : Exit\n"
    )

    try:
        while True:
            if stop_flag is not None and stop_flag[0]:
                break

            key = get_key_nonblock()

            if key == "UP":
                if current_direction == "backward":
                    smooth_stop()
                else:
                    control_motor("forward")

            elif key == "DOWN":
                if current_direction == "forward":
                    smooth_stop()
                else:
                    control_motor("backward")

            elif key == "LEFT":
                SERVO_INDEX = max(0, SERVO_INDEX - 1)
                set_servo_angle(SERVO_STEPS[SERVO_INDEX])

            elif key == "RIGHT":
                SERVO_INDEX = min(len(SERVO_STEPS)-1, SERVO_INDEX + 1)
                set_servo_angle(SERVO_STEPS[SERVO_INDEX])

            elif key in ("s", "S"):
                SERVO_INDEX = SERVO_STEPS.index(90)
                set_servo_angle(90)

            elif key in ("a", "A"):
                motor_speed = min(100, motor_speed + MOTOR_STEP)
                print("[MOTOR] speed:", motor_speed)
                
                if current_direction is not None:
                    motor_pwm.ChangeDutyCycle(motor_speed)

            elif key in ("z", "Z"):
                motor_speed = max(0, motor_speed - MOTOR_STEP)
                print("[MOTOR] speed:", motor_speed)
                
                if current_direction is not None:
                    motor_pwm.ChangeDutyCycle(motor_speed)
            
            elif key in ("t", "T"):
                smooth_stop()
                
            elif key in ("ESC", "CTRL_C"):
                break

            time.sleep(KEY_DELAY)

    finally:
        smooth_stop()
        motor_pwm.stop()
        servo_pwm.stop()
        GPIO.cleanup()
        print("GPIO cleaned up.")

# ================= MAIN =========================
if __name__ == "__main__":
    print("[DEBUG] Simple Controller Started")

     # 단독 실행 시
    run_drive_control(stop_flag=None)