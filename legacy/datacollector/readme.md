# 📂 Data Collector – Autonomous Driving Dataset Tool

## 📌 개요
이 디렉토리는 **카메라 영상 + 차량 조향/모터 제어 정보**를  
**동시에 수집하여 자율주행 학습용 데이터셋을 생성**하는 도구를 제공합니다.

- 자동차를 **키보드로 수동 조작**하면서  
- 매 프레임마다 **이미지 + 조향각 + 모터 속도**가 저장됩니다.  
- PyTorch/TensorFlow 학습에 바로 활용 가능한 형태의 데이터셋 생성이 목적입니다.

---

# 📁 디렉토리 구조

```
data-collector/
├─ camera/
│   ├─ __init__.py
│   ├─ camera_capture.py    # OpenCV 기반 영상 캡처 + 저장 모듈
│   └─ webcam_test.py       # 카메라 테스트 유틸
│
├─ hw_control/
│   ├─ __init__.py
│   ├─ drive.py             # 차량 주행 제어 (키보드 입력 처리 포함)
│   └─ input_utils.py       # Raw 키 입력 처리 모듈
│
├─ img-collector.py         # 🚗 + 📷 데이터 수집 통합 실행 스크립트
└─ readme.md
```

---

# ⚙️ 구성 요소 설명

## 1. **drive.py — 차량 주행 제어 모듈**
- Jetson Nano + L298N + MG996R 기반 제어 로직
- 기능  
  - 전진/후진 제어  
  - 조향(서보) 제어  
  - 속도 조절  
  - 역방향 입력 시 안전 정지  
- 외부에 제공하는 주요 API  
  - `run_drive_control(stop_flag)`  
  - `get_current_state()` → `(servo_angle, motor_speed)` 반환  

---

## 2. **camera_capture.py — 영상 캡처 모듈**
- OpenCV로 웹캠 프레임을 읽고 저장
- 기능  
  - 일정 주기(`save_interval`)마다 이미지 저장  
  - 라벨 자동 기록(CSV)  
  - 파일명에 각도/속도 포함 → 학습 라벨링 자동화

---

## 3. **img-collector.py — 통합 실행 스크립트**
- 🚗 주행 스레드 + 📷 카메라 스레드를 동시에 실행  
- 기능  
  - stop_flag 공유로 깨끗한 종료  
  - 주행 상태와 이미지가 시간적으로 동기화  
- 실행 예시  

```bash
sudo python3 img-collector.py
```

---

# 🔌 H/W Wiring Guide  

## ▶️ L298N DC Motor Wiring

| Jetson Nano Pin (BOARD) | L298N Pin | 기능 |
|--------------------------|-----------|------|
| 33 | ENA / PWM | DC Motor 속도 제어 |
| 31 | IN1 | 방향 제어 1 |
| 29 | IN2 | 방향 제어 2 |
| GND | GND | 공통 접지 |

⚠️ **주의**  
- Jetson의 5V와 L298N의 5V OUT을 **절대 연결하지 말 것** (전압 충돌 위험)  
- DC 모터 전원은 외부 12V 또는 7.4V 배터리를 사용하길 권장

---

## ▶️ MG996R Servo Wiring

| 서보 배선 | 연결 위치 |
|-----------|-----------|
| 갈색 (GND) | Jetson GND |
| 빨강 (VCC) | Jetson 5V |
| 노랑 (PWM) | Jetson BOARD 핀 32 |

---

# 🧩 전체 동작 흐름

```
img-collector.py
   ├─ Thread 1 → drive.run_drive_control()
   │               • 키보드 입력 처리
   │               • DC/Servo 값 업데이트
   │
   └─ Thread 2 → camera_capture_loop()
                   • 프레임 캡처
                   • 이미지 저장
                   • state_getter()로 라벨 조회
```

→ 결과적으로 **이미지와 조향각/속도가 동기화된 데이터셋**이 생성됩니다.

---

# 📦 데이터셋 구조

```
dataset/
├─ 20240210_153015123456_angle120_speed50.png
├─ 20240210_153015225678_angle90_speed40.png
└─ data_labels.csv
```

### CSV 예시

| timestamp | image_path | servo_angle | dc_motor_speed |
|-----------|------------|-------------|----------------|
| 20240210_153015123456 | 20240210_153015123456_angle120_speed50.png | 120 | 50 |
| 20240210_153015225678 | 20240210_153015225678_angle90_speed40.png | 90 | 40 |

---

# ▶️ 실행 방법

## 1️⃣ 카메라 테스트
```bash
python3 camera/webcam_test.py
```

## 2️⃣ 차량 조종 테스트
```bash
sudo python3 hw_control/drive.py
```

## 3️⃣ 데이터 수집 전체 실행
```bash
sudo python3 img-collector.py
```

---