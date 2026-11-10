# 📂 Hardware

## ▶️ **drive.py — 차량 주행 제어 모듈**
- Jetson Nano + L298N + MG996R 기반 제어 로직
- 기능  
  - 전진/후진 제어  
  - 조향(서보) 제어  
  - 속도 조절  
  - 역방향 입력 시 안전 정지  

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

## ▶️ 차량 조종 테스트
```bash
sudo python3 drive.py
```