# 🚗 Capstone Design - Mini Autonomous Car

**영진전문대 글로벌시스템융합과 1학년 캡스톤디자인(I) 프로젝트**
딥러닝과 임베디드 시스템을 활용한 **미니 자율주행 자동차 제작** (Jetson Orin Nano)

이전 프로젝트 결과 영상: https://www.youtube.com/watch?v=0_gV1N-Q7Gw

## 🎯 프로젝트 목표
- 카메라 입력 기반 **차선 인식 및 자율주행 (Lane Keeping)**
- 팀별 확장 기능 제안 및 구현 (신호등 인식, 장애물 감지, 속도 제어 등)

## 📂 구성

| 디렉토리 | 실행 위치 | 역할 |
|---|---|---|
| [common/](common/) | 공용 | 설정, 파일 이름 규약, 조향 모델, 전처리, 하드웨어·입력 장치 드라이버 |
| [collector/](collector/) | Jetson | DualShock 4 로 주행하며 데이터 수집 → `dataset/<세션>/` |
| curator/ (예정) | PC | 세션 검토·불량 제외 → 데이터셋 zip |
| trainer/ (예정) | PC | 학습 → ONNX + model_meta.json |
| inference/ (예정) | Jetson | TensorRT 추론 자율주행 |
| [docs/](docs/) | | [Jetson 설정](docs/jetson-setup.md), 배선도, 보고서 |

설정은 저장소 루트의 [config.yaml](config.yaml) 하나다. 조향 클래스, PCA9685 주소·채널, 카메라, 녹화 간격을 여기서 바꾼다.

## ▶️ 빠른 시작

의존성은 [uv](https://docs.astral.sh/uv/) 로 관리한다 (`pyproject.toml` + `uv.lock`). Python 3.10 (Jetson 과 동일).

```bash
# PC: 의존성 + 테스트 + 모의 실행(가짜 차량/카메라, 키보드)
uv sync --extra pc                 # PySide6, torch(CUDA 12.8), pytest 까지 설치 (첫 실행은 수 GB 다운로드)
uv run pytest
uv run python -m collector --mock

# Jetson: git clone 후 설치 스크립트 (패키지, 권한, uv). 자세한 절차는 docs/jetson-setup.md
bash scripts/jetson_setup.sh install
bash scripts/jetson_setup.sh verify     # I2C 의 PCA9685 두 장(0x40, 0x41), 권한, 패드, 카메라 점검
bash scripts/jetson_setup.sh pair       # DualShock 4 페어링
uv run python -m collector
```

학생은 GitHub 에서 `git clone` 으로 받고, 갱신은 `git pull && uv sync` 으로 한다. 수집한 `dataset/` 은 git 에 올리지 않고 scp 나 USB 로 PC 에 가져온다.

## 🔌 하드웨어 요약

서보와 모터 신호는 Jetson 의 I2C(40핀 헤더 핀 3 SDA, 5 SCL, 1 3.3 V, 6 GND)에 연결한 **PCA9685 두 장**이 낸다:
서보용 0x40(50 Hz, 채널 0 → LD-1501MG, 전원 UBEC 6 V), 모터용 0x41(1.5 kHz, 채널 0 속도·1 방향 → Cytron MDD10A → 후륜 모터 2개, 7.2 V NiMH).
Jetson 헤더의 PWM 핀은 쓰지 않는다. 배선 전체는 [docs/wiring.pdf](docs/wiring.pdf).
