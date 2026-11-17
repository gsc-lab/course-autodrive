# 🚗 Capstone Design - Mini Autonomous Car

**영진전문대 글로벌시스템융합과 1학년 캡스톤디자인(I) 프로젝트**
딥러닝과 임베디드 시스템으로 **미니 자율주행 자동차**를 만듭니다 (Jetson Orin Nano).

이전 프로젝트 결과 영상: https://www.youtube.com/watch?v=0_gV1N-Q7Gw

## 🎯 프로젝트 목표

- 카메라 영상으로 차선을 따라 달리는 **자율주행 (Lane Keeping)**
- 팀별 확장 기능 제안과 구현 (신호등 인식, 장애물 감지, 속도 제어 등)

## 🧭 전체 흐름

1. **수집** (Jetson): 패드로 차를 직접 운전하며 사진을 저장합니다.
2. **가공** (PC): 사진을 검토해 잘못 찍힌 것을 빼고 데이터셋 zip 을 만듭니다.
3. **학습** (PC): 데이터셋으로 모델을 학습합니다. (준비 중)
4. **자율주행** (Jetson): 학습한 모델로 차가 스스로 달립니다. (준비 중)

## 📂 구성

| 폴더 | 실행 위치 | 역할 |
|---|---|---|
| [collector/](collector/) | Jetson | 패드로 운전하며 사진 수집 → `dataset/<세션>/` |
| [curator/](curator/) | PC | 세션 검토, 불량 사진 제외 → 데이터셋 zip |
| trainer/ (예정) | PC | 학습 → ONNX 모델 |
| inference/ (예정) | Jetson | TensorRT 추론으로 자율주행 |
| [common/](common/) | 공용 | 설정, 파일 이름 규칙, 조향 모델, 하드웨어·입력 장치 드라이버 |
| [docs/](docs/) | | [Jetson 설정](docs/jetson-setup.md), [배선도](docs/wiring.pdf) |

설정은 저장소 루트의 [config.yaml](config.yaml) 한 파일에 모여 있습니다. 조향 단계, 속도 단계, 저장 간격을 여기서 바꿉니다.

## ▶️ 시작하기

### Jetson (차)

교수님이 나눠 주신 SD 카드에는 설치가 끝나 있습니다. 보드를 켜고 SSH 로 접속한 뒤 아래 순서로 진행합니다.

```bash
cd ~/course-autodrive
git pull && uv sync                     # 최신 코드로 갱신
bash scripts/jetson_setup.sh pair       # 처음 한 번: 자기 패드 페어링 (Share + PS 를 길게 눌러 라이트바가 빠르게 깜빡일 때)
uv run python -m collector --check      # 점검: 카메라, 보드 두 장, 패드. 서보가 중앙으로 한 번 움직이면 정상
uv run python -m collector              # 수집 시작
```

SD 카드를 새로 구웠다면 [docs/jetson-setup.md](docs/jetson-setup.md) 의 설치 절차를 먼저 따릅니다.

### PC

의존성은 [uv](https://docs.astral.sh/uv/) 로 관리합니다. Python 3.10 을 씁니다 (Jetson 과 같은 버전).

```bash
git clone https://github.com/gsc-lab/course-autodrive.git
cd course-autodrive
uv sync --extra gui                     # 가공 프로그램용. 학습까지 하려면 --extra pc (torch 포함, 수 GB)
uv run -m curator dataset/20261110_1906 # 세션 폴더를 열어 검토 (여러 개를 함께 열 수 있음)
```

차 없이 수집기를 연습하려면 `uv run python -m collector --mock` 을 실행합니다. 가짜 차와 카메라를 키보드로 조종합니다.

## 🎮 운전 방법 (DualShock 4)

| 조작 | 동작 |
|---|---|
| 왼쪽 스틱 좌우 | 조향 (D패드 ←→ 는 한 단계씩) |
| R2 | 전진. 당긴 깊이에 따라 3단계 속도, 놓으면 정지 |
| L2 | 후진 (사진은 저장되지 않음) |
| ○ | 촬영 시작 / 정지 |
| □ | 방금 3초를 불량으로 표시 (실수한 구간에서 바로 누르기) |
| △ | 조향 중앙 |
| Options | 종료 |

차에 달린 도트 매트릭스로 상태를 확인합니다.

| 화면 | 뜻 |
|---|---|
| ○ | 패드 연결됨, 촬영 안 함 |
| ● 과 ○ 가 번갈아 | 촬영 중 |
| X | 패드 연결이 끊기는 등 문제가 생겨 멈춤 |
| 꺼짐 | 수집기가 실행 중이 아님 |

## ⚠️ 안전 수칙

- 조립하거나 배선을 바꾼 뒤 첫 시험은 **바퀴를 띄운 채로** 합니다.
- 차에는 전원 스위치가 없습니다. 차가 말을 듣지 않으면 **구동 배터리 커넥터를 뽑습니다.**
- 패드로 운전할 때는 SSH 가 끊겨도 차가 계속 달립니다. 끝낼 때는 **Options 버튼**을 누릅니다.

## 💾 수집한 데이터 옮기기

`dataset/` 폴더는 git 에 올리지 않습니다. 세션 폴더를 통째로 PC 로 가져옵니다 (USB 메모리로 옮겨도 됩니다).

```bash
# PC 에서 실행. Jetson 의 IP 와 세션 이름은 자기 것으로 바꿉니다
scp -r gsc@192.168.50.119:~/course-autodrive/dataset/20261110_1906 ./dataset/
```

사진 파일 이름이 곧 라벨이므로 **파일 이름과 폴더 구조를 손으로 바꾸지 않습니다.**

## 🔌 하드웨어 요약

서보와 모터는 Jetson 의 I2C 에 연결한 **PCA9685 보드 두 장**이 구동합니다.

| 보드 | 주소 | 연결 |
|---|---|---|
| 서보용 | 0x40 | 채널 0 → 조향 서보 (LD-1501MG, 전원은 UBEC 6 V) |
| 모터용 | 0x41 | 채널 0 속도, 채널 1 방향 → 모터 드라이버 (Cytron MDD10A) → 뒷바퀴 모터 2개 |

Jetson 40핀 헤더에서는 핀 1 (3.3 V), 3 (SDA), 5 (SCL), 6 (GND) 를 씁니다. 헤더의 PWM 핀은 쓰지 않습니다.
전체 배선은 [docs/wiring.pdf](docs/wiring.pdf) 에 있습니다.
