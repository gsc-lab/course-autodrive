# 🚗 Capstone Design - Mini Autonomous Car

**영진전문대 글로벌시스템융합과 1학년 캡스톤디자인(I) 프로젝트**
딥러닝과 임베디드 시스템으로 **미니 자율주행 자동차**를 만듭니다 (Jetson Orin Nano).

이전 프로젝트 결과 영상: https://www.youtube.com/watch?v=0_gV1N-Q7Gw

## 🎯 프로젝트 목표

- 카메라 영상으로 차선을 따라 달리는 **자율주행 (Lane Keeping)**
- 팀별 확장 기능 제안과 구현 (신호등 인식, 장애물 감지, 속도 제어 등)

## 🧭 전체 흐름

1. **수집** (Jetson): 패드로 차를 직접 운전하며 사진을 저장합니다.
2. **가공** (PC): 사진을 검토해 잘못 찍힌 것을 빼고, 모델에 넣을 범위를 정해 데이터셋 zip 을 만듭니다.
3. **학습** (PC): 데이터셋으로 모델을 학습하고, 시험 사진으로 결과를 분석한 뒤 ONNX 모델로 내보냅니다.
4. **자율주행** (Jetson): 학습한 모델로 차가 스스로 달립니다. (준비 중)

## 📂 구성

| 폴더 | 실행 위치 | 역할 |
|---|---|---|
| [collector/](collector/) | Jetson | 패드로 운전하며 사진 수집 → `dataset/<세션>/` |
| [curator/](curator/) | PC | 세션 검토, 불량 사진 제외, 자르는 범위 정하기 → 데이터셋 zip |
| [trainer/](trainer/) | PC | 데이터셋 zip 으로 학습 → `models/<실행 이름>/` (모델, 학습 곡선, 시험 분석, ONNX) |

## ⚙️ 설정

조정하는 값은 [config.yaml](config.yaml) 한 파일에 있습니다. 같은 파일이 차와 PC 에 하나씩 있으므로, 각 부분은 그 값을 쓰는 쪽에서 고쳐야 적용됩니다.

| 구역 | 언제 쓰는 값인가 | 내용 | 고치는 곳 |
|---|---|---|---|
| 1부 데이터 수집 | 차를 운전하며 사진을 찍을 때 | 조향 단계와 범위, 속도 단계, 사진 저장 간격, 카메라 노출 | 차 (SSH 로 접속해서) |
| 2부 모델 학습 | 수집한 사진으로 모델을 학습할 때 | 전처리(`preprocess`), 학습(`training`) | PC |

- 값을 잘못 적으면 프로그램을 시작할 때 `[설정 오류]` 로 알려 줍니다.
- 차를 조립한 뒤 `uv run python scripts/servo_calibrate.py` 로 조향 범위를 재서, 알려 주는 네 값(`classes`, `center`, `min`, `max`)을 적습니다.

## ▶️ 시작하기

### Jetson (차)

교수님이 나눠 주신 SD 카드에는 설치가 끝나 있습니다. 보드를 켜고 SSH 로 접속한 뒤 아래 순서로 진행합니다.

```bash
cd ~/course-autodrive
git pull --autostash && uv sync         # 최신 코드로 갱신
bash scripts/jetson_setup.sh pair       # 처음 한 번: 자기 패드 페어링 (Share + PS 를 길게 눌러 라이트바가 빠르게 깜빡일 때)
uv run python -m collector --check      # 점검: 카메라, 보드 두 장, 패드. 서보가 중앙으로 한 번 움직이면 정상
uv run python -m collector              # 수집 시작
```

`--autostash` 는 고쳐 둔 `config.yaml` 을 새 코드 위에 다시 얹어 줍니다. 얹지 못하면 프로그램이 `[설정 오류]` 로 알려 주니, 그 안내대로 파일을 되돌리고 값을 다시 적습니다.

SD 카드를 새로 구웠다면 교수님이 나눠 주신 실습 자료의 설치 절차를 먼저 따릅니다.

### PC

[uv](https://docs.astral.sh/uv/) 를 먼저 설치합니다.

```bash
git clone https://github.com/gsc-lab/course-autodrive.git
cd course-autodrive
uv sync --extra gui --extra train       # 가공 + 학습 프로그램 설치 (NVIDIA GPU 가 있는 PC, 수 GB)
uv sync --extra gui --extra train-cpu   # GPU 가 없는 PC 는 위 대신 이것 (둘 중 하나만)
uv run -m curator dataset/20261110_1906 # 세션 폴더를 열어 검토 (여러 개를 함께 열 수 있음)
```

`uv sync` 는 적어 준 것만 남기고 나머지를 지웁니다. 가공과 학습을 모두 쓰려면 위처럼 두 가지를 한 줄에 함께 적습니다.

사진 위의 가로선 두 개를 끌어 **모델에 넣을 범위**(위, 아래)를 정합니다. 어두운 부분이 잘려 나가고, 사진 아래의 "모델이 보는 사진"이 그 결과입니다. 사진을 실제로 자르지는 않습니다: 범위가 데이터셋 zip 에 적혀, 학습할 때와 차에서 달릴 때 똑같이 적용됩니다.

조향 클래스(`steering.classes`)는 그 세션을 **촬영할 때 차에 있던 설정**을 따릅니다. PC 의 `config.yaml` 을 맞춰 고칠 필요가 없습니다. 조향 클래스가 서로 다른 세션은 함께 열 수 없습니다 (한 데이터셋으로 묶을 수 없기 때문입니다).

## 🧠 학습

가공 프로그램이 만든 데이터셋 zip 으로 학습합니다. 한 줄로 학습, 시험 분석, ONNX 내보내기까지 이어집니다.

```bash
uv run -m trainer train dataset/dataset_20261110_1906.zip               # 학습 → 시험 분석 → ONNX
uv run -m trainer train dataset/dataset_20261110_1906.zip --name lr0.01 --lr 0.01   # 이름을 붙이고 학습률만 바꿔서
uv run -m trainer test models/<실행 이름>                               # 시험 분석만 다시
uv run -m trainer test models/<실행 이름> --data dataset/다른날.zip     # 다른 날 찍은 데이터로 시험
```

zip 을 여러 개 적으면 합쳐서 학습합니다. 단, 같은 사진이 든 zip 은 함께 줄 수 없습니다 (같은 세션으로 만든 zip 두 개 등): 같은 사진이 학습과 시험에 함께 들어가면 시험 정확도가 실제보다 높게 나오기 때문입니다. 같은 이유로, `test --data` 에 넣은 zip 에 학습에 쓴 사진이 들어 있으면 몇 장인지 `주의:` 로 알려 줍니다.

결과는 `models/<실행 이름>/` 에 모입니다.

| 파일 | 내용 |
|---|---|
| `loss_curve.png`, `history.csv` | 에폭마다의 학습 손실(train loss)과 검증 손실(validation loss), 정확도, 학습률 |
| `test_report.txt`, `confusion_matrix.png` | 시험 사진으로 잰 혼동 행렬과 클래스별 정확도, 정밀도, 재현율, F1. 전체 정확도 아래의 **기준선**은 가장 많은 각도만 답해도 나오는 정확도입니다 |
| `test_predictions.csv` | 시험 사진마다의 정답과 예측 (어떤 사진을 틀렸는지) |
| `model.onnx`, `model_meta.json` | 차에 싣는 모델과 그 설명서 |
| `model.pth`, `split.csv`, `train_config.json` | PyTorch 가중치, 사진을 어떻게 나눴는지, 이번 학습의 설정 |

학습 설정은 PC 의 [config.yaml](config.yaml) 2부에서 바꿉니다.

- `training`: 데이터 나누기, 입력 스케일링, 좌우 반전 증강, 배치 크기, 에폭 수와 조기 종료, 학습률
- `preprocess`: 모델에 넣는 사진의 크기. 사진에서 잘라 낼 범위는 여기가 아니라 가공 프로그램에서 정합니다

값을 하나씩 바꿔 가며 `loss_curve.png` 와 `test_report.txt` 가 어떻게 달라지는지 비교해 보세요. 조향 클래스와 자르는 범위는 데이터셋을 따라가므로 따로 맞추지 않습니다. 자르는 범위가 다른 데이터셋은 함께 학습할 수 없습니다.

코드는 [trainer/](trainer/) 에서 `split.py` (나누기) → `data.py` (Dataset, DataLoader) → `models.py` (모델) → `losses.py` (손실) → `optim.py` (옵티마이저) → `engine.py` (학습 루프) 순서로 읽으면 학습 한 번이 거치는 순서와 같습니다. 새 모델은 `models.py` 에 클래스를 만들어 `MODELS` 에 등록하고 `training.model` 에 이름을 적으면 됩니다.

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
- `config.yaml` 의 `deadman` 을 `true` 로 바꾸면 **L1 을 누르고 있는 동안만** 달립니다.

## 💾 수집한 데이터 옮기기

`dataset/` 폴더는 git 에 올리지 않습니다. 세션 폴더를 통째로 PC 로 가져옵니다 (USB 메모리로 옮겨도 됩니다).

```bash
# PC 에서 실행. Jetson 의 IP 와 세션 이름은 자기 것으로 바꿉니다
scp -r gsc@192.168.50.119:~/course-autodrive/dataset/20261110_1906 ./dataset/
```

사진 파일 이름이 곧 라벨이므로 **파일 이름과 폴더 구조를 손으로 바꾸지 않습니다.**
