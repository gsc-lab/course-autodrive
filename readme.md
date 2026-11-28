# 🚗 Capstone Design - Mini Autonomous Car

**영진전문대 글로벌시스템융합과 1학년 캡스톤디자인(I) 프로젝트**
카메라로 차선을 인식해 스스로 주행하는 **미니 자율주행 자동차** (Jetson Orin Nano).

이전 프로젝트 결과 영상: https://www.youtube.com/watch?v=0_gV1N-Q7Gw

## 🎯 목표

- 카메라 영상으로 차선을 따라 주행하는 **자율주행 (Lane Keeping)**
- 팀별 확장 기능 (신호등 인식, 장애물 감지, 속도 제어 등)

## 🧭 전체 흐름

사람이 운전하며 찍은 사진으로 모델을 학습하고, 그 모델이 차를 조향한다.

| 단계 | 프로그램 | 실행 위치 | 하는 일 | 결과물 |
|---|---|---|---|---|
| 1. 수집 | [collector/](collector/) | 차 (Jetson) | 패드로 운전하며 사진 촬영 | 세션 폴더 `dataset/<날짜_시각>/` |
| 2. 가공 | [curator/](curator/) | PC | 불량 사진 제외, 데이터셋 생성 | `dataset_<날짜_시각>.zip` |
| 3. 학습 | [trainer/](trainer/) | PC | 모델 학습과 시험 | `models/<실행 이름>/` |
| 4. 자율주행 | [inference/](inference/) | 차 (Jetson) | 학습한 모델로 주행 | 예측 기록 `dataset/drives/` |

용어

- **세션**: 한 번의 주행에서 찍은 사진 폴더. 파일 이름에 그 순간의 조향 각도와 속도가 적혀 있으며, 이것이 모델의 정답이다.
- **조향 단계** (`steering.classes`, 기본 30° 60° 90° 120° 150°): 모델이 고르는 답의 목록. 프로그램 화면에서는 **클래스**라고 한다.

## ▶️ 준비

### 차 (Jetson)

설치는 실습 자료를 따른다. 설치를 마친 차에 SSH 로 접속해 실행한다.

```bash
cd ~/course-autodrive
git pull --autostash && uv sync         # 최신 코드 받기
bash scripts/jetson_setup.sh pair       # 처음 한 번: 패드 연결 (Share + PS 를 길게 눌러 라이트바가 빠르게 깜빡일 때)
```

### PC

[uv](https://docs.astral.sh/uv/) 를 설치한 뒤 실행한다.

```bash
git clone https://github.com/gsc-lab/course-autodrive.git
cd course-autodrive
uv sync --extra gui --extra train       # NVIDIA GPU 가 있는 PC (수 GB)
uv sync --extra gui --extra train-cpu   # GPU 가 없는 PC (위 줄 대신. 둘 중 하나만)
```

- `gui` 는 가공 프로그램, `train` 은 학습 프로그램이다. 반드시 **한 줄에 함께** 적는다. 하나만 적으면 다른 하나가 제거된다.
- 코드 갱신: `git pull --autostash` 실행 후 위의 `uv sync` 를 다시 실행한다. `--autostash` 는 수정한 `config.yaml` 을 유지한 채 새 코드를 받는다.
- 갱신 후 `[설정 오류]` 가 나오면 안내에 따라 파일을 되돌리고 값을 다시 적는다.

## 📷 1단계: 수집 (차)

```bash
uv run python -m collector --check      # 점검: 카메라, 보드, 패드. 정상이면 서보가 가운데로 한 번 움직인다
uv run python -m collector              # 수집 시작
```

○ 로 촬영을 켜면 **전진하는 동안** 사진이 저장된다. 정지 중이거나 후진 중에는 저장되지 않는다.

### 패드 조작 (DualShock 4)

| 조작 | 동작 |
|---|---|
| 왼쪽 스틱 좌우 | 조향 (D패드 ←→ 는 한 단계씩) |
| R2 | 전진. 당긴 깊이에 따라 3단계 속도, 놓으면 정지 |
| L2 | 후진 |
| ○ | 촬영 시작 / 정지 |
| □ | 직전 3초를 불량으로 표시 |
| △ | 조향 중앙 |
| Options | 종료 |

### 상태 표시 (도트 매트릭스)

| 화면 | 의미 |
|---|---|
| ○ | 패드 연결됨, 촬영 안 함 |
| ● 과 ○ 가 번갈아 | 촬영 중 |
| X | 패드 연결 끊김 등 오류로 정지 |
| 꺼짐 | 수집 프로그램이 실행 중이 아님 |

### ⚠️ 안전 수칙

- 차에는 전원 스위치가 없다. 차가 제어되지 않으면 **구동 배터리 커넥터를 뽑는다.**
- SSH 가 끊겨도 차는 패드로 계속 움직인다. 종료는 반드시 **Options 버튼**으로 한다.
- `config.yaml` 의 `deadman` 을 `true` 로 하면 **L1 을 누르고 있는 동안만** 주행한다.

### 사진을 PC 로 옮기기

세션 폴더 전체를 PC 의 `dataset/` 폴더에 복사한다 (USB 메모리도 가능). 사진은 git 에 올리지 않는다.

```bash
# PC 에서 실행. IP 주소와 세션 이름은 본인 환경에 맞게 바꾼다
scp -r gsc@192.168.50.119:~/course-autodrive/dataset/20261110_1906 ./dataset/
```

파일 이름이 정답이므로 **파일 이름과 폴더 구조를 바꾸지 않는다.**

## ✂️ 2단계: 가공 (PC)

```bash
uv run -m curator dataset/20261110_1906     # 세션 폴더 열기 (여러 개 가능)
```

1. `←` `→` 로 사진을 확인한다 (`Space`: 자동 재생).
2. 불량 사진(차선 이탈, 잘못된 조향 등)은 `Delete` 로 제외한다. `Insert` 는 복원, `Ctrl+Z` 는 되돌리기. 원본은 삭제되지 않는다.
3. 사진 위의 가로선 두 개를 끌어 **모델에 넣을 범위**를 정한다. 어두운 부분이 잘리며, 사진 아래의 "모델이 보는 사진"이 그 결과다. 범위는 데이터셋에 저장되어 학습과 주행에 똑같이 적용된다.
4. **데이터셋 만들기**를 누르고 저장 폴더를 고르면 `dataset_<날짜_시각>.zip` 이 생성된다.

- 운전 중 □ 로 표시한 구간은 처음부터 제외되어 있다.
- 조향 단계가 다른 세션은 함께 열 수 없다. 조향 단계는 촬영 당시 차의 설정을 따른다.
- 단축키 목록: `F1`

## 🧠 3단계: 학습 (PC)

`train` 한 줄로 학습, 시험, 차에 실을 모델(ONNX) 생성까지 진행된다.

```bash
uv run -m trainer train dataset/dataset_20261110_1906.zip                           # 학습 → 시험 → ONNX
uv run -m trainer train dataset/dataset_20261110_1906.zip --name lr0.01 --lr 0.01   # 이름을 붙이고 학습률만 변경
uv run -m trainer test models/<실행 이름>                                           # 시험만 다시
uv run -m trainer test models/<실행 이름> --data dataset/다른날.zip                 # 다른 데이터셋으로 시험
uv run -m trainer export models/<실행 이름>                                         # ONNX 만 다시 생성
uv run -m trainer train -h                                                          # 옵션 설명 (test -h, export -h)
```

### 데이터 나누기

| 묶음 | 기본 비율 | 용도 |
|---|---|---|
| 학습 (train) | 70 % | 가중치 갱신 |
| 검증 (val) | 15 % | 에폭마다 평가해 저장할 모델 선택 |
| 시험 (test) | 15 % | 학습이 끝난 뒤 한 번만 평가 |

- 각도별로 섞어 나누므로 세 묶음의 각도 비율이 같다. 비율은 `config.yaml` 의 `training.split` 에서 바꾼다.
- 한 세션의 사진은 세 묶음에 **분산**된다. 연속한 사진은 서로 유사해 test 점수가 실제보다 다소 높게 나온다.
- 학습에 쓰지 않은 **다른 주행**의 데이터셋으로 `test --data` 를 실행한 점수가 실제 주행에 더 가깝다.

### 결과

`models/<실행 이름>/` 폴더에 저장된다.

| 파일 | 내용 |
|---|---|
| `loss_curve.png` | 학습 곡선 (에폭별 손실, 정확도, 학습률) |
| `test_report.txt`, `confusion_matrix.png` | 시험 결과: 정확도, 클래스별 정밀도·재현율·F1, 혼동 행렬 |
| `test_predictions.csv` | 시험 사진별 정답과 예측 |
| `model.onnx`, `model_meta.json` | 차에 싣는 모델과 그 정보 |
| `history.csv`, `model.pth`, `split.csv`, `train_config.json` | 에폭별 기록, PyTorch 가중치, 사진 분할 내역, 학습 설정 |

- 에폭 표의 `← 지금까지 최저`: 그 에폭의 검증 손실(val loss)이 그때까지 가장 낮다는 표시. 마지막으로 표시된 에폭의 모델이 저장된다.
- **전체 정확도**는 **기준선**과 비교한다. 기준선은 가장 많은 각도만 답할 때의 정확도이며, 전체 정확도가 기준선과 같으면 학습되지 않은 것이다.

### 옵션

- **설정 변경**: 학습 설정은 PC 의 [config.yaml](config.yaml) 2부(`training`)에 있다. 값을 **하나만** 바꿔 다시 학습하고 결과를 비교한다. `--lr`, `--epochs`, `--batch-size`, `--model` 은 명령에 직접 적을 수 있다.
- **zip 여러 개**: 합쳐서 학습한다. 같은 세션으로 만든 zip, 자르는 범위나 조향 단계가 다른 zip 은 함께 쓸 수 없다.
- **`test --data`**: zip 에 학습에 쓴 사진이 있으면 그 장수를 `주의:` 로 알린다.
- **`--no-test`, `--no-export`**: 시험 또는 ONNX 생성을 건너뛴다. 나중에 `trainer test`, `trainer export` 로 실행한다. `--no-test` 를 써도 test 사진은 학습에 쓰이지 않는다.

## 🏁 4단계: 자율주행 (차)

PC 의 `models/<실행 이름>/` 폴더 전체를 차에 복사한다. 폴더에 `model.onnx` 와 `model_meta.json` 이 있어야 한다 (없으면 PC 에서 `trainer export` 실행).

```bash
mkdir -p ~/course-autodrive/models       # 차에서 한 번
# PC 에서 실행. IP 주소와 실행 이름은 본인 환경에 맞게 바꾼다
scp -r models/20261123_183035_lr0.01 gsc@192.168.50.119:~/course-autodrive/models/
```

차에 SSH 로 접속해 실행한다.

```bash
uv run python -m inference models/20261123_183035_lr0.01 --check   # 점검 (첫 실행은 약 10초 소요)
uv run python -m inference models/20261123_183035_lr0.01           # 시작
```

수동 모드로 시작한다. 차를 트랙에 놓고 **○** 를 누르면 모델이 조향하며 주행한다.

| 조작 | 동작 |
|---|---|
| ○ | 자율주행 시작 / 정지 |
| D패드 ↑ ↓ | 자율주행 속도 한 단계 증가 / 감소 (가장 느린 단계로 시작) |
| R2, L2, 왼쪽 스틱, △ | 조작 즉시 자율주행 해제, 수동 운전으로 전환 |
| □ | 현재 시점을 기록에 표시 |
| Options | 종료 |

| 화면 | 의미 |
|---|---|
| 가운데 점 | 준비 중 |
| ○ | 대기 (수동 운전) |
| 회전하는 막대 | 자율주행 중 |
| X | 패드 연결 끊김 등 오류로 정지 |

- 자율주행 중에는 **패드를 조작하지 않아도 차가 주행한다.** 패드를 들고 차를 따라가며, 차선을 벗어나면 즉시 ○ 또는 R2·L2 를 누른다.
- 패드가 꺼지거나 연결이 끊기면 차가 정지한다.
- 수동 운전 중에도 모델은 예측을 계속한다. 터미널 상태줄에서 조작한 각도와 예측을 비교할 수 있다.
- 사진별 예측은 `dataset/drives/` 에 CSV 로 저장된다 (모드, 예측 각도와 확률, 적용된 각도와 속도).

차 없이 PC 에서 모의 실행으로 동작을 확인할 수 있다 (키보드 `R` 자율주행 시작/정지, `Q` 종료).

```bash
uv run python -m inference models/20261123_183035_lr0.01 --mock
```

## ⚙️ 설정 (config.yaml)

조정할 수 있는 값은 [config.yaml](config.yaml) 한 파일에 있으며, 값마다 설명이 적혀 있다. 파일은 차와 PC 에 각각 있으므로 **그 값을 쓰는 쪽의 파일**을 수정한다.

| 부분 | 내용 | 수정 위치 |
|---|---|---|
| 1부 데이터 수집 | 조향 단계, 속도 단계, 사진 저장 간격, 카메라 노출 | 차 (SSH 접속) |
| 2부 모델 학습 | 모델 입력 크기, 학습 설정 | PC |

값이 잘못되면 프로그램 시작 시 `[설정 오류]` 로 위치를 알린다.
