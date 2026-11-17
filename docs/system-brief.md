# 자율주행 RC카 프로젝트 현황 — 강의 자료 작성용 정리

영진전문대 글로벌시스템융합과 1학년 캡스톤디자인(I). 카메라 영상과 작은 CNN 으로 트랙을 따라가는(Lane-Keeping) 미니 자율주행 자동차.
이 문서는 2026-11-12 기준 **실제로 만들어져 동작하는 내용**만 적는다. 강의 자료(설치·실습·보고서 안내)는 이 문서를 근거로 쓴다.
이전 학년도 자료(Jetson Nano, JetPack 4.6, L298N, 키보드 조종, sudo python3)는 모두 폐기되었고 아래 내용으로 대체된다.

---

## 1. 전체 흐름

```
[Jetson 차량]  collector  : DualShock 4 로 주행하며 사진 + 라벨 수집  →  dataset/<세션>/
      ↓ scp / USB
[PC]           curator    : 사진 검토, 불량 제외, 데이터셋 zip 생성
[PC]           trainer    : PyTorch 학습 → ONNX + model_meta.json               (예정)
      ↓ git / scp
[Jetson 차량]  inference  : ONNX → TensorRT 10 엔진, 자율주행                    (예정)
```

- 완성: `collector` (수집기), `curator` (가공 도구), 하드웨어 드라이버, 상태 표시, 설치 스크립트, 배선도.
- 미완성: `trainer`, `inference`. 단계 사이의 파일 계약(7절)은 확정되어 있다.
- 조향 라벨은 **5단계 분류** (서보 각도 30, 60, 90, 120, 150°; 90° 가 직진). 속도 라벨은 30/45/60 %.

## 2. 하드웨어

| 부품 | 모델 | 역할 |
|---|---|---|
| 컴퓨터 | NVIDIA Jetson Orin Nano Developer Kit (Super), JetPack 6.2.1, Python 3.10, TensorRT 10.3 | 영상 처리·제어 |
| Jetson 전원 | USB-PD 보조배터리 + USB-C→DC 트리거 케이블 | 차량 전원과 분리 |
| 카메라 | USB 웹캠, 640×480 MJPG 30 fps | 전방 영상 |
| 조종기 | Sony DualShock 4 (CUH-ZCT2G), 블루투스 | 주행·녹화 조작 |
| PWM 보드 | PCA9685 ×2 (I2C). 서보용 0x40 50 Hz, 모터용 0x41 1.5 kHz (A0 패드 납땜) | Jetson 헤더 출력이 약해 서보를 못 움직이므로 모든 신호는 이 보드가 냄 |
| 모터 드라이버 | Cytron MDD10A (2채널, PWM+DIR) | 후륜 DC 모터 2개, 두 채널에 같은 신호 |
| 조향 서보 | LD-1501MG (500~2500 µs, 중립 1500 µs) | 앞바퀴 조향 |
| 서보 전원 | Hobbywing UBEC 6 V | 주행 배터리 → 6 V |
| 주행 배터리 | NiMH 7.2 V 1200 mAh + 10 A 퓨즈 | 모터·서보. **주전원 스위치 없음** (배터리 커넥터를 뽑는 것이 전원 차단) |
| 상태 표시 | MAX7219 8×8 도트 매트릭스, Jetson SPI 직결 (선택) | 전진/후진/정지, 조향, 속도, 녹화, 오류 |
| 분배 | WAGO 221 3구 ×4 (J2 +, J3 접지, J4 속도 분기, J5 방향 분기) | 22 AWG 이상만 물린다 |

Jetson 40핀 헤더에서 쓰는 핀: 1 (3.3 V), 3 (SDA), 5 (SCL), 6 (GND) → PCA9685. 도트 매트릭스: 2 (5 V), 20 (GND), 19 (DIN), 23 (CLK), 24 (CS).
헤더의 PWM/GPIO 핀은 쓰지 않는다. 전체 배선은 `docs/wiring.pdf` (자율주행자동차 배선도 v1.0, 전선 36가닥, 그림 4장, 조립 후 동작 시험).

안전 관련 사실
- PCA9685 는 통신이 끊겨도 마지막 출력을 유지한다 → 소프트웨어가 멈추면 차가 계속 달릴 수 있다. 첫 시험은 반드시 바퀴를 띄우고 한다.
- 차를 세우는 최후 수단은 주행 배터리 커넥터를 뽑는 것.

## 3. 학생 보드 준비 (골든 이미지)

교수가 기준 보드 한 대를 완성한 뒤 SD 카드를 통째로 복제한다. 학생은 **설치 과정 없이** 보드를 켜고 자기 패드만 페어링한다.

```bash
cd ~/course-autodrive
bash scripts/jetson_setup.sh pair       # DualShock 4 페어링: Share + PS 를 라이트바가 빠르게 깜빡일 때까지 누른 채 실행
bash scripts/jetson_setup.sh verify     # 점검: I2C 의 PCA9685 두 장, SPI, 권한, 패드, 카메라
git pull && uv sync                     # 코드 갱신 (필요할 때)
```

기준 보드를 만들 때(교수): `bash scripts/jetson_setup.sh install` (패키지, i2c/input/gpio 그룹, bluez 설정, uv 가상환경, SPI1 켜기 → 재부팅) → `verify` → `prepare-image` → 카드 복제. 상세: `docs/jetson-setup.md`.

PC 쪽 접속: Wi-Fi 같은 공유기, SSH (`ssh gsc@<IP>`), 파일 전송은 `scp`. 의존성 관리는 `uv` (`pyproject.toml` + `uv.lock`), `pip install` 은 쓰지 않는다.

## 4. 조립 후 동작 시험 (배선도 4절)

바퀴를 띄우고 배터리를 연결한 뒤 순서대로:

| 순서 | 명령 | 정상 |
|---|---|---|
| 1 | `uv run python scripts/pin_test.py scan` | 0x40 서보용, 0x41 모터용 보드 응답 |
| 2 | `uv run python scripts/pin_test.py servo` | 서보 60° → 120° → 90° |
| 3 | `uv run python scripts/pin_test.py motor` | 두 바퀴가 같은 방향으로 10초, 정지, 반대 방향 10초 |
| 4 | `uv run python scripts/dotmatrix_test.py status` | 도트 매트릭스에 상태 화면 9가지 |
| 5 | `uv run python -m collector --check` | "점검 완료", 서보가 중앙으로 한 번 |

전진 명령에 뒤로 가면 선을 바꾸지 말고 `config.yaml` 의 `hardware.dir_forward_high: false`.

## 5. 수집기(collector) 사용법

```bash
uv run -m collector                     # 패드로 주행·수집 시작
uv run -m collector --input keyboard    # 패드가 없을 때 SSH 키보드로
```

DualShock 4 조작

| 조작 | 동작 |
|---|---|
| 왼쪽 스틱 좌우 | 조향. 가까운 클래스(30/60/90/120/150°)로 스냅, D패드 ←→ 는 한 단계씩 |
| R2 (당긴 깊이) | 전진. 깊이에 따라 30 / 45 / 60 % 세 단계. 놓으면 정지 |
| L2 | 후진 (고정 30 %). 후진 중에는 사진을 저장하지 않음 |
| ○ | 녹화 켜기/끄기 (정지 상태에서만 바뀜) |
| □ | 직전 3초를 "불량"으로 표시 (marks.csv). 사진은 지우지 않고 가공 단계에서 제외 후보가 됨 |
| △ | 조향 중앙 |
| Options | 종료 (정지 → 서보 중앙 → 세션 닫기) |

키보드(SSH): ↑ 전진 토글, ↓ 후진, Space/T 정지, ←→ 조향, S 중앙, A/Z 속도, R 녹화, M 불량 표시, [ ] 저장 간격, Q/ESC 종료. 키보드 모드는 30초 동안 입력이 없으면 자동 정지.

사진이 저장되는 조건: **녹화 ON + 전진 중(속도 > 0)**. 기본 0.5초마다 한 장. 정지·후진 중에는 저장하지 않는다.

터미널 상태줄 예: `[REC●] a=090 v=045 forward | saved=120 int=0.5s | 30:10 60:25 90:60 120:20 150:5`
(조향각, 속도, 방향, 저장 장수, 간격, 클래스별 저장 수).

도트 매트릭스: 위 5행 ▲ 전진 / ▼ 후진 / ■ 정지 (녹화 중이면 위 모서리 점멸), 7행 조향 위치 점, 8행 속도 막대, 오류 시 X 점멸.

안전 동작
- 패드 연결이 끊기면 0.5초 안에 정지하고 종료한다 (패드가 보내는 신호 흐름을 감시).
- 카메라가 끊기면 정지하고 오류 표시.
- SSH 가 끊겨도(노트북 절전, Wi-Fi) 패드 주행은 계속된다. 끝내려면 Options 버튼, 또는 다시 접속해 `kill <PID>` (PID 는 시작 화면에 표시).
- 수집기는 한 번에 하나만 돈다. 이미 돌고 있으면 시작하지 않고 끝내는 명령을 알려 준다.
- 주 루프가 1초 넘게 멈추면 감시 스레드가 모터를 끈다.

## 6. 수집 결과: 데이터셋이 어디에, 어떤 구조로 저장되는가

### 6.1 위치

Jetson 의 저장소 폴더 안 `dataset/` 아래. 수집기를 한 번 실행할 때마다 **세션 폴더** 하나가 생긴다 (이름 = 시작 시각 `YYYYMMDD_HHMM`).

```
/home/gsc/course-autodrive/dataset/
├── collector.log                 수집기 실행 로그 (모든 세션 공통, 문제 생겼을 때 봄)
├── 20261110_1752/                세션 1 (11월 10일 17:52 시작)
├── 20261110_1758/                세션 2
└── 20261110_1842/                세션 3
    ├── session_meta.json         세션 정보 + 설정 스냅샷 + 각도별 장수
    ├── events.csv                조작 기록 (모든 입력 이벤트)
    ├── marks.csv                 □ 버튼으로 표시한 불량 구간
    ├── 20261110_184301_250_a090_s045.jpg
    ├── 20261110_184301_750_a090_s045.jpg
    ├── 20261110_184302_250_a120_s045.jpg
    └── …                         사진 (녹화 ON + 전진 중에만 저장)
```

`dataset/` 은 git 에 올라가지 않는다 (.gitignore). 세션 폴더는 통째로 PC 로 복사한다.

### 6.2 사진 파일 = 학습 데이터 한 장

- 크기 640×480, JPEG 품질 95, 색은 카메라 원본(BGR 로 저장되는 보통의 JPEG).
- **파일 이름이 곧 라벨**이다. 별도의 라벨 파일이 필요 없고, 이름만 읽으면 그 사진을 찍을 때의 조향각과 속도를 안다.

```
20261110_184301_250_a090_s045.jpg
│        │      │   │    └── s045 : 그때의 모터 속도 45 % (30 / 45 / 60 중 하나)
│        │      │   └─────── a090 : 그때의 서보 각도 90° (30 / 60 / 90 / 120 / 150 중 하나, 90 = 직진)
│        │      └─────────── 250  : 밀리초
│        └────────────────── 184301 : 18시 43분 01초
└─────────────────────────── 20261110 : 2026년 11월 10일
```

- 저장 조건: **녹화 ON(○) 이고 전진 중(속도 > 0)** 일 때 0.5초마다 한 장 (`recording.interval_sec`, 주행 중 `[` `]` 로 조정).
  정지 중, 후진 중에는 저장하지 않는다. 그래서 세션 안의 모든 사진은 "앞으로 달리며 그 각도로 조향하던 순간"이다.
- 파일 이름을 손으로 바꾸면 라벨이 깨진다. 폴더 안에 다른 파일이나 하위 폴더를 만들지 않는다.

### 6.3 session_meta.json

세션이 끝날 때(그리고 중간에도 주기적으로) 쓰는 JSON. 주요 항목:

```json
{
  "schema_version": 1,
  "session": "20261110_1842",
  "started_at": "2026-11-10T18:42:37", "ended_at": "2026-11-10T18:47:30",
  "tool": {"name": "collector", "version": "0.1.0"},
  "host": {"node": "gsc-desktop", "platform": "Linux-5.15.148-tegra-aarch64", "python": "3.10.12"},
  "frames_saved": 312,
  "counts_by_angle": {"30": 12, "60": 48, "90": 180, "120": 55, "150": 17},
  "marks": 2,
  "input_device": "ds4",
  "camera": {"device": 0, "width": 640, "height": 480, "fps": 30.0, "fourcc": "MJPG"},
  "config": { …config.yaml 전체 스냅샷… }
}
```

`counts_by_angle` 로 각도별 균형을 바로 확인할 수 있다 (직진 90° 가 압도적으로 많으면 곡선 데이터를 더 찍어야 한다).

### 6.4 events.csv — 조작 기록

운전자가 한 모든 조작이 시각과 함께 남는다 (사진과 조작을 대조하거나, 주행을 되짚어 볼 때 사용).

```
t_iso,t_mono,kind,value
2026-11-10T18:42:40.620,3700.629,steer_axis,0.0
2026-11-10T18:42:41.102,3701.111,throttle_axis,0.62
2026-11-10T18:42:41.350,3701.359,record_toggle,
2026-11-10T18:43:05.004,3725.013,mark,
```

- `t_iso` 시각, `t_mono` 프로그램 기준 초, `kind` 이벤트 종류, `value` 값.
- 종류: `steer_axis`(스틱 −1~1), `steer_step`(D패드 ±1), `steer_center`, `throttle_axis`(R2 0~1), `reverse_axis`(L2 0~1),
  `record_toggle`, `mark`, `interval_step`, `quit`, `device_lost`, 키보드 모드의 `forward_toggle`/`reverse_toggle`/`stop`/`speed_step`.

### 6.5 marks.csv — 불량 구간

□ 버튼을 누를 때마다 "직전 3초"가 한 줄로 기록된다. 사진은 지우지 않는다. 가공 단계(curator)가 이 구간의 사진을 제외 후보로 미리 표시한다.

```
t_start_iso,t_end_iso,note
2026-11-10T18:43:02.004,2026-11-10T18:43:05.004,frame=20261110_184304_750_a060_s045.jpg
```

### 6.6 PC 로 가져오기

```bash
# PC 에서 (Jetson IP 가 192.168.50.119, 사용자 gsc 인 경우). 세션 폴더를 통째로
scp -r gsc@192.168.50.119:~/course-autodrive/dataset/20261110_1842 ./dataset/
```
USB 메모리로 복사해도 된다. 가공 도구(curator)는 세션 폴더(또는 그것을 압축한 zip)를 그대로 연다.

### 6.7 데이터셋 원칙 (이전 자료에서 유지)

각도별 최소 400장, 각도 간 균형(직진만 많이 찍히기 쉬움), 트랙 양 끝·중앙·완만한 곡선·급한 곡선을 골고루, 급조작은 피한다,
한 번에 끝내지 말고 여러 번 반복 주행, 파일명과 폴더 구조를 손으로 바꾸지 않는다, 실수한 구간은 그 자리에서 □ 로 표시한다.

## 6b. 가공 도구 curator (PC) 사용법

```bash
uv sync --extra gui                          # 처음 한 번 (PySide6)
uv run -m curator dataset/20261110_1906      # 세션 폴더(여러 개 가능) 또는 세션 zip
```

| 조작 | 동작 |
|---|---|
| ←/→, PageUp/PageDown, Home/End | 한 장 / 10장 / 처음·끝 이동. Space 자동 재생, 배속 1 · 2 · 5 · 10배 (버튼 또는 + / - 키, 1배 = 실제 주행 속도, 기본 2배) |
| 화면 | 사진 위에 조향 화살표, 오른쪽에 세션·선택·클래스 분포(세로 막대)·현재 프레임, 아래에 타임라인(세션 경계·제외·선택·현재 위치)과 필름스트립 |
| Shift+←/→, Shift+클릭 | 연속 선택 (탐색기와 같음). Ctrl+클릭은 떨어진 낱장 선택, Esc 해제 |
| Delete / Insert | 선택한 사진(없으면 현재 장)을 제외 / 복구. X 는 현재 장 토글 |
| Ctrl+Z | 되돌리기 |
| 데이터셋 만들기 (Ctrl+B) | 포함된 사진을 수집 순서로 번호를 매겨 `dataset_YYYYMMDD_HHMM.zip` 생성 |

- 수집 중 □ 로 표시한 구간(`marks.csv`)의 사진은 열 때 **미리 제외**되어 빨간색으로 보인다. 잘못됐으면 Insert 로 복구.
- 제외는 소프트다: 원본 사진은 그대로 두고 세션 폴더의 `curation.json` 에만 기록한다 (바뀔 때마다 자동 저장). 다시 열면 그대로 복원.
- 여러 세션을 함께 열어 한 zip 으로 합칠 수 있다.

## 7. 단계 사이 파일 계약 (curator·trainer·inference 작성 시 기준)

- curator → trainer: `dataset_YYYYMMDD_HHMM.zip` = `images/…_{index:06d}.jpg` + `labels.csv` (`index,image,servo_angle,motor_speed,source`) + `meta.json` (클래스, 이미지 크기, 세션 목록, 클래스별 수, 제외 수). 원본은 수정하지 않고 `curation.json` 에 제외 목록만 남긴다. 여러 세션을 한 zip 으로 합칠 수 있다.
- trainer → inference: `models/<run_id>/model.pth`, `model.onnx` (배치 1, 입력 `input`, 출력 `output`), `model_meta.json` (task, classes, 전처리 파라미터 — 상단 40 % 자르기, 200×66 리사이즈, RGB, 0~1, CHW —, 학습 정보). **추론기는 전처리를 model_meta.json 에서 읽는다.**
- inference: Jetson 에서 ONNX → TensorRT 10 엔진을 만들어 캐시(보드·버전마다 다시 생성), 예측 평활(EMA/다수결), 예측 CSV 기록, 패드로 수동 개입.

## 8. 설정 파일 config.yaml (학생이 바꿀 만한 값)

| 키 | 기본 | 뜻 |
|---|---|---|
| `steering.classes` | [30, 60, 90, 120, 150] | 조향 클래스(각도). 7단계 등으로 바꿀 수 있음 |
| `steering.invert` | false | 스틱 방향과 차가 반대면 true |
| `throttle.levels` | [30, 45, 60] | R2 속도 단계(%) |
| `throttle.reverse_max_speed` | 30 | 후진 속도 |
| `hardware.dir_forward_high` | true | 전진인데 뒤로 가면 false |
| `recording.interval_sec` | 0.5 | 저장 간격(초) |
| `recording.mark_seconds` | 3.0 | □ 로 표시할 직전 시간 |
| `display.rotate` | 90 | 도트 매트릭스 단 방향 |

## 9. 저장소 구조

```
config.yaml          모든 프로그램이 공유하는 설정 (단일 출처)
common/              설정 로더, 파일명 규약, 조향 클래스, 전처리, PCA9685 드라이버(common/hw), 입력 장치(common/inputs), 표시(common/display)
collector/           수집기 (상태 머신 collector/state.py 에 주행·녹화 규칙)
curator/             가공 도구 (core.py 로직 + app.py PySide6 화면)
scripts/             jetson_setup.sh(설치), pin_test.py(동작 시험), dotmatrix_test.py, make_wiring_doc.py(배선도 생성)
docs/                jetson-setup.md, wiring.pdf, system-brief.md(이 문서)
tests/               pytest 97개 (PC 에서 하드웨어 없이 실행 가능, `uv run pytest`)
```

## 10. 이전 자료와의 차이 (강의 자료에서 고쳐야 할 것)

| 이전 (2025, Jetson Nano) | 현재 |
|---|---|
| JetPack 4.6, Python 3.6, SD 이미지 직접 설치 | JetPack 6.2.1, Python 3.10, 골든 이미지 복제 |
| L298N, 헤더 PWM 직결, Jetson.GPIO, `sudo python3` | PCA9685 ×2 (I2C), MDD10A, sudo 불필요 |
| 키보드 조종 (`drive.py`) | DualShock 4 (키보드는 대체 수단) |
| `img-collector.py` → png + data_labels.csv | `uv run -m collector` → 세션 폴더, jpg(파일명 = 라벨), events.csv, marks.csv |
| `img-cleaner.py` 로 정제 | curator (PySide6, 예정) 로 검토·제외·zip |
| pip + venv, PyTorch 2.1 고정 | uv (`uv sync --extra pc`), PyTorch cu128 |
| PyCUDA 설치, TensorRT 7 API | TensorRT 10 API (cuda-python, JetPack 내장) |
| tar 로 전송 | scp 또는 USB |
| 상태 표시 없음 | 도트 매트릭스 |
