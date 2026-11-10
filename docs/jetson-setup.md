# Jetson Orin Nano 설정 (JetPack 6.2.x)

새 보드나 새로 구운 이미지에서 수집기·추론기를 돌릴 수 있게 만드는 절차다.
**스크립트 하나로 끝나고, 같은 스크립트를 다시 실행해도 안전하다(멱등).** 수동 절차는 스크립트가 실패했을 때만 본다.

검증한 환경: JetPack 6.2.1 (L4T R36.4.7), Ubuntu 22.04, Python 3.10, TensorRT 10.3, Orin Nano Developer Kit Super.

서보와 모터 신호는 모두 **I2C 에 연결한 PCA9685 두 장**이 낸다 (서보용 0x40, 모터용 0x41). Jetson 40핀 헤더에서는 I2C(핀 3, 5),
3.3 V(핀 1), GND(핀 6) 네 핀을 쓰고, 선택 부품인 상태 표시 도트 매트릭스가 SPI1(핀 19/23/24)과 5 V(핀 2), GND(핀 20)을 쓴다.
배선은 `docs/wiring.pdf`.

## 빠른 절차

```bash
# 0. (JetPack 6.2.1 SD 이미지로 부팅, 네트워크 연결, 터미널 또는 SSH)
git clone https://github.com/gsc-lab/course-autodrive.git
cd course-autodrive
bash scripts/jetson_setup.sh install    # 패키지, 그룹/udev, uv+venv. sudo 암호 1회. "로그아웃 후 다시 접속" 이 나오면 재로그인
bash scripts/jetson_setup.sh verify     # 전체 점검 (읽기 전용): I2C 의 PCA9685 두 장, 권한, 패드, 카메라
bash scripts/jetson_setup.sh pair       # DualShock 4 페어링 (Share+PS 를 눌러 라이트바가 빠르게 깜빡일 때 실행)
uv run python -m common.inputs.ds4      # 패드 매핑 확인 (스틱·버튼을 움직이면 코드와 해석 결과가 보임)
uv run python scripts/pin_test.py servo # 서보 60° → 120° → 90°.  바퀴를 띄우고  pin_test.py motor  도
uv run python -m collector --check      # 카메라·보드·입력 초기화 점검. 서보가 중앙으로 한 번 움직이면 정상
uv run python -m collector              # 수집 시작
```

코드가 바뀌면 `git pull && uv sync` 만 하면 된다.

## 학생 배포: 골든 이미지 복제 (권장)

학생이 설치 과정을 거치지 않게 하려면 **교수가 보드 한 대를 완성한 뒤 SD 카드를 통째로 복제**한다.
이미지는 모두 학생 개인용이라 계정·비밀번호·설정이 같아도 문제없다.

1. 기준 보드에서 위 "빠른 절차" 를 끝까지 수행하고 `uv run python -m collector --check` 까지 통과시킨다.
2. 복제 직전에 정리한다 (수집 데이터, 패드 페어링 기록, machine-id 초기화):
   ```bash
   bash scripts/jetson_setup.sh prepare-image
   sudo poweroff
   ```
3. SD 카드를 PC 에 꽂고 이미지로 읽는다.
   - Windows: Win32 Disk Imager → Read (또는 balenaEtcher 의 "Clone drive")
   - Linux/macOS: `sudo dd if=/dev/sdX of=autodrive-golden.img bs=4M status=progress`
   - 카드 전체를 읽으므로 용량이 큰 카드(이 보드는 234 GB)는 그만큼 이미지가 커진다. **골든 카드는 64 GB 로 만들고 학생 카드는 같은 용량 이상**을 쓰면 편하다.
4. 학생 카드에 같은 이미지를 쓴다 (balenaEtcher / Win32 Disk Imager Write / dd).
5. 학생은 보드를 켜고 자기 패드만 페어링한다:
   ```bash
   cd ~/course-autodrive && bash scripts/jetson_setup.sh pair
   ```
   코드 갱신은 `git pull && uv sync`.

주의: 복제한 카드가 골든 카드보다 크면 남는 공간은 쓰이지 않는다(첫 부팅 자동 확장은 이미 끝난 상태). 필요하면 `sudo parted /dev/mmcblk0 resizepart 1 100%` 후 `sudo resize2fs /dev/mmcblk0p1`.

## 스크립트가 하는 일

| 단계 | 내용 | 비고 |
|---|---|---|
| 패키지 | git curl python3-dev build-essential bluez v4l-utils i2c-tools | evdev 가 소스에서 빌드되므로 python3-dev 필요. i2c-tools 는 `i2cdetect` |
| 그룹 | i2c, input, bluetooth, video, gpio 에 사용자 추가 | 새로 추가된 그룹은 재로그인 후 적용. i2c 는 `/dev/i2c-7`, gpio 는 `/dev/spidev0.0`(도트 매트릭스) 때문 |
| SPI1 | `config-by-function.py -o dt spi1` (도트 매트릭스용, 헤더 핀 19/21/23/24/26) | 바뀌면 재부팅 필요. jetson-io 를 쓰는 유일한 항목 |
| bluez | `/etc/bluetooth/input.conf` 에 `ClassicBondedOnly=false` | bluez 5.56+ 기본값은 본딩 전 DS4 의 연결을 거부해 페어링이 실패한다(로그: `Access denied: org.bluez.Error.Rejected`). 수업용 개인 이미지이므로 허용 (2026-11-04 결정) |
| udev | DS4 라이트바 90-ds4-leds.rules | sony 계열 HID 드라이버가 있을 때만 의미 있음 |
| uv | 설치 후 `uv venv --system-site-packages --python /usr/bin/python3` → `uv sync` | JetPack 의 tensorrt·cuda 가 venv 안에서 보이게. evdev 는 기본 의존성(리눅스 마커)이라 `uv run` 만 해도 유지된다. torch/PySide6 는 받지 않음 |

`verify` 는 위 항목에 더해 `i2cdetect -y -r 7` 로 0x40(서보용)과 0x41(모터용)이 모두 응답하는지 본다. 0x70 이 함께 보이는 것은
모든 PCA9685 가 듣는 공통 주소라 정상이다.

## I2C 연결 확인

| 증상 | 원인 | 확인 |
|---|---|---|
| `/dev/i2c-7` 이 없다 | 버스 번호가 다르다 | 핀 3/5 는 i2c-7, 핀 27/28 은 i2c-1. `config.yaml` 의 `hardware.pca9685.bus` |
| 권한 오류 | i2c 그룹이 없다 | `sudo usermod -aG i2c $USER` 후 재로그인 |
| 0x40, 0x41 둘 다 없다 | Jetson ↔ 서보용 보드 케이블(I1~I4) 또는 보드 VCC 3.3 V | 핀 1 ↔ VCC, 3 ↔ SDA, 5 ↔ SCL, 6 ↔ GND |
| 0x40 만 있다 | 보드 사이 케이블(C1~C4) 또는 A0 납땜 | `docs/wiring.pdf` 그림 4 |
| 둘 다 응답하는데 서보가 안 움직인다 | 서보 전원(UBEC 6 V)이 단자대에 없다 | 채널 헤더 V+ ↔ GND 가 6 V 인지 잰다 |

```bash
i2cdetect -y -r 7                              # 40 과 41 이 보여야 한다
uv run python scripts/pin_test.py scan         # 같은 내용을 해석해서 보여 준다 (어느 보드가 없는지, 무엇을 볼지)
uv run python -m collector --check             # 시작 점검: i2c_bus, pca9685:서보, pca9685:모터 가 모두 OK 여야 한다
```

PCA9685 는 통신이 끊겨도 마지막 출력을 유지한다. 주행 중 I2C 가 끊기면 모터는 그 속도로 계속 돌고, 세울 방법은 배터리 커넥터를
뽑는 것뿐이다 (차에 주전원 스위치가 없다). 첫 시험은 반드시 바퀴를 띄우고 한다.

## 상태 표시 도트 매트릭스 (선택)

MAX7219 8×8 모듈을 Jetson 헤더에 직결한다 (추가 부품 없음): VCC → 핀 2 (5 V), GND → 핀 20, DIN → 핀 19, CS → 핀 24, CLK → 핀 23.
`install` 이 jetson-io 로 SPI1 을 켜고(재부팅 필요) `/dev/spidev0.0` 이 생긴다. 사용자는 `gpio` 그룹이어야 한다.
Jetson 의 3.3 V 신호는 MAX7219 규격(HIGH 3.5 V)보다 조금 낮지만 2026-11-10 실기에서 10분 연속 전송에 깨짐이 없어 직결로 쓴다.

| 화면 | 뜻 |
|---|---|
| 위 5행 ▲ / ▼ / ■ | 전진 / 후진 / 정지. 녹화 중이면 위 양 모서리 점이 깜빡인다 |
| 7행 점 2개 | 조향 위치 (왼쪽 끝 = `steering.min`, 가운데 = 직진) |
| 8행 막대 | 속도 (8칸 = 100 %) |
| X 점멸 | 오류 (카메라 끊김 등) |

```bash
uv run python scripts/dotmatrix_test.py          # 무늬 10개
uv run python scripts/dotmatrix_test.py status   # 수집기가 그리는 상태 화면 9가지
```
화살표가 돌아가 보이면 `config.yaml` 의 `display.rotate`(90/180/270) 또는 `flip` 을 바꾼다. 모듈이 없으면 수집기가 경고만 남기고
표시 없이 주행한다 (`display.enabled: false` 로 끌 수도 있다).

## 수동 절차 (스크립트가 실패했을 때)

### 그룹
```bash
sudo usermod -aG i2c,input,bluetooth,video $USER    # 재로그인 필요
```

### DualShock 4 (CUH-ZCT2G) 페어링
```bash
bluetoothctl
  power on
  agent on
  scan on            # 패드의 Share + PS 를 라이트바가 빠르게 깜빡일 때까지 길게
  pair XX:XX:XX:XX:XX:XX
  trust XX:XX:XX:XX:XX:XX
  connect XX:XX:XX:XX:XX:XX
  quit
```
이후에는 PS 버튼만 누르면 자동 재연결된다. `pair` 가 "Failed to pair" 로 끝나도 `trust` 만 되어 있으면 패드가 스스로
걸어오는 연결을 받아 동작한다(`ClassicBondedOnly=false` 가 전제). 반대로 `[NEW] Device` 가 60초 넘게 안 뜨면 패드가
페어링 모드가 아니다: 라이트바가 빠르게 두 번씩 깜빡여야 하고, 느리게 깜빡이면 PS 를 10초 눌러 끈 뒤 Share+PS 를 다시 누른다.

**JetPack 6.2.1 커널에는 hid-sony / hid-playstation 드라이버가 없다.** 패드는 generic HID 로 붙어 스틱·버튼·트리거는 동작하지만 **라이트바와 진동은 쓸 수 없다.** 코드는 드라이버를 감지해 자동으로 generic 레이아웃을 쓰고, 상태 표시는 터미널 상태줄로 한다. 매핑이 다르면 `uv run python -m common.inputs.ds4` 로 코드를 확인하고 `config.yaml` 의 `input.ds4.codes` 에 적는다.

## 주행 중 SSH 가 끊겨도 계속 달린다

패드로 조종할 때 수집기는 SSH 가 닫혀도(노트북 뚜껑, Wi-Fi 끊김) 멈추지 않고 계속 달린다. 조종은 블루투스로 Jetson 에 직접
들어오고 상태는 도트 매트릭스로 보기 때문이다. 끝내는 방법은 패드의 **Options 버튼**, 또는 다시 접속해서 `kill <PID>`
(PID 는 시작할 때 화면에 찍힌다). 키보드 모드는 입력이 SSH 로 오므로 SSH 가 닫히면 정지한다.

수집기는 한 번에 하나만 돈다. 이미 돌고 있는데 또 실행하면 시작하지 않고 끝내는 명령을 알려 준다:
```
[시작 불가] 수집기가 이미 실행 중입니다: PID 9123 (2026-11-10 18:02:11 시작).
먼저 그 수집기를 끝내세요. 패드의 Options 버튼을 누르거나, 이 터미널에서:
    kill 9123
```

## 수집한 데이터를 PC 로 가져오기

`dataset/` 은 git 에 올리지 않는다 (.gitignore). 세션 폴더를 통째로 PC 로 옮긴다:

```bash
# PC 에서 (Jetson IP 가 192.168.50.119, 사용자 gsc 인 경우)
scp -r gsc@192.168.50.119:~/course-autodrive/dataset/20261104_1430 ./dataset/
```

USB 메모리로 복사해도 된다. 가공 도구(curator)는 세션 폴더나 그것을 압축한 zip 을 그대로 연다.

## 조립 후 동작 시험

`docs/wiring.pdf` 4절을 따른다. 바퀴를 띄우고 `pin_test.py scan` → `servo` → `motor` → `collector --check` 순서로 돌려
두 보드 응답, 서보 움직임, 두 바퀴가 같은 방향으로 10초씩 전진·후진하는지 본다.

## JetPack 7.x 를 쓰려면

2026-11 기준 권장하지 않는다. 7.2.x 는 Python 3.12, TensorRT 10.16 으로 바뀌고 SD 카드 이미지가 없다. 전환할 때는 `.python-version` 을 3.12 로 바꾸고 `uv lock` 을 다시 만들어야 한다.
