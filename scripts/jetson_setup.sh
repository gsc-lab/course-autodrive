#!/usr/bin/env bash
# =============================================================================
# Jetson Orin Nano (JetPack 6.x) 설정 스크립트 — 새 이미지에서도 같은 결과를 낸다 (멱등).
#
#   bash scripts/jetson_setup.sh install   # 패키지, 그룹/udev, uv + venv → 필요 시 재로그인 안내
#   bash scripts/jetson_setup.sh verify    # 전체 상태 점검 (읽기 전용): I2C 의 PCA9685 두 장, 권한, 패드, 카메라
#   bash scripts/jetson_setup.sh pair      # DualShock 4 블루투스 페어링 (패드를 페어링 모드로 두고 실행)
#
# 서보·모터 신호는 I2C(핀 3 SDA, 5 SCL = i2c-7)에 연결한 PCA9685 두 장(0x40 서보, 0x41 모터)이 낸다.
# 40핀 헤더의 PWM 핀먹스·GPIO 설정은 필요 없다 (배선: docs/wiring.pdf). 헤더에서 jetson-io 로 켜는 것은
# 상태 표시용 도트 매트릭스(MAX7219)가 쓰는 SPI1(핀 19/21/23/24/26) 하나뿐이다.
# 일반 사용자로 실행한다. 필요한 곳에서만 sudo 를 부른다 (비밀번호를 한 번 묻는다).
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
I2C_BUS=7                    # Jetson 핀 3/5 (config.yaml hardware.pca9685.bus 와 같아야 함)
ADDR_SERVO=40                # 서보용 PCA9685 (config.yaml hardware.pca9685.address)
ADDR_MOTOR=41                # 모터용 PCA9685, A0 납땜 (config.yaml hardware.pca9685_motor.address)
SPI_DEV=/dev/spidev0.0       # 도트 매트릭스 (config.yaml display.spi_dev). SPI1 CS0 = 헤더 핀 19/23/24
JIO=/opt/nvidia/jetson-io
SYS_PY=/usr/bin/python3      # JetPack 시스템 파이썬 (tensorrt 가 여기에 있다)
NEED_RELOGIN=0
NEED_REBOOT=0

spi1_enabled() {   # 부팅된 디바이스 트리에 jetson-io 의 SPI1 핀 설정이 들어 있는지
    [[ -d /proc/device-tree/bus@0/pinmux@2430000/exp-header-pinmux/hdr40-pin19 ]]
}

ok()   { printf '  \033[32m[OK]\033[0m   %s\n' "$*"; }
warn() { printf '  \033[33m[경고]\033[0m %s\n' "$*"; }
fail() { printf '  \033[31m[실패]\033[0m %s\n' "$*"; }
step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die()  { fail "$*"; exit 1; }

require_jetson() {
    [[ -f /etc/nv_tegra_release ]] || die "Jetson 이 아닙니다 (/etc/nv_tegra_release 없음)"
    local rel; rel=$(head -1 /etc/nv_tegra_release)
    echo "  $rel"
    if ! grep -q '^# R36' /etc/nv_tegra_release; then
        warn "이 스크립트는 JetPack 6.x (L4T R36) 기준으로 검증했습니다. 다른 버전은 I2C 장치 번호나 드라이버가 다를 수 있습니다."
    fi
}

i2c_has() {   # $1=주소(16진 두 자리, 예: 40) → i2cdetect 표에 그 주소가 보이면 0
    local addr=$1 row=${1:0:1}0
    i2cdetect -y -r "$I2C_BUS" 2>/dev/null | grep -E "^${row}:" | grep -qw "$addr"
}

# -----------------------------------------------------------------------------
cmd_install() {
    step "1/5 플랫폼 확인"; require_jetson

    step "2/5 시스템 패키지"
    sudo apt-get update -qq
    sudo apt-get install -y -qq git curl python3-dev build-essential bluez v4l-utils i2c-tools >/dev/null
    ok "git curl python3-dev build-essential bluez v4l-utils i2c-tools"

    # 재부팅 원인을 나중에 볼 수 있게 저널을 디스크에 남긴다 (기본은 메모리라 재부팅하면 지워진다. 2026-11-10 주행 중 보드가
    # 재시작했는데 이전 부팅 기록이 없어 원인을 못 찾았다)
    if [[ ! -d /var/log/journal ]]; then
        sudo mkdir -p /var/log/journal && sudo systemd-tmpfiles --create --prefix /var/log/journal >/dev/null 2>&1 || true
        sudo systemctl restart systemd-journald || true
        ok "systemd 저널 영구 저장 (/var/log/journal)"
    fi

    step "3/5 그룹과 udev 규칙"
    sudo groupadd -f gpio                      # /dev/spidev* 가 root:gpio 다
    local me; me=$(id -un)
    for g in i2c input bluetooth video gpio; do
        if id -nG "$me" | tr ' ' '\n' | grep -qx "$g"; then ok "$me ∈ $g"
        else sudo usermod -aG "$g" "$me"; warn "$me 를 $g 그룹에 추가했습니다 (재로그인 후 적용)"; NEED_RELOGIN=1; fi
    done
    # DS4 라이트바(hid-sony 계열 드라이버가 있을 때만 의미 있음). 없어도 무해.
    if [[ ! -f /etc/udev/rules.d/90-ds4-leds.rules ]]; then
        sudo tee /etc/udev/rules.d/90-ds4-leds.rules >/dev/null <<'EOF'
SUBSYSTEM=="leds", KERNEL=="*054C:09CC*", RUN+="/bin/chmod 0666 /sys/class/leds/%k/brightness"
EOF
        ok "udev: 90-ds4-leds.rules"
    fi
    # DS4 hidraw: 수집기가 HID 보고 흐름으로 패드 연결 생사를 감시한다 (끊기면 0.5초 안에 정지). 기본 권한은 root 전용.
    if [[ ! -f /etc/udev/rules.d/91-ds4-hidraw.rules ]]; then
        sudo tee /etc/udev/rules.d/91-ds4-hidraw.rules >/dev/null <<'EOF'
SUBSYSTEM=="hidraw", KERNELS=="*054C:09CC*", MODE="0660", GROUP="input"
EOF
        ok "udev: 91-ds4-hidraw.rules"
    fi

    # DualShock 4 페어링: bluez 5.56+ 기본값(ClassicBondedOnly=true)은 본딩 전 클래식 HID 연결을 거부해
    # DS4 가 붙지 못한다. 수업용 개인 이미지이므로 허용한다 (관리자 결정, 2026-11-04).
    if ! grep -qE '^\s*ClassicBondedOnly\s*=\s*false' /etc/bluetooth/input.conf 2>/dev/null; then
        if grep -qE '^\s*#?\s*ClassicBondedOnly\s*=' /etc/bluetooth/input.conf 2>/dev/null; then
            sudo sed -i -E 's/^\s*#?\s*ClassicBondedOnly\s*=.*/ClassicBondedOnly=false/' /etc/bluetooth/input.conf
        elif grep -q '^\[General\]' /etc/bluetooth/input.conf 2>/dev/null; then
            sudo sed -i 's/^\[General\]/[General]\nClassicBondedOnly=false/' /etc/bluetooth/input.conf
        else
            printf '[General]\nClassicBondedOnly=false\n' | sudo tee -a /etc/bluetooth/input.conf >/dev/null
        fi
        sudo systemctl restart bluetooth
        ok "bluetooth input.conf: ClassicBondedOnly=false 적용 (DS4 연결 허용)"
    else
        ok "bluetooth input.conf: ClassicBondedOnly=false"
    fi

    step "4/5 uv + 가상환경 + 의존성"
    if ! command -v uv >/dev/null 2>&1 && [[ ! -x "$HOME/.local/bin/uv" ]]; then
        curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
        ok "uv 설치"
    fi
    export PATH="$HOME/.local/bin:$PATH"
    cd "$REPO_DIR"
    if [[ ! -x .venv/bin/python ]]; then
        uv venv --system-site-packages --python "$SYS_PY" >/dev/null
        ok ".venv 생성 (--system-site-packages, $SYS_PY)"
    elif ! .venv/bin/python -c 'import sys; sys.exit(0 if any("dist-packages" in p for p in sys.path) else 1)'; then
        warn ".venv 가 시스템 패키지를 보지 못합니다 → 다시 만듭니다"
        rm -rf .venv; uv venv --system-site-packages --python "$SYS_PY" >/dev/null
    fi
    uv sync --group dev >/dev/null
    ok "uv sync (evdev 포함)"
    sudo udevadm control --reload-rules && sudo udevadm trigger || true

    step "5/5 도트 매트릭스용 SPI1 (헤더 핀 19/21/23/24/26)"
    if spi1_enabled; then
        ok "SPI1 이 이미 켜져 있음 ($SPI_DEV)"
    elif [[ ! -x "$JIO/config-by-function.py" ]]; then
        warn "jetson-io 도구가 없어 SPI 를 켜지 못했습니다. 도트 매트릭스를 쓰지 않으면 config.yaml display.enabled: false"
    elif sudo "$JIO/config-by-function.py" -o dt spi1 >/dev/null 2>&1; then
        ok "jetson-io: spi1 추가 → 재부팅 필요"; NEED_REBOOT=1
    else
        warn "jetson-io 로 spi1 을 켜지 못했습니다 (sudo $JIO/jetson-io.py 에서 수동으로). 도트 매트릭스 없이도 주행은 된다"
    fi

    step "완료"
    if [[ $NEED_REBOOT == 1 ]]; then
        echo "  SPI 설정이 바뀌어 재부팅이 필요합니다:  sudo reboot   → 다시 접속해서  bash scripts/jetson_setup.sh verify"
    elif [[ $NEED_RELOGIN == 1 ]]; then
        echo "  그룹이 바뀌었습니다. 로그아웃 후 다시 접속해서  bash scripts/jetson_setup.sh verify"
    else
        echo "  바로 사용 가능합니다. 확인:  bash scripts/jetson_setup.sh verify"
    fi
}

# -----------------------------------------------------------------------------
cmd_verify() {
    local bad=0
    step "플랫폼"; require_jetson
    echo "  $(uname -r) / $(python3 --version) / $(grep PRETTY_NAME /etc/os-release | cut -d'"' -f2)"
    nvpmodel -q 2>/dev/null | head -1 | sed 's/^/  /' || true

    step "I2C: PCA9685 두 장 (i2c-$I2C_BUS, 서보 0x$ADDR_SERVO · 모터 0x$ADDR_MOTOR)"
    if ! command -v i2cdetect >/dev/null; then
        fail "i2c-tools 없음 (install 먼저)"; bad=1
    elif [[ ! -e /dev/i2c-$I2C_BUS ]]; then
        fail "/dev/i2c-$I2C_BUS 없음. 핀 3/5 가 i2c-7 이다"; bad=1
    elif [[ ! -r /dev/i2c-$I2C_BUS || ! -w /dev/i2c-$I2C_BUS ]]; then
        fail "/dev/i2c-$I2C_BUS 접근 권한 없음 (i2c 그룹, 재로그인)"; bad=1
    else
        local servo_ok=0 motor_ok=0
        i2c_has "$ADDR_SERVO" && servo_ok=1
        i2c_has "$ADDR_MOTOR" && motor_ok=1
        if [[ $servo_ok == 1 ]]; then ok "0x$ADDR_SERVO 서보용 보드 응답"; else fail "0x$ADDR_SERVO 서보용 보드 없음 → Jetson 핀 1/3/5/6 ↔ 보드 VCC/SDA/SCL/GND 케이블"; bad=1; fi
        if [[ $motor_ok == 1 ]]; then ok "0x$ADDR_MOTOR 모터용 보드 응답"
        elif [[ $servo_ok == 1 ]]; then fail "0x$ADDR_MOTOR 모터용 보드 없음 → 보드 사이 케이블(C1~C4)과 A0 납땜 (docs/wiring.pdf 그림 4)"; bad=1
        else fail "0x$ADDR_MOTOR 모터용 보드 없음"; bad=1; fi
        i2c_has 70 && echo "  (0x70 은 모든 PCA9685 가 함께 듣는 주소라 보이는 것이 정상)"
    fi

    step "SPI: 도트 매트릭스 ($SPI_DEV)"
    if spi1_enabled; then ok "SPI1 핀 설정이 부팅에 적용됨"; else warn "SPI1 이 꺼져 있음 → install 후 재부팅 (도트 매트릭스를 안 쓰면 무시)"; fi
    if [[ -e $SPI_DEV ]]; then
        if [[ -r $SPI_DEV && -w $SPI_DEV ]]; then ok "$SPI_DEV 접근 가능"; else warn "$SPI_DEV 권한 없음 (gpio 그룹, 재로그인)"; fi
    else warn "$SPI_DEV 없음"; fi

    step "권한"
    local me; me=$(id -un)
    for g in i2c input gpio; do
        if id -nG "$me" | tr ' ' '\n' | grep -qx "$g"; then ok "$me ∈ $g"; else fail "$me ∉ $g (install 후 재로그인했는지 확인)"; bad=1; fi
    done

    step "파이썬 환경"
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv >/dev/null && ok "uv $(uv --version | cut -d' ' -f2)" || { fail "uv 없음"; bad=1; }
    cd "$REPO_DIR"
    if [[ -x .venv/bin/python ]]; then
        for m in tensorrt evdev cv2 numpy yaml; do
            if v=$(.venv/bin/python -c "import $m,sys; print(getattr($m,'__version__','ok'))" 2>/dev/null); then ok "$m $v"; else fail "import $m 실패"; bad=1; fi
        done
    else fail ".venv 없음 (install 먼저)"; bad=1; fi

    step "입력 장치 / HID 드라이버"
    if modinfo -n hid_sony >/dev/null 2>&1 || modinfo -n hid_playstation >/dev/null 2>&1; then
        ok "hid-sony/hid-playstation 있음 → layout=sony (라이트바·진동 가능)"
    else
        warn "hid-sony/hid-playstation 없음 → DS4 는 generic HID (스틱·버튼만, 라이트바·진동 없음). 코드가 자동 처리함"
    fi
    systemctl is-active --quiet bluetooth && ok "bluetooth 서비스 동작" || { fail "bluetooth 서비스 꺼짐"; bad=1; }
    if bluetoothctl devices 2>/dev/null | grep -qi 'wireless controller'; then
        ok "페어링된 DS4: $(bluetoothctl devices | grep -i 'wireless controller' | head -1)"
        if [[ -x .venv/bin/python ]] && .venv/bin/python - <<'EOF' 2>/dev/null
import evdev, sys
for p in evdev.list_devices():
    d = evdev.InputDevice(p)
    if d.info.vendor == 0x054C and d.info.product in (0x09CC, 0x05C4) and "Touchpad" not in d.name and "Motion" not in d.name:
        print(f"  evdev: {p} {d.name}"); sys.exit(0)
sys.exit(1)
EOF
        then ok "DS4 evdev 장치 보임 (연결됨)"; else warn "DS4 가 페어링됐지만 지금은 연결되어 있지 않음 (PS 버튼)"; fi
    else
        warn "페어링된 DS4 없음 →  bash scripts/jetson_setup.sh pair"
    fi

    step "카메라"
    if ls /dev/video0 >/dev/null 2>&1; then
        ok "/dev/video0 $(ls -l /dev/v4l/by-id/ 2>/dev/null | grep -o 'usb-[^ ]*index0' | head -1)"
        command -v v4l2-ctl >/dev/null && v4l2-ctl -d /dev/video0 --list-formats-ext 2>/dev/null | grep -A3 MJPG | grep -q '640x480' && ok "MJPG 640x480 지원" || warn "MJPG 640x480 확인 불가 (v4l2-ctl 없음 또는 미지원)"
    else fail "/dev/video0 없음 (USB 카메라 연결 확인)"; bad=1; fi

    step "디스크"
    df -h "$REPO_DIR" | tail -1 | awk '{print "  여유 " $4 " / " $2}'

    step "결과"
    if [[ $bad == 0 ]]; then
        echo "  모두 통과. 다음:  uv run python scripts/pin_test.py servo   (바퀴를 띄우고 motor 도)  →  uv run python -m collector --check"
    else
        echo "  실패 항목이 있습니다. 위 [실패] 줄을 해결한 뒤 다시 실행하세요."; return 1
    fi
}

# -----------------------------------------------------------------------------
cmd_pair() {
    step "DualShock 4 페어링"
    systemctl is-active --quiet bluetooth || sudo systemctl start bluetooth
    echo "  패드의 Share 와 PS 버튼을 라이트바가 빠르게 깜빡일 때까지 함께 누르세요 (약 5초). 60초 동안 찾습니다."
    bluetoothctl --timeout 2 power on >/dev/null 2>&1 || true
    bluetoothctl --timeout 2 pairable on >/dev/null 2>&1 || true
    bluetoothctl --timeout 2 agent NoInputNoOutput >/dev/null 2>&1 || true
    if ! grep -qE '^\s*ClassicBondedOnly\s*=\s*false' /etc/bluetooth/input.conf 2>/dev/null; then
        die "/etc/bluetooth/input.conf 에 ClassicBondedOnly=false 가 없습니다. 먼저  bash scripts/jetson_setup.sh install"
    fi
    local mac=""
    # 이미 페어링된 패드가 있으면 그대로 연결만 시도
    mac=$(bluetoothctl devices Paired 2>/dev/null | grep -i 'wireless controller' | head -1 | awk '{print $2}' || true)
    if [[ -n "$mac" ]]; then
        ok "이미 페어링된 패드: $mac (PS 버튼을 누르면 자동 연결됩니다)"
    else
        # 지난 실패로 남은 미페어링 기록은 지우고 새로 시작
        for old in $(bluetoothctl devices 2>/dev/null | grep -i 'wireless controller' | awk '{print $2}'); do
            bluetoothctl --timeout 3 remove "$old" >/dev/null 2>&1 || true
        done
        bluetoothctl --timeout 60 scan on >/dev/null 2>&1 &
        local scan_pid=$!
        for i in $(seq 1 60); do
            # set -e 때문에 grep 이 실패해도 대입문이 스크립트를 끝내지 않도록 || true
            mac=$(bluetoothctl devices 2>/dev/null | grep -i 'wireless controller' | head -1 | awk '{print $2}' || true)
            [[ -n "$mac" ]] && break
            printf '\r  탐색 중... %2ds' "$i"
            sleep 1
        done
        printf '\r%40s\r' ''
        kill "$scan_pid" 2>/dev/null || true
        [[ -n "$mac" ]] || die "'Wireless Controller' 를 찾지 못했습니다. 라이트바가 빠르게 깜빡이는 상태(페어링 모드)에서 다시 실행하세요."
        ok "발견: $mac"
        # DS4 는 페어링 모드에서 스스로 연결을 걸어온다. 먼저 trust 해서 그 연결을 받아들이고, 본딩(pair)은 되면 좋고 안 돼도 무방.
        bluetoothctl --timeout 5  trust "$mac" 2>&1 | tail -1 | sed 's/^/  /' || true
        bluetoothctl --timeout 20 pair "$mac"  2>&1 | grep -iE 'successful|fail|error' | tail -1 | sed 's/^/  /' || true
    fi
    for _ in $(seq 1 15); do
        bluetoothctl info "$mac" 2>/dev/null | grep -q 'Connected: yes' && break
        sleep 1
    done
    if ! bluetoothctl info "$mac" 2>/dev/null | grep -q 'Connected: yes'; then
        bluetoothctl --timeout 20 connect "$mac" 2>&1 | grep -iE 'successful|fail|error' | tail -1 | sed 's/^/  /' || true
        sleep 2
    fi
    if bluetoothctl info "$mac" 2>/dev/null | grep -q 'Connected: yes'; then
        ok "연결됨. 이후에는 PS 버튼만 누르면 자동 재연결됩니다."
        echo "  매핑 확인:  uv run python -m common.inputs.ds4"
    else
        warn "아직 연결되지 않았습니다. 패드의 PS 버튼을 한 번 누른 뒤 확인:  bluetoothctl info $mac"
    fi
}

# -----------------------------------------------------------------------------
cmd_prepare_image() {
    step "골든 이미지 준비 (SD 카드를 복제하기 직전에 실행)"
    cd "$REPO_DIR"
    rm -rf dataset models .pytest_cache; find . -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
    ok "수집 데이터·모델·캐시 삭제"
    for mac in $(bluetoothctl devices 2>/dev/null | grep -i 'wireless controller' | awk '{print $2}'); do
        bluetoothctl --timeout 3 remove "$mac" >/dev/null 2>&1 || true
    done
    ok "패드 페어링 기록 삭제 (각 팀이 자기 패드를 pair 한다)"
    sudo apt-get clean >/dev/null 2>&1 || true
    history -c 2>/dev/null || true; : > "$HOME/.bash_history"
    sudo truncate -s 0 /etc/machine-id            # 첫 부팅 때 새로 생성 → 보드마다 다른 ID (DHCP 충돌 방지)
    ok "apt 캐시·셸 기록 정리, machine-id 초기화"
    echo "  이제 전원을 끄고 SD 카드를 PC 에서 이미지로 읽어 복제하세요 (docs/jetson-setup.md '학생 배포')."
    echo "  sudo poweroff"
}

case "${1:-}" in
    install)       cmd_install ;;
    verify)        cmd_verify ;;
    pair)          cmd_pair ;;
    prepare-image) cmd_prepare_image ;;
    *) echo "사용법: bash scripts/jetson_setup.sh install | verify | pair | prepare-image"; exit 2 ;;
esac
