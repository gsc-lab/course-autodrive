"""시작 전 점검. 각 함수는 CheckResult 를 돌려주고 예외를 던지지 않는다."""
from __future__ import annotations

import glob
import importlib
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CheckResult:
    name: str
    ok: Optional[bool]   # True 통과 / False 실패 / None 판단 불가(또는 경고)
    message: str


def check_module(name: str, pip_name: Optional[str] = None) -> CheckResult:
    try:
        importlib.import_module(name)
        return CheckResult(f"module:{name}", True, "ok")
    except ImportError as e:
        pip = pip_name or name
        return CheckResult(f"module:{name}", False, f"{name} 을(를) import 할 수 없습니다 ({e}). uv add {pip}")
    except Exception as e:  # 일부 하드웨어 라이브러리는 보드를 못 알아보면 일반 Exception 을 던진다
        return CheckResult(f"module:{name}", False, f"{name} import 중 오류: {e}")


def check_i2c_bus(bus: int) -> CheckResult:
    """/dev/i2c-N 이 있고 현재 사용자가 읽고 쓸 수 있는지."""
    if not sys.platform.startswith("linux"):
        return CheckResult("i2c_bus", None, "리눅스가 아니므로 건너뜀")
    dev = f"/dev/i2c-{bus}"
    if not os.path.exists(dev):
        return CheckResult("i2c_bus", False, f"{dev} 가 없습니다. 핀 3/5 는 i2c-7, 핀 27/28 은 i2c-1 (hardware.pca9685.bus)")
    if not os.access(dev, os.R_OK | os.W_OK):
        user = os.environ.get("USER", str(os.getuid()))
        return CheckResult("i2c_bus", False, f"{dev} 에 접근 권한이 없습니다: sudo usermod -aG i2c {user} 후 재로그인")
    return CheckResult("i2c_bus", True, f"{dev} 접근 가능")


def check_pca9685(bus: int, address: int, role: str, hint: str = "") -> CheckResult:
    """PCA9685 보드 하나가 응답하는지 (MODE1 읽기). role 은 '서보' / '모터' 처럼 결과 이름에 붙는다."""
    name = f"pca9685:{role}"
    if not sys.platform.startswith("linux"):
        return CheckResult(name, None, "리눅스가 아니므로 건너뜀")
    dev = f"/dev/i2c-{bus}"
    if not os.path.exists(dev) or not os.access(dev, os.R_OK | os.W_OK):
        return CheckResult(name, False, f"{role}용 PCA9685 를 확인할 수 없습니다: {dev} 를 열 수 없음 (i2c_bus 점검 참조)")
    try:
        from common.hw.pca9685 import probe
        info = probe(bus, address)
    except OSError as e:
        return CheckResult(name, False,
                           f"{dev} 주소 0x{address:02x} 에서 {role}용 PCA9685 가 응답하지 않습니다 ({e}). "
                           f"`i2cdetect -y -r {bus}` 또는 `uv run python scripts/pin_test.py scan` 으로 확인. {hint}".rstrip())
    return CheckResult(name, True, f"{dev} 0x{address:02x} 응답 ({info})")


def check_disk(path: str | Path, min_free_gb: float) -> CheckResult:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(p).free / (1024 ** 3)
    ok = free_gb >= min_free_gb
    return CheckResult("disk", ok, f"여유 {free_gb:.1f} GB (최소 {min_free_gb} GB)")


def check_input_group(group: str = "input") -> CheckResult:
    """evdev 장치 읽기 권한: 현재 프로세스가 input 그룹을 실제로 가지고 있는지 (usermod 후 재로그인 필요)."""
    if not sys.platform.startswith("linux"):
        return CheckResult("input_group", None, "리눅스가 아니므로 건너뜀")
    if os.geteuid() == 0:
        return CheckResult("input_group", True, "root")
    try:
        import grp
        gid = grp.getgrnam(group).gr_gid
    except KeyError:
        return CheckResult("input_group", None, f"{group} 그룹이 없음")
    if gid in os.getgroups():
        return CheckResult("input_group", True, f"현재 프로세스가 {group} 그룹 보유")
    user = os.environ.get("USER", str(os.getuid()))
    return CheckResult("input_group", False,
                       f"현재 세션에 {group} 그룹이 없습니다: sudo usermod -aG {group} {user} 후 재로그인 (이미 했다면 로그아웃 후 다시 접속)")


def check_frame_size(shape: tuple, width: int, height: int) -> CheckResult:
    """카메라가 실제로 준 프레임 크기가 설정과 같은지. 데이터셋 계약(640×480)을 지키기 위해 필수."""
    h, w = int(shape[0]), int(shape[1])
    if (w, h) == (width, height):
        return CheckResult("camera_size", True, f"{w}x{h}")
    return CheckResult("camera_size", False,
                       f"카메라가 {w}x{h} 프레임을 줍니다 (설정 {width}x{height}). 카메라가 이 해상도/포맷(MJPG)을 지원하는지 확인하세요")


def find_ds4_lightbar_dirs() -> list:
    """hid-sony 가 만든 라이트바 LED 디렉토리 (…:red, …:green, …:blue 순). 세 색이 다 없으면 빈 목록."""
    out = []
    for color in ("red", "green", "blue"):
        found = sorted(glob.glob(f"/sys/class/leds/*054[cC]:09[cC][cC]*:{color}"))
        if not found:
            return []
        out.append(found[-1])  # 가장 최근 연결 장치
    return out
