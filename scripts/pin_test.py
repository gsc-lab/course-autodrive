"""하드웨어 점검 도구 (패드·카메라 없이 PCA9685 두 장과 서보·모터만). 모터 시험은 바퀴를 띄운 상태에서 실행한다.

  uv run python scripts/pin_test.py scan           I2C 버스를 훑어 응답하는 주소를 보여 준다 (40 = 서보용, 41 = 모터용, 70 = 정상인 공통 주소)
  uv run python scripts/pin_test.py motor [초]     전진 30% N초 → 정지 → 후진 30% N초 → 정지 (기본 10초)
  uv run python scripts/pin_test.py servo          60° → 120° → 90° (각 1초)
  uv run python scripts/pin_test.py servo 110 90   적은 각도를 순서대로 (조향 한계·복귀 확인용)
                                                   --backlash=N 을 붙이면 유격 보정값을 N° 로 바꿔 시험 (0 = 끔)
  uv run python scripts/pin_test.py servo-all      서보용 보드 16채널 전부에 60° → 120° → 90° (서보를 몇 번 채널에 꽂았는지 모를 때)
  uv run python scripts/pin_test.py servo-hold [초] [각도]   서보 신호(기본 90°, 1.5 ms)를 N초 유지 — 멀티미터로 서보용 보드
                                                   채널 PWM 핀 ↔ GND 핀 DC 평균 약 0.25 V, 서보 커넥터 신호선에서도 같으면 신호 도달
  uv run python scripts/pin_test.py all            motor + servo

설정은 config.yaml 의 hardware 절을 따른다 (서보용 0x40 채널 0, 모터용 0x41 채널 0=속도·1=방향).
보드가 응답하지 않으면 `scan` 으로 먼저 어느 보드가 안 보이는지 가린다. 40 만 보이면 보드 사이 케이블이나 A0 납땜,
둘 다 안 보이면 Jetson 핀 1/3/5/6 케이블과 i2c 그룹 권한이다.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_config          # noqa: E402
from common.hw.base import BACKWARD, FORWARD   # noqa: E402


def banner(msg: str) -> None:
    print(f"\n>>> {msg}", flush=True)


def mode_scan(cfg) -> int:
    from common.hw.pca9685 import PCA9685_ALLCALL, scan
    hw = cfg.hardware
    expected = {hw.pca9685.address: "서보용 보드", hw.pca9685_motor.address: "모터용 보드", PCA9685_ALLCALL: "PCA9685 공통 주소 (정상)"}
    rc = 0
    for bus in dict.fromkeys((hw.pca9685.bus, hw.pca9685_motor.bus)):
        try:
            found = scan(bus)
        except OSError as e:
            print(f"/dev/i2c-{bus}: 열 수 없음 ({e}). Jetson 핀 3/5 가 i2c-7 이고, 사용자가 i2c 그룹인지 확인")
            return 2
        print(f"/dev/i2c-{bus}: " + (", ".join(f"0x{a:02x}" for a in found) if found else "응답하는 장치 없음"))
        for addr, role in expected.items():
            if addr == PCA9685_ALLCALL:
                continue
            if addr in found:
                print(f"  0x{addr:02x} {role}: 응답")
            else:
                rc = 1
                hint = ("서보용 보드도 없으므로 Jetson ↔ 서보용 보드 케이블(핀 1/3/5/6)을 본다"
                        if hw.pca9685.address not in found else
                        "서보용 보드는 보이므로 보드 사이 케이블(C1~C4)과 A0 납땜을 본다")
                print(f"  0x{addr:02x} {role}: 없음 → {hint}")
        others = [a for a in found if a not in expected]
        if others:
            print("  그 밖의 장치: " + ", ".join(f"0x{a:02x}" for a in others))
    return rc


def mode_motor(v, seconds: float = 10.0) -> None:
    banner(f"모터 전진 30% ({seconds:g}초)")
    v.set_throttle(30, FORWARD); time.sleep(seconds)
    banner("정지"); v.stop(smooth=True); time.sleep(1.0)
    banner(f"모터 후진 30% ({seconds:g}초)")
    v.set_throttle(30, BACKWARD); time.sleep(seconds)
    banner("정지"); v.stop(smooth=True)


def mode_servo(v, angles=(60, 120, 90)) -> None:
    for a in angles:
        banner(f"서보 {a}°")
        v.set_steering(a); v.settle(); time.sleep(1.0)
    time.sleep(2.0)                                 # 마지막 각도에서 멈춘 위치를 볼 시간


def mode_servo_all(v) -> None:
    """서보용 보드의 16채널 전부에 같은 펄스. 채널 번호를 모를 때 쓴다. 끝나면 설정된 채널만 남기고 나머지는 끈다."""
    chip = v.servo.chip
    for a in (60, 120, 90):
        banner(f"16채널 전부 서보 {a}°")
        chip.set_pulse_us(-1, v._angle_to_us(a)); time.sleep(1.0)
    for ch in range(16):
        if ch != v.servo.channel:
            chip.channel_off(ch)


def mode_servo_hold(v, seconds: float, angle: int) -> None:
    us = v._angle_to_us(angle)
    banner(f"서보 {angle}° 신호를 {seconds:.0f}초 유지 (50 Hz, 펄스 {us:.0f} µs). "
           f"보드 채널 PWM 핀 ↔ GND 핀 DC 평균 ≈ {3.3 * us / 20000:.2f} V")
    v.set_steering(angle)
    time.sleep(seconds)


def main(argv) -> int:
    backlash = None
    for a in list(argv[1:]):
        if a.startswith("--backlash="):
            backlash = float(a.split("=", 1)[1]); argv.remove(a)
    mode = argv[1] if len(argv) > 1 else "all"
    cfg = load_config()
    if mode == "scan":
        return mode_scan(cfg)
    if backlash is not None:
        cfg.hardware.servo.backlash_comp_deg = backlash
        print(f"[임시] 조향 유격 보정 {backlash:g}°")
    from common.hw.jetson import JetsonVehicle
    v = JetsonVehicle(cfg.hardware, 90)
    print(f"모터: {v.motor_desc()} | 서보: {v.servo_desc()}")
    try:
        if mode == "motor":
            mode_motor(v, float(argv[2]) if len(argv) > 2 else 10.0)
        elif mode == "servo":
            angles = [int(x) for x in argv[2:]]      # 각도를 적으면 그 순서대로 (예: servo 110 90)
            mode_servo(v, angles) if angles else mode_servo(v)
        elif mode == "servo-all":
            mode_servo_all(v)
        elif mode == "servo-hold":
            seconds = float(argv[2]) if len(argv) > 2 else 30.0
            mode_servo_hold(v, seconds, int(argv[3]) if len(argv) > 3 else 90)
        elif mode == "all":
            mode_motor(v); mode_servo(v)
        else:
            print(__doc__); return 2
    finally:
        v.close()
        print("\n완료 (모터 정지, 서보 중앙)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
