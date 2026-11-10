"""조향 기구 한계 찾기. 서보를 조금씩 움직이며 좌우 한계와 정면 각도를 표시하면 config.yaml 에 넣을 값을 알려 준다.

  uv run python scripts/servo_calibrate.py        (Jetson 에서, SSH 터미널로 직접 실행)

왜 필요한가: 서보는 0~180° 를 돌 수 있지만 조향 링크는 그보다 훨씬 좁은 범위에서 걸린다. 한계를 넘는 각도를 주면
서보가 걸린 채 힘을 주다가(스톨) 돌아오지 못하거나 과열된다. 차마다 조립 상태가 달라 한 번은 재야 한다.

키:  ← →  1° 씩      a d  5° 씩      c  90° 로
     [    지금 각도를 "낮은 쪽 한계" 로 표시        ]  "높은 쪽 한계" 로 표시
     m    지금 각도를 "정면(직진)" 으로 표시          q / Enter  끝내고 결과 출력

요령: 한쪽으로 5° 씩 가다가 서보에서 웅 소리가 나거나 바퀴가 더 안 꺾이면, 소리가 안 나는 곳까지 1° 씩 되돌린 뒤 표시한다.
한계에 오래 대고 있지 않는다.
"""
from __future__ import annotations

import select
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import load_config          # noqa: E402
from common.hw.jetson import JetsonVehicle      # noqa: E402
from common.steering import recommend_steering  # noqa: E402

ESC = chr(27)
STEPS = {"LEFT": -1, "RIGHT": 1, "a": -5, "A": -5, "d": 5, "D": 5}


def read_key(timeout: float = 0.2) -> str:
    """cbreak 모드 터미널에서 키 하나. 화살표는 'LEFT' / 'RIGHT' 로 돌려준다."""
    if not select.select([sys.stdin], [], [], timeout)[0]:
        return ""
    ch = sys.stdin.read(1)
    if ch == ESC:
        if select.select([sys.stdin], [], [], 0.02)[0] and sys.stdin.read(1) == "[":
            return {"D": "LEFT", "C": "RIGHT"}.get(sys.stdin.read(1), "")
        return "ESC"
    return ch


def show(angle: int, low, high, straight) -> None:
    def f(v):
        return "--" if v is None else str(v)
    sys.stdout.write(chr(13) + f"각도 {angle:3d}°   낮은 쪽 한계 [ {f(low)}   정면 m {f(straight)}   높은 쪽 한계 ] {f(high)}     ")
    sys.stdout.flush()


def main() -> int:
    import termios
    import tty
    cfg = load_config()
    sv = cfg.hardware.servo
    sv.backlash_comp_deg = 0          # 한계를 잴 때는 유격 보정을 끈다 (지나치는 동작이 한계를 넘을 수 있다)
    v = JetsonVehicle(cfg.hardware, 90)
    angle, low, high, straight = 90, None, None, None
    print("키:" + __doc__.split("키:")[1])
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            show(angle, low, high, straight)
            k = read_key()
            if not k:
                continue
            if k in STEPS:
                angle = max(sv.min_deg, min(sv.max_deg, angle + STEPS[k]))
                v.set_steering(angle)
            elif k in ("c", "C"):
                angle = 90
                v.set_steering(angle)
            elif k == "[":
                low = angle
            elif k == "]":
                high = angle
            elif k in ("m", "M"):
                straight = angle
            elif k in ("q", "Q", chr(10), chr(13), "ESC"):
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        try:
            v.set_steering(straight if straight is not None else 90)
            time.sleep(0.4)
        finally:
            v.close()
    print()
    print()
    if low is None or high is None:
        print("낮은 쪽([)과 높은 쪽(]) 한계를 둘 다 표시해야 추천값을 계산할 수 있습니다.")
        return 1
    try:
        r = recommend_steering(low, high, straight)
    except ValueError as e:
        print(f"추천값을 만들 수 없습니다: {e}")
        return 1
    print(f"실측: 낮은 쪽 한계 {low}°, 높은 쪽 한계 {high}°, 정면 {'표시 안 함(가운데 사용)' if straight is None else straight}")
    print("config.yaml 의 steering 에 넣을 값 (좁은 쪽에 맞춘 대칭 범위, 한계에서 3° 안쪽):")
    print()
    print(f"  classes: {r['classes']}")
    print(f"  center: {r['center']}")
    print(f"  min: {r['min']}")
    print(f"  max: {r['max']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
