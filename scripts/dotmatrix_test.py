"""MAX7219 8×8 도트 매트릭스 시험 (Jetson SPI1 직결). 드라이버는 common/display/max7219.py 를 그대로 쓴다.

배선: 모듈 VCC → 핀 2 (5 V), GND → 핀 20, DIN → 핀 19, CS → 핀 24, CLK → 핀 23. SPI1 은 jetson_setup.sh install 이 켠다.

  uv run python scripts/dotmatrix_test.py              무늬 10개를 1초씩 (전체 켜기, 끄기, 화살표 4방향, 사각형, X, 테두리, 체크)
  uv run python scripts/dotmatrix_test.py status       수집기가 그리는 상태 화면을 차례로 (정지 → 전진 3단계 → 후진 → 조향 좌우 → 녹화 → 오류)
  uv run python scripts/dotmatrix_test.py stress [초]  무늬를 0.1초마다 계속 바꾼다 (기본 120초). 깨지는 점이 보이면 직결이 불안한 것
  uv run python scripts/dotmatrix_test.py off          화면 끄기
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_config                        # noqa: E402
from common.display.base import DisplayState                 # noqa: E402
from common.display.glyphs import CROSS                      # noqa: E402

PATTERNS = {
    "full": [0xFF] * 8, "blank": [0x00] * 8,
    "up": [0x18, 0x3C, 0x7E, 0xFF, 0x18, 0x18, 0x18, 0x18], "down": [0x18, 0x18, 0x18, 0x18, 0xFF, 0x7E, 0x3C, 0x18],
    "left": [0x10, 0x30, 0x7F, 0xFF, 0xFF, 0x7F, 0x30, 0x10], "right": [0x08, 0x0C, 0xFE, 0xFF, 0xFF, 0xFE, 0x0C, 0x08],
    "stop": [0x00, 0x7E, 0x7E, 0x7E, 0x7E, 0x7E, 0x7E, 0x00], "x": CROSS,
    "frame": [0xFF, 0x81, 0x81, 0x81, 0x81, 0x81, 0x81, 0xFF], "check": [0xAA, 0x55] * 4,
}


def main(argv) -> int:
    mode = argv[1] if len(argv) > 1 else "demo"
    cfg = load_config().display
    if not os.path.exists(cfg.spi_dev):
        print(f"{cfg.spi_dev} 가 없습니다. SPI 를 켜세요:  bash scripts/jetson_setup.sh install  → 재부팅")
        return 2
    from common.display.max7219 import Max7219Display
    d = Max7219Display(cfg.spi_dev, cfg.intensity, cfg.rotate, cfg.flip, cfg.blink_hz)
    chip = d.chip
    try:
        if mode == "off":
            print("끔"); return 0
        if mode == "status":
            lo, hi = 30, 150
            scenes = [("정지", DisplayState(0, None, 90, lo, hi)),
                      ("전진 30 %", DisplayState(30, "forward", 90, lo, hi)),
                      ("전진 45 %", DisplayState(45, "forward", 90, lo, hi)),
                      ("전진 60 %", DisplayState(60, "forward", 90, lo, hi)),
                      ("후진 30 %", DisplayState(30, "backward", 90, lo, hi)),
                      ("좌회전", DisplayState(45, "forward", 30, lo, hi)),
                      ("우회전", DisplayState(45, "forward", 150, lo, hi)),
                      ("녹화 중 (모서리 점멸)", DisplayState(45, "forward", 90, lo, hi, recording=True)),
                      ("오류 (X 점멸)", DisplayState(0, None, 90, lo, hi, error=True))]
            for name, st in scenes:
                print(f">>> {name}", flush=True)
                t0 = time.monotonic()
                while time.monotonic() - t0 < 2.0:
                    d.update(st, time.monotonic()); time.sleep(0.05)
            return 0
        if mode == "stress":
            seconds = float(argv[2]) if len(argv) > 2 else 120.0
            names = list(PATTERNS)
            print(f"{seconds:.0f}초 동안 무늬를 0.1초마다 바꿉니다. Ctrl+C 로 중단", flush=True)
            t0, n = time.monotonic(), 0
            while time.monotonic() - t0 < seconds:
                chip.show(PATTERNS[names[n % len(names)]]); n += 1
                time.sleep(0.1)
            print(f"무늬 {n}회 전송. 깨짐이 없었다면 직결 사용 가능")
            return 0
        for name, rows in PATTERNS.items():
            print(f">>> {name}", flush=True)
            chip.show(rows); time.sleep(1.0)
        chip.show(PATTERNS["stop"]); time.sleep(1.0)
        print("완료. 화살표 방향이 틀리면 config.yaml display.rotate / flip 을 조정한다")
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        d.close()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
