"""8×8 상태 화면을 만드는 순수 함수 (하드웨어 없음, 테스트 가능).

화면은 행 8개의 정수 목록이고 bit7 이 왼쪽 열이다 (MAX7219 의 DIG0~7 레지스터에 그대로 쓴다).
운전 중에는 화면을 읽을 수 없으므로 화면 전체의 모양과 움직임만으로 상태를 알린다.

  꺼짐              프로그램이 돌고 있지 않다
  ○ 고정            패드 연결됨, 대기 (수집기: 촬영 안 함. 추론기: 수동 운전)
  ● ↔ ○ 번갈아      촬영 중 (display.blink_hz). 테두리는 계속 켜져 있어 화면이 꺼지는 순간이 없다
  막대가 돈다        자율주행 중 (추론기). 깜빡이는 화면들과 달리 밝기가 일정하고 모양이 돈다
  가운데 점 고정     준비 중 (추론기가 모델을 여는 동안. 아직 패드를 받지 않는다)
  전체 켜짐 한 번    □ 를 눌렀다
  X 빠른 깜빡임      패드 끊김·오류. 이 이유로 끝나면 X 를 켜 둔 채 종료한다

모든 화면은 90° 돌려도 같은 화면(또는 같은 움직임)이다: 모듈을 어느 방향으로 달았든 똑같이 보인다.
"""
from __future__ import annotations

from typing import List

from common.display.base import DisplayState

RING = [0x3C, 0x42, 0x81, 0x81, 0x81, 0x81, 0x42, 0x3C]
DISC = [0x3C, 0x7E, 0xFF, 0xFF, 0xFF, 0xFF, 0x7E, 0x3C]
CROSS = [0x81, 0x42, 0x24, 0x18, 0x18, 0x24, 0x42, 0x81]
FULL = [0xFF] * 8
BLANK = [0x00] * 8
DOT = [0x00, 0x00, 0x00, 0x18, 0x18, 0x00, 0x00, 0x00]
# 가운데를 지나는 두꺼운 막대가 45° 씩 돈다 (가로, 대각선, 세로, 반대 대각선). 네 장을 90° 돌리면 다시 이 네 장이므로
# 모듈을 다는 방향과 상관없다
SPIN = [
    [0x00, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0x00],
    [0xC0, 0xE0, 0x70, 0x38, 0x1C, 0x0E, 0x07, 0x03],
    [0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18],
    [0x03, 0x07, 0x0E, 0x1C, 0x38, 0x70, 0xE0, 0xC0],
]
ERROR_BLINK_HZ = 4.0        # X 는 촬영 표시보다 빠르게 깜빡여 속도만으로도 구분된다
SPIN_FPS = 6.0              # 자율주행 막대: 1초에 여섯 장 (반 바퀴에 0.67초)


def spin_phase(now: float) -> int:
    """지금 보여 줄 막대 그림의 번호 (0~3)."""
    return int(now * SPIN_FPS) % len(SPIN)


def render(st: DisplayState, blink_on: bool, fast_on: bool = True, phase: int = 0) -> List[int]:
    """blink_on = 촬영 깜빡임의 켜진 단계, fast_on = 오류 깜빡임의 켜진 단계, phase = 자율주행 막대의 그림 번호."""
    if st.error:
        return list(CROSS) if fast_on else list(BLANK)
    if st.flash:
        return list(FULL)
    if st.busy:
        return list(DOT)
    if st.auto:
        return list(SPIN[phase % len(SPIN)])
    if st.recording and blink_on:
        return list(DISC)
    return list(RING)


def rotate(rows: List[int], degrees: int) -> List[int]:
    """시계 방향 회전 (0/90/180/270). 모듈을 어느 방향으로 달았는지에 맞춘다."""
    bits = [[(r >> (7 - c)) & 1 for c in range(8)] for r in rows]
    for _ in range((degrees // 90) % 4):
        bits = [[bits[7 - r][c] for r in range(8)] for c in range(8)]
    return [sum(b << (7 - c) for c, b in enumerate(row)) for row in bits]


def flip_horizontal(rows: List[int]) -> List[int]:
    return [int(f"{r:08b}"[::-1], 2) for r in rows]
