"""8×8 상태 화면을 만드는 순수 함수 (하드웨어 없음, 테스트 가능).

화면은 행 8개의 정수 목록이고 bit7 이 왼쪽 열이다 (MAX7219 의 DIG0~7 레지스터에 그대로 쓴다).
운전 중에는 화면을 읽을 수 없으므로 화면 전체의 모양과 깜빡임만으로 두 가지를 알린다: 패드 연결, 촬영 중.

  꺼짐              프로그램이 돌고 있지 않다
  ○ 고정            패드 연결됨, 촬영 안 함
  ● ↔ ○ 번갈아      촬영 중 (display.blink_hz). 테두리는 계속 켜져 있어 화면이 꺼지는 순간이 없다
  전체 켜짐 한 번    □ 불량 표시를 눌렀다
  X 빠른 깜빡임      패드 끊김·오류. 이 이유로 끝나면 X 를 켜 둔 채 종료한다
"""
from __future__ import annotations

from typing import List

from common.display.base import DisplayState

RING = [0x3C, 0x42, 0x81, 0x81, 0x81, 0x81, 0x42, 0x3C]
DISC = [0x3C, 0x7E, 0xFF, 0xFF, 0xFF, 0xFF, 0x7E, 0x3C]
CROSS = [0x81, 0x42, 0x24, 0x18, 0x18, 0x24, 0x42, 0x81]
FULL = [0xFF] * 8
BLANK = [0x00] * 8
ERROR_BLINK_HZ = 4.0        # X 는 촬영 표시보다 빠르게 깜빡여 속도만으로도 구분된다


def render(st: DisplayState, blink_on: bool, fast_on: bool = True) -> List[int]:
    """blink_on = 촬영 깜빡임의 켜진 단계, fast_on = 오류 깜빡임의 켜진 단계."""
    if st.error:
        return list(CROSS) if fast_on else list(BLANK)
    if st.flash:
        return list(FULL)
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
