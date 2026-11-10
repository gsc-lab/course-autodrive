"""8×8 상태 화면을 만드는 순수 함수 (하드웨어 없음, 테스트 가능).

화면은 행 8개의 정수 목록이고 bit7 이 왼쪽 열이다 (MAX7219 의 DIG0~7 레지스터에 그대로 쓴다).

  행 0~4  방향: ▲ 전진, ▼ 후진, ■ 정지. 녹화 중이면 위쪽 양 모서리 점이 깜빡인다
  행 5    비움
  행 6    조향: 각도 위치에 점 2개 (왼쪽 끝 = steering.min, 가운데 = 직진, 오른쪽 끝 = steering.max)
  행 7    속도: 왼쪽부터 채워지는 막대 (8칸 = 100 %)
  오류    X 가 깜빡인다
"""
from __future__ import annotations

from typing import List

from common.display.base import DisplayState

ARROW_UP = [0x18, 0x3C, 0x7E, 0x18, 0x18]
ARROW_DOWN = [0x18, 0x18, 0x7E, 0x3C, 0x18]
SQUARE = [0x00, 0x3C, 0x3C, 0x3C, 0x3C]
CROSS = [0x81, 0x42, 0x24, 0x18, 0x18, 0x24, 0x42, 0x81]
BLANK = [0x00] * 8


def steering_row(angle: int, lo: int, hi: int) -> int:
    """각도를 0~6 위치로 바꿔 2픽셀 점을 놓는다. 가운데 각도면 가운데(열 3~4)."""
    if hi <= lo:
        return 0x18
    pos = round((angle - lo) / (hi - lo) * 6)
    pos = max(0, min(6, pos))
    return (0xC0 >> pos) & 0xFF


def speed_row(speed: int) -> int:
    n = max(0, min(8, round(speed / 100.0 * 8)))
    return (0xFF << (8 - n)) & 0xFF if n else 0


def render(st: DisplayState, blink_on: bool) -> List[int]:
    if st.error:
        return list(CROSS) if blink_on else list(BLANK)
    if st.speed > 0 and st.direction == "forward":
        top = list(ARROW_UP)
    elif st.speed > 0 and st.direction == "backward":
        top = list(ARROW_DOWN)
    else:
        top = list(SQUARE)
    if st.recording and blink_on:
        top[0] |= 0x81
    return top + [0x00, steering_row(st.angle, st.angle_min, st.angle_max), speed_row(st.speed)]


def rotate(rows: List[int], degrees: int) -> List[int]:
    """시계 방향 회전 (0/90/180/270). 모듈을 어느 방향으로 달았는지에 맞춘다."""
    bits = [[(r >> (7 - c)) & 1 for c in range(8)] for r in rows]
    for _ in range((degrees // 90) % 4):
        bits = [[bits[7 - r][c] for r in range(8)] for c in range(8)]
    return [sum(b << (7 - c) for c, b in enumerate(row)) for row in bits]


def flip_horizontal(rows: List[int]) -> List[int]:
    return [int(f"{r:08b}"[::-1], 2) for r in rows]
