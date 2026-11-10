"""상태 표시 장치 인터페이스. 수집기(나중에는 추론기)가 주기적으로 update() 를 부른다."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class DisplayState:
    speed: int                  # 듀티 % (0 = 정지)
    direction: Optional[str]    # "forward" | "backward" | None
    angle: int                  # 현재 조향 각도
    angle_min: int              # steering.min
    angle_max: int              # steering.max
    recording: bool = False
    error: bool = False


class Display:
    name = "none"

    def update(self, st: DisplayState, now: float) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class NullDisplay(Display):
    """표시 장치가 없거나 꺼져 있을 때. 아무것도 하지 않는다."""

    def update(self, st: DisplayState, now: float) -> None:
        pass
