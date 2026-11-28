"""상태 표시 장치 인터페이스. 수집기와 추론기가 주기적으로 update() 를 부른다."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DisplayState:
    recording: bool = False     # 촬영(녹화) 중
    error: bool = False         # 패드 끊김·카메라 끊김 등으로 멈춘 상태
    flash: bool = False         # □ 불량 표시를 누른 직후 잠깐 True (화면 전체가 한 번 번쩍인다)
    auto: bool = False          # 자율주행 중 (추론기)
    busy: bool = False          # 준비 중: 모델을 여는 동안 (추론기. 처음에는 엔진을 만드느라 수십 초가 걸린다)


class Display:
    name = "none"

    def update(self, st: DisplayState, now: float) -> None:
        raise NotImplementedError

    def close(self, keep_error: bool = False) -> None:
        """keep_error 면 화면을 끄지 않고 X 를 켜 둔 채 닫는다 (왜 멈췄는지 프로그램이 끝난 뒤에도 보이게)."""


class NullDisplay(Display):
    """표시 장치가 없거나 꺼져 있을 때. 아무것도 하지 않는다."""

    def update(self, st: DisplayState, now: float) -> None:
        pass
