from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

FORWARD = "forward"
BACKWARD = "backward"


@dataclass
class VehicleState:
    angle: int
    speed: int = 0                      # 0~100, 현재 듀티
    direction: Optional[str] = None     # FORWARD | BACKWARD | None(정지)


class Vehicle(ABC):
    """서보 조향 + DC 모터 구동 추상화. 수집기와 추론기가 공유한다."""

    @abstractmethod
    def set_steering(self, angle: int) -> None: ...

    @abstractmethod
    def set_throttle(self, speed: int, direction: str) -> None:
        """speed 0~100 듀티, direction FORWARD|BACKWARD. speed 0 이면 정지."""

    @abstractmethod
    def stop(self, smooth: bool = True) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @property
    @abstractmethod
    def state(self) -> VehicleState: ...

    def service(self, now: Optional[float] = None) -> None:
        """주기 호출 훅 (제어 루프마다). 시간이 지나야 끝나는 동작(조향 유격 보정의 두 번째 단계)을 마무리한다."""

    def settle(self) -> None:
        """진행 중인 조향 동작이 끝날 때까지 기다린다 (점검 스크립트용, 블로킹)."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
