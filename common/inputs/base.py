from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from common.events import InputEvent

STATUS_IDLE = "idle"
STATUS_RECORDING = "recording"
STATUS_ERROR = "error"


class InputDevice(ABC):
    name: str = "input"
    requires_deadman: bool = False   # True 면 DEADMAN(True) 가 유지되는 동안만 주행

    @abstractmethod
    def poll(self) -> List[InputEvent]:
        """쌓인 입력을 이벤트 목록으로. 블로킹하지 않는다."""

    def set_status(self, status: str) -> None:
        """idle | recording | error — 라이트바 등 표시 장치가 있으면 반영."""

    def notify(self, kind: str) -> None:
        """record_on | record_off | mark — 진동 등 촉각 알림."""

    def battery(self) -> int | None:
        return None

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
