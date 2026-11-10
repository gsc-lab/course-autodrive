"""PC 에서 로직만 돌려볼 때 쓰는 가짜 차량. 명령을 로그로 남긴다."""
from __future__ import annotations

import logging
import time

from common.config import HardwareConfig
from common.hw.base import BACKWARD, FORWARD, Vehicle, VehicleState

log = logging.getLogger(__name__)


class MockVehicle(Vehicle):
    def __init__(self, hw: HardwareConfig, initial_angle: int):
        self.hw = hw
        self._state = VehicleState(angle=initial_angle)
        self.calls: list = []  # 테스트 검증용
        log.info("MockVehicle 준비 (하드웨어 없음)")

    def set_steering(self, angle: int) -> None:
        self._state.angle = int(angle)
        self.calls.append(("steer", int(angle)))
        log.debug("servo -> %d", angle)

    def set_throttle(self, speed: int, direction: str) -> None:
        if direction not in (FORWARD, BACKWARD):
            raise ValueError(direction)
        speed = int(max(0, min(100, speed)))
        self._state.speed = speed
        self._state.direction = direction if speed > 0 else None
        self.calls.append(("throttle", speed, direction))
        log.debug("motor -> %s %d%%", direction, speed)

    def stop(self, smooth: bool = True) -> None:
        if smooth and self._state.speed > 0:
            time.sleep(min(0.05, self.hw.stop_decay_delay))
        self._state.speed = 0
        self._state.direction = None
        self.calls.append(("stop", smooth))
        log.debug("motor -> stop")

    def close(self) -> None:
        self.stop(smooth=False)
        self.calls.append(("close",))

    @property
    def state(self) -> VehicleState:
        return self._state
