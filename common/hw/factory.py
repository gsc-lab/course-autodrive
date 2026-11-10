from __future__ import annotations

from common.config import HardwareConfig
from common.hw.base import Vehicle


def create_vehicle(hw: HardwareConfig, initial_angle: int, angle_limits=None) -> Vehicle:
    if hw.backend == "mock":
        from common.hw.mock import MockVehicle
        return MockVehicle(hw, initial_angle)
    if hw.backend == "jetson":
        from common.hw.jetson import JetsonVehicle
        return JetsonVehicle(hw, initial_angle, angle_limits)
    raise ValueError(f"알 수 없는 hardware.backend: {hw.backend}")
