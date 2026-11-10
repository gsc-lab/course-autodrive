"""입력 장치(DualShock 4, 키보드)가 만들어 내는 공통 명령 이벤트.

주행 로직은 어떤 장치에서 왔는지 모른다. 장치별 버튼 매핑은 common/inputs/ 안에서만 바뀐다.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventKind(str, Enum):
    STEER_AXIS = "steer_axis"          # value: -1.0 ~ 1.0 (스틱)
    STEER_STEP = "steer_step"          # value: -1 | +1 (D패드, ←→)
    STEER_CENTER = "steer_center"
    THROTTLE_AXIS = "throttle_axis"    # value: 0.0 ~ 1.0 (R2)
    REVERSE_AXIS = "reverse_axis"      # value: 0.0 ~ 1.0 (L2)
    FORWARD_TOGGLE = "forward_toggle"  # 키보드 ↑
    REVERSE_TOGGLE = "reverse_toggle"  # 키보드 ↓
    STOP = "stop"                      # Space / T
    SPEED_STEP = "speed_step"          # value: -1 | +1 (A/Z)
    DEADMAN = "deadman"                # value: True(누름) / False(뗌)  (L1)
    RECORD_TOGGLE = "record_toggle"    # ○ / R
    MARK = "mark"                      # □ / M
    INTERVAL_STEP = "interval_step"    # value: -1 | +1 ([ ])
    QUIT = "quit"                      # Options / Q / ESC
    DEVICE_LOST = "device_lost"        # 패드 연결 끊김


@dataclass
class InputEvent:
    kind: EventKind
    value: Any = None
    t: float = field(default_factory=time.monotonic)

    def __repr__(self) -> str:
        return f"InputEvent({self.kind.value}, {self.value!r})"
