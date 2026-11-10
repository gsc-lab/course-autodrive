from __future__ import annotations

import logging
from typing import Optional

from common.config import InputConfig
from common.inputs.base import InputDevice

log = logging.getLogger(__name__)


def create_input(cfg: InputConfig, override: Optional[str] = None) -> InputDevice:
    """primary 장치를 열고, 실패하면 설정에 따라 키보드로 대체한다."""
    kind = override or cfg.primary
    if kind == "ds4":
        try:
            from common.inputs.ds4 import DS4Input
            return DS4Input(cfg.ds4)
        except Exception as e:
            if not cfg.keyboard_fallback or override == "ds4":
                raise
            log.warning("DS4 를 열지 못해 키보드로 대체합니다: %s", e)
            kind = "keyboard"
    if kind == "keyboard":
        from common.inputs.keyboard import KeyboardInput
        return KeyboardInput()
    raise ValueError(f"알 수 없는 입력 장치: {kind}")
