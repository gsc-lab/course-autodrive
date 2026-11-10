"""상태 표시 장치 (선택). 기본은 Jetson SPI1 에 직결한 MAX7219 8×8 도트 매트릭스.

표시 장치는 주행에 필수가 아니므로 열지 못하면 경고만 남기고 NullDisplay 로 계속 간다."""
from __future__ import annotations

import logging
import os

from common.config import DisplayConfig
from common.display.base import Display, DisplayState, NullDisplay

log = logging.getLogger(__name__)

__all__ = ["Display", "DisplayState", "NullDisplay", "create_display"]


def create_display(cfg: DisplayConfig, mock: bool = False) -> Display:
    if mock or not cfg.enabled:
        return NullDisplay()
    if cfg.driver != "max7219":
        log.warning("알 수 없는 display.driver %r → 표시 없음", cfg.driver)
        return NullDisplay()
    if not os.path.exists(cfg.spi_dev):
        log.warning("%s 가 없어 상태 표시를 끕니다 (SPI 를 켜려면 bash scripts/jetson_setup.sh install 후 재부팅)", cfg.spi_dev)
        return NullDisplay()
    try:
        from common.display.max7219 import Max7219Display
        return Max7219Display(cfg.spi_dev, cfg.intensity, cfg.rotate, cfg.flip, cfg.blink_hz)
    except Exception as e:  # 표시 장치 고장은 주행을 막지 않는다
        log.warning("상태 표시 장치를 열지 못해 끕니다: %s", e)
        return NullDisplay()
