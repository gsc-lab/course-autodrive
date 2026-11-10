"""MAX7219 8×8 도트 매트릭스를 Jetson 의 SPI1 (/dev/spidev0.0) 에 직결해 구동한다. 추가 패키지 없이 ioctl 로 쓴다.

배선 (Jetson 40핀 헤더): VCC → 핀 2 (5 V), GND → 핀 20, DIN → 핀 19 (MOSI), CS → 핀 24 (CS0), CLK → 핀 23 (SCK).
SPI 는 jetson-io 로 켠다 (scripts/jetson_setup.sh install 이 한다). Jetson 의 3.3 V 신호는 MAX7219 규격(HIGH 3.5 V)보다
조금 낮지만 2026-11-10 실기에서 10분 연속 전송에 깨짐이 없어 직결로 쓴다. 안 되는 모듈이 있으면 display.enabled 를 끈다.
"""
from __future__ import annotations

import array
import logging
import os
import struct
from typing import List, Optional

from common.display.base import Display, DisplayState
from common.display.glyphs import flip_horizontal, render, rotate

log = logging.getLogger(__name__)

SPI_IOC_WR_MAX_SPEED_HZ = 0x40046B04
SPI_IOC_WR_MODE = 0x40016B01
SPI_IOC_MESSAGE_1 = 0x40206B00
NOOP, DECODE, INTENSITY, SCANLIMIT, SHUTDOWN, DISPLAYTEST = 0x00, 0x09, 0x0A, 0x0B, 0x0C, 0x0F


class SpiDevice:
    """/dev/spidevB.C 하나. write(bytes) 만 제공한다 (테스트에서 가짜로 바꾼다)."""

    def __init__(self, path: str, hz: int):
        import fcntl  # 리눅스 전용: 지연 import (PC 에서도 모듈은 import 되어야 한다)
        self._ioctl = fcntl.ioctl
        self.fd = os.open(path, os.O_RDWR)
        self._ioctl(self.fd, SPI_IOC_WR_MODE, struct.pack("B", 0))
        self._ioctl(self.fd, SPI_IOC_WR_MAX_SPEED_HZ, struct.pack("I", hz))
        self.hz = hz

    def write(self, data: bytes) -> None:
        buf = array.array("B", data)
        # struct spi_ioc_transfer (32 바이트): tx_buf, rx_buf, len, speed_hz, delay_usecs, bits_per_word, cs_change, …
        msg = struct.pack("QQIIHBBBBBB", buf.buffer_info()[0], 0, len(buf), self.hz, 0, 8, 0, 0, 0, 0, 0)
        self._ioctl(self.fd, SPI_IOC_MESSAGE_1, msg)

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class Max7219:
    def __init__(self, dev, intensity: int = 3):
        self.dev = dev
        self.write_reg(DISPLAYTEST, 0)
        self.write_reg(SCANLIMIT, 7)
        self.write_reg(DECODE, 0)
        self.write_reg(INTENSITY, max(0, min(15, intensity)))
        self.write_reg(SHUTDOWN, 1)
        self.show([0] * 8)

    def write_reg(self, reg: int, value: int) -> None:
        self.dev.write(bytes([reg & 0x0F, value & 0xFF]))

    def show(self, rows: List[int]) -> None:
        for i, bits in enumerate(rows):
            self.write_reg(i + 1, bits)

    def shutdown(self) -> None:
        self.show([0] * 8)
        self.write_reg(SHUTDOWN, 0)

    def close(self) -> None:
        self.dev.close()


class Max7219Display(Display):
    name = "max7219"

    def __init__(self, spi_dev: str, intensity: int, rotate_deg: int, flip: bool, blink_hz: float,
                 spi_hz: int = 1_000_000, dev=None):
        self.rotate_deg, self.flip, self.blink_hz = rotate_deg, flip, blink_hz
        self.chip = Max7219(dev if dev is not None else SpiDevice(spi_dev, spi_hz), intensity)
        self._last: Optional[List[int]] = None
        log.info("MAX7219 준비: %s rotate=%d flip=%s", spi_dev, rotate_deg, flip)

    def update(self, st: DisplayState, now: float) -> None:
        blink_on = int(now * self.blink_hz * 2) % 2 == 0
        rows = render(st, blink_on)
        if self.rotate_deg:
            rows = rotate(rows, self.rotate_deg)
        if self.flip:
            rows = flip_horizontal(rows)
        if rows != self._last:            # 바뀐 프레임만 보낸다
            self.chip.show(rows)
            self._last = rows

    def close(self) -> None:
        try:
            self.chip.shutdown()
        finally:
            self.chip.close()
