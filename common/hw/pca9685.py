"""PCA9685 16채널 PWM 드라이버 (I2C). 추가 패키지 없이 /dev/i2c-N 을 직접 쓴다.

왜 쓰나 (2026-11-06 실기 확인): Orin Nano 개발자 키트 40핀 헤더의 PWM 은 레벨 시프터를 거쳐 나와 서보를 구동하지
못했고, PCA9685 는 푸시풀 출력이라 LD-1501MG 가 바로 반응했다. 배선은 docs/wiring.pdf
(핀 3 SDA, 5 SCL, 1 VCC 3.3 V, 6 GND → i2c-7; 서보용 0x40, 모터용 0x41; 서보 전원은 단자대 V+ 에 UBEC 6 V).

레지스터: MODE1 0x00, MODE2 0x01, LED0_ON_L 0x06 (채널당 4바이트), ALL_LED_ON_L 0xFA, PRESCALE 0xFE.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

log = logging.getLogger(__name__)

I2C_SLAVE = 0x0703          # linux/i2c-dev.h ioctl
MODE1, MODE2, PRESCALE = 0x00, 0x01, 0xFE
LED0_ON_L, ALL_LED_ON_L = 0x06, 0xFA
MODE1_SLEEP, MODE1_AI, MODE1_RESTART = 0x10, 0x20, 0x80
MODE2_OUTDRV = 0x04
FULL_OFF = 0x10             # LEDn_OFF_H bit4: 채널 완전 끔
FULL_ON = 0x10              # LEDn_ON_H bit4: 채널 완전 켬 (OFF_H 의 FULL_OFF 가 우선하므로 OFF_H 는 0 으로 쓴다)
DEFAULT_OSC_HZ = 25_000_000
I2C_RETRIES = 4             # 쓰기/읽기 한 번당 최대 시도 횟수
I2C_RETRY_DELAY = 0.003     # 재시도 사이 대기(초)


class I2CDevice:
    """/dev/i2c-N 의 한 슬레이브 주소. write(bytes) / read(n) 만 제공한다 (테스트에서 가짜로 바꿀 수 있는 최소 인터페이스)."""

    def __init__(self, bus: int, address: int):
        import fcntl  # 리눅스 전용: 지연 import
        self.path = f"/dev/i2c-{bus}"
        self.fd = os.open(self.path, os.O_RDWR)
        fcntl.ioctl(self.fd, I2C_SLAVE, address)

    def write(self, data: bytes) -> None:
        os.write(self.fd, data)

    def read(self, n: int) -> bytes:
        return os.read(self.fd, n)

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def prescale_for(hz: float, osc_hz: float = DEFAULT_OSC_HZ) -> int:
    """PRESCALE = round(osc / (4096 * hz)) - 1 (데이터시트 7.3.5). 3~255."""
    value = round(osc_hz / (4096.0 * hz)) - 1
    if not (3 <= value <= 255):
        raise ValueError(f"PCA9685 로 만들 수 없는 주파수입니다: {hz} Hz (prescale {value})")
    return value


def pulse_to_counts(us: float, hz: float) -> int:
    """펄스 폭(µs) → 12비트 OFF 카운트 (ON 은 0 에서 시작)."""
    return int(round(us * 4096.0 * hz / 1_000_000.0))


class PCA9685:
    def __init__(self, dev, hz: float, osc_hz: float = DEFAULT_OSC_HZ):
        self.dev = dev
        self.hz = float(hz)
        self.prescale = prescale_for(hz, osc_hz)
        self.io_errors = 0          # 재시도를 포함해 지금까지 난 I2C 오류 수 (진단용)
        self._init()

    # ------------------------------------------------------------------ 저수준
    def _retry(self, op):
        """일시적인 I2C 오류(커넥터 접촉, 진동으로 인한 NACK)는 몇 번 다시 시도한다. 계속 실패하면 마지막 예외를 올린다."""
        for attempt in range(I2C_RETRIES):
            try:
                return op()
            except OSError as e:
                self.io_errors += 1
                if attempt == I2C_RETRIES - 1:
                    raise
                log.warning("PCA9685 I2C 오류, 재시도 %d/%d: %s", attempt + 1, I2C_RETRIES - 1, e)
                time.sleep(I2C_RETRY_DELAY)

    def write_reg(self, reg: int, *vals: int) -> None:
        data = bytes([reg, *vals])
        self._retry(lambda: self.dev.write(data))

    def read_reg(self, reg: int) -> int:
        def op() -> int:
            self.dev.write(bytes([reg]))
            return self.dev.read(1)[0]
        return self._retry(op)

    def _init(self) -> None:
        self.write_reg(MODE1, 0x00)                       # 리셋: 정상 모드, 올콜 끔
        time.sleep(0.005)
        mode1 = self.read_reg(MODE1) & ~MODE1_RESTART
        self.write_reg(MODE1, mode1 | MODE1_SLEEP)        # 프리스케일은 sleep 중에만 쓸 수 있다
        self.write_reg(PRESCALE, self.prescale)
        self.write_reg(MODE1, mode1 & ~MODE1_SLEEP)
        time.sleep(0.005)                                 # 오실레이터 안정 (≥500 µs)
        self.write_reg(MODE1, mode1 | MODE1_RESTART | MODE1_AI)
        self.write_reg(MODE2, MODE2_OUTDRV)               # 토템폴 출력
        actual = self.read_reg(PRESCALE)
        if actual != self.prescale:
            raise RuntimeError(f"PCA9685 PRESCALE 쓰기 실패: 요청 {self.prescale}, 읽힘 {actual}")

    # ------------------------------------------------------------------ 채널
    def set_pulse_us(self, channel: int, us: float) -> int:
        """channel 0~15, 또는 -1 이면 16채널 전부. 돌려주는 값은 OFF 카운트."""
        off = max(0, min(4095, pulse_to_counts(us, self.hz)))
        reg = ALL_LED_ON_L if channel < 0 else LED0_ON_L + 4 * channel
        self.write_reg(reg, 0, 0, off & 0xFF, off >> 8)
        return off

    def set_duty(self, channel: int, percent: float) -> int:
        """0 % 는 완전 OFF, 100 % 는 완전 ON (펄스 없이 정상태), 그 사이는 12비트 카운트."""
        percent = max(0.0, min(100.0, percent))
        if percent <= 0.0:
            self.channel_off(channel); return 0
        if percent >= 100.0:
            self.channel_on(channel); return 4096
        period_us = 1_000_000.0 / self.hz
        return self.set_pulse_us(channel, period_us * percent / 100.0)

    def channel_on(self, channel: int) -> None:
        reg = ALL_LED_ON_L if channel < 0 else LED0_ON_L + 4 * channel
        self.write_reg(reg, 0, FULL_ON, 0, 0)

    def channel_off(self, channel: int) -> None:
        reg = ALL_LED_ON_L if channel < 0 else LED0_ON_L + 4 * channel
        self.write_reg(reg, 0, 0, 0, FULL_OFF)

    def close(self) -> None:
        self.dev.close()


class Pca9685Servo:
    """서보 한 채널. JetsonVehicle 이 쓰는 서보 출력 인터페이스: set_pulse_us / set_duty / close."""

    def __init__(self, bus: int, address: int, channel: int, hz: float, osc_hz: float = DEFAULT_OSC_HZ,
                 dev=None):
        if not (0 <= channel <= 15):
            raise ValueError(f"PCA9685 채널은 0~15: {channel}")
        self.channel = channel
        self.chip = PCA9685(dev if dev is not None else I2CDevice(bus, address), hz, osc_hz)
        self.last_us: Optional[float] = None
        log.info("PCA9685 준비: bus=%d addr=0x%02x ch=%d %g Hz prescale=%d", bus, address, channel, hz, self.chip.prescale)

    def set_pulse_us(self, us: float) -> None:
        self.chip.set_pulse_us(self.channel, us)
        self.last_us = us

    def set_duty(self, percent: float) -> None:
        self.chip.set_duty(self.channel, percent)

    def close(self) -> None:
        try:
            self.chip.channel_off(self.channel)   # 펄스를 끊어 서보 힘을 푼다
        finally:
            self.chip.close()


class Pca9685Motor:
    """MDD10A 한 채널을 PCA9685 두 채널(PWM, DIR)로 구동. JetsonVehicle 의 모터 출력 인터페이스:
    set_duty / set_direction_high / close."""

    def __init__(self, bus: int, address: int, pwm_channel: int, dir_channel: int, hz: float,
                 osc_hz: float = DEFAULT_OSC_HZ, dev=None, pwm_channel_2: int = -1, dir_channel_2: int = -1):
        """pwm_channel/dir_channel 은 MDD10A 1번 채널(PWM1/DIR1)로 간다. pwm_channel_2/dir_channel_2 를 주면
        2번 채널(PWM2/DIR2)에 항상 같은 값을 내서, 분기 커넥터 없이 선 한 가닥씩으로 두 모터를 함께 구동한다."""
        self.pwm_channels = [pwm_channel] + ([pwm_channel_2] if pwm_channel_2 >= 0 else [])
        self.dir_channels = [dir_channel] + ([dir_channel_2] if dir_channel_2 >= 0 else [])
        used = self.pwm_channels + self.dir_channels
        if (len(self.pwm_channels) != len(self.dir_channels) or len(set(used)) != len(used)
                or not all(0 <= c <= 15 for c in used)):
            raise ValueError(f"PCA9685 모터 채널이 올바르지 않습니다: pwm={self.pwm_channels} dir={self.dir_channels}")
        self.pwm_channel, self.dir_channel = pwm_channel, dir_channel
        self.chip = PCA9685(dev if dev is not None else I2CDevice(bus, address), hz, osc_hz)
        for ch in used:
            self.chip.channel_off(ch)
        log.info("PCA9685 모터 준비: bus=%d addr=0x%02x pwm_ch=%s dir_ch=%s %g Hz prescale=%d",
                 bus, address, self.pwm_channels, self.dir_channels, hz, self.chip.prescale)

    def set_duty(self, percent: float) -> None:
        for ch in self.pwm_channels:
            self.chip.set_duty(ch, percent)

    def set_direction_high(self, high: bool) -> None:
        for ch in self.dir_channels:
            if high:
                self.chip.channel_on(ch)
            else:
                self.chip.channel_off(ch)

    def close(self) -> None:
        try:
            for ch in self.pwm_channels + self.dir_channels:
                self.chip.channel_off(ch)
        finally:
            self.chip.close()


def probe(bus: int, address: int) -> str:
    """시작 전 점검용: 장치 파일 접근 + MODE1 읽기. 실패하면 예외 메시지를 그대로 올린다."""
    dev = I2CDevice(bus, address)
    try:
        dev.write(bytes([MODE1]))
        mode1 = dev.read(1)[0]
    finally:
        dev.close()
    return f"MODE1=0x{mode1:02x}"


PCA9685_ALLCALL = 0x70      # 모든 PCA9685 가 함께 듣는 주소 (리셋 기본 활성). i2cdetect 에 보여도 정상


def scan(bus: int, addresses=range(0x03, 0x78)) -> list:
    """버스에서 응답하는 7비트 주소 목록 (i2cdetect 와 같은 '읽기 1바이트' 방식). 장치 파일이 없거나 권한이 없으면 OSError."""
    found = []
    for addr in addresses:
        try:
            dev = I2CDevice(bus, addr)
        except OSError:
            raise
        try:
            dev.read(1)
            found.append(addr)
        except OSError:
            pass            # NACK: 그 주소에 장치 없음
        finally:
            dev.close()
    return found
