"""PCA9685 서보 드라이버를 가짜 I2C 장치로 검증한다 (PC 에서 실행)."""
import sys

import pytest

from common.config import HardwareConfig
from common.hw import pca9685 as m


class FakeI2C:
    """write/read 를 기록하고 레지스터 맵을 흉내 낸다 (자동 증가 포함)."""

    def __init__(self, *a, **k):
        self.regs = {m.MODE1: 0x11, m.MODE2: 0x04, m.PRESCALE: 0x1E}   # 전원 투입 기본값
        self.writes = []
        self._ptr = 0
        self.closed = False

    def write(self, data: bytes) -> None:
        self.writes.append(bytes(data))
        reg, vals = data[0], data[1:]
        self._ptr = reg
        for i, v in enumerate(vals):
            self.regs[reg + i] = v

    def read(self, n: int) -> bytes:
        return bytes([self.regs.get(self._ptr, 0)])

    def close(self) -> None:
        self.closed = True


def test_prescale_and_counts():
    assert m.prescale_for(50) == 121                 # 25 MHz / (4096 * 50) - 1
    assert m.pulse_to_counts(1500, 50) == 307        # 1.5 ms / 20 ms * 4096
    assert m.pulse_to_counts(500, 50) == 102 and m.pulse_to_counts(2500, 50) == 512
    with pytest.raises(ValueError):
        m.prescale_for(5000)                         # 프리스케일 3 미만


def test_init_sequence_writes_prescale_in_sleep_and_enables_auto_increment():
    dev = FakeI2C()
    chip = m.PCA9685(dev, 50)
    regs = [w[0] for w in dev.writes]
    i_sleep = next(i for i, w in enumerate(dev.writes) if len(w) > 1 and w[0] == m.MODE1 and w[1] & m.MODE1_SLEEP)
    i_pre = next(i for i, w in enumerate(dev.writes) if w[0] == m.PRESCALE)
    assert i_sleep < i_pre, "PRESCALE 은 sleep 상태에서 써야 한다"
    assert dev.regs[m.PRESCALE] == 121
    assert dev.regs[m.MODE1] & m.MODE1_AI and not dev.regs[m.MODE1] & m.MODE1_SLEEP
    assert dev.regs[m.MODE2] & m.MODE2_OUTDRV
    assert m.PRESCALE in regs and chip.prescale == 121


def test_channel_registers_and_all_call():
    dev = FakeI2C()
    chip = m.PCA9685(dev, 50)
    assert chip.set_pulse_us(3, 1500) == 307
    assert dev.writes[-1] == bytes([m.LED0_ON_L + 12, 0, 0, 307 & 0xFF, 307 >> 8])
    chip.set_pulse_us(-1, 2500)
    assert dev.writes[-1][0] == m.ALL_LED_ON_L and dev.writes[-1][3:] == bytes([512 & 0xFF, 512 >> 8])
    chip.set_duty(0, 100)                             # 100 % → 완전 ON
    assert dev.writes[-1] == bytes([m.LED0_ON_L, 0, m.FULL_ON, 0, 0])
    chip.channel_off(0)
    assert dev.writes[-1] == bytes([m.LED0_ON_L, 0, 0, 0, m.FULL_OFF])


def test_servo_wrapper_turns_channel_off_on_close():
    dev = FakeI2C()
    s = m.Pca9685Servo(7, 0x40, 2, 50, dev=dev)
    s.set_pulse_us(1500)
    assert s.last_us == 1500 and dev.writes[-1][0] == m.LED0_ON_L + 8
    s.close()
    assert dev.writes[-1] == bytes([m.LED0_ON_L + 8, 0, 0, 0, m.FULL_OFF]) and dev.closed
    with pytest.raises(ValueError):
        m.Pca9685Servo(7, 0x40, 16, 50, dev=FakeI2C())


@pytest.fixture
def fake_board(monkeypatch):
    """가짜 I2C 장치. JetsonVehicle 이 보드 두 장(0x40, 0x41)을 여는지 본다."""
    devices = []

    def fake_i2c(bus, address):
        d = FakeI2C(); d.bus, d.address = bus, address; devices.append(d); return d
    monkeypatch.setattr(m, "I2CDevice", fake_i2c)
    return devices


def test_full_on_off_and_motor_wrapper():
    dev = FakeI2C()
    chip = m.PCA9685(dev, 1500)
    assert chip.prescale == 3                                     # 25 MHz / (4096 * 1500) - 1 = 3.07 → 3
    chip.channel_on(1)
    assert dev.writes[-1] == bytes([m.LED0_ON_L + 4, 0, m.FULL_ON, 0, 0])
    assert chip.set_duty(0, 100) == 4096 and dev.writes[-1][2] == m.FULL_ON
    assert chip.set_duty(0, 0) == 0 and dev.writes[-1][4] == m.FULL_OFF
    assert chip.set_duty(0, 30) == round(4096 * 0.3)              # 30 % → 1229 카운트
    mt = m.Pca9685Motor(7, 0x41, 0, 1, 1500, dev=FakeI2C())
    d = mt.chip.dev
    mt.set_direction_high(True);  assert d.writes[-1] == bytes([m.LED0_ON_L + 4, 0, m.FULL_ON, 0, 0])
    mt.set_direction_high(False); assert d.writes[-1] == bytes([m.LED0_ON_L + 4, 0, 0, 0, m.FULL_OFF])
    mt.set_duty(60);              assert d.writes[-1][0] == m.LED0_ON_L
    mt.close();                   assert d.closed and d.writes[-1][4] == m.FULL_OFF
    with pytest.raises(ValueError):
        m.Pca9685Motor(7, 0x41, 2, 2, 1500, dev=FakeI2C())


def _last(dev, channel):
    """채널의 LEDn 레지스터 4바이트에 마지막으로 쓴 값 (ON_L, ON_H, OFF_L, OFF_H)."""
    reg = m.LED0_ON_L + 4 * channel
    return tuple(next(w for w in reversed(dev.writes) if len(w) == 5 and w[0] == reg)[1:])


def test_vehicle_drives_two_pca9685_boards(fake_board):
    devices = fake_board
    from common.hw.jetson import JetsonVehicle
    from common.hw.base import BACKWARD, FORWARD
    hw = HardwareConfig()                                         # 기본값: 서보 0x40, 모터 0x41
    v = JetsonVehicle(hw, initial_angle=90)
    assert "Jetson.GPIO" not in sys.modules and "Jetson" not in sys.modules   # GPIO 라이브러리는 쓰지 않는다
    by_addr = {d.address: d for d in devices}
    assert set(by_addr) == {0x40, 0x41}
    servo, motor = by_addr[0x40], by_addr[0x41]
    assert servo.regs[m.PRESCALE] == 121 and motor.regs[m.PRESCALE] == 3   # 50 Hz / 1500 Hz
    assert servo.writes[-1] == bytes([m.LED0_ON_L, 0, 0, 307 & 0xFF, 307 >> 8])   # 90° = 1500 µs
    v.set_steering(0);   assert servo.writes[-1][3:] == bytes([102 & 0xFF, 102 >> 8])
    v.set_throttle(60, FORWARD)                                   # 채널 0 = 속도, 채널 1 = 방향 (기본 배선은 커넥터 안에서 두 채널로 나눔)
    assert _last(motor, 1) == (0, m.FULL_ON, 0, 0)                # DIR HIGH (dir_forward_high)
    assert _last(motor, 0) == (0, 0, 2458 & 0xFF, 2458 >> 8)      # 60 % = 2458 카운트
    v.set_throttle(30, BACKWARD)                                  # 정지를 거쳐 DIR LOW
    assert _last(motor, 1) == (0, 0, 0, m.FULL_OFF)
    assert _last(motor, 0) == (0, 0, 1229 & 0xFF, 1229 >> 8)
    v.set_throttle(40, FORWARD)
    v.stop(smooth=False)                                          # 정지하면 속도 0 과 함께 방향 출력도 LOW
    assert all(_last(motor, ch) == (0, 0, 0, m.FULL_OFF) for ch in range(2))
    assert not any(len(w) == 5 and w[0] == m.LED0_ON_L + 4 * ch for ch in (2, 3) for w in motor.writes)   # 채널 2·3 은 안 씀
    v.close()
    assert servo.closed and motor.closed
    assert all(_last(motor, ch) == (0, 0, 0, m.FULL_OFF) for ch in range(2))


def test_servo_and_motor_cannot_share_one_pca9685(tmp_path):
    import yaml
    from common.config import DEFAULT_CONFIG_PATH, ConfigError, load_config
    base = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    base["hardware"]["pca9685_motor"]["address"] = base["hardware"]["pca9685"]["address"]
    p = tmp_path / "same.yaml"; p.write_text(yaml.safe_dump(base), encoding="utf-8")
    with pytest.raises(ConfigError, match="같은 PCA9685"):
        load_config(p)


def test_config_validation_for_pca9685(tmp_path):
    import yaml
    from common.config import DEFAULT_CONFIG_PATH, ConfigError, load_config
    base = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    for section, key, bad in (("servo", "pwm_hz", 0), ("pca9685", "channel", 16), ("pca9685", "address", 0x80),
                              ("pca9685_motor", "address", 0x80), ("pca9685_motor", "bus", -1)):
        cfg = yaml.safe_load(yaml.safe_dump(base))
        cfg["hardware"][section][key] = bad
        p = tmp_path / f"{section}_{key}.yaml"
        p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(p)


class FlakyI2C(FakeI2C):
    """지정한 횟수만큼 write 에서 OSError(121) 를 낸 뒤 정상 동작한다."""

    def __init__(self, fail_times: int = 0):
        super().__init__()
        self.fail_times = fail_times

    def write(self, data: bytes) -> None:
        if self.fail_times > 0:
            self.fail_times -= 1
            raise OSError(121, "Remote I/O error")
        super().write(data)


def test_transient_i2c_error_is_retried(monkeypatch):
    monkeypatch.setattr(m, "I2C_RETRY_DELAY", 0)
    dev = FlakyI2C()
    chip = m.PCA9685(dev, 50)
    dev.fail_times = 2                                # 두 번 실패 후 성공
    assert chip.set_pulse_us(0, 1500) == 307
    assert dev.writes[-1] == bytes([m.LED0_ON_L, 0, 0, 307 & 0xFF, 307 >> 8])
    assert chip.io_errors == 2


def test_persistent_i2c_error_raises(monkeypatch):
    monkeypatch.setattr(m, "I2C_RETRY_DELAY", 0)
    dev = FlakyI2C()
    chip = m.PCA9685(dev, 50)
    dev.fail_times = 100
    with pytest.raises(OSError):
        chip.set_duty(0, 30)
    assert chip.io_errors == m.I2C_RETRIES


def test_stop_writes_zero_duty_even_if_ramp_fails(fake_board, monkeypatch):
    """감속 단계에서 I2C 오류가 나도 마지막에 듀티 0(완전 OFF)은 반드시 쓴다."""
    monkeypatch.setattr(m, "I2C_RETRY_DELAY", 0)
    devices = fake_board
    from common.hw.jetson import JetsonVehicle
    from common.hw.base import FORWARD
    v = JetsonVehicle(HardwareConfig(), initial_angle=90)
    motor = {d.address: d for d in devices}[0x41]
    v.set_throttle(60, FORWARD)
    real_write, state = motor.write, {"left": m.I2C_RETRIES}       # 감속 첫 쓰기를 재시도까지 모두 실패시킨다

    def flaky(data):
        if state["left"] > 0:
            state["left"] -= 1
            raise OSError(121, "Remote I/O error")
        real_write(data)
    monkeypatch.setattr(motor, "write", flaky)
    v.stop(smooth=True)
    assert all(_last(motor, ch) == (0, 0, 0, m.FULL_OFF) for ch in range(2))
    assert v._state.speed == 0


def _servo_counts(dev):
    """채널 0 에 쓴 OFF 카운트 목록."""
    return [w[3] | (w[4] << 8) for w in dev.writes if len(w) == 5 and w[0] == m.LED0_ON_L and not w[4] & m.FULL_OFF]


def test_backlash_compensation_overshoots_then_returns(fake_board):
    """높은 각도에서 내려올 때만 목표를 지나쳤다가(−8°) 시간이 지난 뒤 목표로 돌아온다."""
    devices = fake_board
    from common.hw.jetson import JetsonVehicle
    hw = HardwareConfig()
    hw.servo.backlash_comp_deg = 8
    v = JetsonVehicle(hw, initial_angle=90, angle_limits=(30, 150))
    servo = {d.address: d for d in devices}[0x40]
    us = lambda deg: m.pulse_to_counts(500 + 2000 * deg / 180, 50)
    v.set_steering(120)                                   # 올라가는 방향: 바로 목표
    assert _servo_counts(servo)[-1] == us(120) and v._steer_pending is None
    v.set_steering(90)                                    # 내려오는 방향: 82° 를 먼저
    assert _servo_counts(servo)[-1] == us(82) and v.state.angle == 90
    due = v._steer_pending[0]
    v.service(now=due - 0.01)
    assert _servo_counts(servo)[-1] == us(82)             # 아직 이르다
    v.service(now=due + 0.01)
    assert _servo_counts(servo)[-1] == us(90) and v._steer_pending is None
    v.set_steering(30)                                    # 기구 한계(30°) 아래로는 지나치지 않는다
    assert _servo_counts(servo)[-1] == us(30) and v._steer_pending is None
    v.set_steering(60); v.set_steering(40)                # 보정 도중 새 명령이 오면 이전 대기는 버린다
    assert _servo_counts(servo)[-1] == us(32)
    v.set_steering(90)
    assert _servo_counts(servo)[-1] == us(90) and v._steer_pending is None
    v.close()


def test_backlash_settle_blocks_until_final_position(fake_board, monkeypatch):
    devices = fake_board
    import common.hw.jetson as jm
    monkeypatch.setattr(jm, "SERVO_SEC_PER_DEG", 0.0)
    hw = HardwareConfig()
    hw.servo.backlash_comp_deg = -5                       # 음수: 올라가는 방향에 적용
    hw.servo.backlash_settle_sec = 0.01
    v = jm.JetsonVehicle(hw, initial_angle=90)
    servo = {d.address: d for d in devices}[0x40]
    us = lambda deg: m.pulse_to_counts(500 + 2000 * deg / 180, 50)
    v.set_steering(60)
    assert _servo_counts(servo)[-1] == us(60)             # 내려가는 방향은 그대로
    v.set_steering(100)
    assert _servo_counts(servo)[-1] == us(105)
    v.settle()
    assert _servo_counts(servo)[-1] == us(100) and v._steer_pending is None
    v.close()


def test_motor_second_channel_pair_can_be_disabled():
    """pwm_channel_2/dir_channel_2 가 -1 이면 채널 0·1 만 쓴다 (분기 커넥터로 나눈 배선)."""
    dev = FakeI2C()
    mt = m.Pca9685Motor(7, 0x41, 0, 1, 1500, dev=dev)
    mt.set_direction_high(True); mt.set_duty(50)
    regs = {w[0] for w in dev.writes if len(w) == 5}
    assert regs == {m.LED0_ON_L, m.LED0_ON_L + 4}
    dev = FakeI2C()
    mt = m.Pca9685Motor(7, 0x41, 0, 1, 1500, dev=dev, pwm_channel_2=2, dir_channel_2=3)
    mt.set_direction_high(True); mt.set_duty(50)
    assert _last(dev, 0) == _last(dev, 2) and _last(dev, 1) == _last(dev, 3) == (0, m.FULL_ON, 0, 0)
    with pytest.raises(ValueError):
        m.Pca9685Motor(7, 0x41, 0, 1, 1500, dev=FakeI2C(), pwm_channel_2=1, dir_channel_2=3)   # 채널 중복
    with pytest.raises(ValueError):
        m.Pca9685Motor(7, 0x41, 0, 1, 1500, dev=FakeI2C(), pwm_channel_2=2)                    # 한쪽만 지정


def test_config_validation_for_second_motor_channels(tmp_path):
    import yaml
    from common.config import DEFAULT_CONFIG_PATH, ConfigError, load_config
    base = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    pm = base["hardware"]["pca9685_motor"]
    assert (pm["pwm_channel"], pm["dir_channel"], pm["pwm_channel_2"], pm["dir_channel_2"]) == (0, 1, -1, -1)
    for bad in ((2, 1), (0, 3), (2, -1), (16, 3)):
        pm["pwm_channel_2"], pm["dir_channel_2"] = bad
        p = tmp_path / "bad.yaml"; p.write_text(yaml.safe_dump(base), encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(p)
    pm["pwm_channel_2"], pm["dir_channel_2"] = -1, -1
    p = tmp_path / "ok.yaml"; p.write_text(yaml.safe_dump(base), encoding="utf-8")
    assert load_config(p).hardware.pca9685_motor.pwm_channel_2 == -1

