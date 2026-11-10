"""Jetson Orin Nano 실제 하드웨어: PCA9685 두 장(I2C) → LD-1501MG 서보 + Cytron MDD10A (PWM+DIR).

- 서보용 보드 0x40, 50 Hz: 채널 0 에 서보.
- 모터용 보드 0x41, 1.5 kHz: 채널 0 = 속도(PWM), 채널 1 = 방향(DIR, 완전 ON/OFF). 보드당 주파수가 하나라 두 장이 필요하다.
Jetson 40핀 헤더로는 I2C(핀 3, 5)만 쓴다. 헤더의 PWM 출력은 레벨 시프터를 거쳐 구동력이 약해 서보를 움직이지 못했다
(2026-11-06 실측, 핀 15 LOW 가 1.4 V 에 머묾). 그래서 헤더 직결 드라이버는 없다. Jetson.GPIO 도 쓰지 않는다.
배선은 docs/wiring.pdf.
"""
from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

from common.config import HardwareConfig
from common.hw.base import BACKWARD, FORWARD, Vehicle, VehicleState
from common.hw.pca9685 import Pca9685Motor, Pca9685Servo

log = logging.getLogger(__name__)

SERVO_SEC_PER_DEG = 0.004   # 서보 이동 시간 어림 (LD-1501MG 6 V 에서 약 0.16 s/60° 보다 넉넉하게)


class JetsonVehicle(Vehicle):
    def __init__(self, hw: HardwareConfig, initial_angle: int, angle_limits: Optional[Tuple[int, int]] = None):
        self.hw = hw
        self._state = VehicleState(angle=initial_angle)
        self._angle_limits = angle_limits     # 유격 보정으로 지나칠 때도 넘지 않을 기구 한계 (steering.min, max)
        self._steer_pending = None            # (마감 시각, 목표 펄스 µs): 유격 보정의 두 번째 단계
        pm, pc = hw.pca9685_motor, hw.pca9685
        self.motor = Pca9685Motor(pm.bus, pm.address, pm.pwm_channel, pm.dir_channel, hw.motor.pwm_hz, pm.osc_hz,
                                  pwm_channel_2=pm.pwm_channel_2, dir_channel_2=pm.dir_channel_2)
        self.servo = Pca9685Servo(pc.bus, pc.address, pc.channel, hw.servo.pwm_hz, pc.osc_hz)
        self.servo.set_pulse_us(self._angle_to_us(initial_angle))
        log.info("JetsonVehicle 준비: motor=%s(%d Hz) servo=%s(%d Hz)",
                 self.motor_desc(), hw.motor.pwm_hz, self.servo_desc(), hw.servo.pwm_hz)

    def servo_desc(self) -> str:
        pc = self.hw.pca9685
        return f"pca9685 bus={pc.bus} addr=0x{pc.address:02x} ch={pc.channel}"

    def motor_desc(self) -> str:
        pm = self.hw.pca9685_motor
        second = f"+{pm.pwm_channel_2}/{pm.dir_channel_2}" if pm.pwm_channel_2 >= 0 else ""
        return f"pca9685 bus={pm.bus} addr=0x{pm.address:02x} pwm_ch={pm.pwm_channel} dir_ch={pm.dir_channel}{second}"

    # ------------------------------------------------------------------ servo
    def _angle_to_us(self, angle: float) -> float:
        s = self.hw.servo
        angle = max(s.min_deg, min(s.max_deg, angle))
        return s.min_us + (angle - s.min_deg) / (s.max_deg - s.min_deg) * (s.max_us - s.min_us)

    def set_steering(self, angle: int) -> None:
        """조향 각도. 유격 보정이 켜져 있고 보정 방향으로 움직이는 경우에는 목표를 지나친 각도를 먼저 보내고,
        서보가 도착할 시간이 지난 뒤 service() 가 목표 각도를 보낸다 (최종 접근 방향을 항상 같게)."""
        sv = self.hw.servo
        comp = float(sv.backlash_comp_deg)
        prev, angle = self._state.angle, int(angle)
        self._steer_pending = None
        self._state.angle = angle
        if comp and ((comp > 0 and angle < prev) or (comp < 0 and angle > prev)):
            lo, hi = self._angle_limits if self._angle_limits else (sv.min_deg, sv.max_deg)
            over = max(lo, min(hi, angle - comp))
            if over != angle:
                self.servo.set_pulse_us(self._angle_to_us(over))
                due = time.monotonic() + abs(prev - over) * SERVO_SEC_PER_DEG + float(sv.backlash_settle_sec)
                self._steer_pending = (due, self._angle_to_us(angle))
                return
        self.servo.set_pulse_us(self._angle_to_us(angle))

    def service(self, now: Optional[float] = None) -> None:
        pending = self._steer_pending
        if pending is not None and (time.monotonic() if now is None else now) >= pending[0]:
            self._steer_pending = None
            self.servo.set_pulse_us(pending[1])

    def settle(self) -> None:
        pending = self._steer_pending
        if pending is not None:
            time.sleep(max(0.0, pending[0] - time.monotonic()))
            self.service()

    # ------------------------------------------------------------------ motor
    def _set_direction(self, direction: str) -> None:
        forward = direction == FORWARD
        self.motor.set_direction_high(forward == self.hw.dir_forward_high)

    def set_throttle(self, speed: int, direction: str) -> None:
        if direction not in (FORWARD, BACKWARD):
            raise ValueError(direction)
        speed = int(max(0, min(100, speed)))
        if speed == 0:
            self.stop(smooth=False)
            return
        if self._state.direction not in (None, direction) and self._state.speed > 0:
            # 방향 전환은 반드시 정지를 거친다 (드라이버·기어 보호)
            self.stop(smooth=True)
        self._set_direction(direction)
        self.motor.set_duty(speed)
        self._state.speed = speed
        self._state.direction = direction

    def stop(self, smooth: bool = True) -> None:
        try:
            if smooth and self._state.speed > 0:
                for sp in range(self._state.speed, 0, -max(1, self.hw.stop_decay_step)):
                    self.motor.set_duty(sp)
                    time.sleep(self.hw.stop_decay_delay)
        except OSError as e:
            # 감속 단계에서 I2C 오류가 나면 감속을 포기하고 바로 듀티 0 을 쓴다 (정지가 최우선)
            log.warning("감속 정지 중 오류, 즉시 정지로 전환: %s", e)
        finally:
            self.motor.set_duty(0)
            self._state.speed = 0
            self._state.direction = None
            # 정지 중에는 방향 출력도 LOW 로 둔다: 배터리가 꺼진 MDD10A 입력에 HIGH 를 계속 걸어 두지 않는다
            self.motor.set_direction_high(False)

    def close(self) -> None:
        errors = []
        for name, step in (("stop", lambda: self.stop(smooth=True)),
                           ("motor.close", self.motor.close),
                           ("servo.close", self.servo.close)):
            try:
                step()
            except Exception as e:  # 한 단계가 실패해도 나머지 정리는 계속
                errors.append(f"{name}: {e}")
        if errors:
            log.warning("하드웨어 정리 중 오류: %s", "; ".join(errors))
        else:
            log.info("하드웨어 정리 완료")

    @property
    def state(self) -> VehicleState:
        return self._state
