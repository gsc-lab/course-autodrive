"""config.yaml 로드와 검증.

사용법:
    from common.config import load_config
    cfg = load_config()              # 저장소 루트의 config.yaml
    cfg = load_config("other.yaml")

알 수 없는 키가 있으면 오류를 낸다(오타 방지).
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, get_type_hints

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
ENV_CONFIG = "AUTODRIVE_CONFIG"


@dataclass
class SteeringConfig:
    mode: str = "discrete"
    classes: list = field(default_factory=lambda: [30, 60, 90, 120, 150])
    center: int = 90
    min: int = 30
    max: int = 150
    step_deg: int = 5
    invert: bool = False
    hysteresis: float = 0.15


@dataclass
class ThrottleConfig:
    max_speed: int = 60
    default_speed: int = 40
    step: int = 10
    reverse_max_speed: int = 30
    levels: list = field(default_factory=lambda: [30, 45, 60])   # 전진 속도 단계(듀티 %). 빈 목록이면 연속(트리거 비례)
    trigger_deadzone: float = 0.10      # 트리거를 이만큼 당기기 전에는 정지
    level_hysteresis: float = 0.15      # 단계 경계에서 떨림 방지 (단계 폭 대비 비율)


@dataclass
class ServoConfig:
    pwm_hz: int = 50
    min_us: int = 500
    max_us: int = 2500
    min_deg: int = 0
    max_deg: int = 180
    backlash_comp_deg: float = 0.0     # 조향 유격 보정. 양수: 각도를 낮추는 쪽으로 움직일 때 목표를 이만큼 지나쳤다가 되돌아온다. 음수: 반대 방향
    backlash_settle_sec: float = 0.08  # 지나친 위치에 도착한 뒤 머무는 시간


@dataclass
class MotorConfig:
    pwm_hz: int = 1500            # PCA9685 가 낼 수 있는 24~1526 Hz (프리스케일 한계). MDD10A 는 20 kHz 까지 받는다


@dataclass
class Pca9685Config:
    """서보용 PCA9685 보드. I2C 는 Jetson 핀 3(SDA)·5(SCL) = i2c-7, 로직 전원은 핀 1(3.3 V), 접지 핀 6."""
    bus: int = 7                  # 핀 3/5 = i2c-7, 핀 27/28 = i2c-1
    address: int = 0x40
    channel: int = 0
    osc_hz: int = 25_000_000      # 내부 오실레이터. 펄스 폭이 측정과 다르면 보정


@dataclass
class Pca9685MotorConfig:
    bus: int = 7
    address: int = 0x41           # 두 번째 보드: A0 점퍼 납땜. 서보 보드와 주소가 달라야 한다 (주파수가 보드당 하나)
    pwm_channel: int = 0          # -> MDD10A PWM1
    dir_channel: int = 1          # -> MDD10A DIR1 (완전 ON/OFF 로 씀)
    # 기본 배선(R5.1)은 채널 0 → J4 → PWM1·PWM2, 채널 1 → J5 → DIR1·DIR2 로 WAGO 에서 나누어 채널 0·1 만 쓴다. 채널 넷으로 따로 꽂은
    # 보드라면 2, 3 으로 두면 채널 0·1 과 같은 값을 낸다. -1 이면 쓰지 않음
    pwm_channel_2: int = -1       # -> MDD10A PWM2 (따로 꽂았을 때만)
    dir_channel_2: int = -1       # -> MDD10A DIR2 (따로 꽂았을 때만)
    osc_hz: int = 25_000_000


@dataclass
class HardwareConfig:
    backend: str = "jetson"
    motor: MotorConfig = field(default_factory=MotorConfig)
    dir_forward_high: bool = True
    servo: ServoConfig = field(default_factory=ServoConfig)
    pca9685: Pca9685Config = field(default_factory=Pca9685Config)
    pca9685_motor: Pca9685MotorConfig = field(default_factory=Pca9685MotorConfig)
    stop_decay_step: int = 5
    stop_decay_delay: float = 0.02


@dataclass
class CameraConfig:
    device: Any = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    fourcc: str = "MJPG"
    jpeg_quality: int = 95
    auto_exposure: Optional[int] = None
    exposure: Optional[int] = None


@dataclass
class RecordingConfig:
    root: str = "dataset"
    interval_sec: float = 0.5
    interval_min: float = 0.1
    interval_max: float = 1.0
    mark_seconds: float = 3.0
    min_free_gb: float = 1.0


@dataclass
class DS4Config:
    device_name: str = "Wireless Controller"
    layout: str = "auto"            # auto | sony | generic
    codes: dict = field(default_factory=dict)   # 역할별 evdev 코드 덮어쓰기 (예: {record: 0x132})
    deadman: bool = False           # false: R2 만으로 주행 (놓으면 정지). true: L1 을 누르고 있는 동안만 주행
    stick_deadzone: float = 0.08
    rumble: bool = True
    lightbar: bool = True
    link_timeout_sec: float = 0.5   # 패드의 HID 보고(초당 수백 개)가 이 시간 동안 없으면 링크 끊김 → 정지. 0 이면 감시 안 함


@dataclass
class InputConfig:
    primary: str = "ds4"
    keyboard_fallback: bool = True
    ds4: DS4Config = field(default_factory=DS4Config)


@dataclass
class SafetyConfig:
    deadman_sec: float = 30.0


@dataclass
class DisplayConfig:
    """상태 표시용 8×8 도트 매트릭스 (MAX7219, Jetson SPI1 직결). 없어도 주행에는 지장이 없다."""
    enabled: bool = True
    driver: str = "max7219"
    spi_dev: str = "/dev/spidev0.0"   # 헤더 핀 19/23/24 = SPI1 CS0
    intensity: int = 3                # 밝기 0~15
    rotate: int = 0                   # 모듈을 돌려 단 방향에 맞춤: 0 | 90 | 180 | 270
    flip: bool = False                # 좌우가 거울처럼 보이면 true
    blink_hz: float = 2.0             # 녹화·오류 표시 깜빡임


@dataclass
class PreprocessConfig:
    crop_top_ratio: float = 0.4
    crop_bottom_ratio: float = 1.0
    out_width: int = 200
    out_height: int = 66


@dataclass
class Config:
    schema_version: int = 1
    steering: SteeringConfig = field(default_factory=SteeringConfig)
    throttle: ThrottleConfig = field(default_factory=ThrottleConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    input: InputConfig = field(default_factory=InputConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    source_path: Optional[str] = None  # 어디서 읽었는지 (스냅샷에는 포함하지 않음)


class ConfigError(ValueError):
    pass


def _build(cls, data: Any, path: str = "") -> Any:
    """dict 를 dataclass 로 재귀 변환. 모르는 키는 오류."""
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"{path or cls.__name__}: 매핑(dict)이어야 합니다. 받은 값: {data!r}")
    known = {f.name: f for f in fields(cls)}
    hints = get_type_hints(cls)  # `from __future__ import annotations` 때문에 f.type 은 문자열
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(f"{path or cls.__name__}: 알 수 없는 키 {sorted(unknown)}. 허용: {sorted(known)}")
    kwargs = {}
    for name in known:
        if name not in data:
            continue
        value = data[name]
        sub_cls = hints.get(name)
        if isinstance(sub_cls, type) and is_dataclass(sub_cls):
            kwargs[name] = _build(sub_cls, value, f"{path}.{name}" if path else name)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def validate(cfg: Config) -> None:
    s = cfg.steering
    if s.mode not in ("discrete", "continuous"):
        raise ConfigError(f"steering.mode 는 discrete 또는 continuous 여야 합니다: {s.mode!r}")
    if not (0 <= s.min < s.max <= 180):
        raise ConfigError(f"steering.min/max 가 올바르지 않습니다: {s.min}, {s.max}")
    if not (s.min <= s.center <= s.max):
        raise ConfigError("steering.center 는 min~max 안에 있어야 합니다")
    classes = list(s.classes)
    if not classes or any(not isinstance(c, int) for c in classes):
        raise ConfigError("steering.classes 는 정수 목록이어야 합니다")
    if len(set(classes)) != len(classes):
        raise ConfigError("steering.classes 에 중복이 있습니다")
    if any(c < s.min or c > s.max for c in classes):
        raise ConfigError("steering.classes 의 값은 모두 min~max 안에 있어야 합니다")
    if s.mode == "discrete" and s.center not in classes:
        raise ConfigError("discrete 모드에서는 steering.center 가 classes 에 포함되어야 합니다")
    if not (0.0 <= s.hysteresis < 0.5):
        raise ConfigError("steering.hysteresis 는 0 이상 0.5 미만이어야 합니다")
    if s.step_deg <= 0:
        raise ConfigError("steering.step_deg 는 양수여야 합니다")
    s.classes = sorted(classes)

    t = cfg.throttle
    for name in ("max_speed", "default_speed", "reverse_max_speed"):
        v = getattr(t, name)
        if not (0 <= v <= 100):
            raise ConfigError(f"throttle.{name} 은 0~100 이어야 합니다: {v}")
    if t.step <= 0:
        raise ConfigError("throttle.step 은 양수여야 합니다")
    lv = t.levels
    if not isinstance(lv, list) or any(isinstance(v, bool) or not isinstance(v, int) for v in lv):
        raise ConfigError(f"throttle.levels 는 정수 목록이어야 합니다: {lv}")
    if lv:
        if lv != sorted(set(lv)) or lv[0] < 1:
            raise ConfigError(f"throttle.levels 는 1 이상의 값을 작은 것부터 중복 없이 적어야 합니다: {lv}")
        if lv[-1] > t.max_speed:
            raise ConfigError(f"throttle.levels 의 최댓값 {lv[-1]} 이 throttle.max_speed {t.max_speed} 보다 큽니다")
    if not (0.0 <= t.trigger_deadzone < 0.9):
        raise ConfigError(f"throttle.trigger_deadzone 은 0 이상 0.9 미만이어야 합니다: {t.trigger_deadzone}")
    if not (0.0 <= t.level_hysteresis < 0.5):
        raise ConfigError(f"throttle.level_hysteresis 는 0 이상 0.5 미만이어야 합니다: {t.level_hysteresis}")

    h = cfg.hardware
    if not (-30.0 <= float(h.servo.backlash_comp_deg) <= 30.0):
        raise ConfigError(f"hardware.servo.backlash_comp_deg 는 -30~30 이어야 합니다: {h.servo.backlash_comp_deg}")
    if not (0.0 <= float(h.servo.backlash_settle_sec) <= 1.0):
        raise ConfigError(f"hardware.servo.backlash_settle_sec 는 0~1 이어야 합니다: {h.servo.backlash_settle_sec}")
    if h.backend not in ("jetson", "mock"):
        raise ConfigError(f"hardware.backend 는 jetson 또는 mock 이어야 합니다: {h.backend!r}")
    mt = h.motor
    if not (24 <= mt.pwm_hz <= 1526):
        raise ConfigError("hardware.motor.pwm_hz 는 PCA9685 가 낼 수 있는 24~1526 Hz 여야 합니다")
    pm = h.pca9685_motor
    if pm.bus < 0:
        raise ConfigError("hardware.pca9685_motor.bus 는 0 이상이어야 합니다")
    if not (0 <= pm.pwm_channel <= 15 and 0 <= pm.dir_channel <= 15 and pm.pwm_channel != pm.dir_channel):
        raise ConfigError("hardware.pca9685_motor 의 pwm_channel/dir_channel 은 0~15 이고 서로 달라야 합니다")
    second = (pm.pwm_channel_2, pm.dir_channel_2)
    if second != (-1, -1):
        used = [pm.pwm_channel, pm.dir_channel, *second]
        if not all(0 <= c <= 15 for c in second) or len(set(used)) != 4:
            raise ConfigError("hardware.pca9685_motor 의 pwm_channel_2/dir_channel_2 는 둘 다 -1 이거나, "
                              "0~15 이면서 네 채널이 모두 달라야 합니다")
    if not (0x03 <= pm.address <= 0x77):
        raise ConfigError(f"hardware.pca9685_motor.address 가 I2C 범위를 벗어납니다: 0x{pm.address:x}")
    if (h.pca9685.bus, h.pca9685.address) == (pm.bus, pm.address):
        raise ConfigError("서보와 모터가 같은 PCA9685(버스/주소)를 쓸 수 없습니다: 보드당 주파수가 하나라 50 Hz 와 모터 PWM 을 "
                          "함께 낼 수 없습니다. 모터용 보드의 A0 패드를 납땜해 0x41 로 만드세요 (docs/wiring.pdf 그림 4)")
    sv = h.servo
    if not (sv.min_us < sv.max_us and sv.min_deg < sv.max_deg):
        raise ConfigError("hardware.servo 의 us/deg 범위가 올바르지 않습니다")
    if not (1 <= sv.pwm_hz <= 1000):
        raise ConfigError("hardware.servo.pwm_hz 는 1~1000 이어야 합니다 (서보 표준 50)")
    pc = h.pca9685
    if not (0 <= pc.channel <= 15):
        raise ConfigError("hardware.pca9685.channel 은 0~15 여야 합니다")
    if not (0x03 <= pc.address <= 0x77):
        raise ConfigError(f"hardware.pca9685.address 가 I2C 범위를 벗어납니다: 0x{pc.address:x}")
    if pc.bus < 0:
        raise ConfigError("hardware.pca9685.bus 는 0 이상이어야 합니다")
    if not (sv.min_deg <= s.min and s.max <= sv.max_deg):
        raise ConfigError(f"steering.min/max ({s.min}~{s.max}) 는 hardware.servo.min_deg/max_deg ({sv.min_deg}~{sv.max_deg}) 안에 있어야 합니다")

    r = cfg.recording
    if not (0 < r.interval_min <= r.interval_sec <= r.interval_max):
        raise ConfigError("recording.interval_min <= interval_sec <= interval_max 이어야 합니다")

    d = cfg.display
    if d.driver not in ("max7219",):
        raise ConfigError(f"display.driver 는 max7219 여야 합니다: {d.driver!r}")
    if not (0 <= d.intensity <= 15):
        raise ConfigError("display.intensity 는 0~15 여야 합니다")
    if d.rotate not in (0, 90, 180, 270):
        raise ConfigError("display.rotate 는 0, 90, 180, 270 중 하나여야 합니다")
    if not (0.2 <= float(d.blink_hz) <= 10):
        raise ConfigError("display.blink_hz 는 0.2~10 이어야 합니다")

    if cfg.input.primary not in ("ds4", "keyboard"):
        raise ConfigError("input.primary 는 ds4 또는 keyboard 여야 합니다")
    if cfg.input.ds4.layout not in ("auto", "sony", "generic"):
        raise ConfigError("input.ds4.layout 은 auto | sony | generic 중 하나여야 합니다")
    if not isinstance(cfg.input.ds4.codes, dict):
        raise ConfigError("input.ds4.codes 는 매핑이어야 합니다")

    p = cfg.preprocess
    if not (0.0 <= p.crop_top_ratio < p.crop_bottom_ratio <= 1.0):
        raise ConfigError("preprocess.crop_top_ratio < crop_bottom_ratio (0~1) 이어야 합니다")


def resolve_config_path(path: Optional[str | Path] = None) -> Path:
    if path is not None:
        return Path(path)
    env = os.environ.get(ENV_CONFIG)
    if env:
        return Path(env)
    return DEFAULT_CONFIG_PATH


def load_config(path: Optional[str | Path] = None) -> Config:
    p = resolve_config_path(path)
    if not p.exists():
        raise ConfigError(f"설정 파일이 없습니다: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = _build(Config, data)
    validate(cfg)
    cfg.source_path = str(p)
    return cfg


def config_snapshot(cfg: Config) -> dict:
    """세션/모델 메타에 넣을 설정 사본 (source_path 제외)."""
    d = asdict(cfg)
    d.pop("source_path", None)
    return d
