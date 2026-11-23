"""설정 읽기와 검증.

값이 적힌 곳은 둘이다.
  - config.yaml          학생이 조정하는 값만 (조향·속도 단계, 내 차에 맞추는 값, 학습 설정).
  - 이 파일의 dataclass   모든 값의 이름·자료형·기본값. config.yaml 에 없는 값은 여기 적힌 기본값을 쓴다
                         (보드 주소, 서보 펄스 폭, 안전 한계처럼 부품과 배선으로 정해져 바꿀 일이 없는 값).
config.yaml 에 없는 값도 같은 이름으로 적으면 그 값이 쓰인다 (배선이 다른 차 등).
지금 쓰이는 값을 모두 보려면:  uv run python -m common.config

사용법:
    from common.config import load_config
    cfg = load_config()                        # 저장소 루트의 config.yaml, 모든 절을 검사
    cfg = load_config(program="collector")     # 그 프로그램이 읽는 절만 검사 (PROGRAM_SECTIONS)
    cfg = load_config("other.yaml")

알 수 없는 키가 있으면 어느 프로그램이든 오류를 낸다(오타 방지).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Union, get_args, get_origin, get_type_hints

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
ENV_CONFIG = "AUTODRIVE_CONFIG"


@dataclass
class SteeringConfig:
    mode: str = "discrete"          # discrete(단계 조향) | continuous(연속 조향)
    classes: list = field(default_factory=lambda: [30, 60, 90, 120, 150])   # 조향 단계 (서보 각도, 도)
    center: int = 90                # 직진 각도. discrete 모드에서는 classes 에 들어 있어야 한다
    min: int = 30                   # 기구 한계: 서보가 이 범위를 넘지 않는다
    max: int = 150
    step_deg: int = 5               # continuous 모드에서 D패드/키보드 한 번에 움직이는 각도
    invert: bool = False            # 스틱을 왼쪽으로 밀었는데 차가 오른쪽으로 꺾이면 true
    hysteresis: float = 0.15        # discrete 모드: 인접 클래스 경계에서 떨림 방지 (클래스 간격 대비 비율)


@dataclass
class ThrottleConfig:
    """구동(DC 모터) 속도. 단위는 PWM 듀티 %."""
    max_speed: int = 60                 # 전진 속도 상한. levels 의 값은 이보다 클 수 없다
    default_speed: int = 40             # 키보드 모드의 처음 속도 (levels 가 있으면 가장 가까운 단계로 맞춘다)
    step: int = 10                      # 키보드 A/Z 한 번당 속도 변화 (levels 가 있으면 한 단계씩)
    reverse_max_speed: int = 30         # 후진 속도. levels 가 있으면 L2 를 당기는 즉시 이 값, 없으면 깊이에 비례
    levels: list = field(default_factory=lambda: [30, 45, 60])   # 전진 속도 단계(듀티 %). 빈 목록이면 연속(트리거 비례)
    trigger_deadzone: float = 0.10      # 트리거를 이만큼 당기기 전에는 정지
    level_hysteresis: float = 0.15      # 단계 경계에서 떨림 방지 (단계 폭 대비 비율)


@dataclass
class ServoConfig:
    pwm_hz: int = 50
    min_us: int = 500                  # LD-1501MG 펄스 폭
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
    """서보와 모터 신호는 모두 I2C(Jetson 핀 3 SDA, 5 SCL = i2c-7)에 연결한 PCA9685 두 장이 낸다. 배선은 docs/wiring.pdf."""
    backend: str = "jetson"         # jetson | mock (collector --mock 이 mock 으로 바꾼다)
    motor: MotorConfig = field(default_factory=MotorConfig)
    dir_forward_high: bool = True   # "전진"인데 뒤로 가면 false 로 (모터선 교환 대신 쓸 수 있다)
    servo: ServoConfig = field(default_factory=ServoConfig)
    pca9685: Pca9685Config = field(default_factory=Pca9685Config)
    pca9685_motor: Pca9685MotorConfig = field(default_factory=Pca9685MotorConfig)
    stop_decay_step: int = 5        # 감속 정지: 한 단계에 줄이는 듀티 %
    stop_decay_delay: float = 0.02  # 감속 단계 사이 시간(초)


@dataclass
class CameraConfig:
    """USB 웹캠. 크기와 JPEG 품질은 수집 → 가공 → 학습이 함께 쓰는 사진 규격이다."""
    device: Any = 0                         # /dev/video0 → 0. 경로 문자열도 가능
    width: int = 640
    height: int = 480
    fps: int = 30
    fourcc: str = "MJPG"
    jpeg_quality: int = 95
    auto_exposure: Optional[float] = None   # None 이면 카메라에 맡긴다. V4L2 값 (1 = 수동, 3 = 자동)
    exposure: Optional[float] = None        # auto_exposure 를 수동으로 둔 경우의 노출값 (단위와 범위는 카메라마다 다르다)


@dataclass
class RecordingConfig:
    root: str = "dataset"           # 세션 폴더 dataset/YYYYMMDD_HHMM/ 이 여기에 생긴다
    interval_sec: float = 0.5       # 저장 간격(초). 키보드 주행 중에는 [ ] 키로 0.1 초씩 바꿀 수 있다
    interval_min: float = 0.1       # interval_sec 가 가질 수 있는 범위
    interval_max: float = 1.0
    mark_seconds: float = 3.0       # □ / M 키: 직전 몇 초를 불량으로 표시할지
    min_free_gb: float = 1.0        # 디스크 여유가 이보다 적으면 시작하지 않는다


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
    primary: str = "ds4"            # ds4 | keyboard (collector --input 으로 한 번만 바꿀 수 있다)
    keyboard_fallback: bool = True  # 패드를 못 찾으면 키보드(SSH 터미널)로 대체
    ds4: DS4Config = field(default_factory=DS4Config)


@dataclass
class SafetyConfig:
    deadman_sec: float = 30.0       # 키보드 모드(토글 주행): 입력 없이 주행이 이 시간을 넘으면 자동 정지. 트리거 주행에는 적용 안 함


@dataclass
class DisplayConfig:
    """상태 표시용 8×8 도트 매트릭스 (MAX7219, Jetson SPI1 직결). 없어도 주행에는 지장이 없다."""
    enabled: bool = True
    driver: str = "max7219"
    spi_dev: str = "/dev/spidev0.0"   # 헤더 핀 19/23/24 = SPI1 CS0
    intensity: int = 3                # 밝기 0~15
    rotate: int = 90                  # 모듈을 돌려 단 방향: 0 | 90 | 180 | 270. 기준 차는 90 (핀 헤더가 위).
                                      #   지금의 화면(○ ● X 전체)은 돌려도 똑같아서 어느 값이든 보이는 것은 같다
    flip: bool = False                # 좌우가 거울처럼 보이면 true
    blink_hz: float = 2.0             # 녹화·오류 표시 깜빡임


@dataclass
class CuratorConfig:
    """가공 도구(PC) 화면 설정."""
    arrow_max_deg: float = 60.0     # 조향 끝(steering.min/max)일 때 화살표가 기울어지는 각도
    arrow_invert: bool = False      # 화살표가 실제 차의 방향과 반대로 보이면 true (steering.invert 와 별개)
    thumb_px: int = 96              # 필름스트립 썸네일 크기
    autoplay_speed: float = 2.0     # 자동 재생 배속의 처음 값 (실제 촬영 시간 기준. 1, 2, 5, 10 중 하나)


@dataclass
class PreprocessConfig:
    """이미지 전처리 (학습·추론 공용). 학습할 때 model_meta.json 에 복사되어 모델과 함께 다닌다.

    자르는 범위는 가공 프로그램에서 사진을 보며 정하고 데이터셋 zip 이 가지고 다닌다 (2026-11-23). 아래 crop_* 는
    가공 프로그램이 처음 보여 주는 선의 위치이고, 범위가 적혀 있지 않은 예전 zip 을 학습할 때의 값이다.
    """
    crop_top_ratio: float = 0.4     # 위에서 이 비율까지 잘라 낸다 (하늘·배경)
    crop_bottom_ratio: float = 1.0  # 이 비율 아래를 잘라 낸다 (1.0 = 자르지 않음)
    out_width: int = 200            # 모델 입력 크기 (PilotNet 200×66)
    out_height: int = 66


@dataclass
class SplitConfig:
    """데이터셋을 학습(train) / 검증(val) / 시험(test) 으로 나누는 방법."""
    train: float = 0.7
    val: float = 0.15
    test: float = 0.15              # 0 이면 시험 분석을 하지 않는다
    stratify: bool = True           # 조향 클래스 비율을 세 묶음에서 같게 유지
    block_size: int = 1             # 연속한 사진 N장을 한 덩어리로 묶어 같은 쪽에 넣는다 (1 = 사진 한 장씩)


@dataclass
class AugmentConfig:
    hflip: Any = "random"           # none | random | both (좌우 반전. 조향 라벨도 함께 뒤집는다)
                                    #   자료형이 Any 인 이유: YAML 은 따옴표 없는 off 를 false 로 읽는다 (validate_training 이 none 으로 바꾼다)


@dataclass
class LoaderConfig:
    batch_size: int = 32
    shuffle: bool = True            # 학습 데이터를 에폭마다 섞을지
    num_workers: int = 0


@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 10              # 검증 손실이 이 에폭 수만큼 나아지지 않으면 멈춘다
    min_delta: float = 0.0          # 이만큼 이상 줄어야 "나아졌다"고 본다


@dataclass
class OptimizerConfig:
    name: str = "adam"              # adam | sgd
    lr: float = 0.001
    weight_decay: float = 0.0
    momentum: float = 0.9           # sgd 에서만 사용
    schedule: str = "constant"      # constant | warmup | decay | warmup_decay
    warmup_epochs: int = 5
    min_lr_ratio: float = 0.01      # decay 의 마지막 학습률 = lr × 이 값


@dataclass
class LossConfig:
    name: str = "cross_entropy"
    class_weight: str = "none"      # none | balanced (사진이 적은 클래스의 오답을 더 크게 벌한다)


@dataclass
class TrainingConfig:
    """학습기(PC, uv run -m trainer) 설정."""
    task: str = "classification"    # 지금은 classification 만 (regression 은 뒤에 추가)
    model: str = "pilotnet"         # trainer/models.py 의 MODELS 에 등록된 이름
    device: str = "auto"            # auto | cuda | cpu
    seed: int = 42
    epochs: int = 50
    rescale: str = "minmax"         # none | minmax | standardize
    output_root: str = "models"
    split: SplitConfig = field(default_factory=SplitConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss: LossConfig = field(default_factory=LossConfig)


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
    curator: CuratorConfig = field(default_factory=CuratorConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
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


# ---------------------------------------------------------------------------
# 검증
# ---------------------------------------------------------------------------
SECTIONS = tuple(name for name, hint in get_type_hints(Config).items() if is_dataclass(hint))   # 절 이름, Config 에 적힌 순서

_KIND = {str: "글자를", list: "목록([a, b, c])을", dict: "'이름: 값' 묶음을"}


def _coerce(obj: Any, path: str) -> None:
    """dataclass 에 적어 둔 자료형대로 값을 맞춘다. 맞출 수 없으면 어느 값이 왜 틀렸는지 ConfigError 로 알린다.

    사람이 고친 YAML 에서는 값을 비워 두거나(None) 수 자리에 글자를 적는 일이 흔하다. 여기서 걸러 두면 뒤의 검사가
    `0 <= None` 같은 비교에서 TypeError 로 죽지 않는다 (TypeError 는 "[설정 오류]" 한 줄이 아니라 영어 오류 더미가 된다).
    """
    hints = get_type_hints(type(obj))
    for f in fields(obj):
        name, value, hint = f"{path}.{f.name}", getattr(obj, f.name), hints.get(f.name)
        if is_dataclass(value):
            _coerce(value, name)
            continue
        if get_origin(hint) is Union:                   # Optional[X]: 비워 두어도 된다
            if value is None:
                continue
            hint = next(a for a in get_args(hint) if a is not type(None))
        if hint is bool:
            value = _flag(name, value)
        elif hint is int:
            value = _number(name, value, int)
        elif hint is float:
            value = _number(name, value)
        elif hint in _KIND and not isinstance(value, hint):
            raise ConfigError(f"{name} 에는 {_KIND[hint]} 적어야 합니다: {value!r}")
        setattr(obj, f.name, value)


def _check_steering(cfg: Config) -> None:
    s = cfg.steering
    if s.mode not in ("discrete", "continuous"):
        raise ConfigError(f"steering.mode 는 discrete 또는 continuous 여야 합니다: {s.mode!r}")
    if not (0 <= s.min < s.max <= 180):
        raise ConfigError(f"steering.min/max 가 올바르지 않습니다: {s.min}, {s.max}")
    if not (s.min <= s.center <= s.max):
        raise ConfigError("steering.center 는 min~max 안에 있어야 합니다")
    classes = list(s.classes)
    if not classes or any(isinstance(c, bool) or not isinstance(c, int) for c in classes):
        raise ConfigError(f"steering.classes 는 정수 목록이어야 합니다: {classes}")
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


def _check_throttle(cfg: Config) -> None:
    t = cfg.throttle
    for name in ("max_speed", "default_speed", "reverse_max_speed"):
        v = getattr(t, name)
        if not (0 <= v <= 100):
            raise ConfigError(f"throttle.{name} 은 0~100 이어야 합니다: {v}")
    if t.step <= 0:
        raise ConfigError("throttle.step 은 양수여야 합니다")
    lv = t.levels
    if any(isinstance(v, bool) or not isinstance(v, int) for v in lv):
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


def _check_hardware(cfg: Config) -> None:
    h = cfg.hardware
    if not (-30.0 <= h.servo.backlash_comp_deg <= 30.0):
        raise ConfigError(f"hardware.servo.backlash_comp_deg 는 -30~30 이어야 합니다: {h.servo.backlash_comp_deg}")
    if not (0.0 <= h.servo.backlash_settle_sec <= 1.0):
        raise ConfigError(f"hardware.servo.backlash_settle_sec 는 0~1 이어야 합니다: {h.servo.backlash_settle_sec}")
    if h.backend not in ("jetson", "mock"):
        raise ConfigError(f"hardware.backend 는 jetson 또는 mock 이어야 합니다: {h.backend!r}")
    if h.stop_decay_step < 1 or not (0.0 <= h.stop_decay_delay <= 0.1) \
            or (100.0 / h.stop_decay_step) * h.stop_decay_delay > 0.6:
        raise ConfigError("hardware.stop_decay_step 은 1 이상, stop_decay_delay 는 0~0.1 초이고, 100 % 에서 서는 데 0.6초를 "
                          "넘으면 안 됩니다 (감속하는 동안 조종이 멈춘다): "
                          f"step={h.stop_decay_step}, delay={h.stop_decay_delay}")
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


def _check_steering_fits_servo(cfg: Config) -> None:
    """두 절에 걸친 검사: steering 과 hardware 를 둘 다 검사한 뒤에만 부른다."""
    s, sv = cfg.steering, cfg.hardware.servo
    if not (sv.min_deg <= s.min and s.max <= sv.max_deg):
        raise ConfigError(f"steering.min/max ({s.min}~{s.max}) 는 hardware.servo.min_deg/max_deg ({sv.min_deg}~{sv.max_deg}) 안에 있어야 합니다")


def _check_recording(cfg: Config) -> None:
    r = cfg.recording
    if not (0 < r.interval_min <= r.interval_max):
        raise ConfigError(f"recording.interval_min 은 0 보다 크고 interval_max 이하여야 합니다: {r.interval_min:g}, {r.interval_max:g}")
    if not (r.interval_min <= r.interval_sec <= r.interval_max):
        raise ConfigError(f"recording.interval_sec 는 {r.interval_min:g}~{r.interval_max:g} 초여야 합니다: {r.interval_sec:g}")
    if not (0 < r.mark_seconds <= 60):
        raise ConfigError(f"recording.mark_seconds 는 0 보다 크고 60 이하여야 합니다: {r.mark_seconds:g}")
    if r.min_free_gb < 0:
        raise ConfigError(f"recording.min_free_gb 는 0 이상이어야 합니다: {r.min_free_gb:g}")


def _check_input(cfg: Config) -> None:
    if cfg.input.primary not in ("ds4", "keyboard"):
        raise ConfigError("input.primary 는 ds4 또는 keyboard 여야 합니다")
    if cfg.input.ds4.layout not in ("auto", "sony", "generic"):
        raise ConfigError("input.ds4.layout 은 auto | sony | generic 중 하나여야 합니다")
    lt = cfg.input.ds4.link_timeout_sec
    if lt != 0 and not (0.2 <= lt <= 10):
        raise ConfigError(f"input.ds4.link_timeout_sec 는 0(끔) 또는 0.2~10 초여야 합니다: {lt} "
                          "(너무 짧으면 블루투스가 잠깐 늦을 때마다 차가 선다)")


def _check_display(cfg: Config) -> None:
    d = cfg.display
    if d.driver not in ("max7219",):
        raise ConfigError(f"display.driver 는 max7219 여야 합니다: {d.driver!r}")
    if not (0 <= d.intensity <= 15):
        raise ConfigError("display.intensity 는 0~15 여야 합니다")
    if d.rotate not in (0, 90, 180, 270):
        raise ConfigError("display.rotate 는 0, 90, 180, 270 중 하나여야 합니다")
    if not (0.2 <= d.blink_hz <= 10):
        raise ConfigError("display.blink_hz 는 0.2~10 이어야 합니다")


def _check_curator(cfg: Config) -> None:
    cu = cfg.curator
    if not (1 <= cu.arrow_max_deg <= 90) or cu.thumb_px < 32 or cu.autoplay_speed not in (1, 2, 5, 10):
        raise ConfigError("curator 설정 범위: arrow_max_deg 1~90, thumb_px ≥ 32, autoplay_speed 1·2·5·10 중 하나")


def _check_preprocess(cfg: Config) -> None:
    p = cfg.preprocess
    if not (0.0 <= p.crop_top_ratio < p.crop_bottom_ratio <= 1.0):
        raise ConfigError("preprocess.crop_top_ratio < crop_bottom_ratio (0~1) 이어야 합니다")
    if p.out_width < 1 or p.out_height < 1:
        raise ConfigError(f"preprocess.out_width / out_height 는 1 이상이어야 합니다: {p.out_width}, {p.out_height}")


def _check_training(cfg: Config) -> None:
    validate_training(cfg.training)


# 절 이름 → 그 절의 값 범위 검사. 여기에 없는 절(camera, safety)은 자료형만 본다 (_coerce).
_CHECKS = {
    "steering": _check_steering, "throttle": _check_throttle, "hardware": _check_hardware, "recording": _check_recording,
    "input": _check_input, "display": _check_display, "curator": _check_curator, "preprocess": _check_preprocess,
    "training": _check_training,
}

# 프로그램마다 자기가 읽는 절만 검사한다. 차(수집기)가 PC 에서만 쓰는 학습 값의 오타 때문에 출발하지 못하거나, PC 프로그램이
# 차에서만 쓰는 값 때문에 열리지 않는 일이 없게 한다. 모르는 키와 YAML 문법 오류는 어느 프로그램이든 파일 전체에서 잡는다 (_build).
PROGRAM_SECTIONS = {
    "collector": ("steering", "throttle", "hardware", "camera", "recording", "input", "safety", "display"),
    "curator": ("steering", "throttle", "camera", "recording", "curator", "preprocess"),
    "trainer": ("preprocess", "training"),
}


def validate(cfg: Config, sections: Optional[Iterable[str]] = None) -> None:
    """sections 에 든 절만 검사한다 (None 이면 전부). 검사하면서 수는 int/float 로 바꿔 다시 넣는다."""
    wanted = set(SECTIONS if sections is None else sections)
    if wanted - set(SECTIONS):
        raise ValueError(f"없는 절 이름입니다: {sorted(wanted - set(SECTIONS))}. 있는 절: {list(SECTIONS)}")
    for name in SECTIONS:
        if name in wanted:
            _coerce(getattr(cfg, name), name)
            if name in _CHECKS:
                _CHECKS[name](cfg)
    if {"steering", "hardware"} <= wanted:
        _check_steering_fits_servo(cfg)


def _choice(name: str, value, allowed) -> None:
    if value not in allowed:
        raise ConfigError(f"{name} 은(는) {' | '.join(map(str, allowed))} 중 하나여야 합니다: {value!r}")


def _number(name: str, value, kind=float):
    """수로 바꿔 돌려준다. YAML 은 소수점 없는 `1e-3` 을 문자열로 읽으므로 문자열로 적힌 수도 받는다."""
    if value is None or isinstance(value, bool):
        raise ConfigError(f"{name} 에는 수를 적어야 합니다: {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{name} 에는 수를 적어야 합니다: {value!r}") from None
    if number != number or number in (float("inf"), float("-inf")):
        raise ConfigError(f"{name} 에는 유한한 수를 적어야 합니다: {value!r}")
    if kind is int:
        if number != int(number):
            raise ConfigError(f"{name} 에는 정수를 적어야 합니다: {value!r}")
        return int(number)
    return number


def _flag(name: str, value) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} 은(는) true 또는 false 여야 합니다: {value!r}")
    return value


def validate_training(t: TrainingConfig) -> None:
    """검사하면서 수는 int/float 로 바꿔 다시 넣는다 (뒤에서 문자열 '1e-3' 이 옵티마이저까지 흘러가지 않게)."""
    if t.augment.hflip is False or t.augment.hflip == "off":    # YAML 은 따옴표 없는 off 를 false 로 읽는다
        t.augment.hflip = "none"
    _choice("training.task", t.task, ("classification",))
    _choice("training.device", t.device, ("auto", "cuda", "cpu"))
    _choice("training.rescale", t.rescale, ("none", "minmax", "standardize"))
    _choice("training.augment.hflip", t.augment.hflip, ("none", "random", "both"))
    _choice("training.optimizer.name", t.optimizer.name, ("adam", "sgd"))
    _choice("training.optimizer.schedule", t.optimizer.schedule, ("constant", "warmup", "decay", "warmup_decay"))
    _choice("training.loss.name", t.loss.name, ("cross_entropy",))
    _choice("training.loss.class_weight", t.loss.class_weight, ("none", "balanced"))
    if not isinstance(t.model, str) or not t.model:
        raise ConfigError(f"training.model 은 모델 이름(문자열)이어야 합니다: {t.model!r}")
    if not isinstance(t.output_root, str) or not t.output_root:
        raise ConfigError(f"training.output_root 는 폴더 경로(문자열)여야 합니다: {t.output_root!r}")
    t.seed = _number("training.seed", t.seed, int)
    if not (0 <= t.seed <= 2 ** 32 - 1):        # numpy 가 받는 범위. 벗어나면 학습을 시작할 때 영어 오류로 죽는다
        raise ConfigError(f"training.seed 는 0~4294967295 의 정수여야 합니다: {t.seed}")
    t.epochs = _number("training.epochs", t.epochs, int)
    if not (1 <= t.epochs <= 10000):
        raise ConfigError(f"training.epochs 는 1~10000 이어야 합니다: {t.epochs}")

    sp = t.split
    sp.train, sp.val, sp.test = (_number(f"training.split.{n}", getattr(sp, n)) for n in ("train", "val", "test"))
    ratios = (sp.train, sp.val, sp.test)
    if min(ratios) < 0:
        raise ConfigError(f"training.split 의 train/val/test 는 0 이상이어야 합니다: {ratios}")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ConfigError(f"training.split 의 train + val + test 는 1 이어야 합니다: {ratios} (합 {sum(ratios):.3f})")
    if sp.train <= 0 or sp.val <= 0:
        raise ConfigError("training.split 의 train 과 val 은 0 보다 커야 합니다 (검증 손실로 가장 좋은 모델을 고른다)")
    sp.stratify = _flag("training.split.stratify", sp.stratify)
    sp.block_size = _number("training.split.block_size", sp.block_size, int)
    if sp.block_size < 1:
        raise ConfigError(f"training.split.block_size 는 1 이상이어야 합니다: {sp.block_size}")

    lo = t.loader
    lo.batch_size = _number("training.loader.batch_size", lo.batch_size, int)
    lo.num_workers = _number("training.loader.num_workers", lo.num_workers, int)
    lo.shuffle = _flag("training.loader.shuffle", lo.shuffle)
    if not (1 <= lo.batch_size <= 4096) or lo.num_workers < 0:
        raise ConfigError("training.loader: batch_size 는 1~4096, num_workers 는 0 이상이어야 합니다")

    es = t.early_stopping
    es.enabled = _flag("training.early_stopping.enabled", es.enabled)
    es.patience = _number("training.early_stopping.patience", es.patience, int)
    es.min_delta = _number("training.early_stopping.min_delta", es.min_delta)
    if es.patience < 1 or es.min_delta < 0:
        raise ConfigError("training.early_stopping: patience 는 1 이상, min_delta 는 0 이상이어야 합니다")

    o = t.optimizer
    o.lr = _number("training.optimizer.lr", o.lr)
    o.weight_decay = _number("training.optimizer.weight_decay", o.weight_decay)
    o.momentum = _number("training.optimizer.momentum", o.momentum)
    o.warmup_epochs = _number("training.optimizer.warmup_epochs", o.warmup_epochs, int)
    o.min_lr_ratio = _number("training.optimizer.min_lr_ratio", o.min_lr_ratio)
    if not (0 < o.lr <= 10):
        raise ConfigError(f"training.optimizer.lr 는 0 보다 크고 10 이하여야 합니다: {o.lr}")
    if o.weight_decay < 0 or not (0 <= o.momentum < 1):
        raise ConfigError("training.optimizer: weight_decay 는 0 이상, momentum 은 0 이상 1 미만이어야 합니다")
    if o.warmup_epochs < 0 or not (0 <= o.min_lr_ratio <= 1):
        raise ConfigError("training.optimizer: warmup_epochs 는 0 이상, min_lr_ratio 는 0~1 이어야 합니다")


def resolve_config_path(path: Optional[str | Path] = None) -> Path:
    if path is not None:
        return Path(path)
    env = os.environ.get(ENV_CONFIG)
    if env:
        return Path(env)
    return DEFAULT_CONFIG_PATH


def _read_yaml(path: Optional[str | Path] = None) -> tuple:
    """(파일 경로, 파일을 읽은 dict)"""
    p = resolve_config_path(path)
    if not p.exists():
        raise ConfigError(f"설정 파일이 없습니다: {p}")
    try:
        text = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise ConfigError(f"{p.name} 을 읽을 수 없습니다 (UTF-8 로 저장한 글 파일이어야 합니다): {e}") from None
    # `git pull --autostash` 로 새 파일을 받았는데 직접 고친 줄과 겹치면 git 이 파일 안에 두 내용을 나란히 남긴다.
    # 그대로 두면 "YAML 문법 오류"로만 보여 무엇을 해야 할지 알 수 없다.
    if any(line.startswith(("<<<<<<<", ">>>>>>>")) for line in text.splitlines()):
        raise ConfigError(f"{p.name} 에 git 충돌 표시(<<<<<<<)가 남아 있습니다: 새로 받은 파일과 직접 고친 값이 같은 줄에서 겹쳤습니다. "
                          f"`git checkout HEAD -- {p.name}` 로 새 파일로 되돌린 뒤 값을 다시 적으세요 "
                          "(고쳐 두었던 값은 `git stash show -p` 로 볼 수 있습니다)")
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"{p.name} 의 YAML 문법이 올바르지 않습니다: {e}") from None
    return p, data


def load_config(path: Optional[str | Path] = None, program: Optional[str] = None) -> Config:
    """설정 파일을 읽어 검사한다. program 을 주면 그 프로그램이 읽는 절만 검사한다 (PROGRAM_SECTIONS). 주지 않으면 모든 절."""
    p, data = _read_yaml(path)
    cfg = _build(Config, data)
    validate(cfg, PROGRAM_SECTIONS[program] if program else None)
    cfg.source_path = str(p)
    return cfg


def config_snapshot(cfg: Config) -> dict:
    """세션/모델 메타에 넣을 설정 사본 (source_path 제외). JSON 으로 쓸 수 있는 값만 담는다.

    프로그램이 검사하지 않는 절에는 YAML 이 날짜로 읽은 값처럼 json 이 받지 않는 것이 남아 있을 수 있다. 그런 값은 글자로 바꾼다
    (수집기가 세션 기록을 쓰다가 죽지 않게).
    """
    d = asdict(cfg)
    d.pop("source_path", None)
    return json.loads(json.dumps(d, default=str))


# ---------------------------------------------------------------------------
# 전체 값 보기:  uv run python -m common.config
# config.yaml 에는 학생이 조정하는 값만 있으므로, 나머지 값이 지금 얼마인지는 이 명령으로 확인한다.
# ---------------------------------------------------------------------------
def _show(key: str, value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if key == "address" and isinstance(value, int):
        return f"0x{value:02x}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def describe(cfg: Config, raw: Optional[dict] = None) -> str:
    """모든 값을 절별로 적은 글. raw(설정 파일을 읽은 dict)에 적혀 있는 값은 줄 앞에 * 를 붙인다."""
    lines: list = []

    def walk(obj: Any, given: Any, depth: int) -> None:
        given = given if isinstance(given, dict) else {}
        for f in fields(obj):
            value, pad = getattr(obj, f.name), "  " * depth
            if is_dataclass(value):
                lines.append(f"  {pad}{f.name}:")
                walk(value, given.get(f.name), depth + 1)
            else:
                lines.append(f"{'*' if f.name in given else ' '} {pad}{f.name}: {_show(f.name, value)}")

    for name in SECTIONS:
        lines.append(f"  {name}:")
        walk(getattr(cfg, name), (raw or {}).get(name), 1)
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="python -m common.config",
                                description="모든 설정 값과, 그 값이 config.yaml 에 적힌 것인지 기본값인지 보여 준다")
    p.add_argument("--config", help="config.yaml 경로 (기본: 저장소 루트)")
    args = p.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):             # 한글을 못 찍는 터미널에서도 죽지 않게
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    try:
        path, raw = _read_yaml(args.config)
        cfg = load_config(path)
    except ConfigError as e:
        print(f"[설정 오류] {e}", file=sys.stderr)
        return 1
    print(f"설정 파일: {path}")
    print("* = 이 파일에 적힌 값. 나머지는 기본값이며, 같은 이름으로 파일에 적으면 그 값이 쓰입니다.\n")
    print(describe(cfg, raw))
    return 0


if __name__ == "__main__":
    sys.exit(main())
