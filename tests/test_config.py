from pathlib import Path

import pytest
import yaml

from common.config import DEFAULT_CONFIG_PATH, ConfigError, config_snapshot, load_config


def test_default_config_loads():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert cfg.steering.mode == "discrete"
    assert cfg.steering.classes == [30, 60, 90, 120, 150]
    assert (cfg.hardware.pca9685.bus, cfg.hardware.pca9685.address) == (7, 0x40)
    assert (cfg.hardware.pca9685_motor.bus, cfg.hardware.pca9685_motor.address) == (7, 0x41)
    snap = config_snapshot(cfg)
    assert "source_path" not in snap and snap["steering"]["center"] == 90


def _write(tmp_path: Path, override: dict) -> Path:
    base = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    for k, v in override.items():
        section, key = k.split(".")
        base[section][key] = v
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(base), encoding="utf-8")
    return p


def test_unknown_key_rejected(tmp_path):
    p = _write(tmp_path, {"steering.clases": [1]})
    with pytest.raises(ConfigError, match="알 수 없는 키"):
        load_config(p)


def test_center_must_be_a_class_in_discrete(tmp_path):
    p = _write(tmp_path, {"steering.center": 100})
    with pytest.raises(ConfigError):
        load_config(p)


def test_classes_sorted(tmp_path):
    p = _write(tmp_path, {"steering.classes": [150, 90, 30], "steering.center": 90})
    assert load_config(p).steering.classes == [30, 90, 150]


def test_pwm_hz_limit(tmp_path):
    """모터 PWM 은 PCA9685 가 낼 수 있는 24~1526 Hz 안이어야 한다."""
    for bad in (5000, 10):
        p = _write(tmp_path, {"hardware.motor": {"pwm_hz": bad}})
        with pytest.raises(ConfigError):
            load_config(p)


def test_header_pwm_keys_are_gone(tmp_path):
    """헤더 직결 시절의 설정 키(pins, driver, pwm_check)는 더 이상 받지 않는다."""
    for override in ({"hardware.pins": {"servo_pwm": 15}}, {"hardware.motor": {"driver": "pca9685", "pwm_hz": 1500}},
                     {"safety.pwm_check": "fail"}):
        p = _write(tmp_path, override)
        with pytest.raises(ConfigError, match="알 수 없는 키"):
            load_config(p)
