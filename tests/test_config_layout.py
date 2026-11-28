"""config.yaml 에는 학생이 조정하는 값만 두고 나머지는 코드의 기본값(common/config.py)이 맡는다 (2026-11-23).

여기서 지키는 것:
  - 파일에서 뺀 값은 기본값으로 똑같이 쓰이고, 예전처럼 모든 값을 적은 파일도 그대로 읽힌다 (차마다 config.yaml 이 다르다).
  - 프로그램은 자기가 읽는 절만 검사한다: PC 에서만 쓰는 학습 값의 오타가 차를 세우지 않는다.
  - 어느 값을 잘못 적어도 TypeError 가 아니라 그 키 이름이 든 ConfigError 로 끝난다.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Union, get_origin, get_type_hints

import pytest
import yaml

from common.config import (DEFAULT_CONFIG_PATH, PROGRAM_SECTIONS, REPO_ROOT, SECTIONS, Config, ConfigError,
                           config_snapshot, describe, load_config, main, validate)

OLD_STUDENT_CONFIG = Path(__file__).parent / "data" / "config_before_20261123.yaml"   # 모든 값을 적던 때 학생에게 나간 파일

# 학생이 조정하는 값. config.yaml 에는 이 값들만 있어야 한다. 값을 넣거나 뺄 때는 여기도 함께 고친다 (일부러 정한 목록이다).
# hardware 절의 값은 하나도 없다: 학생은 하드웨어 설정을 건드리지 않는다 (2026-11-23).
STUDENT_KEYS = {
    # 차: 운전과 촬영
    "steering.mode", "steering.classes", "steering.center", "steering.min", "steering.max", "steering.invert",
    "throttle.levels", "throttle.max_speed", "throttle.reverse_max_speed",
    "recording.interval_sec", "recording.mark_seconds",
    "camera.auto_exposure", "camera.exposure",
    "input.ds4.deadman",
    # PC: 전처리와 학습. 사진에서 자를 범위(preprocess.crop_*)는 가공 프로그램에서 정하므로 파일에 없다 (2026-11-23)
    "preprocess.out_width", "preprocess.out_height",
    "training.model", "training.seed", "training.epochs", "training.rescale",
    "training.split.train", "training.split.val", "training.split.test", "training.split.stratify", "training.split.block_size",
    "training.augment.hflip", "training.loader.batch_size", "training.loader.shuffle",
    "training.early_stopping.enabled", "training.early_stopping.patience",
    "training.optimizer.name", "training.optimizer.lr", "training.optimizer.weight_decay", "training.optimizer.momentum",
    "training.optimizer.schedule", "training.optimizer.warmup_epochs", "training.optimizer.min_lr_ratio",
    "training.loss.class_weight",
}


def _raw() -> dict:
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def _leaves(tree: dict, prefix: str = "") -> list:
    out = []
    for key, value in tree.items():
        out += _leaves(value, f"{prefix}{key}.") if isinstance(value, dict) and value else [prefix + key]
    return out


def _typed_leaves(cls=Config, prefix: str = "") -> list:
    """코드에 정의된 모든 값: (키 경로, 자료형)."""
    out, hints = [], get_type_hints(cls)
    for f in fields(cls):
        if is_dataclass(hints[f.name]):
            out += _typed_leaves(hints[f.name], f"{prefix}{f.name}.")
        elif prefix:                                    # 맨 위의 schema_version, source_path 는 절이 아니다
            out.append((prefix + f.name, hints[f.name]))
    return out


def _nested(path: str, value) -> dict:
    """'a.b.c', v → {a: {b: {c: v}}}"""
    for key in reversed(path.split(".")):
        value = {key: value}
    return value


def _write(tmp_path: Path, tree: dict, name: str = "c.yaml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(tree, allow_unicode=True), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# 파일에는 학생이 조정하는 값만, 나머지는 기본값
# ---------------------------------------------------------------------------
def test_config_yaml_holds_only_the_values_students_adjust():
    in_file = set(_leaves(_raw()))
    assert in_file == STUDENT_KEYS, f"더 있음 {sorted(in_file - STUDENT_KEYS)}, 빠짐 {sorted(STUDENT_KEYS - in_file)}"
    assert STUDENT_KEYS < {path for path, _ in _typed_leaves()}       # 모두 실제로 있는 키이고, 파일에 없는 값이 더 있다


def test_values_in_config_yaml_equal_the_defaults_in_code():
    """파일에 남긴 값도 코드에 기본값으로 한 번 더 적혀 있다. 둘이 어긋나면 파일에서 한 줄을 지웠을 때 차의 동작이 달라진다."""
    defaults = Config()
    validate(defaults)
    assert config_snapshot(load_config(DEFAULT_CONFIG_PATH)) == config_snapshot(defaults)


def test_an_empty_file_and_a_file_with_every_value_give_the_same_config(tmp_path):
    shipped = config_snapshot(load_config(DEFAULT_CONFIG_PATH))
    empty = tmp_path / "empty.yaml"
    empty.write_text("# 아무 값도 적지 않음\n", encoding="utf-8")
    assert config_snapshot(load_config(empty)) == shipped
    assert config_snapshot(load_config(_write(tmp_path, shipped, "full.yaml"))) == shipped


def test_the_config_file_students_already_have_still_works():
    """git pull 을 하지 않았거나 예전 파일을 그대로 둔 차도 새 코드로 출발해야 하고, 값도 하나도 달라지면 안 된다."""
    old = yaml.safe_load(OLD_STUDENT_CONFIG.read_text(encoding="utf-8"))
    assert "pca9685_motor" in old["hardware"] and "training" not in old        # 정말 예전 모양의 파일인지
    for program in (None, *PROGRAM_SECTIONS):
        assert config_snapshot(load_config(OLD_STUDENT_CONFIG, program=program)) == \
            config_snapshot(load_config(DEFAULT_CONFIG_PATH, program=program))


def test_a_value_missing_from_the_file_can_still_be_written(tmp_path):
    """배선이 다른 차의 비상구: 파일에서 뺀 값도 같은 이름으로 적으면 그 값이 쓰인다."""
    cfg = load_config(_write(tmp_path, {"hardware": {"pca9685_motor": {"pwm_channel_2": 2, "dir_channel_2": 3}},
                                        "display": {"enabled": False}, "input": {"ds4": {"codes": {"record": 0x132}}}}))
    pm = cfg.hardware.pca9685_motor
    assert (pm.pwm_channel, pm.dir_channel, pm.pwm_channel_2, pm.dir_channel_2) == (0, 1, 2, 3) and pm.address == 0x41
    assert cfg.display.enabled is False and cfg.input.ds4.codes == {"record": 0x132}
    assert cfg.input.ds4.link_timeout_sec == 0.5                    # 같은 절의 다른 값은 기본값 그대로


# ---------------------------------------------------------------------------
# 프로그램마다 자기가 읽는 절만 검사
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tree, stopped", [
    ({"training": {"epochs": "many"}}, {"trainer"}),
    ({"training": {"optimizer": {"lr": 0}}}, {"trainer"}),
    ({"preprocess": {"out_height": 0}}, {"trainer", "curator"}),        # 가공 프로그램도 "모델이 보는 사진"을 그리려고 읽는다
    ({"preprocess": {"crop_top_ratio": 1.0}}, {"trainer", "curator"}),
    ({"hardware": {"motor": {"pwm_hz": 5000}}}, {"collector", "inference"}),      # 차에서 도는 두 프로그램
    ({"input": {"ds4": {"link_timeout_sec": 0.01}}}, {"collector", "inference"}),
    ({"display": {"rotate": 45}}, {"collector", "inference"}),
    ({"safety": {"deadman_sec": "soon"}}, {"collector", "inference"}),
    ({"curator": {"thumb_px": 8}}, {"curator"}),
    ({"steering": {"center": 100}}, {"collector", "inference", "curator"}),
    ({"throttle": {"levels": [60, 30]}}, {"collector", "inference", "curator"}),
    ({"recording": {"interval_sec": 5}}, {"collector", "inference", "curator"}),
    ({"camera": {"width": "wide"}}, {"collector", "inference", "curator"}),
    ({"inference": {"smoothing": "average"}}, {"inference"}),                       # 추론 값의 오타는 수집기를 세우지 않는다
    ({"inference": {"window": 0}}, {"inference"}),
    ({"inference": {"max_frame_age_sec": 10}}, {"inference"}),
])
def test_each_program_checks_only_the_sections_it_reads(tmp_path, tree, stopped):
    p = _write(tmp_path, tree)
    for program in PROGRAM_SECTIONS:
        if program in stopped:
            with pytest.raises(ConfigError):
                load_config(p, program=program)
        else:
            load_config(p, program=program)             # 이 프로그램은 그 값을 읽지 않는다: 시작해야 한다
    with pytest.raises(ConfigError):
        load_config(p)                                  # 프로그램을 지정하지 않으면 전부 검사한다


def test_unknown_keys_and_broken_yaml_stop_every_program(tmp_path):
    """오타 난 키 이름은 '조용히 무시된 설정'이 되므로, 어느 절에 있든 모든 프로그램이 알린다."""
    typo = _write(tmp_path, {"training": {"optimizer": {"learning_rate": 0.1}}}, "typo.yaml")
    broken = tmp_path / "broken.yaml"
    broken.write_text("training:\n  epochs: [1, 2\n", encoding="utf-8")
    for program in (None, *PROGRAM_SECTIONS):
        with pytest.raises(ConfigError, match="알 수 없는 키"):
            load_config(typo, program=program)
        with pytest.raises(ConfigError, match="YAML"):
            load_config(broken, program=program)
    with pytest.raises(KeyError):
        load_config(DEFAULT_CONFIG_PATH, program="autopilot")       # 없는 프로그램 이름: 조용히 넘어가지 않는다
    with pytest.raises(ValueError, match="없는 절"):
        validate(Config(), ["steering", "trainning"])


def test_every_section_a_program_reads_is_checked_for_it():
    """프로그램이 새 절을 읽기 시작했는데 PROGRAM_SECTIONS 에 넣지 않으면, 검사하지 않은 값을 쓰게 된다."""
    pattern = re.compile(r"\bcfg\.(%s)\b" % "|".join(SECTIONS))
    for program, sections in PROGRAM_SECTIONS.items():
        read = set()
        for source in (REPO_ROOT / program).rglob("*.py"):
            read |= set(pattern.findall(source.read_text(encoding="utf-8")))
        assert read, program                            # 찾는 방법이 틀려 아무것도 못 찾은 것은 아닌지
        assert read <= set(sections), f"{program} 이 읽지만 검사하지 않는 절: {sorted(read - set(sections))}"


def test_collector_runs_with_broken_pc_only_values_and_still_writes_its_session_record(tmp_path, monkeypatch):
    """차의 config.yaml 에서 2부(학습 값)가 틀려 있어도 수집기는 달리고, 세션 기록(JSON)도 쓴다.
    검사하지 않은 절에는 YAML 이 날짜로 읽은 값처럼 json 이 받지 않는 것이 남을 수 있다."""
    import collector.app as app_mod
    import common.driveapp as drive_mod
    from common.events import EventKind as K, InputEvent as E
    from test_session_and_collector import ScriptedInput

    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    for old, new in (("epochs: 50 ", "epochs: many "), ("seed: 42 ", "seed: 2026-11-23 "), ("lr: 0.001 ", "lr: ")):
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    p = tmp_path / "config.yaml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="training"):
        load_config(p)                                  # 전부 검사하면 걸리는 파일이다

    cfg = load_config(p, program="collector")
    cfg.recording.root = str(tmp_path / "dataset")
    cfg.recording.interval_sec = cfg.recording.interval_min = 0.1
    scripted = ScriptedInput([E(K.DEADMAN, True), E(K.THROTTLE_AXIS, 1.0), E(K.RECORD_TOGGLE)])
    monkeypatch.setattr(drive_mod, "create_input", lambda cfg_, kind: scripted)
    col = app_mod.Collector(cfg, mock=True, skip_checks=True)
    t_end = time.monotonic() + 0.4
    while time.monotonic() < t_end:
        col._step(time.monotonic())
        time.sleep(0.02)
    col.logic.state.quit = True
    col.close()

    meta = json.loads((col.session.path / "session_meta.json").read_text(encoding="utf-8"))
    assert meta["frames_saved"] >= 1 and meta["config"]["steering"]["classes"] == [30, 60, 90, 120, 150]
    assert meta["config"]["training"]["epochs"] == "many" and meta["config"]["training"]["seed"] == "2026-11-23"


def test_collector_reports_a_car_side_mistake_in_one_line(tmp_path, capsys):
    from collector.__main__ import main as collector_main
    assert collector_main(["--config", str(_write(tmp_path, {"steering": {"min": None}})), "--mock", "--check"]) == 1
    err = capsys.readouterr().err
    assert "[설정 오류] steering.min" in err and "Traceback" not in err


# ---------------------------------------------------------------------------
# 잘못 적은 값
# ---------------------------------------------------------------------------
def _wrong_values(hint) -> list:
    if get_origin(hint) is Union:                       # Optional[float]: 비워 두는 것(None)은 된다
        return ["abc", [1], True]
    return {bool: ["abc", 3, None, [1]], int: ["abc", None, [1], 1.5, True], float: ["abc", None, [1], True],
            str: [3, None, [1]], list: ["abc", 3, None], dict: ["abc", 3, [1], None]}.get(hint, [])


def test_a_wrongly_typed_value_is_a_config_error_for_every_key(tmp_path):
    """값을 비워 두거나 수 자리에 글자를 적는 것은 가장 흔한 실수다. 어느 키에서든 TypeError 로 죽지 않고, 그 키 이름이 든
    ConfigError 로 끝나야 한다 (프로그램은 ConfigError 만 "[설정 오류]" 한 줄로 바꿔 보여 준다)."""
    problems, checked = [], 0
    for path, hint in _typed_leaves():
        for bad in _wrong_values(hint):
            checked += 1
            p = _write(tmp_path, _nested(path, bad))
            try:
                load_config(p)
            except ConfigError as e:
                if path not in str(e):
                    problems.append(f"{path} = {bad!r}: 오류 문구에 키 이름이 없다: {e}")
            except Exception as e:                      # noqa: BLE001  (바로 이것을 잡으려는 테스트)
                problems.append(f"{path} = {bad!r}: {type(e).__name__}: {e}")
            else:
                problems.append(f"{path} = {bad!r}: 받아들였다")
    assert not problems, "\n".join(problems)
    assert checked > 300 and len(STUDENT_KEYS) == 38


def test_numbers_are_normalised_for_car_values_too(tmp_path):
    """수는 적힌 모양과 상관없이 정해진 자료형으로 맞춘다: 따옴표가 붙은 수, 소수점 없는 실수, .0 이 붙은 정수."""
    cfg = load_config(_write(tmp_path, {"steering": {"center": "90"}, "recording": {"interval_sec": 1},
                                        "throttle": {"max_speed": 60.0}, "camera": {"auto_exposure": 1, "exposure": 120}}))
    assert cfg.steering.center == 90 and isinstance(cfg.steering.center, int)
    assert cfg.recording.interval_sec == 1.0 and isinstance(cfg.recording.interval_sec, float)
    assert cfg.throttle.max_speed == 60 and isinstance(cfg.throttle.max_speed, int)
    assert (cfg.camera.auto_exposure, cfg.camera.exposure) == (1.0, 120.0)
    assert load_config(DEFAULT_CONFIG_PATH).camera.exposure is None         # 비워 두면 카메라에 맡긴다


@pytest.mark.parametrize("tree, words", [
    ({"recording": {"interval_sec": 2.0}}, "0.1~1 초"),         # 파일에 없는 interval_min / max 가 아니라 범위를 말해 준다
    ({"recording": {"interval_sec": 0.05}}, "0.1~1 초"),
    ({"recording": {"mark_seconds": 0}}, "recording.mark_seconds"),
    ({"recording": {"mark_seconds": 600}}, "recording.mark_seconds"),
    ({"preprocess": {"out_width": 0}}, "out_width"),
    ({"preprocess": {"out_height": 66.5}}, "preprocess.out_height"),
    ({"camera": {"exposure": "bright"}}, "camera.exposure"),
    ({"input": {"ds4": {"deadman": "on please"}}}, "input.ds4.deadman"),
    ({"hardware": {"dir_forward_high": 1}}, "hardware.dir_forward_high"),
    ({"hardware": {"servo": {"backlash_comp_deg": None}}}, "hardware.servo.backlash_comp_deg"),
    ({"steering": {"center": 90.5}}, "steering.center"),
    ({"steering": {"classes": [30, True, 150]}}, "steering.classes"),
    ({"steering": {"mode": None}}, "steering.mode"),
    ({"throttle": {"levels": None}}, "throttle.levels"),
])
def test_mistakes_in_student_values_name_the_value(tmp_path, tree, words):
    with pytest.raises(ConfigError, match=re.escape(words)):
        load_config(_write(tmp_path, tree))


# ---------------------------------------------------------------------------
# 전체 값 보기:  uv run python -m common.config
# ---------------------------------------------------------------------------
def test_show_all_values_marks_what_the_file_sets(tmp_path, capsys):
    lines = describe(load_config(DEFAULT_CONFIG_PATH), _raw()).splitlines()
    values = [ln for ln in lines if not ln.rstrip().endswith(":")]
    assert len(values) == len(_typed_leaves())                          # 값이 하나도 빠지지 않는다
    assert sum(ln.startswith("*") for ln in values) == len(STUDENT_KEYS)

    assert main(["--config", str(DEFAULT_CONFIG_PATH)]) == 0
    out = capsys.readouterr().out
    assert "*   classes: [30, 60, 90, 120, 150]" in out                 # 파일에 적힌 값
    assert "\n    hysteresis: 0.15" in out                              # 파일에 없는 값 (기본값)
    assert "\n      address: 0x40" in out and "\n      address: 0x41" in out
    assert "*   auto_exposure: null" in out and "\n*     deadman: false" in out   # 한 단계 안쪽(input.ds4)의 값

    changed = _write(tmp_path, {"hardware": {"pca9685": {"address": 0x42}}})
    assert main(["--config", str(changed)]) == 0
    assert "\n*     address: 0x42" in capsys.readouterr().out               # 파일에 적으면 표시가 붙는다
    assert main(["--config", str(_write(tmp_path, {"steering": {"max": 999}}, "bad.yaml"))]) == 1
    assert "[설정 오류]" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 읽을 수 없는 파일
# ---------------------------------------------------------------------------
def test_leftover_git_conflict_markers_are_explained(tmp_path):
    """학생이 고친 줄과 새로 받은 파일이 겹치면 git 이 config.yaml 안에 <<<<<<< 표시를 남긴다 (git pull --autostash).
    'YAML 문법 오류'로만 알리면 무엇을 해야 할지 알 수 없으므로, 되돌리는 명령까지 알려 준다."""
    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    ours = "  levels: [30, 45, 60]      #"
    assert text.count(ours) == 1
    conflicted = text.replace(ours, "<<<<<<< Updated upstream\n" + ours.replace("#", " #")
                              + "\n=======\n  levels: [35, 50, 60]      #\n>>>>>>> Stashed changes\n" + ours)
    p = tmp_path / "config.yaml"
    p.write_text(conflicted, encoding="utf-8")
    for program in (None, *PROGRAM_SECTIONS):
        with pytest.raises(ConfigError, match="git 충돌 표시") as e:
            load_config(p, program=program)
        assert "git checkout HEAD -- config.yaml" in str(e.value) and "git stash show -p" in str(e.value)


def test_a_file_that_is_not_utf8_text_or_not_a_mapping_is_a_config_error(tmp_path):
    utf16 = tmp_path / "utf16.yaml"
    utf16.write_text("steering:\n  center: 90   # 직진\n", encoding="utf-16")     # 메모장에서 다른 인코딩으로 저장한 경우
    with pytest.raises(ConfigError, match="읽을 수 없습니다"):
        load_config(utf16)
    for body in ("- 30\n- 60\n", "just some words\n", "steering: 90\n"):
        p = tmp_path / "shape.yaml"
        p.write_text(body, encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(p)
    with pytest.raises(ConfigError, match="설정 파일이 없습니다"):
        load_config(tmp_path / "missing.yaml")
