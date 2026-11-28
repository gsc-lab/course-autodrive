"""추론기: 모델 번들 읽기, 예측, 엔진, 주 루프 (가짜 차량·카메라·패드). 하드웨어 없이 돈다.

모델은 tests/data/tiny_model/ 의 아주 작은 번들을 쓴다 (tests/data/make_tiny_model.py 가 만든 것. 학습한 모델이 아니다).
TensorRT 가 있는 곳(Jetson)에서는 실제로 엔진을 만들어 PC 의 답과 견주고, 없는 곳에서는 그 테스트만 건너뛴다.
"""
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

import common.driveapp as drive_mod
import inference.app as app_mod
from common.config import DEFAULT_CONFIG_PATH, load_config
from common.display.base import DisplayState
from common.display.glyphs import CROSS, DOT, FULL, RING, SPIN, render, rotate, spin_phase
from common.driveapp import WATCHDOG_SEC
from common.events import EventKind as K, InputEvent as E
from common.hw.base import FORWARD
from common.inputs.base import InputDevice
from inference.bundle import BundleError, load_bundle
from inference.engine import ENGINE_FILE, ENGINE_INFO, Backend, EngineError, TensorRTBackend, open_backend
from inference.predict import PredictError, Prediction, Predictor, Smoother, softmax

TINY = Path(__file__).parent / "data" / "tiny_model"
REFERENCE = json.loads((TINY / "reference.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 도구
class FakeBackend(Backend):
    """정해 준 클래스가 가장 큰 점수를 받게 하는 가짜 모델."""
    name, note = "fake", "가짜 모델"

    def __init__(self, n=5, index=2):
        self.n, self.index, self.calls, self.closed, self.fail = n, index, 0, False, None

    def infer(self, x):
        self.calls += 1
        assert x.shape == (1, 3, 8, 16) and x.dtype == np.float32 and 0.0 <= x.min() and x.max() <= 1.0
        if self.fail is not None:
            if isinstance(self.fail, Exception):
                raise self.fail
            return np.full(self.n, self.fail, dtype=np.float32)
        out = np.zeros(self.n, dtype=np.float32)
        out[self.index] = 5.0
        return out

    def close(self):
        self.closed = True


class Scripted(InputDevice):
    """미리 정한 이벤트를 한 주기에 하나씩 내보낸다. push() 로 더 넣을 수 있다."""

    def __init__(self, script=(), name="scripted", requires_deadman=False, link="watched"):
        self.script, self.name, self.requires_deadman = list(script), name, requires_deadman
        self.statuses, self.closed = [], False
        if name == "ds4":
            self.link = link                    # 실제 DS4Input 처럼: 연결 감시가 없으면 None

    def push(self, *events):
        self.script += events

    def poll(self):
        return [self.script.pop(0)] if self.script else []

    def set_status(self, status):
        self.statuses.append(status)

    def close(self):
        self.closed = True


class Screen:
    """도트 매트릭스 자리에 끼워, 어떤 상태를 그리라고 했는지 모아 둔다."""
    name = "screen"

    def __init__(self):
        self.states, self.closed_with_error = [], None

    def update(self, st, now):
        self.states.append(st)

    def close(self, keep_error=False):
        self.closed_with_error = keep_error


def _cfg(tmp_path, edit=None):
    cfg = load_config(DEFAULT_CONFIG_PATH)
    cfg.recording.root = str(tmp_path / "dataset")
    if edit:
        edit(cfg)
    return cfg


def _app(tmp_path, monkeypatch, pad=None, backend=None, edit=None, **kw):
    """가짜 차량·카메라(mock), 가짜 패드, 가짜 모델로 추론기를 만든다. 돌려주는 값: (앱, 패드, 모델)."""
    pad = pad or Scripted()
    backend = backend or FakeBackend()
    monkeypatch.setattr(drive_mod, "create_input", lambda cfg_, override: pad)
    monkeypatch.setattr(app_mod, "open_backend", lambda bundle, kind, precision, say=None: backend)
    app = app_mod.Autopilot(_cfg(tmp_path, edit), load_bundle(TINY), mock=True, skip_checks=True, **kw)
    if not kw.get("check_only"):
        app.display = Screen()
    return app, pad, backend


def _run(app, seconds=0.12, dt=0.01):
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        app._step(time.monotonic())
        time.sleep(dt)


def _run_script(app, pad, deadline=3.0):
    """넣어 둔 이벤트가 모두 처리될 때까지 돌린다 (Scripted 는 한 주기에 하나씩 낸다).

    _run 처럼 시간을 정해 두면, 차를 세우는 주기가 길어지는 PC(Windows 의 sleep 은 16 ms 단위일 때가 있다)에서는 그 시간 안에
    주기가 한 번밖에 돌지 않아 두 번째 이벤트가 처리되지 않는다 (2026-11-24, 7번에 1~2번 실패)."""
    t_end = time.monotonic() + deadline
    while pad.script and time.monotonic() < t_end:
        app._step(time.monotonic())
        time.sleep(0.01)
    app._step(time.monotonic())
    assert not pad.script


def _finish(app):
    app.logic.state.quit = True
    app.close()


def _rows(app):
    with open(app.drive_log.path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- 모델 번들
def test_bundle_reads_classes_and_preprocessing_from_the_model_folder():
    b = load_bundle(TINY)
    assert b.classes == [30, 60, 90, 120, 150] and b.input_shape == (1, 3, 8, 16) and b.run_id == "tiny_model"
    assert (b.pre.crop_top_ratio, b.pre.crop_bottom_ratio, b.pre.out_width, b.pre.out_height) == (0.4, 1.0, 16, 8)
    assert load_bundle(TINY / "model.onnx").path == b.path              # 폴더 안의 파일을 가리켜도 된다


def _copy(tmp_path, meta_edit=None, drop=None, onnx_bytes=None, raw_meta=None):
    d = tmp_path / "model"
    shutil.copytree(TINY, d)
    if meta_edit:
        meta = json.loads((d / "model_meta.json").read_text(encoding="utf-8"))
        meta_edit(meta)
        (d / "model_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if raw_meta is not None:
        (d / "model_meta.json").write_text(raw_meta, encoding="utf-8")
    if onnx_bytes is not None:
        (d / "model.onnx").write_bytes(onnx_bytes)
    if drop:
        (d / drop).unlink()
    return d


@pytest.mark.parametrize("damage, words", [
    (dict(drop="model.onnx"), "model.onnx"),
    (dict(drop="model_meta.json"), "model_meta.json"),
    (dict(raw_meta="{ broken"), "읽을 수 없습니다"),
    (dict(raw_meta="[1, 2]"), "읽을 수 없습니다"),
    (dict(onnx_bytes=b"another model"), "서로 다른 학습"),                  # 두 파일을 따로 복사했다
    (dict(meta_edit=lambda m: m.update(task="regression")), "분류"),
    (dict(meta_edit=lambda m: m.update(schema_version=2)), "형식 번호"),
    (dict(meta_edit=lambda m: m.update(classes=[90])), "classes"),
    (dict(meta_edit=lambda m: m.update(classes=[30, 30, 90, 120, 150])), "classes"),
    (dict(meta_edit=lambda m: m["preprocess"].pop("out_width")), "preprocess"),
    (dict(meta_edit=lambda m: m["preprocess"].update(crop_top_ratio=0.95)), "preprocess"),
    (dict(meta_edit=lambda m: m["preprocess"].update(out_width=200)), "입출력 모양"),     # 전처리와 모델 입력이 어긋난다
    (dict(meta_edit=lambda m: m.update(classes=[30, 90, 150])), "입출력 모양"),
])
def test_a_model_folder_that_cannot_be_used_is_refused_with_one_line(tmp_path, damage, words):
    with pytest.raises(BundleError, match=words):
        load_bundle(_copy(tmp_path, **damage))


def test_missing_model_folder_says_what_to_copy(tmp_path):
    with pytest.raises(BundleError, match="모델 폴더가 없습니다"):
        load_bundle(tmp_path / "models" / "nope")


# ---------------------------------------------------------------- 예측
def test_predictor_preprocesses_like_training_and_maps_the_answer_to_an_angle():
    backend = FakeBackend(index=3)
    predict = Predictor(load_bundle(TINY), backend)
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    p = predict(frame)
    assert (p.index, p.angle, p.raw_angle) == (3, 120, 120) and p.confidence == pytest.approx(float(softmax(np.array([0, 0, 0, 5, 0.]))[3]))
    assert p.prob == p.confidence and p.probs.sum() == pytest.approx(1.0, abs=1e-5) and p.ms >= 0 and predict.count == 1
    predict.reset()
    assert predict.count == 0                                   # 시작할 때 미리 돌려 본 것은 예측 수에 넣지 않는다
    predict.close()
    assert backend.closed


def test_the_probability_shown_belongs_to_the_angle_that_was_chosen():
    """흔들림 줄이기를 쓰면 '쓰기로 한 각도'와 '이 사진만의 각도'가 다를 수 있다. 화면에는 쓰기로 한 각도의 확률을 보여 준다."""
    backend = FakeBackend(index=2)
    predict = Predictor(load_bundle(TINY), backend, smoothing="vote", window=3)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    predict(frame); predict(frame)
    backend.index = 3                                           # 한 장만 120° 로 튄다
    p = predict(frame)
    assert (p.angle, p.raw_angle) == (90, 120)                  # 다수결은 아직 90°
    assert p.confidence == pytest.approx(float(p.probs[3])) and p.confidence > 0.9          # 이 사진의 가장 큰 확률은 120° 의 것
    assert p.prob == pytest.approx(float(p.probs[2])) and p.prob < 0.05                     # 90° 의 확률은 낮다


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_model_output_that_is_not_a_number_is_never_turned_into_steering(bad):
    backend = FakeBackend()
    backend.fail = bad
    with pytest.raises(PredictError):
        Predictor(load_bundle(TINY), backend)(np.zeros((480, 640, 3), dtype=np.uint8))
    with pytest.raises(PredictError):                                   # 클래스 수가 다른 출력
        Predictor(load_bundle(TINY), FakeBackend(n=3, index=1))(np.zeros((480, 640, 3), dtype=np.uint8))


def _onehot(i, n=5, p=0.9):
    v = np.full(n, (1 - p) / (n - 1), dtype=np.float32)
    v[i] = p
    return v


def test_smoothing_none_vote_and_ema():
    seq = [2, 2, 3, 2, 3, 3, 3]
    assert [Smoother("none")(_onehot(i)) for i in seq] == seq
    vote = Smoother("vote", window=3)
    assert [vote(_onehot(i)) for i in seq] == [2, 2, 2, 2, 3, 3, 3]     # 한 장짜리 튐(3)은 넘기고, 두 장이 모이면 따라간다
    tie = Smoother("vote", window=2)
    assert [tie(_onehot(i)) for i in (1, 4)] == [1, 4]                  # 같은 표면 가장 최근 것
    ema = Smoother("ema", alpha=0.5)
    out = [ema(_onehot(i)) for i in (2, 3, 3)]
    assert out == [2, 2, 3]                                             # 첫 3 은 아직 2 를 못 넘는다 (0.9·0.5 대 0.9·0.5+작은 값)
    vote.reset(); ema.reset()
    assert vote(_onehot(4)) == 4 and ema(_onehot(0)) == 0               # 지난 기록을 버린다


# ---------------------------------------------------------------- 엔진 (모델을 실제로 실행)
def _reference_pairs():
    return [(np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32))
            for x, y in zip(REFERENCE["inputs"], REFERENCE["outputs"])]


def test_onnxruntime_backend_gives_the_recorded_answers():
    pytest.importorskip("onnxruntime")
    backend = open_backend(load_bundle(TINY), "onnxruntime")
    try:
        for x, want in _reference_pairs():
            got = backend.infer(x)
            assert got.shape == (5,) and np.abs(got - want).max() < 1e-5
    finally:
        backend.close()


def test_tensorrt_engine_is_built_cached_and_agrees_with_the_pc(tmp_path):
    """Jetson 에서만 돈다. 차가 만든 엔진이 PC(onnxruntime)의 답과 같은 답을 내고, 두 번째부터는 저장한 엔진을 읽는다."""
    pytest.importorskip("tensorrt")
    folder = _copy(tmp_path)
    said = []
    first = open_backend(load_bundle(folder), "tensorrt", say=said.append)
    try:
        assert "새로 만듦" in first.note and said and (folder / "model.engine").is_file()
        key = json.loads((folder / "model.engine.json").read_text(encoding="utf-8"))
        assert key["onnx_sha256"] == load_bundle(folder).onnx_sha256 and key["precision"] == "fp32" and key["tf32"] is False
        for x, want in _reference_pairs():
            got = first.infer(x)
            assert np.abs(got - want).max() < 1e-4 and int(got.argmax()) == int(want.argmax())
        with pytest.raises(EngineError):
            first.infer(np.zeros((1, 3, 4, 4), dtype=np.float32))       # 크기가 다른 입력
    finally:
        first.close()
    said.clear()
    again = open_backend(load_bundle(folder), "tensorrt", say=said.append)
    try:
        assert "저장된 엔진을 읽음" in again.note and not said
        x, want = _reference_pairs()[0]
        assert np.abs(again.infer(x) - want).max() < 1e-4
    finally:
        again.close()
    (folder / "model.engine.json").write_text(json.dumps({**key, "tensorrt": "0.0"}), encoding="utf-8")
    rebuilt = open_backend(load_bundle(folder), "tensorrt")             # 다른 TensorRT 판에서 만든 엔진이면 다시 만든다
    try:
        assert "새로 만듦" in rebuilt.note
    finally:
        rebuilt.close()


def test_saved_engine_is_reused_only_for_the_same_model_and_tensorrt(tmp_path):
    key = {"onnx_sha256": "abc", "tensorrt": "10.3.0", "precision": "fp32", "tf32": False}
    assert TensorRTBackend._cached_engine(tmp_path, key) == (None, "저장된 엔진 없음")
    TensorRTBackend._save_engine(tmp_path, b"engine-bytes", key)
    assert sorted(p.name for p in tmp_path.iterdir()) == [ENGINE_FILE, ENGINE_INFO]            # 만들다 만 파일(.part)이 없다
    assert TensorRTBackend._cached_engine(tmp_path, key) == (b"engine-bytes", "저장된 엔진을 읽음")
    for changed in ({"onnx_sha256": "other"}, {"tensorrt": "10.4.0"}, {"precision": "fp16"}):
        assert TensorRTBackend._cached_engine(tmp_path, {**key, **changed})[0] is None         # 모델이나 TensorRT 가 바뀌면 다시 만든다
    (tmp_path / ENGINE_INFO).write_text("{ broken", encoding="utf-8")
    assert TensorRTBackend._cached_engine(tmp_path, key)[0] is None


def test_an_engine_that_cannot_be_saved_does_not_stop_the_car(tmp_path, monkeypatch, caplog):
    """읽기 전용 폴더(USB 등): 저장도, 만들다 만 파일을 지우는 것도 실패한다. 경고만 남기고 방금 만든 엔진으로 달린다."""
    def read_only(self, *a, **k):
        raise OSError(30, "Read-only file system")
    monkeypatch.setattr(Path, "unlink", read_only)
    monkeypatch.setattr(Path, "write_bytes", read_only)
    with caplog.at_level("WARNING"):
        TensorRTBackend._save_engine(tmp_path, b"engine-bytes", {"onnx_sha256": "abc"})         # 예외 없이 지나간다
    assert "저장하지 못했습니다" in caplog.text


def test_asking_for_a_backend_that_is_not_installed_is_one_clear_line():
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        with pytest.raises(EngineError, match="tensorrt"):
            open_backend(load_bundle(TINY), "tensorrt")
    with pytest.raises(EngineError, match="알 수 없는"):
        open_backend(load_bundle(TINY), "openvino")


# ---------------------------------------------------------------- 화면과 패드
def test_display_shows_a_dot_while_loading_and_a_turning_bar_while_driving_itself():
    assert render(DisplayState(busy=True), True) == DOT
    frames = [render(DisplayState(auto=True), True, True, ph) for ph in range(4)]
    assert frames == SPIN and len({tuple(f) for f in frames}) == 4
    assert [spin_phase(t) for t in (0.0, 0.17, 0.34, 0.51, 0.67)] == [0, 1, 2, 3, 0]         # 1초에 여섯 장
    assert render(DisplayState(auto=True, error=True), True, True, 1) == CROSS               # 오류가 가장 우선
    assert render(DisplayState(auto=True, flash=True), True, True, 1) == FULL
    assert render(DisplayState(), True, True, 3) == RING                                     # 수동(대기)은 수집기와 같은 ○
    # 모듈을 어느 방향으로 달아도 같은 움직임: 네 장을 90° 돌리면 다시 그 네 장이다
    assert sorted(map(tuple, (rotate(f, 90) for f in SPIN))) == sorted(map(tuple, SPIN)) and rotate(DOT, 90) == DOT


def test_dpad_up_and_down_step_the_speed_level_and_mean_nothing_to_the_collector():
    from collector.state import DriveLogic
    from common.inputs.ds4 import ABS_HAT0Y, EV_ABS, DS4Mapper, resolve_codes
    for layout in ("generic", "sony"):
        m = DS4Mapper(resolve_codes(layout), {ABS_HAT0Y: (-1, 1)})
        up, down = m.translate(EV_ABS, ABS_HAT0Y, -1), m.translate(EV_ABS, ABS_HAT0Y, 1)
        assert (up.kind, up.value, down.kind, down.value) == (K.LEVEL_STEP, 1, K.LEVEL_STEP, -1)
        assert m.translate(EV_ABS, ABS_HAT0Y, 0) is None
    logic = DriveLogic(load_config(DEFAULT_CONFIG_PATH), requires_deadman=False)
    logic.apply(E(K.THROTTLE_AXIS, 0.3), 0.0)
    before = (logic.state.angle, logic.effective_speed, logic.state.keyboard_speed, logic.state.recording)
    logic.apply(E(K.LEVEL_STEP, 1), 0.1)
    assert (logic.state.angle, logic.effective_speed, logic.state.keyboard_speed, logic.state.recording) == before


@pytest.mark.skipif(not __import__("sys").platform.startswith("linux"), reason="pty 는 리눅스에서만 쓴다")
def test_keys_typed_while_the_program_was_starting_are_thrown_away(monkeypatch):
    """준비하는 동안 눌린 키가 남아 있으면 준비가 끝나는 순간 한꺼번에 처리된다: 기다리다 눌러 본 ↑ 에 차가 출발하고,
    R 과 Q 를 미리 눌렀으면 자율주행이 켜지자마자 끝난다 (2026-11-23 보드에서 실제로 봄)."""
    import os
    import sys

    from common.inputs.keyboard import KeyboardInput
    master, slave = os.openpty()
    try:
        monkeypatch.setattr(sys, "stdin", os.fdopen(slave, "r", closefd=False))
        os.write(master, b"r\x1b[Aq")                           # 준비 중에 누른 키: R, ↑, Q
        time.sleep(0.05)
        kb = KeyboardInput()
        try:
            assert kb.poll() == []
            os.write(master, b"r")                              # 준비가 끝난 뒤에 누른 키는 받는다
            time.sleep(0.05)
            assert [e.kind for e in kb.poll()] == [K.RECORD_TOGGLE]
        finally:
            kb.close()
    finally:
        os.close(master)
        os.close(slave)


def test_collector_and_inference_share_one_lock():
    import collector.lock
    import common.lock
    assert collector.lock.InstanceLock is common.lock.InstanceLock
    assert "추론기" in common.lock.AlreadyRunning(1, "").message()


@pytest.mark.skipif(not __import__("sys").platform.startswith("linux"), reason="flock 은 리눅스 전용")
def test_lock_is_taken_on_the_file_that_carries_the_name_now(tmp_path, monkeypatch):
    """앞 프로그램이 끝나며 잠금 파일을 지우는 바로 그 순간에 시작하면, 연 파일은 이미 이름이 없는 옛 파일이다. 그것을 잠그면
    뒤에 오는 프로그램이 새 파일을 잠가 둘이 함께 돈다. 지금 그 이름으로 있는 파일을 잠글 때까지 다시 한다."""
    import os

    import common.lock as lock_mod
    path = tmp_path / "collector.lock"
    path.write_text("old\n", encoding="utf-8")
    real_open, opens = os.open, []

    def racing_open(p, flags, mode=0o777):
        fd = real_open(p, flags, mode)
        opens.append(fd)
        if len(opens) == 1:                                     # 처음 연 직후, 앞 프로그램이 파일을 지운다
            os.unlink(p)
        return fd
    monkeypatch.setattr(lock_mod.os, "open", racing_open)
    lock = lock_mod.InstanceLock(path).acquire()
    try:
        assert len(opens) == 2 and lock.fd == opens[1]
        assert os.fstat(lock.fd).st_ino == os.stat(path).st_ino
        with pytest.raises(lock_mod.AlreadyRunning):            # 뒤에 오는 프로그램은 거절된다
            lock_mod.InstanceLock(path).acquire()
    finally:
        lock.release()


# ---------------------------------------------------------------- 주 루프
def test_autopilot_end_to_end_with_mocks(tmp_path, monkeypatch):
    app, pad, model = _app(tmp_path, monkeypatch)
    assert model.calls == app_mod.WARMUP_RUNS and not pad.closed        # 시작할 때 미리 돌려 본다 (첫 추론은 느리다)
    _run(app)
    assert not app.logic.auto and app.vehicle.state.speed == 0 and app.last.angle == 90      # 수동 중에도 예측은 계속한다

    model.index = 4
    pad.push(E(K.RECORD_TOGGLE))                                # ○: 자율주행 시작
    _run(app)
    assert app.logic.auto and app.vehicle.state.speed == 30 and app.vehicle.state.direction == FORWARD
    assert app.vehicle.state.angle == 150 and app.display.states[-1].auto
    model.index = 0
    _run(app)
    assert app.vehicle.state.angle == 30                        # 예측이 바뀌면 조향이 따라간다
    pad.push(E(K.LEVEL_STEP, 1))                                # D패드 ↑
    _run(app, 0.05)
    assert app.vehicle.state.speed == 45

    pad.push(E(K.THROTTLE_AXIS, 1.0))                           # R2: 바로 수동
    _run(app, 0.05)
    assert not app.logic.auto and app.vehicle.state.speed == 60 and app.vehicle.state.angle == 90
    pad.push(E(K.THROTTLE_AXIS, 0.0), E(K.RECORD_TOGGLE), E(K.MARK))
    _run(app)
    assert app.logic.auto and any(s.flash for s in app.display.states)
    pad.push(E(K.RECORD_TOGGLE))                                # ○: 정지
    _run(app, 0.05)
    assert not app.logic.auto and app.vehicle.state.speed == 0 and not app.display.states[-1].auto
    _finish(app)

    assert model.closed and pad.closed and app.vehicle.calls[-1] == ("close",)
    assert pad.statuses[0] == "idle" and "auto" in pad.statuses          # 라이트바(있는 패드에서만): 대기 → 자율주행
    rows = _rows(app)
    assert {r["mode"] for r in rows} == {"manual", "auto"} and len(rows) == app.drive_log.rows > 20
    assert rows[0]["pred_angle"] == "90" and float(rows[0]["p_90"]) > 0.9 and rows[0]["speed"] == "0"
    notes = "; ".join(r["note"] for r in rows)
    for word in ("auto_on speed=30", "level speed=45", "auto_off takeover:R2", "mark", "auto_off button"):
        assert word in notes, word
    assert app.drive_log.path.read_bytes().isascii()            # 한글이 없다: Excel 로 열어도 깨지지 않는다
    assert app.predictor.count == len(rows)                     # 미리 돌려 본 것은 세지 않는다
    auto_rows = [r for r in rows if r["mode"] == "auto"]
    assert all(r["applied_angle"] == r["pred_angle"] for r in auto_rows)           # 자율주행 중: 차에 낸 각도 = 예측
    assert {r["applied_angle"] for r in rows if r["mode"] == "manual"} == {"90"}   # 수동 중: 사람이 꺾은 각도


def test_a_failing_model_stops_the_car_and_ends_the_run(tmp_path, monkeypatch):
    for failure in (RuntimeError("GPU 가 응답하지 않음"), float("nan")):
        app, pad, model = _app(tmp_path, monkeypatch, pad=Scripted([E(K.RECORD_TOGGLE)]))
        _run(app, 0.06)
        assert app.logic.auto and app.vehicle.state.speed == 30
        model.fail = failure
        _run(app, 0.06)
        st = app.logic.state
        assert st.quit and "추론 실패" in st.error and not app.logic.auto and app.vehicle.state.speed == 0
        assert app.display.states[-1].error
        screen = app.display
        app.close()
        assert screen.closed_with_error is True                 # X 를 켜 둔 채 끝난다


def test_pad_loss_during_autonomous_driving_stops_the_car(tmp_path, monkeypatch):
    app, pad, _ = _app(tmp_path, monkeypatch, pad=Scripted([E(K.RECORD_TOGGLE)]))
    _run(app, 0.06)
    assert app.vehicle.state.speed == 30
    pad.push(E(K.DEVICE_LOST))
    _run(app, 0.04)
    assert app.logic.state.quit and app.vehicle.state.speed == 0 and "끊김" in app.logic.state.error
    app.close()


class _FrozenCamera:
    """읽기 스레드가 멈춘 카메라: 같은 사진만 계속 준다."""
    error = False

    def __init__(self, t):
        self.t = t

    def read(self):
        return np.zeros((480, 640, 3), dtype=np.uint8), self.t

    def age(self, now=None):
        return (time.monotonic() if now is None else now) - self.t

    def close(self):
        pass


def test_a_camera_that_stops_delivering_frames_ends_autonomous_driving(tmp_path, monkeypatch):
    app, pad, model = _app(tmp_path, monkeypatch, pad=Scripted([E(K.RECORD_TOGGLE)]))
    _run(app, 0.06)
    assert app.logic.auto and app.vehicle.state.speed == 30
    calls = model.calls
    app.camera = _FrozenCamera(time.monotonic())
    _run(app, app.cfg.inference.max_frame_age_sec + 0.15)
    assert not app.logic.auto and app.vehicle.state.speed == 0 and "예측이 끊김" in app.logic.note
    assert model.calls == calls + 1 and not app.logic.state.quit        # 같은 사진을 다시 예측하지 않는다. 아직 끝내지는 않는다
    app.camera = _FrozenCamera(time.monotonic() - drive_mod.CAMERA_STALE_SEC - 1)
    _run(app, 0.03)
    assert app.logic.state.quit and "카메라" in app.logic.state.error    # 오래 멈추면 수집기와 같이 고장으로 보고 끝낸다
    app.close()


def test_watchdog_stops_the_motor_when_the_loop_stalls_in_autonomous_mode(tmp_path, monkeypatch):
    app, pad, _ = _app(tmp_path, monkeypatch, pad=Scripted([E(K.RECORD_TOGGLE)]))
    _run(app, 0.06)
    assert app.vehicle.state.speed == 30
    app._last_tick = 100.0
    assert app._watchdog_check(100.0 + WATCHDOG_SEC + 0.5) is True
    assert app.vehicle.state.speed == 0 and app.vehicle.calls[-1] == ("stop", False)
    assert app.logic.state.quit and "비상 정지" in app.logic.state.error
    app.close()


def test_autonomous_driving_needs_a_pad_whose_link_is_watched(tmp_path, monkeypatch):
    app, pad, _ = _app(tmp_path, monkeypatch, pad=Scripted([E(K.RECORD_TOGGLE)], name="ds4", link=None))
    _run(app, 0.06)
    assert not app.logic.auto and app.vehicle.state.speed == 0 and "감시" in app.logic.note
    pad.push(E(K.THROTTLE_AXIS, 1.0))                           # 수동 운전은 된다
    _run(app, 0.04)
    assert app.vehicle.state.speed == 60
    _finish(app)
    app, pad, _ = _app(tmp_path, monkeypatch, pad=Scripted([E(K.RECORD_TOGGLE)], name="ds4"))
    _run(app, 0.06)
    assert app.logic.auto
    _finish(app)


def test_on_a_real_car_the_keyboard_can_drive_by_hand_but_never_starts_autonomous_driving(tmp_path, monkeypatch):
    """패드를 못 찾으면 키보드로 대체된다. 키보드는 SSH 로 오므로 Wi-Fi 가 끊겨도 차가 모른다: 진짜 차에서는 켜지 않는다."""
    app, pad, _ = _app(tmp_path, monkeypatch, pad=Scripted(name="keyboard"))
    assert app.logic.block_reason == ""                         # 가짜 차량(--mock): 무입력 한도를 걸고 허용한다
    app.cfg.hardware.backend = "jetson"
    real = app._make_logic()
    assert "패드로만" in real.block_reason
    real.set_prediction(90, 1.0)
    real.apply(E(K.RECORD_TOGGLE), 1.0)
    assert not real.auto and real.changes[-1][:2] == ("refused", "input_not_watched")
    real.apply(E(K.FORWARD_TOGGLE), 1.1)                        # 수동 운전은 된다
    assert real.effective_speed > 0
    _finish(app)


def test_keyboard_gets_a_no_input_limit_even_when_the_safety_timer_is_off(tmp_path, monkeypatch):
    app, _, _ = _app(tmp_path, monkeypatch, pad=Scripted(name="keyboard"))
    assert app.logic.no_input_timeout == app.cfg.safety.deadman_sec == 30.0
    _finish(app)

    def off(cfg):
        cfg.safety.deadman_sec = 0.0
    app, _, _ = _app(tmp_path, monkeypatch, pad=Scripted(name="keyboard"), edit=off)
    assert app.logic.no_input_timeout == app_mod.KEYBOARD_AUTO_SEC
    _finish(app)
    app, _, _ = _app(tmp_path, monkeypatch)
    assert app.logic.no_input_timeout == 0.0                    # 패드
    _finish(app)


def test_a_model_that_cannot_be_opened_touches_no_pad_and_leaves_the_car_stopped(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(drive_mod, "create_input", lambda cfg_, override: opened.append(1) or Scripted())

    def broken(bundle, kind, precision, say=None):
        raise EngineError("엔진을 만들지 못했습니다")
    monkeypatch.setattr(app_mod, "open_backend", broken)
    vehicles = []
    real = drive_mod.create_vehicle
    monkeypatch.setattr(drive_mod, "create_vehicle", lambda *a, **k: vehicles.append(real(*a, **k)) or vehicles[-1])
    with pytest.raises(EngineError):
        app_mod.Autopilot(_cfg(tmp_path), load_bundle(TINY), mock=True, skip_checks=True)
    assert not opened and vehicles[0].calls[-1] == ("close",)


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_interrupting_the_start_releases_what_was_opened(tmp_path, monkeypatch, interrupt):
    """엔진을 만드는 수십 초 동안 Ctrl+C 나 kill 을 하는 일은 흔하다. 정리 없이 죽으면 도트 매트릭스에 '준비 중' 화면이
    남고 서보에도 신호가 계속 간다."""
    vehicles, screens = [], []
    real_vehicle = drive_mod.create_vehicle
    monkeypatch.setattr(drive_mod, "create_vehicle", lambda *a, **k: vehicles.append(real_vehicle(*a, **k)) or vehicles[-1])
    monkeypatch.setattr(drive_mod, "create_display", lambda *a, **k: screens.append(Screen()) or screens[-1])
    monkeypatch.setattr(drive_mod, "create_input", lambda cfg_, override: pytest.fail("패드를 열면 안 된다"))

    def interrupted(bundle, kind, precision, say=None):
        raise interrupt()
    monkeypatch.setattr(app_mod, "open_backend", interrupted)
    with pytest.raises(interrupt):
        app_mod.Autopilot(_cfg(tmp_path), load_bundle(TINY), mock=True, skip_checks=True)
    assert vehicles[0].calls[-1] == ("close",)
    assert screens[0].states[-1].busy and screens[0].closed_with_error is False       # 화면을 끄고 닫았다


@pytest.mark.parametrize("name, raised", [("SIGTERM", SystemExit), ("SIGINT", KeyboardInterrupt)])
def test_a_stop_signal_during_the_engine_build_is_not_lost(tmp_path, monkeypatch, name, raised):
    """엔진을 만드는 동안의 kill / Ctrl+C. 신호 처리기에서 바로 예외를 던지면 TensorRT 가 부르는 파이썬 함수 안에서 나서
    TensorRT 가 삼켜 버린다: 2026-11-23 보드에서 kill 이 무시되고 프로그램이 그대로 시작했다. 그래서 처리기는 표시만 하고,
    그 단계가 끝난 뒤에 멈춘다. 패드는 열지 않고, 연 장치는 놓는다."""
    import signal
    sig = getattr(signal, name)
    installed, vehicles = {}, []
    monkeypatch.setattr(drive_mod.signal, "signal", lambda s, handler: installed.setdefault(s, handler))   # 처음 건 것만
    real_vehicle = drive_mod.create_vehicle
    monkeypatch.setattr(drive_mod, "create_vehicle", lambda *a, **k: vehicles.append(real_vehicle(*a, **k)) or vehicles[-1])
    monkeypatch.setattr(drive_mod, "create_input", lambda cfg_, override: pytest.fail("패드를 열면 안 된다"))
    backend = FakeBackend()

    def building(bundle, kind, precision, say=None):
        try:
            installed[sig](sig, None)                           # 엔진을 만드는 도중에 신호가 온다
        except BaseException as err:                            # 여기서 난 예외는 진짜 TensorRT 안에서는 사라진다
            pytest.fail(f"signal handler raised {err!r}")
        return backend
    monkeypatch.setattr(app_mod, "open_backend", building)
    with pytest.raises(raised) as e:
        app_mod.Autopilot(_cfg(tmp_path), load_bundle(TINY), mock=True, skip_checks=True)
    if raised is SystemExit:
        assert e.value.code == 128 + sig
    assert vehicles[0].calls[-1] == ("close",) and backend.closed


def test_after_the_start_a_kill_means_stop_and_quit(tmp_path, monkeypatch):
    import signal
    installed = []
    monkeypatch.setattr(drive_mod.signal, "signal", lambda sig, handler: installed.append((sig, handler)))
    app, _, _ = _app(tmp_path, monkeypatch)
    assert not app.logic.state.quit
    dict(installed)[signal.SIGTERM](signal.SIGTERM, None)       # 신호마다 마지막에 건 것: 준비가 끝난 뒤의 처리기
    assert app.logic.state.quit                                 # 예외가 아니라 '정지 후 종료'
    app.close()

    class Late(app_mod.Autopilot):                              # 마지막 확인과 처리기를 바꿔 거는 사이에 온 신호
        def _install_signals(self):
            self._stop_signal = signal.SIGTERM
            super()._install_signals()
    monkeypatch.setattr(drive_mod, "create_input", lambda cfg_, override: Scripted())
    late = Late(_cfg(tmp_path), load_bundle(TINY), mock=True, skip_checks=True)
    assert late.logic.state.quit                                # 루프가 돌자마자 끝난다
    late.close()


def test_command_stops_cleanly_when_interrupted_before_it_starts(tmp_path, monkeypatch, capsys):
    from inference.__main__ import main

    def interrupted(bundle, kind, precision, say=None):
        raise KeyboardInterrupt()
    monkeypatch.setattr(app_mod, "open_backend", interrupted)
    assert main([str(TINY), "--mock", "--config", str(_config_file(tmp_path))]) == 130
    err = capsys.readouterr().err
    assert "[중단]" in err and "Traceback" not in err


def test_command_reports_a_model_that_answers_nan_as_a_model_error(tmp_path, monkeypatch, capsys):
    from inference.__main__ import main
    broken = FakeBackend()
    broken.fail = float("nan")
    monkeypatch.setattr(app_mod, "open_backend", lambda bundle, kind, precision, say=None: broken)
    monkeypatch.setattr(drive_mod, "create_input", lambda cfg_, override: pytest.fail("패드를 열면 안 된다"))
    assert main([str(TINY), "--mock", "--config", str(_config_file(tmp_path))]) == 6
    err = capsys.readouterr().err
    assert "[모델 오류]" in err and "Traceback" not in err and broken.closed


def _config_file(tmp_path):
    """기록 폴더만 임시 폴더로 바꾼 설정 파일 (명령을 통째로 부르는 테스트가 저장소의 dataset/ 에 쓰지 않게)."""
    p = tmp_path / "config.yaml"
    p.write_text(f"recording:\n  root: '{(tmp_path / 'dataset').as_posix()}'\n", encoding="utf-8")
    return p


def test_check_mode_fails_when_autonomous_driving_could_not_be_started(tmp_path, monkeypatch, capsys):
    """모델과 장치가 준비돼도 ○ 를 눌렀을 때 달리지 않을 차라면 '[점검 완료]' 라고 하면 안 된다 (패드 연결 감시가 없는 이미지)."""
    from inference.__main__ import main
    monkeypatch.setattr(app_mod, "open_backend", lambda bundle, kind, precision, say=None: FakeBackend())
    args = [str(TINY), "--mock", "--check", "--config", str(_config_file(tmp_path))]
    monkeypatch.setattr(drive_mod, "create_input", lambda cfg_, override: Scripted(name="ds4", link=None))
    assert main(args) == 7
    err = capsys.readouterr().err
    assert "[점검 실패]" in err and "감시" in err and "tiny_model" in err
    monkeypatch.setattr(drive_mod, "create_input", lambda cfg_, override: Scripted(name="ds4"))
    assert main(args) == 0
    assert "[점검 완료]" in capsys.readouterr().out


def test_a_prediction_log_that_fails_mid_drive_is_dropped_and_the_car_keeps_driving(tmp_path, monkeypatch, caplog):
    app, pad, _ = _app(tmp_path, monkeypatch, pad=Scripted([E(K.RECORD_TOGGLE)]))
    _run(app, 0.06)
    assert app.logic.auto and app.drive_log.rows > 0

    def full(*a, **k):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(app.drive_log, "row", full)
    with caplog.at_level("WARNING", logger="inference"):
        _run(app, 0.06)
    assert app.drive_log is None and "기록을 그만둡니다" in caplog.text
    assert app.logic.auto and app.vehicle.state.speed == 30 and not app.logic.state.quit and not app.logic.state.error
    _finish(app)


def test_a_main_loop_stall_is_written_to_the_drive_record_and_the_final_summary(tmp_path, monkeypatch, caplog):
    """감시 한계보다 짧은 멈춤(2026-11-24 실차: 자율주행 중 0.68초)은 차를 세우지 않는다. 그래도 기록에는 남아야
    그 구간의 줄 사이가 왜 비어 있는지, 차가 그동안 직전 명령으로 움직였다는 것을 나중에 알 수 있다."""
    app, pad, _ = _app(tmp_path, monkeypatch, pad=Scripted([E(K.RECORD_TOGGLE)]))
    _run(app, 0.06)
    with caplog.at_level("INFO", logger="inference"):
        app._stall_where = "pca9685.py:83 _retry ← jetson.py:99 set_throttle"      # 감시 스레드가 적어 둔 위치
        app._slow_step(0.684)
        assert "주 루프가 0.68초 멈췄습니다 (멈춰 있던 곳: pca9685.py:83 _retry ← jetson.py:99 set_throttle)" in caplog.text
        _run(app, 0.06)
        assert app.logic.auto and not app.logic.state.quit              # 차를 세우지는 않는다
        path = app.drive_log.path
        _finish(app)
    assert "주 루프 멈춤 1회 (가장 긴 것 0.68초)" in caplog.text
    with open(path, newline="", encoding="utf-8") as f:
        notes = [r["note"] for r in csv.DictReader(f) if r["note"]]
    assert "loop_stall ms=684" in notes and path.read_bytes().isascii()


def test_status_line_fits_the_terminal_and_what_happened_gets_a_line_of_its_own(tmp_path, monkeypatch):
    assert app_mod.fit_line("abc", 10) == "abc" and app_mod.fit_line("abcdef", 4) == "abcd"
    assert app_mod.fit_line("자율주행 시작", 7) == "자율주"      # 한글은 두 칸: 일곱 칸에는 세 글자까지
    assert app_mod.fit_line("90°", 3) == "90"                    # 터미널마다 폭이 다른 글자도 두 칸으로 센다

    written = []
    monkeypatch.setattr(app_mod, "_safe_write", written.append)
    app, pad, _ = _app(tmp_path, monkeypatch, pad=Scripted([E(K.RECORD_TOGGLE)]))
    _run(app, 0.06)
    now = time.monotonic()
    line = app._status_line()
    assert line.startswith("[자율] a=090 v=030 forward") and "예측 090 (0.9" in line and line.endswith("자율주행 속도 1/3 (30%)")
    assert sum(2 if not ch.isascii() else 1 for ch in line) <= 79                     # 80칸 터미널에 통째로 들어간다
    # 있었던 일은 상태줄 끝에 붙이지 않고 한 줄씩 따로 남긴다: 80칸에서는 잘려서 이유를 읽을 수 없었다 (보드에서 확인)
    assert "\r자율주행 시작 (속도 30 %)\x1b[K\r\n" in written
    pad.push(E(K.THROTTLE_AXIS, 1.0))
    _run(app, 0.04)
    assert "\r자율주행 끝: 수동 조작 (R2)\x1b[K\r\n" in written
    pad.push(E(K.RECORD_TOGGLE))                                  # R2 를 당긴 채로 ○: 왜 켜지지 않는지가 통째로 보인다
    _run(app, 0.04)
    assert "\r자율주행을 켤 수 없음: 차를 세우고 R2·L2 를 놓은 뒤 누르세요\x1b[K\r\n" in written
    pad.push(E(K.THROTTLE_AXIS, 0.0), E(K.RECORD_TOGGLE))
    _run_script(app, pad)
    assert app.logic.auto
    app.last = Prediction(index=2, angle=90, raw_index=3, raw_angle=120, confidence=0.97, prob=0.02,
                          probs=np.array([0, 0, 0.02, 0.97, 0.01], dtype=np.float32), ms=2.0)
    assert "예측 090 (0.02)" in app._status_line()               # 90° 옆에 120° 의 확률(0.97)을 붙이지 않는다

    monkeypatch.setattr(app_mod.shutil, "get_terminal_size", lambda fallback=None: __import__("os").terminal_size((40, 24)))
    app._print_status(now)
    text = written[-1]
    assert text.startswith("\r[자율]") and text.endswith("\x1b[K")                   # 줄 끝까지 지운다: 앞 줄의 꼬리가 남지 않는다
    body = text[1:-3]
    assert sum(2 if not ch.isascii() else 1 for ch in body) <= 39                     # 40칸 터미널에서 줄이 바뀌지 않는다
    _finish(app)


def test_model_classes_outside_this_cars_range_are_clamped_with_a_warning(tmp_path, monkeypatch, caplog):
    def narrow(cfg):
        cfg.steering.min, cfg.steering.max, cfg.steering.classes = 40, 140, [40, 65, 90, 115, 140]
    with caplog.at_level("WARNING", logger="inference"):
        app, pad, model = _app(tmp_path, monkeypatch, pad=Scripted([E(K.RECORD_TOGGLE)]), backend=FakeBackend(index=4), edit=narrow)
    assert "조향 범위 40~140 밖" in caplog.text
    _run(app, 0.06)
    assert app.logic.auto and app.vehicle.state.angle == 140    # 모델은 150° 라고 하지만 이 차는 140° 까지
    _finish(app)


def test_check_mode_reports_the_model_and_opens_no_log(tmp_path, monkeypatch):
    app, pad, model = _app(tmp_path, monkeypatch, check_only=True)
    report = app.check_report()
    app.close()
    assert "tiny_model" in report and "90°" in report and "가짜 모델" in report
    assert app.drive_log is None and model.closed and pad.closed
    assert not (tmp_path / "dataset" / "drives").exists()


def test_prediction_log_can_be_turned_off_and_a_log_that_cannot_be_written_does_not_stop_the_car(tmp_path, monkeypatch):
    def no_log(cfg):
        cfg.inference.log = False
    app, pad, _ = _app(tmp_path, monkeypatch, pad=Scripted([E(K.RECORD_TOGGLE)]), edit=no_log)
    _run(app, 0.06)
    assert app.drive_log is None and app.logic.auto
    _finish(app)

    def boom(*a, **k):
        raise OSError("디스크가 가득 참")
    monkeypatch.setattr(app_mod, "DriveLog", boom)
    app, pad, _ = _app(tmp_path, monkeypatch, pad=Scripted([E(K.RECORD_TOGGLE)]))
    _run(app, 0.06)
    assert app.drive_log is None and app.logic.auto and app.vehicle.state.speed == 30
    _finish(app)


# ---------------------------------------------------------------- 명령
def test_command_reports_a_bad_model_folder_or_config_in_one_line(tmp_path, capsys):
    from inference.__main__ import main
    assert main([str(tmp_path / "models" / "nope"), "--mock"]) == 6
    err = capsys.readouterr().err
    assert "[모델 오류]" in err and "Traceback" not in err
    bad = tmp_path / "c.yaml"
    bad.write_text("inference:\n  smoothing: average\n", encoding="utf-8")
    assert main([str(TINY), "--mock", "--config", str(bad)]) == 1
    assert "[설정 오류] inference.smoothing" in capsys.readouterr().err
    car = tmp_path / "car.yaml"
    car.write_text("training:\n  epochs: many\n", encoding="utf-8")     # PC 에서만 쓰는 값의 오타는 차를 세우지 않는다
    assert main([str(tmp_path / "nope"), "--mock", "--config", str(car)]) == 6


# ---------------------------------------------------------------------------
# 주행 기록은 뒤의 스레드가 쓴다 (2026-11-24: 자율주행 중 주 루프가 0.68초 멈췄다. SD 카드가 바쁠 때 flush 가 1.19초까지 막혔다)
# ---------------------------------------------------------------------------
def test_autopilot_keeps_driving_while_the_disk_is_stuck(tmp_path, monkeypatch, caplog):
    """디스크가 1초 넘게 멈춰도 주 루프는 제 주기로 돌고, 그동안에도 자율주행이 조종되고 R2 가 바로 듣는다."""
    from test_bgio import StuckDisk
    app, pad, _ = _app(tmp_path, monkeypatch, pad=Scripted([E(K.RECORD_TOGGLE)]))
    disk = StuckDisk()
    monkeypatch.setattr(app.drive_log, "_write", disk.hold(app.drive_log._write))
    steps, state = [], {}
    real_step = app._step

    def step(now):
        steps.append(now)
        elapsed = now - steps[0]
        if elapsed > 1.2 and "at_release" not in state:         # 1.2초 동안 디스크가 멈춰 있다가 풀린다
            state["at_release"] = (app.logic.auto, app.vehicle.state.speed, app.drive_log.rows, app.drive_log._io.pending)
            disk.release()
        if elapsed > 1.3 and "r2" not in state:
            state["r2"] = True
            pad.push(E(K.THROTTLE_AXIS, 1.0))                   # 사람이 넘겨받는다
        if elapsed > 1.5:
            state["at_end"] = (app.logic.auto, app.vehicle.state.speed)
            app.logic.state.quit = True
        real_step(now)
    monkeypatch.setattr(app, "_step", step)
    path = app.drive_log.path
    with caplog.at_level("WARNING"):
        assert app.run() == 0
    auto, speed, rows, pending = state["at_release"]
    assert disk.entered.is_set() and auto and speed == 30       # 디스크가 멈춘 동안에도 자율주행은 조종되고 있었다
    assert rows > 20 and pending >= rows - 1                    # 그동안의 줄은 쓰지 못한 채 밀려 있었다
    assert state["at_end"] == (False, 60)                       # R2 로 넘겨받았다
    assert app.slow_steps == 0 and "주 루프가" not in caplog.text
    assert max(b - a for a, b in zip(steps, steps[1:])) < 0.25 and len(steps) > 40
    with open(path, newline="", encoding="utf-8") as f:
        written = list(csv.DictReader(f))
    assert len(written) == app.drive_log.rows and app.drive_log.dropped == 0          # 한 줄도 잃지 않았다
    times = [float(r["t_mono"]) for r in written]
    assert times == sorted(times)                                                      # 맡긴 순서 그대로
    notes = [r["note"] for r in written if r["note"]]
    assert notes[0] == "auto_on speed=30" and "auto_off takeover:R2" in notes


def test_drive_log_drops_rows_when_too_many_wait_and_keeps_what_happened(tmp_path, monkeypatch, caplog):
    from test_bgio import StuckDisk
    monkeypatch.setattr(app_mod, "LOG_PENDING_MAX", 5)
    app, pad, _ = _app(tmp_path, monkeypatch)
    disk = StuckDisk()
    monkeypatch.setattr(app.drive_log, "_write", disk.hold(app.drive_log._write))
    with caplog.at_level("WARNING", logger="inference"):
        _run(app, 0.3)
        assert app.drive_log is not None and app.drive_log.dropped > 0 and app.drive_log.rows == 6     # 쓰는 중 1 + 밀린 5
        pad.push(E(K.RECORD_TOGGLE))                                # 줄이 버려지는 동안 자율주행을 켠다
        _run(app, 0.1)
        assert app.logic.auto and app.vehicle.state.speed == 30    # 기록이 밀려도 주행은 그대로다
    assert caplog.text.count("디스크가 느려 예측 기록의 일부 줄을 버리고 있습니다") == 1
    disk.release()
    assert app.drive_log._io.flush(5)
    _run(app, 0.1)
    path, dropped = app.drive_log.path, app.drive_log.dropped
    _finish(app)
    with open(path, newline="", encoding="utf-8") as f:
        written = list(csv.DictReader(f))
    assert dropped > 0 and len(written) == app.drive_log.rows          # 버린 줄은 세지 않는다
    # 버려진 줄에 적히려던 일은 사라지지 않고, 다음에 쓰인 줄에 붙는다
    assert [r["note"] for r in written if r["note"].startswith("auto_on")] == ["auto_on speed=30"]


def test_a_drive_log_that_fails_in_the_writer_thread_is_dropped_and_the_car_keeps_driving(tmp_path, monkeypatch, caplog):
    app, pad, _ = _app(tmp_path, monkeypatch, pad=Scripted([E(K.RECORD_TOGGLE)]))
    _run(app, 0.06)
    assert app.logic.auto and app.drive_log.rows > 0

    def full(fields, flush):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(app.drive_log, "_write", full)
    with caplog.at_level("WARNING", logger="inference"):
        _run(app, 0.3)
    assert app.drive_log is None and "기록을 그만둡니다" in caplog.text and "No space left" in caplog.text
    assert app.logic.auto and app.vehicle.state.speed == 30 and not app.logic.state.quit and not app.logic.state.error
    _finish(app)


def test_closing_a_drive_log_on_a_stuck_disk_gives_up_instead_of_hanging(tmp_path, monkeypatch, caplog):
    from test_bgio import StuckDisk
    dl = app_mod.DriveLog(tmp_path, "run", [30, 90, 150])
    disk = StuckDisk()
    monkeypatch.setattr(dl, "_write", disk.hold(dl._write))
    pred = Prediction(index=1, angle=90, raw_index=1, raw_angle=90, confidence=0.9, prob=0.9,
                      probs=np.array([0.05, 0.9, 0.05], dtype=np.float32), ms=2.0)
    t0 = time.perf_counter()
    assert dl.row(1.0, "manual", pred, 90, 0, 0.01) and dl.row(1.1, "manual", pred, 90, 0, 0.01)
    assert time.perf_counter() - t0 < 0.05 and disk.entered.wait(2)
    with caplog.at_level("WARNING", logger="inference"):
        t0 = time.perf_counter()
        assert dl.close(0.2) is False                               # 프로그램의 종료가 여기서 멈추지 않는다
        assert time.perf_counter() - t0 < 1.0
    assert "예측 기록을 끝까지 쓰지 못했습니다" in caplog.text
    disk.release()
