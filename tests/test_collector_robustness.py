"""2026-11-17 코드 점검에서 찾은 수집기 쪽 결함의 재현 테스트.

- 감시 스레드가 로그를 쓰다 멈추면 모터를 끄지 못한다 → 모터를 먼저 끈다. 화면 로그는 멈추지 않는 쓰기로.
- evdev 이벤트를 놓치면(주 루프가 잠깐 멈춘 사이 커널 큐가 넘침) "R2 를 놓았다"를 모르고 계속 달린다 → 축의 실제 값을 주기적으로 맞춘다.
- 카메라 스레드가 멈추면 옛 프레임이 새 라벨로 계속 저장된다 → 오래된 프레임은 저장하지 않고, 계속 멈춰 있으면 오류로 정지한다.
- 사진을 디스크에 확정하지 않아 전원이 끊기면 마지막 십여 초가 0바이트 파일로 남는다 → 뒤에서 주기적으로 fsync.
"""
import logging
from datetime import datetime

import numpy as np
import pytest

import collector.app as app_mod
import common.session as session_mod
from collector.app import WATCHDOG_SEC
from common.config import DEFAULT_CONFIG_PATH, ConfigError, config_snapshot, load_config
from common.events import EventKind as K, InputEvent
from common.inputs.base import InputDevice
from common.inputs.ds4 import ABS_RY, ABS_RX, ABS_X, AxisSync
from common.session import SessionWriter


class _Quiet(InputDevice):
    name = "ds4"

    def poll(self):
        return []


def _collector(tmp_path, monkeypatch):
    cfg = load_config(DEFAULT_CONFIG_PATH)
    cfg.recording.root = str(tmp_path / "dataset")
    monkeypatch.setattr(app_mod, "create_input", lambda cfg_, override: _Quiet())
    return app_mod.Collector(cfg, mock=True, skip_checks=True)


# ---------------------------------------------------------------- 감시 스레드와 로그
def test_watchdog_stops_the_motor_before_it_logs(tmp_path, monkeypatch):
    col = _collector(tmp_path, monkeypatch)
    try:
        col.logic.apply(InputEvent(K.FORWARD_TOGGLE), 0.0)
        col._apply_vehicle()
        speed_when_logged = []
        monkeypatch.setattr(app_mod.log, "error", lambda *a, **k: speed_when_logged.append(col.vehicle.state.speed))
        col._last_tick = 100.0
        assert col._watchdog_check(100.0 + WATCHDOG_SEC + 0.5) is True
        assert speed_when_logged and speed_when_logged[0] == 0     # 로그가 막혀도(끊긴 SSH) 모터는 이미 꺼져 있다
    finally:
        col.close()


def test_screen_log_handler_drops_lines_instead_of_blocking(monkeypatch):
    written = []
    monkeypatch.setattr(app_mod, "_fd_writable", lambda fd: False)
    monkeypatch.setattr(app_mod.os, "write", lambda fd, data: written.append(data))
    h = app_mod.DroppingStderrHandler()
    h.emit(logging.LogRecord("x", logging.WARNING, __file__, 1, "경고 %d", (1,), None))
    assert written == []                                            # 아무도 읽지 않는 터미널: 버린다
    monkeypatch.setattr(app_mod, "_fd_writable", lambda fd: True)
    h.emit(logging.LogRecord("x", logging.WARNING, __file__, 1, "경고 %d", (2,), None))
    assert len(written) == 1 and "경고 2".encode() in written[0]

    def boom(fd, data):
        raise OSError("EIO")
    monkeypatch.setattr(app_mod.os, "write", boom)
    h.emit(logging.LogRecord("x", logging.WARNING, __file__, 1, "x", (), None))     # 터미널이 사라져도 예외 없음


def test_collector_logs_to_screen_through_the_dropping_handler():
    import collector.__main__ as main_mod
    assert isinstance(main_mod.screen_handler(check=False), app_mod.DroppingStderrHandler)


# ---------------------------------------------------------------- 놓친 입력 이벤트
def test_axis_sync_reports_changes_the_event_stream_missed():
    kernel = {ABS_X: 128, ABS_RY: 200, ABS_RX: 0}                  # 커널이 알고 있는 지금 값
    sync = AxisSync([ABS_X, ABS_RY, ABS_RX], lambda c: kernel[c], interval_sec=0.25, now=0.0)
    assert sync.poll(0.1) == []                                     # 시작 값은 이벤트로 내지 않는다 (R2 를 쥔 채 켜도 출발 안 함)
    sync.note(ABS_RY, 255)                                          # 이벤트로 본 값
    kernel[ABS_RY] = 255
    assert sync.poll(0.3) == []                                     # 이벤트와 커널이 같으면 조용
    kernel[ABS_RY] = 0                                              # 놓았는데 그 이벤트가 큐에서 버려졌다
    assert sync.poll(0.4) == []                                     # 아직 주기가 안 됨
    assert sync.poll(0.56) == [(ABS_RY, 0)]                         # 주기가 되면 실제 값으로 바로잡는다
    assert sync.poll(0.9) == []
    kernel[ABS_X] = 30
    assert sync.poll(0.91, force=True) == [(ABS_X, 30)]             # SYN_DROPPED 를 보면 기다리지 않고 즉시


# ---------------------------------------------------------------- 멈춘 카메라
class _FrozenCamera:
    error = False

    def __init__(self, age):
        self._age = age

    def read(self):
        return np.zeros((48, 64, 3), dtype=np.uint8), 0.0

    def age(self, now=None):
        return self._age

    def props(self):
        return {}

    def close(self):
        pass


def test_stale_camera_frame_is_not_saved_under_a_fresh_label(tmp_path, monkeypatch):
    col = _collector(tmp_path, monkeypatch)
    try:
        col.camera = _FrozenCamera(age=1.0)                         # 1초 전 그림 (정상은 0.03초)
        col.logic.apply(InputEvent(K.RECORD_TOGGLE), 0.0)
        col.logic.apply(InputEvent(K.FORWARD_TOGGLE), 0.0)
        col._step(10.0)
        assert col.session.saved == 0 and not col.logic.state.quit  # 저장만 건너뛴다
        col.camera = _FrozenCamera(age=0.02)
        col._step(10.1)
        assert col.session.saved == 1
    finally:
        col.close()


def test_camera_that_stays_frozen_stops_the_car(tmp_path, monkeypatch):
    col = _collector(tmp_path, monkeypatch)
    try:
        col.logic.apply(InputEvent(K.FORWARD_TOGGLE), 0.0)
        col._apply_vehicle()
        col.camera = _FrozenCamera(age=app_mod.CAMERA_STALE_SEC + 1)
        col._step(10.0)
        st = col.logic.state
        assert st.quit and "카메라" in st.error and col.vehicle.state.speed == 0
    finally:
        col.close()


# ---------------------------------------------------------------- 사진을 디스크에 확정
def test_session_writer_fsyncs_frames_in_the_background(tmp_path, monkeypatch):
    synced = []
    real = session_mod.os.fsync
    monkeypatch.setattr(session_mod.os, "fsync", lambda fd: (synced.append(fd), real(fd))[1])
    s = SessionWriter(tmp_path, config_snapshot(load_config(DEFAULT_CONFIG_PATH)), sync_every_sec=0)   # 0 = 스레드 없이 직접 부른다
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    names = [s.save_frame(frame, 90, 45, datetime(2026, 11, 17, 10, 0, i)) for i in range(3)]
    assert len(s._unsynced) >= 3 and not synced                     # 저장은 fsync 를 기다리지 않는다 (주 루프를 막지 않게)
    assert s.sync() >= 3 and s._unsynced == [] and len(synced) >= 3
    assert all((s.path / n).stat().st_size > 0 for n in names)
    s.close()
    assert s._unsynced == []


def test_session_writer_sync_thread_starts_and_stops(tmp_path):
    s = SessionWriter(tmp_path, {}, sync_every_sec=0.05)
    assert s._sync_thread is not None and s._sync_thread.is_alive()
    s.close()
    assert not s._sync_thread.is_alive()


# ---------------------------------------------------------------- 설정
def _cfg_with(tmp_path, edit):
    import yaml
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    edit(data)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def test_broken_yaml_is_reported_as_a_config_error(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("steering:\n  classes: [30, 60\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_stop_ramp_that_would_block_the_loop_is_rejected(tmp_path):
    def slow(d):
        d["hardware"]["stop_decay_step"] = 1
        d["hardware"]["stop_decay_delay"] = 0.2                     # 100 % 에서 20초: 그동안 조종이 멈춘다
    with pytest.raises(ConfigError):
        load_config(_cfg_with(tmp_path, slow))
    with pytest.raises(ConfigError):
        load_config(_cfg_with(tmp_path, lambda d: d["hardware"].update(stop_decay_delay=-0.01)))


def test_link_timeout_too_short_to_be_real_is_rejected(tmp_path):
    with pytest.raises(ConfigError):
        load_config(_cfg_with(tmp_path, lambda d: d["input"]["ds4"].update(link_timeout_sec=0.01)))
    load_config(_cfg_with(tmp_path, lambda d: d["input"]["ds4"].update(link_timeout_sec=0)))   # 0 = 끔 (허용)
