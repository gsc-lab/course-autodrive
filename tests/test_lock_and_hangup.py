"""단일 실행 잠금과 SIGHUP 처리 (패드 모드는 SSH 가 끊겨도 계속 달린다)."""
import signal
import sys

import pytest

from collector.lock import AlreadyRunning, InstanceLock
from common.config import DEFAULT_CONFIG_PATH, load_config
from common.inputs.base import InputDevice


def test_already_running_message_names_pid_and_kill_command():
    msg = AlreadyRunning(4242, "2026-11-10T18:00:00").message()
    assert "PID 4242" in msg and "kill 4242" in msg and "Options" in msg and "2026-11-10 18:00:00" in msg
    msg2 = AlreadyRunning(None, "").message()
    assert "pkill" in msg2


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="flock 은 리눅스 전용")
def test_second_instance_is_refused_until_first_releases(tmp_path):
    first = InstanceLock(tmp_path / "collector.lock").acquire()
    with pytest.raises(AlreadyRunning) as e:
        InstanceLock(tmp_path / "collector.lock").acquire()
    assert e.value.pid == __import__("os").getpid()
    first.release()
    second = InstanceLock(tmp_path / "collector.lock").acquire()
    second.release()


def test_lock_is_a_noop_off_linux(tmp_path):
    if sys.platform.startswith("linux"):
        pytest.skip("리눅스에서는 실제로 잠근다")
    lock = InstanceLock(tmp_path / "collector.lock").acquire()
    assert lock.fd is None
    lock.release()


class _Fake(InputDevice):
    def __init__(self, name):
        self.name = name

    def poll(self):
        return []

    def close(self):
        pass


@pytest.mark.parametrize("input_name,expect_ignore", [("ds4", True), ("keyboard", False)])
def test_sighup_ignored_only_for_pad_driving(tmp_path, monkeypatch, input_name, expect_ignore):
    import collector.app as app_mod
    cfg = load_config(DEFAULT_CONFIG_PATH)
    cfg.recording.root = str(tmp_path / "dataset")
    installed = {}

    def fake_signal(sig, handler):
        installed[sig] = handler
    monkeypatch.setattr(app_mod.signal, "signal", fake_signal)
    monkeypatch.setattr(app_mod, "create_input", lambda cfg_, override: _Fake(input_name))
    if not hasattr(signal, "SIGHUP"):
        monkeypatch.setattr(app_mod.signal, "SIGHUP", 1, raising=False)
    col = app_mod.Collector(cfg, mock=True, skip_checks=True)
    try:
        hup = installed[app_mod.signal.SIGHUP]
        assert (hup is signal.SIG_IGN) == expect_ignore
        assert col.survives_hangup == expect_ignore
        assert callable(installed[signal.SIGTERM])        # 종료 명령(kill PID)은 항상 정지로 이어진다
    finally:
        col.close()
