"""주 루프가 멈춰도 차가 계속 달리지 않게 하는 두 장치: 멈추지 않는 터미널 쓰기와 감시 스레드."""
import sys

import pytest

import collector.app as app_mod
from collector.app import WATCHDOG_SEC, _safe_write
from common.config import DEFAULT_CONFIG_PATH, load_config
from common.events import EventKind as K, InputEvent
from common.inputs.base import InputDevice


class _Quiet(InputDevice):
    name = "ds4"

    def poll(self):
        return []

    def close(self):
        pass


def _collector(tmp_path, monkeypatch):
    cfg = load_config(DEFAULT_CONFIG_PATH)
    cfg.recording.root = str(tmp_path / "dataset")
    monkeypatch.setattr(app_mod, "create_input", lambda cfg_, override: _Quiet())
    return app_mod.Collector(cfg, mock=True, skip_checks=True)


def test_watchdog_stops_motor_when_loop_stalls(tmp_path, monkeypatch):
    col = _collector(tmp_path, monkeypatch)
    try:
        col.logic.apply(InputEvent(K.FORWARD_TOGGLE), 0.0)
        col._apply_vehicle()
        assert col.vehicle.state.speed > 0
        col._last_tick = 100.0
        assert col._watchdog_check(100.0 + WATCHDOG_SEC) is False          # 아직 한도 안
        assert col.vehicle.state.speed > 0
        assert col._watchdog_check(100.0 + WATCHDOG_SEC + 0.5) is True      # 멈춤 감지 → 즉시 정지
        assert col.vehicle.state.speed == 0 and col.vehicle.calls[-1] == ("stop", False)
        assert col.logic.state.quit and "비상 정지" in col.logic.state.error
    finally:
        col.close()


def test_watchdog_ignores_stall_while_stopped(tmp_path, monkeypatch):
    col = _collector(tmp_path, monkeypatch)
    try:
        col._last_tick = 0.0
        assert col._watchdog_check(1000.0) is False                       # 모터가 서 있으면 할 일 없음
        assert not col.logic.state.quit
    finally:
        col.close()


@pytest.mark.skipif(not hasattr(__import__("select"), "poll"), reason="POSIX 전용")
def test_safe_write_drops_output_when_terminal_is_not_writable(monkeypatch):
    """SSH 가 끊긴 채 매달려 있으면 터미널 버퍼가 차서 쓸 수 없다. 그때 write 로 멈추지 말고 버려야 한다."""
    import select

    class NeverWritable:
        def register(self, fd, mask):
            pass

        def poll(self, timeout):
            return []
    written = []
    monkeypatch.setattr(select, "poll", lambda: NeverWritable())
    monkeypatch.setattr(app_mod.os, "write", lambda fd, data: written.append(data))
    _safe_write("상태줄")
    assert written == []                                                 # 버림: 호출조차 하지 않는다


def test_safe_write_survives_closed_stdout(monkeypatch):
    class Dead:
        def fileno(self):
            raise ValueError("closed")
    monkeypatch.setattr(sys, "stdout", Dead())
    _safe_write("x")                                                      # 예외 없음
