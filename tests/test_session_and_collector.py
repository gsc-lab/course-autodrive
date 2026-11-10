import json
from datetime import datetime

import numpy as np

from common.config import DEFAULT_CONFIG_PATH, load_config, config_snapshot
from common.events import EventKind as K, InputEvent as E
from common.filename import parse_frame_name
from common.inputs.base import InputDevice
from common.session import SessionWriter


def test_session_writer_layout(tmp_path):
    cfg = load_config(DEFAULT_CONFIG_PATH)
    started = datetime(2026, 11, 4, 14, 30, 0)
    with SessionWriter(tmp_path, config_snapshot(cfg), started_at=started) as s:
        assert s.path.name == "20261104_1430"
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        name = s.save_frame(frame, 120, 40, datetime(2026, 11, 4, 14, 30, 1, 500_000))
        assert parse_frame_name(name).angle == 120
        s.log_event("record_toggle", None, 1.0)
        s.mark(3.0, note="x", now=datetime(2026, 11, 4, 14, 30, 5))
    meta = json.loads((s.path / "session_meta.json").read_text(encoding="utf-8"))
    assert meta["frames_saved"] == 1 and meta["counts_by_angle"] == {"120": 1} and meta["ended_at"]
    assert meta["config"]["steering"]["classes"] == [30, 60, 90, 120, 150]
    assert (s.path / name).stat().st_size > 0
    assert "record_toggle" in (s.path / "events.csv").read_text(encoding="utf-8")
    marks = (s.path / "marks.csv").read_text(encoding="utf-8").splitlines()
    assert marks[1].startswith("2026-11-04T14:30:02.000,2026-11-04T14:30:05.000")
    # 같은 분에 두 번째 세션 → _2
    s2 = SessionWriter(tmp_path, {}, started_at=started)
    assert s2.path.name == "20261104_1430_2"
    s2.close()


class ScriptedInput(InputDevice):
    """테스트용: 미리 정한 이벤트를 순서대로 내보낸다."""
    name = "scripted"
    requires_deadman = True

    def __init__(self, script):
        self.script = list(script)
        self.statuses = []
        self.notes = []

    def poll(self):
        return [self.script.pop(0)] if self.script else []

    def set_status(self, status):
        self.statuses.append(status)

    def notify(self, kind):
        self.notes.append(kind)


def test_collector_end_to_end_with_mocks(tmp_path, monkeypatch):
    import collector.app as app_mod
    cfg = load_config(DEFAULT_CONFIG_PATH)
    cfg.recording.root = str(tmp_path / "dataset")
    cfg.recording.interval_sec = 0.1
    cfg.recording.interval_min = 0.1

    script = [E(K.DEADMAN, True), E(K.THROTTLE_AXIS, 1.0), E(K.STEER_AXIS, -1.0), E(K.RECORD_TOGGLE)]
    script += [E(K.STEER_AXIS, 0.0)] + [None] * 0
    scripted = ScriptedInput(script)
    monkeypatch.setattr(app_mod, "create_input", lambda cfg_, kind: scripted)

    col = app_mod.Collector(cfg, mock=True, skip_checks=True)
    # 루프를 직접 돌린다: 0.6초 동안 → 0.1초 간격이면 5장 안팎
    import time
    t_end = time.monotonic() + 0.6
    while time.monotonic() < t_end:
        col._step(time.monotonic())
        time.sleep(0.02)
    col.logic.apply(E(K.MARK), time.monotonic())
    col._step(time.monotonic())
    col.logic.state.quit = True
    col.close()

    saved = sorted(p.name for p in col.session.path.glob("*.jpg"))
    assert 3 <= len(saved) <= 8, saved
    infos = [parse_frame_name(n) for n in saved]
    assert all(i.speed == 60 for i in infos)
    assert {i.angle for i in infos} <= {30, 90}
    assert "record_on" in scripted.notes and "mark" in scripted.notes
    assert "recording" in scripted.statuses
    # 차량에 조향·구동·정지 명령이 갔는지
    kinds = [c[0] for c in col.vehicle.calls]
    assert "steer" in kinds and "throttle" in kinds and "stop" in kinds and kinds[-1] == "close"
    meta = json.loads((col.session.path / "session_meta.json").read_text(encoding="utf-8"))
    assert meta["frames_saved"] == len(saved) and meta["marks"] == 1 and meta["input_device"] == "scripted"
