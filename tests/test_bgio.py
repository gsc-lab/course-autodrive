"""파일 쓰기를 주 루프 밖으로 옮긴 것(common/bgio.py)과, 그것을 쓰는 세션 기록 · 주행 기록 · 로그 파일.

2026-11-24 실차 시험에서 자율주행 중 주 루프가 0.68초 멈췄고, 보드에서 재 보니 SD 카드가 바쁠 때는 파일 쓰기 한 번이
1.19초까지 막혔다. 주 루프는 그동안 조종을 하지 못하고 차는 직전 명령으로 달린다. 그래서 주 루프는 무엇을 쓸지만
넘기고, 쓰는 일은 뒤의 스레드가 한다. 이 파일의 테스트는 대부분 "디스크가 멈춘 것처럼" 쓰기를 붙잡아 두고, 그래도
맡기는 쪽이 기다리지 않는지를 본다.
"""
import csv
import json
import logging
import threading
import time
from datetime import datetime

import numpy as np
import pytest

import collector.app as collector_mod
import common.driveapp as drive_mod
import common.session as session_mod
from common.bgio import BackgroundFileHandler, BackgroundWriter
from common.config import DEFAULT_CONFIG_PATH, config_snapshot, load_config
from common.events import EventKind as K, InputEvent as E
from common.inputs.base import InputDevice
from common.session import SessionWriter

FAST = 0.05         # 맡기는 일이 이 시간(초) 안에 끝나면 "기다리지 않았다"고 본다 (디스크는 1초씩 막힌다)


class StuckDisk:
    """쓰기를 붙잡아 두는 가짜 디스크. hold(fn) 으로 감싼 함수는 release() 할 때까지 돌아오지 않는다."""

    def __init__(self):
        self.gate = threading.Event()
        self.entered = threading.Event()        # 쓰는 스레드가 실제로 붙잡혔다
        self.calls = 0

    def hold(self, fn):
        def held(*a, **k):
            self.calls += 1
            self.entered.set()
            if not self.gate.wait(8):
                self.gate.set()             # 한 번 걸렸으면 뒤의 호출은 붙잡지 않는다 (테스트가 끝없이 늘어지지 않게)
                raise AssertionError("디스크가 풀리지 않았다 (쓰는 일을 주 루프가 직접 하고 있으면 여기서 걸린다)")
            return fn(*a, **k)
        return held

    def release(self):
        self.gate.set()


def _timed(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# BackgroundWriter
# ---------------------------------------------------------------------------
def test_work_is_done_in_order_on_another_thread():
    w = BackgroundWriter("t")
    done, threads = [], set()
    for i in range(200):
        assert w.submit(lambda i=i: (done.append(i), threads.add(threading.get_ident())))
    assert w.close() is True
    assert done == list(range(200)) and threads and threading.get_ident() not in threads
    assert w.dropped == 0 and w.failed is None and w.pending == 0
    assert w.close() is True                                    # 여러 번 불러도 된다
    assert w.submit(lambda: done.append("late")) is False and "late" not in done     # 끝낸 뒤에는 받지 않는다


def test_submit_never_waits_while_the_disk_is_stuck():
    disk, done = StuckDisk(), []
    w = BackgroundWriter("t")
    assert w.submit(disk.hold(lambda: done.append("first")))
    assert disk.entered.wait(2)                                 # 쓰는 스레드가 디스크에 붙잡혔다
    _, took = _timed(lambda: [w.submit(lambda i=i: done.append(i)) for i in range(500)])
    assert took < FAST and done == [] and w.pending == 500      # 맡기는 쪽은 기다리지 않는다
    disk.release()
    assert w.flush(5) and done == ["first", *range(500)]        # 풀리면 맡긴 순서대로 모두 한다
    assert w.close()


def test_a_full_backlog_drops_new_work_and_counts_it():
    disk, done = StuckDisk(), []
    w = BackgroundWriter("t", max_pending=5)
    w.submit(disk.hold(lambda: done.append("first")))
    assert disk.entered.wait(2)
    results, took = _timed(lambda: [w.submit(lambda i=i: done.append(i)) for i in range(9)])
    assert took < FAST                                          # 가득 차도 기다리지 않고 버린다
    assert results == [True] * 5 + [False] * 4 and w.dropped == 4
    disk.release()
    assert w.close() and done == ["first", 0, 1, 2, 3, 4]


def test_a_failed_write_stops_the_writing_and_is_reported_to_the_owner():
    done = []
    w = BackgroundWriter("t")

    def full():
        raise OSError(28, "No space left on device")
    w.submit(lambda: done.append(1))
    w.submit(full)
    w.submit(lambda: done.append(2))                            # 실패한 뒤의 일은 하지 않는다
    assert w.flush(5)                                           # 실패한 뒤에도 flush 가 끝없이 기다리지 않는다
    assert done == [1] and isinstance(w.failed, OSError) and w.failed.errno == 28
    assert w.submit(lambda: done.append(3)) is False and w.dropped == 1
    assert w.close() and done == [1]


def test_close_gives_up_on_a_disk_that_does_not_answer():
    disk, done = StuckDisk(), []
    w = BackgroundWriter("t", max_pending=3)
    w.submit(disk.hold(lambda: done.append("first")))
    assert disk.entered.wait(2)
    for i in range(3):
        w.submit(lambda i=i: done.append(i))
    ok, took = _timed(w.close, 0.2)                             # 줄이 가득 차 끝 표시조차 넣을 수 없다
    assert ok is False and took < 1.0 and done == []
    disk.release()                                              # 뒤늦게 풀려도 밀린 일은 하지 않고 끝난다 (이미 포기했다)
    w._thread.join(3)
    assert done == ["first"] and not w._thread.is_alive()       # 스레드가 남아 떠돌지 않는다


def test_stop_soon_does_not_wait_and_skips_what_is_left():
    disk, done = StuckDisk(), []
    w = BackgroundWriter("t")
    w.submit(disk.hold(lambda: done.append("first")))
    assert disk.entered.wait(2)
    w.submit(lambda: done.append("second"))
    _, took = _timed(w.stop_soon)
    assert took < FAST and w.submit(lambda: done.append("third")) is False
    disk.release()
    w._thread.join(3)
    assert not w._thread.is_alive() and done == ["first"]


# ---------------------------------------------------------------------------
# 로그 파일
# ---------------------------------------------------------------------------
def _logger(tmp_path, name):
    h = BackgroundFileHandler(tmp_path / "run.log")
    h.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    lg.propagate = False
    lg.addHandler(h)
    return lg, h


def test_log_file_gets_every_line_in_order(tmp_path):
    lg, h = _logger(tmp_path, "bgio.order")
    values = {"n": 1}
    lg.info("값 %(n)d", values)
    values["n"] = 2                                             # 글은 남긴 순간의 값으로 만들어져 있어야 한다
    try:
        raise ValueError("깨짐")
    except ValueError:
        lg.exception("예외와 함께")
    for i in range(50):
        lg.warning("줄 %d", i)
    h.close()
    text = (tmp_path / "run.log").read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == "INFO 값 1" and "ValueError: 깨짐" in text and "Traceback" in text
    assert [ln for ln in lines if ln.startswith("WARNING 줄 ")] == [f"WARNING 줄 {i}" for i in range(50)]
    lg.removeHandler(h)


def test_logging_does_not_wait_for_a_stuck_log_file(tmp_path, monkeypatch):
    lg, h = _logger(tmp_path, "bgio.stuck")
    disk = StuckDisk()
    monkeypatch.setattr(h, "_write", disk.hold(h._write))
    lg.warning("첫 줄")
    assert disk.entered.wait(2)
    _, took = _timed(lambda: [lg.warning("경고 %d", i) for i in range(200)])
    assert took < 0.2                                           # 주 루프가 경고를 남기다 SD 카드에 막히지 않는다
    assert (tmp_path / "run.log").read_text(encoding="utf-8") == ""
    done = []
    other = threading.Thread(target=lambda: (lg.error("감시 스레드의 줄"), done.append(1)))     # 다른 스레드도 기다리지 않는다
    other.start(); other.join(1)
    assert done == [1]
    disk.release()
    h.close()
    lines = (tmp_path / "run.log").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "WARNING 첫 줄" and lines[-1] == "ERROR 감시 스레드의 줄" and len(lines) == 202
    lg.removeHandler(h)


def test_closing_a_stuck_log_file_does_not_hang_the_exit(tmp_path, monkeypatch):
    lg, h = _logger(tmp_path, "bgio.exit")
    disk = StuckDisk()
    monkeypatch.setattr(h, "_write", disk.hold(h._write))
    monkeypatch.setattr(h._io, "close", lambda timeout=3.0, real=h._io.close: real(0.2))     # 기다리는 시간만 줄인다
    lg.warning("x")
    assert disk.entered.wait(2)
    _, took = _timed(h.close)
    assert took < 1.5
    disk.release()
    lg.removeHandler(h)


def test_both_car_programs_log_to_file_through_the_background_handler():
    """두 프로그램의 시작 코드가 보통의 FileHandler 로 돌아가지 않게 붙잡아 둔다."""
    import inspect

    import collector.__main__ as collector_main
    import inference.__main__ as inference_main
    for mod in (collector_main, inference_main):
        src = inspect.getsource(mod)
        assert "BackgroundFileHandler(" in src and "logging.FileHandler(" not in src, mod.__name__


# ---------------------------------------------------------------------------
# 세션 기록 (수집기)
# ---------------------------------------------------------------------------
FRAME = np.full((120, 160, 3), 90, dtype=np.uint8)


def _session(tmp_path, **kw):
    return SessionWriter(tmp_path, config_snapshot(load_config(DEFAULT_CONFIG_PATH)), sync_every_sec=0, **kw)


def test_saving_a_frame_does_not_wait_for_the_disk(tmp_path, monkeypatch):
    s = _session(tmp_path)
    disk = StuckDisk()
    monkeypatch.setattr(s, "_write_frame", disk.hold(s._write_frame))
    name, took = _timed(s.save_frame, FRAME, 120, 45, datetime(2026, 11, 24, 18, 27, 2))
    assert took < FAST and name.endswith("_a120_s045.jpg")
    assert disk.entered.wait(2) and not (s.path / name).exists()        # 아직 디스크에 없다
    assert s.saved == 1 and s.counts[120] == 1 and s.last_frame_name == name    # 화면에 보이는 장수는 바로 오른다
    more, took = _timed(lambda: [s.save_frame(FRAME, 90, 45, datetime(2026, 11, 24, 18, 27, 3 + i)) for i in range(10)])
    assert took < FAST * 4 and all(more) and s.saved == 11
    disk.release()
    assert s.flush(5)
    assert (s.path / name).stat().st_size > 100 and len(list(s.path.glob("*.jpg"))) == 11
    assert len([p for p in s._unsynced if p.suffix == ".jpg"]) == 11    # 다 쓴 뒤에야 디스크 확정 목록에 오른다
    s.close()
    meta = json.loads((s.path / "session_meta.json").read_text(encoding="utf-8"))
    assert meta["frames_saved"] == 11 and meta["counts_by_angle"] == {"90": 10, "120": 1}


def test_frames_are_dropped_instead_of_piling_up_when_the_disk_is_stuck(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "PENDING_FRAMES_MAX", 4)
    s = _session(tmp_path)
    disk = StuckDisk()
    monkeypatch.setattr(s, "_write_frame", disk.hold(s._write_frame))
    names, took = _timed(lambda: [s.save_frame(FRAME, 90, 30, datetime(2026, 11, 24, 18, 0, i)) for i in range(10)])
    assert took < FAST * 4                                              # 밀려도 기다리지 않는다
    assert [bool(n) for n in names] == [True] * 4 + [False] * 6         # 넘치는 사진은 버린다 (메모리에 쌓지 않는다)
    assert s.saved == 4 and s.dropped == 6 and s.counts[90] == 4
    disk.release()
    assert s.flush(5)
    assert s.save_frame(FRAME, 90, 30, datetime(2026, 11, 24, 18, 1, 0))        # 디스크가 돌아오면 다시 저장한다
    s.close()
    assert len(list(s.path.glob("*.jpg"))) == 5 == s.saved


def test_events_and_marks_keep_their_order_and_do_not_wait_for_the_disk(tmp_path, monkeypatch):
    s = _session(tmp_path)
    disk = StuckDisk()
    monkeypatch.setattr(SessionWriter, "_write_row", staticmethod(disk.hold(SessionWriter._write_row)))

    def burst():
        for i in range(300):                                            # 스틱을 움직이면 이벤트가 한꺼번에 온다
            s.log_event("steer_axis", i / 300, float(i))
        s.mark(3.0, note="x", now=datetime(2026, 11, 24, 18, 27, 5))
    _, took = _timed(burst)
    assert took < FAST * 2 and s.marks == 1
    assert disk.entered.wait(2)
    disk.release()
    s.close()
    rows = list(csv.reader(open(s.path / "events.csv", encoding="utf-8")))[1:]
    assert [r[1] for r in rows] == [f"{float(i):.3f}" for i in range(300)]      # 맡긴 순서 그대로
    marks = (s.path / "marks.csv").read_text(encoding="utf-8").splitlines()
    assert len(marks) == 2 and marks[1].endswith(",x")


def test_the_periodic_meta_file_is_written_by_the_writer_thread_too(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "META_FLUSH_EVERY", 2)
    s = _session(tmp_path)
    disk = StuckDisk()
    monkeypatch.setattr(s, "_write_meta_file", disk.hold(s._write_meta_file))
    _, took = _timed(lambda: [s.save_frame(FRAME, 90, 30, datetime(2026, 11, 24, 18, 0, i)) for i in range(2)])
    assert took < FAST * 2 and disk.entered.wait(2)                     # 두 번째 사진이 메타 파일을 다시 쓰게 한다
    assert json.loads((s.path / "session_meta.json").read_text(encoding="utf-8"))["frames_saved"] == 0
    disk.release()
    assert s.flush(5)
    assert json.loads((s.path / "session_meta.json").read_text(encoding="utf-8"))["frames_saved"] == 2
    s.close()


def test_a_disk_error_reaches_the_main_loop_as_an_exception(tmp_path, monkeypatch):
    s = _session(tmp_path)

    def full(name, data):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(s, "_write_frame", full)
    assert s.save_frame(FRAME, 90, 30, datetime(2026, 11, 24, 18, 0, 0))        # 맡길 때는 아직 모른다
    assert s.flush(5)
    for call in (lambda: s.save_frame(FRAME, 90, 30, datetime(2026, 11, 24, 18, 0, 1)),
                 lambda: s.log_event("x"), lambda: s.mark(3.0)):
        with pytest.raises(OSError, match="세션 파일을 쓰지 못했습니다"):       # 다음에 부를 때 알린다
            call()


def test_close_takes_frames_that_never_reached_the_disk_out_of_the_counts(tmp_path, monkeypatch, caplog):
    s = _session(tmp_path)
    disk = StuckDisk()
    assert s.save_frame(FRAME, 60, 30, datetime(2026, 11, 24, 18, 0, 0)) and s.flush(5)        # 이 한 장은 디스크에 있다
    monkeypatch.setattr(s, "_write_frame", disk.hold(s._write_frame))
    for i in range(3):
        s.save_frame(FRAME, 90, 30, datetime(2026, 11, 24, 18, 0, 1 + i))
    assert disk.entered.wait(2) and s.saved == 4
    with caplog.at_level(logging.WARNING, logger="common.session"):
        _, took = _timed(s.close, 0.3)                                  # 디스크가 끝내 응답하지 않는다
    assert took < 3.0
    assert s.saved == 1 and dict(s.counts) == {60: 1}                   # 파일이 없는 사진은 장수에 넣지 않는다
    assert "사진 3장을 디스크에 쓰지 못했습니다" in caplog.text
    meta = json.loads((s.path / "session_meta.json").read_text(encoding="utf-8"))
    assert meta["frames_saved"] == 1 and meta["counts_by_angle"] == {"60": 1} and meta["ended_at"]
    disk.release()


# ---------------------------------------------------------------------------
# 수집기의 주 루프
# ---------------------------------------------------------------------------
class _Pad(InputDevice):
    name = "ds4"

    def __init__(self, script=()):
        self.script = list(script)

    def poll(self):
        return [self.script.pop(0)] if self.script else []


def _collector(tmp_path, monkeypatch, pad):
    cfg = load_config(DEFAULT_CONFIG_PATH)
    cfg.recording.root = str(tmp_path / "dataset")
    cfg.recording.interval_sec = cfg.recording.interval_min = 0.1
    monkeypatch.setattr(drive_mod, "create_input", lambda cfg_, override: pad)
    return collector_mod.Collector(cfg, mock=True, skip_checks=True)


def test_collector_keeps_driving_while_the_disk_is_stuck(tmp_path, monkeypatch, caplog):
    """디스크가 1초 넘게 멈춰도 주 루프는 제 주기로 돌고(감시 스레드와 '주 루프 멈춤' 경고가 조용하다), 패드가 그대로 듣는다."""
    monkeypatch.setattr(session_mod, "PENDING_FRAMES_MAX", 5)
    pad = _Pad([E(K.THROTTLE_AXIS, 1.0), E(K.RECORD_TOGGLE)])
    col = _collector(tmp_path, monkeypatch, pad)
    disk = StuckDisk()
    monkeypatch.setattr(col.session, "_write_frame", disk.hold(col.session._write_frame))
    monkeypatch.setattr(SessionWriter, "_write_row", staticmethod(disk.hold(SessionWriter._write_row)))
    steps, state = [], {}
    real_step = col._step

    def step(now):
        steps.append(now)
        elapsed = now - steps[0]
        if elapsed > 1.2 and "released" not in state:       # 1.2초 동안 디스크가 멈춰 있다가 풀린다
            state["released"] = True
            state["at_release"] = (col.session.saved, col.session.dropped, col.vehicle.state.speed)
            disk.release()
        if elapsed > 1.3 and "stopped" not in state:
            state["stopped"] = True
            pad.script.append(E(K.THROTTLE_AXIS, 0.0))      # 손을 놓는다
        if elapsed > 1.6:
            state["speed_at_end"] = col.vehicle.state.speed
            col.logic.state.quit = True
        real_step(now)
    monkeypatch.setattr(col, "_step", step)
    with caplog.at_level(logging.WARNING):
        assert col.run() == 0
    saved, dropped, speed = state["at_release"]
    assert disk.entered.is_set() and speed == 60                        # 디스크가 멈춘 동안에도 차는 조종되고 있었다
    assert saved == 5 and dropped >= 3                                  # 밀린 사진은 5장까지만 쥐고 나머지는 버렸다
    assert state["speed_at_end"] == 0                                   # 그 뒤 손을 놓자 바로 섰다
    assert col.slow_steps == 0 and "주 루프가" not in caplog.text       # 주 루프는 한 번도 오래 머물지 않았다
    gaps = [b - a for a, b in zip(steps, steps[1:])]
    assert max(gaps) < 0.25 and len(steps) > 40
    assert caplog.text.count("디스크가 느려 사진을 저장하지 못하고 있습니다") == 1     # 경고는 한 번만
    files = list(col.session.path.glob("*.jpg"))
    assert len(files) == col.session.saved >= 5                         # 풀린 뒤 밀린 사진을 모두 썼다
    meta = json.loads((col.session.path / "session_meta.json").read_text(encoding="utf-8"))
    assert meta["frames_saved"] == len(files)


def test_collector_stops_with_an_error_when_the_session_cannot_be_written(tmp_path, monkeypatch, caplog):
    pad = _Pad([E(K.THROTTLE_AXIS, 1.0), E(K.RECORD_TOGGLE)])
    col = _collector(tmp_path, monkeypatch, pad)

    def full(name, data):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(col.session, "_write_frame", full)
    started = time.monotonic()
    real_step = col._step

    def step(now):
        assert now - started < 5, "디스크 오류가 주 루프에 전해지지 않았다"
        real_step(now)
    monkeypatch.setattr(col, "_step", step)
    with caplog.at_level(logging.ERROR):
        assert col.run() == 2                                           # 예전과 같이: 저장할 수 없으면 멈춘다
    assert "세션 파일을 쓰지 못했습니다" in caplog.text and col.vehicle.state.speed == 0
