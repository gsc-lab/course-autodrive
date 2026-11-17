"""curator 화면을 화면 없이(offscreen) 띄워 핵심 조작을 검사한다. PySide6 가 없으면 건너뛴다."""
import os
import zipfile

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from common.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from curator.app import MainWindow, arrow_rotation_deg, runs  # noqa: E402
from curator.core import build_dataset  # noqa: E402
from tests.test_curator_core import make_session  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_arrow_rotation_sign_and_limits():
    assert arrow_rotation_deg(1.0, 60, False) == 60 and arrow_rotation_deg(-1.0, 60, False) == -60
    assert arrow_rotation_deg(0.0, 60, False) == 0 and arrow_rotation_deg(2.0, 60, False) == 60
    assert arrow_rotation_deg(0.5, 60, True) == -30


def test_session_when_reads_name_and_duration():
    from curator.app import session_when
    meta = {"started_at": "2026-11-10T19:06:23", "ended_at": "2026-11-10T19:22:23"}
    assert session_when("20261110_1906", meta) == "11월 10일 19:06 촬영 · 16분 주행"
    assert session_when("20261110_1908_2", {}) == "11월 10일 19:08 촬영"
    assert session_when("something", {}) == ""


def test_balance_note_names_the_extremes():
    from curator.app import balance_note
    assert balance_note([]) == ("", False)
    assert balance_note([(30, 10, 0), (90, 80, 5), (150, 20, 0)]) == ("불균형 · 90° 가 30° 의 8.0배", True)
    assert balance_note([(30, 10, 0), (90, 20, 0)]) == ("균형 양호 · 최다 90° 가 최소 30° 의 2.0배", False)
    assert balance_note([(30, 0, 4), (90, 20, 0)]) == ("30° 사진이 없습니다", True)


def test_runs_groups_consecutive_flags():
    assert runs([True, False, True, True, False, True]) == [(0, 0), (2, 3), (5, 5)]
    assert runs([]) == [] and runs([False]) == []


def test_window_opens_navigates_excludes_and_builds(qapp, tmp_path):
    d = make_session(tmp_path / "in", n=12, mark=(2, 3))
    cfg = load_config(DEFAULT_CONFIG_PATH)
    cfg.recording.root = str(tmp_path / "in")
    win = MainWindow(cfg)
    try:
        win.open_paths([str(d)])
        assert len(win.frames) == 12 and win.strip.count() == 12
        win.goto(7); win.open_paths([str(d)])                               # 같은 세션을 또 열면 창 없이 알림만, 첫 장으로
        assert len(win.ws.sessions) == 1 and win.index == 0 and "이미 열려 있는" in win.statusBar().currentMessage()
        assert sum(win.ws.counts()[1].values()) == 2                       # marks 구간 2장이 미리 제외
        win.goto(5); assert win.index == 5
        info = win.info.d
        assert (info["in_session"], info["session_len"], info["overall"], info["total"]) == (6, 12, 6, 12)
        assert info["angle"] == 30 and info["side"] == -1.0 and info["speed"] == 45 and info["reason"] is None   # 5번 = 30°(왼쪽 끝)
        win.goto(2); assert win.info.d["reason"] == "mark"                   # □ 구간이라 미리 제외된 사진
        win.goto(99); assert win.index == 11                                # 끝에서 멈춤
        win.goto(-3); assert win.index == 0
        win.goto(6); win.goto(8, extend=True)                               # Shift+→ 두 번: 6~8 연속 선택
        assert win.selected == {6, 7, 8} and win.timeline.selected == [(6, 8)]
        win.goto(7, extend=True); assert win.selected == {6, 7}             # 되돌아오면 줄어든다
        win.goto(8, extend=True)
        win.exclude_selection()
        assert sum(win.ws.counts()[1].values()) == 5 and win.selected == set()
        win.undo()
        assert sum(win.ws.counts()[1].values()) == 2
        win.goto(5); win.toggle_select(9); win.toggle_select(11)            # Ctrl+클릭: 떨어진 낱장
        assert win.selected == {5, 9, 11} and win.timeline.selected == [(5, 5), (9, 9), (11, 11)]
        win.toggle_select(9); assert win.selected == {5, 11}
        win.goto(4); assert win.selected == set()                           # 그냥 이동하면 선택 해제
        win.goto(0); win.toggle_current(); assert win.ws.is_excluded(win.frames[0])
        assert (d / "curation.json").exists()                               # 바뀔 때마다 저장
        assert [a for a, _, _ in win.chart.data] == [30, 60, 90, 120, 150]
        assert sum(i for _, i, _ in win.chart.data) == 9 and sum(e for _, _, e in win.chart.data) == 3
        assert win.timeline.n == 12 and win.timeline.excluded == [(0, 0), (2, 3)]      # 타임라인의 제외 구간
        assert (win.summary.included, win.summary.excluded) == (9, 3)                 # 포함 + 제외 = 전체 12
        assert win.windowTitle().startswith("영진전문대 글로벌시스템융합과")
        path = build_dataset(win.ws, tmp_path / "out")
        with zipfile.ZipFile(path) as z:
            assert len([n for n in z.namelist() if n.startswith("images/")]) == 9
        from curator.app import BuildDoneDialog
        done = BuildDoneDialog(path, 9, 3, 1, win)
        assert done.windowTitle() == "데이터셋 작성 완료" and done.open_btn.text() == "폴더 열기" and done.path == path
        done.close()
        qapp.processEvents()
    finally:
        win.close()
