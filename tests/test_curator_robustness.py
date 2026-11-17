"""curator 가 깨진 입력·읽기/쓰기 실패에서도 죽지 않고 잘못된 결과를 남기지 않는지 (2026-11-17 코드 점검에서 찾은 것들)."""
import json
import os
import zipfile
from datetime import datetime

import pytest

from common.config import DEFAULT_CONFIG_PATH, load_config
from common.filename import make_frame_name
from curator.core import REASON_MARK, BrokenImage, SessionSource, Workspace, build_dataset, image_is_complete
from tests.test_curator_core import _jpeg, make_session


def _ws():
    return Workspace(load_config(DEFAULT_CONFIG_PATH))


# ---------------------------------------------------------------- core
@pytest.mark.parametrize("content", ["[1, 2, 3]", '"text"', '{"excluded": ["a.jpg"]}', '{"excluded": null}', "{broken"])
def test_broken_curation_json_falls_back_to_marks(tmp_path, content):
    d = make_session(tmp_path)
    (d / "curation.json").write_text(content, encoding="utf-8")
    ws = _ws()
    src = ws.open(d)                                        # 죽지 않고, 처음 여는 것처럼 □ 구간을 미리 제외한다
    assert set(ws.curations[src.name].excluded.values()) == {REASON_MARK} and len(ws.curations[src.name].excluded) == 3
    assert len(ws.sessions) == 1
    ws.close_all()


def test_failed_open_leaves_workspace_unchanged(tmp_path, monkeypatch):
    from curator import core
    d = make_session(tmp_path)
    ws = _ws()
    monkeypatch.setattr(core.Curation, "load_or_init", classmethod(lambda cls, src: (_ for _ in ()).throw(OSError("boom"))))
    with pytest.raises(OSError):
        ws.open(d)
    assert ws.sessions == [] and ws.curations == {}         # 반쯤 열린 세션이 남으면 이후 모든 조작이 KeyError


def test_zip_with_the_same_file_in_two_folders_lists_each_frame_once(tmp_path):
    d = make_session(tmp_path, n=4, mark=(0, 0))
    zpath = tmp_path / "both.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for p in d.iterdir():
            z.write(p, f"a/{p.name}")
            z.write(p, f"b/{p.name}")
    src = SessionSource(zpath)
    assert len(src.frames) == 4 and len({f.name for f in src.frames}) == 4
    src.close()


def test_build_failure_leaves_no_partial_zip_and_keeps_the_previous_one(tmp_path):
    ws = _ws()
    d = make_session(tmp_path / "in", n=6, mark=(0, 0))
    ws.open(d)
    now = datetime(2026, 11, 17, 10, 30)
    good = build_dataset(ws, tmp_path / "out", now=now)
    size = good.stat().st_size
    os.remove(d / ws.frames()[4].name)                      # 열어 둔 뒤에 원본 한 장이 사라졌다
    with pytest.raises(OSError):
        build_dataset(ws, tmp_path / "out", now=now)        # 같은 분에 다시 만들다 실패
    assert good.exists() and good.stat().st_size == size    # 먼저 만든 zip 은 그대로
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [good.name]   # 만들다 만 파일이 없다
    with zipfile.ZipFile(good) as z:
        assert z.testzip() is None
    ws.close_all()


def test_dataset_image_keeps_the_source_extension(tmp_path):
    d = tmp_path / "20261110_1906"; d.mkdir()
    import cv2
    import numpy as np
    png = cv2.imencode(".png", np.full((48, 64, 3), 10, dtype=np.uint8))[1].tobytes()
    (d / make_frame_name(datetime(2026, 11, 10, 19, 6, 1), 90, 45, ext="png")).write_bytes(png)
    (d / make_frame_name(datetime(2026, 11, 10, 19, 6, 2), 60, 45)).write_bytes(_jpeg(20))
    ws = _ws(); ws.open(d)
    path = build_dataset(ws, tmp_path / "out", now=datetime(2026, 11, 17, 10, 30))
    with zipfile.ZipFile(path) as z:
        assert sorted(n for n in z.namelist() if n.startswith("images/")) == [
            "images/20261117_1030_000000.png", "images/20261117_1030_000001.jpg"]
        assert "20261117_1030_000000.png" in z.read("labels.csv").decode()
    ws.close_all()


def test_empty_photos_from_a_power_cut_are_left_out(tmp_path):
    """전원이 끊긴 세션의 끝에는 0바이트 사진이 남는다 (2026-11-10 의 두 세션에 27장씩). 프레임으로 세지 않는다."""
    d = make_session(tmp_path, n=6, mark=(0, 0))
    names = sorted(p.name for p in d.glob("*.jpg"))
    for n in names[4:]:
        (d / n).write_bytes(b"")
    src = SessionSource(d)
    assert [f.name for f in src.frames] == names[:4] and src.empty == names[4:]
    src.close()
    zpath = tmp_path / "s.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for p in d.iterdir():
            z.write(p, p.name)
    src = SessionSource(zpath)
    assert len(src.frames) == 4 and len(src.empty) == 2
    src.close()
    ws = _ws(); ws.open(d)
    path = build_dataset(ws, tmp_path / "out")
    with zipfile.ZipFile(path) as z:
        assert all(i.file_size > 0 for i in z.infolist()) and json.loads(z.read("meta.json"))["empty_files_skipped"] == 2
    ws.close_all()


def test_half_written_photo_stops_the_build_and_names_the_frame(tmp_path):
    good = _jpeg(50)
    assert image_is_complete("a.jpg", good) and not image_is_complete("a.jpg", good[:-40]) and not image_is_complete("a.jpg", b"")
    d = make_session(tmp_path / "in", n=5, mark=(0, 0))
    ws = _ws(); ws.open(d)
    victim = ws.frames()[3]
    (d / victim.name).write_bytes(good[: len(good) // 2])      # 앞부분만 저장된 사진
    with pytest.raises(BrokenImage) as e:
        build_dataset(ws, tmp_path / "out")
    assert e.value.frame == victim and not list((tmp_path / "out").iterdir())
    ws.curation(victim).exclude([victim.name])                 # 제외하면 만들어진다
    assert build_dataset(ws, tmp_path / "out").exists()
    ws.close_all()


# ---------------------------------------------------------------- GUI
pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(qapp, tmp_path):
    from curator.app import MainWindow
    d = make_session(tmp_path, n=8, mark=(0, 0))
    w = MainWindow(load_config(DEFAULT_CONFIG_PATH))
    w.open_paths([str(d)])
    w.session_dir = d
    yield w
    w.ws.save = lambda: None
    w.close()


def test_thumbnail_from_an_older_worker_is_ignored(win):
    img = QImage(8, 6, QImage.Format_RGB32); img.fill(0xFF0000)
    win._stop_thumbs(); win._thumbs = {}
    win._on_thumb(win._thumb_gen - 1, 2, img)               # 세션을 더 열기 전의 작업자가 보낸, 번호가 밀린 썸네일
    assert 2 not in win._thumbs
    win._on_thumb(win._thumb_gen, 2, img)
    assert 2 in win._thumbs


def test_unreadable_photo_does_not_break_navigation(win):
    os.remove(win.session_dir / win.frames[3].name)
    win.goto(3)                                             # 예외 없이 이동하고, 읽지 못했다고 알려 준다
    assert win.index == 3 and win.view.pixmap is None and "읽을 수 없습니다" in win.view.message
    assert win.info.d["overall"] == 4
    win.goto(4); assert win.view.pixmap is not None


def test_save_failure_is_reported_and_does_not_block_editing_or_closing(win):
    def boom():
        raise PermissionError("read-only")
    win.ws.save = boom
    win.goto(2); win.toggle_current()                       # 저장은 실패해도 화면 상태는 따라온다
    assert win.ws.is_excluded(win.frames[2]) and win.summary.excluded == 2
    assert "저장하지 못했습니다" in win.statusBar().currentMessage()
    assert win.close()                                      # 닫히지 않는 창이 되면 안 된다


def test_exclude_message_counts_only_photos_that_changed(win):
    win.goto(0); win.goto(2, extend=True)                   # 0번은 □ 구간이라 이미 제외
    win.exclude_selection()
    assert win.statusBar().currentMessage().startswith("2장을 제외했습니다")
    win.goto(0); win.exclude_selection()
    assert win.statusBar().currentMessage().startswith("이미 제외된 사진입니다") and win._history and len(win._history) == 1
