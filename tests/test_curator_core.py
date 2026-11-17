"""curator 핵심 로직: 세션 읽기, marks 사전 제외, 되돌리기, 데이터셋 zip 계약."""
import csv
import io
import json
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest

from common.config import DEFAULT_CONFIG_PATH, load_config
from common.filename import make_frame_name
from curator.core import REASON_MARK, REASON_USER, Curation, SessionSource, Workspace, build_dataset


def _jpeg(color):
    img = np.full((48, 64, 3), color, dtype=np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


def make_session(root: Path, name="20261110_1906", n=10, mark=(3, 5)) -> Path:
    """프레임 n 장(0.5초 간격, 각도 30/60/90/120/150 순환) + marks.csv 에 프레임 mark[0]~mark[1] 구간."""
    d = root / name
    d.mkdir(parents=True)
    t0 = datetime(2026, 11, 10, 19, 6, 25)
    angles = [30, 60, 90, 120, 150]
    times = []
    for i in range(n):
        ts = t0 + timedelta(milliseconds=500 * i)
        times.append(ts)
        (d / make_frame_name(ts, angles[i % 5], 45)).write_bytes(_jpeg(i * 20))
    with open(d / "marks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_start_iso", "t_end_iso", "note"])
        w.writerow([times[mark[0]].isoformat(timespec="milliseconds"),
                    times[mark[1]].isoformat(timespec="milliseconds"), "x"])
    (d / "events.csv").write_text("t_iso,t_mono,kind,value\n")
    (d / "session_meta.json").write_text(json.dumps({"session": name, "frames_saved": n}))
    return d


def test_session_folder_lists_frames_in_order_and_marks(tmp_path):
    d = make_session(tmp_path)
    src = SessionSource(d)
    assert src.name == "20261110_1906" and len(src.frames) == 10
    assert [f.info.angle for f in src.frames[:5]] == [30, 60, 90, 120, 150]
    assert [f.info.timestamp for f in src.frames] == sorted(f.info.timestamp for f in src.frames)
    assert [src.frames.index(f) for f in src.frames_in_marks()] == [3, 4, 5]
    assert src.meta["frames_saved"] == 10
    assert src.read_bytes(src.frames[0].name)[:2] == b"\xff\xd8"


def test_session_zip_is_read_like_a_folder(tmp_path):
    d = make_session(tmp_path)
    zpath = tmp_path / "20261110_1906.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for p in d.iterdir():
            z.write(p, f"20261110_1906/{p.name}")       # 폴더째 압축한 형태
    src = SessionSource(zpath)
    assert src.name == "20261110_1906" and len(src.frames) == 10 and len(src.frames_in_marks()) == 3
    assert src.read_bytes("marks.csv").startswith(b"t_start_iso")
    assert src.curation_path == tmp_path / "20261110_1906.curation.json"
    src.close()


def test_curation_preflags_marks_then_user_edits_undo_and_persist(tmp_path):
    d = make_session(tmp_path)
    src = SessionSource(d)
    cur = Curation.load_or_init(src)
    names = [f.name for f in src.frames]
    assert cur.excluded == {names[3]: REASON_MARK, names[4]: REASON_MARK, names[5]: REASON_MARK}
    cur.exclude(names[0:2])
    assert cur.is_excluded(names[0]) and cur.excluded[names[1]] == REASON_USER
    cur.restore([names[4]])
    assert not cur.is_excluded(names[4])
    cur.toggle(names[9])
    assert cur.is_excluded(names[9])
    assert cur.undo() == "exclude" and not cur.is_excluded(names[9])           # toggle 의 제외를 되돌림
    assert cur.undo() == "restore" and cur.excluded[names[4]] == REASON_MARK   # 복구를 되돌림
    assert cur.undo() == "exclude" and not cur.is_excluded(names[0])
    assert cur.undo() is None and not cur.can_undo
    cur.exclude([names[7]])
    cur.save(src)
    assert (d / "curation.json").exists()
    again = Curation.load_or_init(src)                     # 저장된 상태가 marks 사전 제외보다 우선
    assert set(again.excluded) == {names[3], names[4], names[5], names[7]}
    assert sorted(p.name for p in d.iterdir() if p.suffix == ".jpg") == sorted(names)   # 원본은 그대로


def test_workspace_counts_and_dataset_zip_contract(tmp_path):
    cfg = load_config(DEFAULT_CONFIG_PATH)
    ws = Workspace(cfg)
    make_session(tmp_path / "in", "20261110_1906", n=10, mark=(3, 5))
    make_session(tmp_path / "in", "20261110_1930", n=5, mark=(0, 0))
    ws.open(tmp_path / "in" / "20261110_1906")
    ws.open(tmp_path / "in" / "20261110_1930")
    with pytest.raises(ValueError):
        ws.open(tmp_path / "in" / "20261110_1930")
    frames = ws.frames()
    assert len(frames) == 15 and [f.session for f in frames[:10]] == ["20261110_1906"] * 10
    inc, exc = ws.counts()
    assert sum(inc.values()) == 11 and sum(exc.values()) == 4                 # marks: 3 + 1 제외
    ws.curation(frames[0]).exclude([frames[0].name])
    zip_path = build_dataset(ws, tmp_path / "out", now=datetime(2026, 11, 17, 10, 30))
    assert zip_path.name == "dataset_20261117_1030.zip"
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        images = sorted(n for n in names if n.startswith("images/"))
        assert len(images) == 10 and images[0] == "images/20261117_1030_000000.jpg" and images[-1].endswith("_000009.jpg")
        rows = list(csv.DictReader(io.StringIO(z.read("labels.csv").decode("utf-8"))))
        assert list(rows[0].keys()) == ["index", "image", "servo_angle", "motor_speed", "source"]
        assert rows[0]["image"] == "20261117_1030_000000.jpg" and rows[0]["servo_angle"] == "60"   # 0번(30°)은 제외됨
        assert rows[0]["source"].startswith("20261110_1906/") and rows[-1]["source"].startswith("20261110_1930/")
        assert [r["index"] for r in rows] == [str(i) for i in range(10)]
        meta = json.loads(z.read("meta.json"))
        assert meta["schema_version"] == 1 and meta["steering"]["classes"] == [30, 60, 90, 120, 150]
        assert meta["image"] == {"width": 640, "height": 480, "format": "jpg"}
        assert meta["sessions"] == ["20261110_1906", "20261110_1930"] and meta["frames"] == 10 and meta["excluded"] == 5
        assert sum(meta["counts_by_angle"].values()) == 10
        assert z.read(images[0])[:2] == b"\xff\xd8"
    ws.close_all()


def test_build_dataset_refuses_when_everything_is_excluded(tmp_path):
    ws = Workspace(load_config(DEFAULT_CONFIG_PATH))
    make_session(tmp_path / "in", n=3, mark=(0, 2))
    ws.open(tmp_path / "in" / "20261110_1906")
    with pytest.raises(ValueError):
        build_dataset(ws, tmp_path / "out")
