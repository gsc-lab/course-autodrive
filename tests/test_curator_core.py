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
from curator.core import (REASON_MARK, REASON_USER, AlreadyOpen, Curation, NoFrames, SessionSource, SteeringMismatch,
                          Workspace, build_dataset)


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
    with pytest.raises(AlreadyOpen):
        ws.open(tmp_path / "in" / "20261110_1930")
    (tmp_path / "in" / "20261110_2000").mkdir()             # 녹화 없이 끝난 빈 세션
    with pytest.raises(NoFrames):
        ws.open(tmp_path / "in" / "20261110_2000")
    assert len(ws.sessions) == 2
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
        assert meta["steering"]["center"] == 90            # 학습기가 좌우 반전의 대칭 축으로 쓴다
        assert meta["image"] == {"width": 640, "height": 480, "format": "jpg"}
        assert meta["sessions"] == ["20261110_1906", "20261110_1930"] and meta["frames"] == 10 and meta["excluded"] == 5
        assert sum(meta["counts_by_angle"].values()) == 10
        assert z.read(images[0])[:2] == b"\xff\xd8"
    ws.close_all()


def _set_recorded_steering(session_dir: Path, **steering) -> None:
    """수집기가 session_meta.json 에 넣는 설정 사본(config.steering)을 흉내 낸다."""
    base = {"mode": "discrete", "classes": [30, 60, 90, 120, 150], "center": 90, "min": 30, "max": 150,
            "step_deg": 5, "invert": False, "hysteresis": 0.15}
    meta = json.loads((session_dir / "session_meta.json").read_text())
    meta["config"] = {"steering": {**base, **steering}}
    (session_dir / "session_meta.json").write_text(json.dumps(meta))


def test_dataset_steering_follows_the_session_not_the_pc_config(tmp_path):
    """학생이 차에서 steering.classes 를 바꿔 촬영해도 PC 의 config.yaml 은 그대로다. 데이터셋의 meta.json 이 PC 의 값을
    적으면, 학습기는 실제로는 없는 클래스로 라벨을 묶고(50·70 → 60) 한 번도 배우지 않은 출력(30·150)을 가진 모델을 만든다."""
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert cfg.steering.classes == [30, 60, 90, 120, 150]
    d = make_session(tmp_path / "in", "20261110_1906", n=10, mark=(0, 0))      # 파일 이름의 각도는 30 60 90 120 150
    _set_recorded_steering(d, classes=[40, 65, 90, 115, 140], min=40, max=140)
    ws = Workspace(cfg)
    assert ws.steering_from is None and not ws.steering_differs_from_config
    ws.open(d)
    assert ws.steering_from == "20261110_1906" and ws.steering_differs_from_config
    assert ws.steering.classes == [40, 65, 90, 115, 140]
    inc, exc = ws.counts()
    assert sorted(inc) == [40, 65, 90, 115, 140] and sum(inc.values()) + sum(exc.values()) == 10
    with zipfile.ZipFile(build_dataset(ws, tmp_path / "out", now=datetime(2026, 11, 23, 9, 0))) as z:
        meta = json.loads(z.read("meta.json"))
    assert meta["steering"] == {"mode": "discrete", "classes": [40, 65, 90, 115, 140], "center": 90, "min": 40, "max": 140}
    assert sorted(meta["counts_by_angle"]) == ["115", "140", "40", "65", "90"]
    ws.close_all()
    assert ws.steering.classes == [30, 60, 90, 120, 150] and ws.steering_from is None   # 닫으면 config.yaml 로 돌아간다


def test_sessions_recorded_with_different_steering_cannot_be_merged(tmp_path):
    a = make_session(tmp_path / "in", "20261110_1906", n=5, mark=(0, 0))
    b = make_session(tmp_path / "in", "20261110_1930", n=5, mark=(0, 0))
    c = make_session(tmp_path / "in", "20261110_2000", n=5, mark=(0, 0))
    plain = make_session(tmp_path / "in", "20261110_2030", n=5, mark=(0, 0))   # 설정 사본이 없는 세션
    _set_recorded_steering(a)
    _set_recorded_steering(b, classes=[45, 90, 135])
    _set_recorded_steering(c, invert=True)                                      # 서보가 반대로 달린 차: 같은 각도가 반대 방향
    ws = Workspace(load_config(DEFAULT_CONFIG_PATH))
    ws.open(a)
    with pytest.raises(SteeringMismatch, match="클래스"):
        ws.open(b)
    with pytest.raises(SteeringMismatch, match="좌우 반대"):
        ws.open(c)
    ws.open(plain)                                                              # 알 수 없는 세션은 막지 않는다
    assert [s.name for s in ws.sessions] == ["20261110_1906", "20261110_2030"]
    ws.close_all()
    ws.open(b)                                                                  # 다 닫고 나면 다른 설정의 세션을 열 수 있다
    assert ws.steering.classes == [45, 90, 135]
    ws.close_all()


def test_unreadable_recorded_steering_falls_back_to_the_config(tmp_path):
    d = make_session(tmp_path / "in", n=5, mark=(0, 0))
    for bad in ({"classes": "abc"}, {"classes": []}, {"center": None}, {"min": 150, "max": 30}):
        _set_recorded_steering(d, **bad)
        ws = Workspace(load_config(DEFAULT_CONFIG_PATH))
        ws.open(d)
        assert ws.steering_from is None and ws.steering.classes == [30, 60, 90, 120, 150], bad
        ws.close_all()


def _meta(zip_path: Path) -> dict:
    with zipfile.ZipFile(zip_path) as z:
        return json.loads(z.read("meta.json"))


def test_crop_lines_snap_to_one_percent_and_keep_a_minimum_height():
    from common.preprocess import crop_from_dict, crop_to_dict, describe_crop, fit_crop
    assert fit_crop(0.153, 1.0) == (0.15, 1.0) and fit_crop(-0.2, 1.7) == (0.0, 1.0)
    assert fit_crop(0.97, 1.0, "top") == (0.9, 1.0)             # 위 선은 아래 선의 10 % 앞에서 멈춘다
    assert fit_crop(0.4, 0.42, "bottom") == (0.4, 0.5)          # 아래 선은 위 선의 10 % 뒤에서 멈춘다
    assert fit_crop(0.02, 0.05, "top") == (0.0, 0.1) and fit_crop(0.97, 0.99, "bottom") == (0.9, 1.0)   # 끝에 몰린 경우
    assert crop_from_dict(crop_to_dict((0.15, 0.9))) == (0.15, 0.9) and crop_from_dict(None) is None
    for bad in ({}, {"crop_top_ratio": 0.5}, {"crop_top_ratio": "x", "crop_bottom_ratio": 1}, [0.1, 1.0],
                {"crop_top_ratio": 0.6, "crop_bottom_ratio": 0.5}, {"crop_top_ratio": -0.1, "crop_bottom_ratio": 1.0}):
        with pytest.raises(ValueError):
            crop_from_dict(bad)
    assert describe_crop((0.15, 1.0)) == "위쪽 15% · 아래쪽 0% 잘라 냄" and describe_crop((0.4, 0.9)) == "위쪽 40% · 아래쪽 10% 잘라 냄"


def test_crop_starts_at_the_default_and_goes_into_the_dataset(tmp_path):
    """자르는 범위는 사진을 보고 정하는 값이라 가공 프로그램이 정하고, 데이터셋 zip 의 meta.json 이 학습기로 가져간다."""
    d = make_session(tmp_path / "in", n=6, mark=(0, 0))
    ws = Workspace(load_config(DEFAULT_CONFIG_PATH))
    ws.open(d)
    assert ws.crop == (0.4, 1.0) and not ws.crop_chosen                    # 기본값: 위 40 % 를 잘라 낸다
    assert _meta(build_dataset(ws, tmp_path / "out", now=datetime(2026, 11, 23, 9, 0)))["preprocess"] == \
        {"crop_top_ratio": 0.4, "crop_bottom_ratio": 1.0}
    ws.save()
    assert "preprocess" not in json.loads((d / "curation.json").read_text(encoding="utf-8"))   # 정한 적 없는 값은 적지 않는다
    assert ws.set_crop(top=0.153) and ws.crop == (0.15, 1.0) and ws.crop_chosen
    assert ws.set_crop(bottom=0.9) and ws.crop == (0.15, 0.9)
    assert not ws.set_crop(bottom=0.9)                                      # 같은 값이면 바뀐 것이 없다
    assert ws.set_crop(top=0.99) and ws.crop == (0.8, 0.9)                  # 아래 선을 넘지 못한다
    ws.set_crop(top=0.15)
    assert ws.preprocessor.crop == (0.15, 0.9) and ws.preprocessor.out_width == 200
    assert _meta(build_dataset(ws, tmp_path / "out", now=datetime(2026, 11, 23, 9, 1)))["preprocess"] == \
        {"crop_top_ratio": 0.15, "crop_bottom_ratio": 0.9}
    ws.save()
    saved = json.loads((d / "curation.json").read_text(encoding="utf-8"))
    assert saved["preprocess"] == {"crop_top_ratio": 0.15, "crop_bottom_ratio": 0.9} and len(saved["excluded"]) == 1
    ws.close_all()
    assert ws.crop == (0.4, 1.0) and not ws.crop_chosen                    # 다 닫으면 기본값으로
    ws.open(d)                                                              # 다시 열면 정해 둔 범위로 시작한다
    assert ws.crop == (0.15, 0.9) and ws.crop_chosen and ws.crop_from == "20261110_1906"
    ws.close_all()


def test_one_crop_for_the_whole_workspace(tmp_path):
    a = make_session(tmp_path / "in", "20261110_1906", n=5, mark=(0, 0))
    b = make_session(tmp_path / "in", "20261110_1930", n=5, mark=(0, 0))
    c = make_session(tmp_path / "in", "20261110_2000", n=5, mark=(0, 0))
    ws = Workspace(load_config(DEFAULT_CONFIG_PATH))
    ws.open(b); ws.set_crop(top=0.3); ws.save(); ws.close_all()             # b 에만 30 % 로 정해 둔다
    ws.open(a)                                                              # 정해 둔 범위가 없는 세션: 기본값
    assert ws.crop == (0.4, 1.0) and not ws.crop_chosen
    ws.open(b)                                                              # 아직 정해지지 않았으면 b 의 값을 따른다
    assert ws.crop == (0.3, 1.0) and ws.crop_from == "20261110_1930" and not ws.crop_replaced
    ws.set_crop(top=0.2)                                                    # 선을 옮기면 열려 있는 세션 모두에 적힌다
    assert ws.curations["20261110_1906"].crop == ws.curations["20261110_1930"].crop == (0.2, 1.0)
    state = ws.crop_state()
    ws.set_crop(bottom=0.8)
    ws.restore_crop_state(state)                                            # 되돌리기
    assert ws.crop == (0.2, 1.0) and ws.curations["20261110_1930"].crop == (0.2, 1.0)
    ws.save(); ws.close_all()
    ws.open(c); ws.set_crop(top=0.1); ws.save(); ws.close_all()
    ws.open(a); ws.open(c)                                                  # a 는 20 %, c 는 10 %: 먼저 연 쪽을 따른다
    assert ws.crop == (0.2, 1.0) and ws.crop_replaced == {"20261110_2000": (0.1, 1.0)}
    assert ws.curations["20261110_2000"].crop == (0.2, 1.0)
    ws.close_all()
    assert not ws.crop_replaced


def test_unreadable_saved_crop_keeps_the_exclusions(tmp_path):
    d = make_session(tmp_path / "in", n=6, mark=(1, 2))
    ws = Workspace(load_config(DEFAULT_CONFIG_PATH))
    src = ws.open(d)
    ws.curation(src.frames[5]).exclude([src.frames[5].name])
    ws.set_crop(top=0.2); ws.save(); ws.close_all()
    path = d / "curation.json"
    for bad in ("넓게", {"crop_top_ratio": 0.9, "crop_bottom_ratio": 0.1}, [1, 2]):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["preprocess"] = bad
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        ws.open(d)
        assert ws.crop == (0.4, 1.0) and not ws.crop_chosen, bad
        assert len(ws.curations[src.name].excluded) == 3                    # □ 구간 2장 + 손으로 뺀 1장은 그대로
        ws.close_all()


def test_model_view_is_what_the_preprocessor_makes(tmp_path):
    """화면의 "모델이 보는 사진"은 학습기·추론기가 쓰는 전처리 그대로다 (따로 만든 흉내가 아니다)."""
    from common.preprocess import Preprocessor
    from curator.core import model_view
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:120] = (255, 0, 0)                                                 # 위 25 % 는 파랑(BGR), 그 아래는 초록
    img[120:] = (0, 255, 0)
    data = cv2.imencode(".png", img)[1].tobytes()
    pre = Preprocessor()
    view = model_view(data, pre.with_crop((0.25, 1.0)))
    assert view.shape == (66, 200, 3) and view.dtype == np.uint8
    assert (view == (0, 255, 0)).all()                                      # 파랑 부분이 모두 잘려 나갔다 (RGB 순서)
    assert (model_view(data, pre.with_crop((0.0, 0.25))) == (0, 0, 255)).all()
    assert np.array_equal(view, pre.with_crop((0.25, 1.0)).to_rgb_uint8(img))
    assert model_view(b"", pre) is None and model_view(b"not an image", pre) is None


def test_build_dataset_refuses_when_everything_is_excluded(tmp_path):
    ws = Workspace(load_config(DEFAULT_CONFIG_PATH))
    make_session(tmp_path / "in", n=3, mark=(0, 2))
    ws.open(tmp_path / "in" / "20261110_1906")
    with pytest.raises(ValueError):
        build_dataset(ws, tmp_path / "out")
