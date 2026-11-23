"""가공 도구의 핵심 로직. GUI 와 무관하고 pytest 로 검증한다.

세션 소스: 수집기가 만든 세션 폴더(dataset/YYYYMMDD_HHMM/) 또는 그 폴더를 압축한 zip.
제외 상태: 세션 옆의 curation.json 에 소프트 기록 (원본은 절대 건드리지 않는다). 처음 열 때 marks.csv 구간의
프레임은 "mark" 이유로 미리 제외된다 (Ctrl+Z 또는 복구로 되돌릴 수 있다).
자르는 범위: 모델에 넣을 세로 범위(위·아래 비율)를 사진을 보며 정한다. 사진을 자르지는 않고 값만 적어 둔다:
작업 중에는 curation.json 에, 데이터셋을 만들 때는 meta.json 에. 학습기가 그 값대로 잘라 모델에 넣는다.
데이터셋 zip: images/ 에 수집 순서대로 다시 번호를 매긴 사진, labels.csv, meta.json (CLAUDE.md 의 계약).
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from common import __version__
from common.config import Config, SteeringConfig
from common.filename import (FrameInfo, make_dataset_image_name, make_dataset_stamp, make_dataset_zip_name,
                             try_parse_frame_name)
from common.preprocess import Preprocessor, crop_from_dict, crop_to_dict, fit_crop
from common.steering import SteeringModel

log = logging.getLogger(__name__)

DATASET_SCHEMA_VERSION = 1
CURATION_SCHEMA_VERSION = 1
REASON_MARK = "mark"        # marks.csv 구간 안이라 자동 제외
REASON_USER = "user"        # 사람이 제외


class AlreadyOpen(ValueError):
    """같은 이름의 세션이 이미 열려 있다. 오류라기보다 알림이다."""


class NoFrames(ValueError):
    """수집기가 만든 사진(…_aNNN_sNNN.jpg)이 하나도 없는 폴더/zip."""


class BrokenImage(ValueError):
    """포함된 사진 하나가 끝까지 저장되지 않았다 (전원이 끊긴 세션). 그 사진을 제외해야 데이터셋을 만들 수 있다."""

    def __init__(self, frame: "Frame"):
        super().__init__(f"저장이 덜 된 사진입니다: {frame.key}")
        self.frame = frame


def image_is_complete(name: str, data: bytes) -> bool:
    """파일이 처음부터 끝까지 저장됐는지 머리와 꼬리 표식으로 본다 (내용을 풀어 보지는 않는다)."""
    if name.lower().endswith(".png"):
        return data.startswith(b"\x89PNG\r\n\x1a\n") and data.rstrip(b"\x00").endswith(b"IEND\xaeB`\x82")
    return data.startswith(b"\xff\xd8") and data.rstrip(b"\x00").endswith(b"\xff\xd9")


# ---------------------------------------------------------------------------
# 세션 소스
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Frame:
    session: str          # 세션 이름 (YYYYMMDD_HHMM)
    info: FrameInfo       # 파일 이름에서 읽은 시각·각도·속도

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def key(self) -> str:
        """작업 공간 안에서 유일한 이름 (세션/파일)."""
        return f"{self.session}/{self.info.name}"


@dataclass
class MarkInterval:
    start: datetime
    end: datetime
    note: str = ""


class SessionSource:
    """세션 폴더 또는 zip 하나. 프레임 목록, 사진 바이트, marks.csv, 메타를 제공한다."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.is_zip = self.path.is_file() and self.path.suffix.lower() == ".zip"
        self._zip: Optional[zipfile.ZipFile] = zipfile.ZipFile(self.path) if self.is_zip else None
        self._members: Dict[str, str] = {}     # 파일 이름 → zip 안의 경로 (zip 일 때)
        self.empty: List[str] = []             # 0바이트 사진 (전원이 끊겨 저장되지 못한 것). 프레임 목록에 넣지 않는다
        self.name = self.path.stem if self.is_zip else self.path.name
        try:
            self.frames: List[Frame] = self._list_frames()
            self.marks: List[MarkInterval] = self._read_marks()
            self.meta: dict = self._read_meta()
        except Exception:
            self.close()
            raise

    # -- 내부 --------------------------------------------------------------
    def _entries(self) -> Iterable[Tuple[str, int]]:
        """(파일 이름, 크기)를 돌려준다. zip 이면 안의 경로를 _members 에 기억한다."""
        if self._zip is None:
            for p in sorted(self.path.iterdir()):
                if p.is_file():
                    yield p.name, p.stat().st_size
        else:
            for zi in self._zip.infolist():
                m = zi.filename
                if m.endswith("/"):
                    continue
                base = m.rsplit("/", 1)[-1]
                if base in self._members:
                    continue
                self._members[base] = m
                yield base, zi.file_size

    def _list_frames(self) -> List[Frame]:
        out = []
        for base, size in self._entries():
            info = try_parse_frame_name(base)
            if info is None:
                continue
            if size == 0:
                self.empty.append(base)
                continue
            out.append(Frame(self.name, info))
        if self.empty:
            log.warning("%s: 0바이트 사진 %d장을 뺐습니다 (전원이 끊겨 저장되지 못한 사진)", self.name, len(self.empty))
        out.sort(key=lambda f: (f.info.timestamp, f.info.name))
        return out

    def read_bytes(self, filename: str) -> bytes:
        if self._zip is None:
            return (self.path / filename).read_bytes()
        member = self._members.get(filename)
        if member is None:
            raise FileNotFoundError(filename)
        return self._zip.read(member)

    def _read_text(self, filename: str) -> Optional[str]:
        try:
            return self.read_bytes(filename).decode("utf-8")
        except (FileNotFoundError, KeyError):
            return None

    def _read_marks(self) -> List[MarkInterval]:
        text = self._read_text("marks.csv")
        if not text:
            return []
        out = []
        for row in csv.DictReader(io.StringIO(text)):
            try:
                out.append(MarkInterval(datetime.fromisoformat(row["t_start_iso"]),
                                        datetime.fromisoformat(row["t_end_iso"]), row.get("note", "")))
            except (KeyError, ValueError, TypeError):
                continue
        return out

    def _read_meta(self) -> dict:
        text = self._read_text("session_meta.json")
        if not text:
            return {}
        try:
            data = json.loads(text)
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}

    def frames_in_marks(self) -> List[Frame]:
        return [f for f in self.frames if any(m.start <= f.info.timestamp <= m.end for m in self.marks)]

    @property
    def curation_path(self) -> Path:
        return self.path.with_suffix(".curation.json") if self.is_zip else self.path / "curation.json"

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()


# ---------------------------------------------------------------------------
# 제외 상태 (소프트, 되돌리기 가능)
# ---------------------------------------------------------------------------
@dataclass
class Curation:
    excluded: Dict[str, str] = field(default_factory=dict)   # 프레임 이름 → 이유
    crop: Optional[Tuple[float, float]] = None               # 사람이 정한 자르는 범위 (위, 아래). 정한 적이 없으면 None
    _undo: List[Tuple[str, Dict[str, Optional[str]]]] = field(default_factory=list, repr=False)

    @classmethod
    def load_or_init(cls, src: SessionSource) -> "Curation":
        p = src.curation_path
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                exc = data.get("excluded") if isinstance(data, dict) else None
                if isinstance(exc, dict):
                    return cls(excluded={str(k): str(v) for k, v in exc.items()}, crop=_saved_crop(data, p.name))
            except (ValueError, OSError):
                pass
            log.warning("%s 를 읽을 수 없어 처음 여는 것으로 다룹니다 (□ 구간만 미리 제외)", p.name)
        c = cls()
        for f in src.frames_in_marks():            # 처음 열 때: □ 로 표시한 구간은 미리 제외
            c.excluded[f.name] = REASON_MARK
        return c

    def save(self, src: SessionSource) -> None:
        data = {"schema_version": CURATION_SCHEMA_VERSION, "session": src.name,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "tool": {"name": "curator", "version": __version__},
                "excluded": dict(sorted(self.excluded.items()))}
        if self.crop is not None:                   # 정한 적이 없으면 적지 않는다: 기본값이 "정한 값"처럼 남지 않게
            data["preprocess"] = crop_to_dict(self.crop)
        src.curation_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def is_excluded(self, name: str) -> bool:
        return name in self.excluded

    def _apply(self, label: str, changes: Dict[str, Optional[str]]) -> None:
        """changes: 이름 → 새 이유(None 이면 복구). 되돌리기 정보를 쌓는다."""
        before = {n: self.excluded.get(n) for n in changes}
        if all(before[n] == v for n, v in changes.items()):
            return
        for n, v in changes.items():
            if v is None:
                self.excluded.pop(n, None)
            else:
                self.excluded[n] = v
        self._undo.append((label, before))

    def exclude(self, names: Iterable[str], reason: str = REASON_USER) -> None:
        self._apply("exclude", {n: reason for n in names})

    def restore(self, names: Iterable[str]) -> None:
        self._apply("restore", {n: None for n in names})

    def toggle(self, name: str) -> None:
        if self.is_excluded(name):
            self.restore([name])
        else:
            self.exclude([name])

    def undo(self) -> Optional[str]:
        if not self._undo:
            return None
        label, before = self._undo.pop()
        for n, v in before.items():
            if v is None:
                self.excluded.pop(n, None)
            else:
                self.excluded[n] = v
        return label

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)


def _saved_crop(data: dict, where: str) -> Optional[Tuple[float, float]]:
    """curation.json 에 적힌 자르는 범위. 없으면 None. 읽을 수 없는 값은 버린다 (제외 목록까지 잃지 않게)."""
    try:
        crop = crop_from_dict(data.get("preprocess"))
    except ValueError as e:
        log.warning("%s: %s. 기본 범위를 씁니다", where, e)
        return None
    return fit_crop(*crop) if crop else None


def model_view(data: bytes, pre: Preprocessor) -> Optional[np.ndarray]:
    """사진 파일의 내용 → 모델에 들어가는 사진 (out_height, out_width, 3) RGB uint8. 풀 수 없는 사진이면 None.

    학습기와 추론기가 쓰는 Preprocessor 를 그대로 쓴다: 화면의 "모델이 보는 사진"이 실제로 모델에 들어가는 것과 같다.
    """
    try:
        bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR) if data else None
        return None if bgr is None else pre.to_rgb_uint8(bgr)
    except (cv2.error, ValueError):
        return None


# ---------------------------------------------------------------------------
# 작업 공간: 세션 여러 개 + 각 세션의 제외 상태
# ---------------------------------------------------------------------------
class SteeringMismatch(ValueError):
    """조향 설정이 다른 세션은 한 데이터셋으로 묶을 수 없다."""


def session_steering(meta: dict) -> Optional[SteeringConfig]:
    """세션을 촬영할 때의 조향 설정 (수집기가 session_meta.json 에 넣어 둔 설정 사본). 없거나 읽을 수 없으면 None."""
    config = meta.get("config")
    snap = config.get("steering") if isinstance(config, dict) else None
    if not isinstance(snap, dict):
        return None
    base = SteeringConfig()
    try:
        cfg = SteeringConfig(
            mode=str(snap.get("mode", base.mode)), classes=sorted({int(c) for c in snap["classes"]}),
            center=int(snap["center"]), min=int(snap["min"]), max=int(snap["max"]),
            step_deg=int(snap.get("step_deg", base.step_deg)), invert=bool(snap.get("invert", base.invert)),
            hysteresis=float(snap.get("hysteresis", base.hysteresis)))
    except (KeyError, TypeError, ValueError):
        return None
    return cfg if cfg.classes and cfg.min < cfg.max else None


def _steering_key(cfg: SteeringConfig) -> tuple:
    """라벨의 뜻을 정하는 값들: 클래스 목록, 직진 각도, 좌우 방향. 이것이 다르면 같은 각도가 다른 뜻이 된다."""
    return tuple(sorted(int(c) for c in cfg.classes)), int(cfg.center), bool(cfg.invert)


class Workspace:
    """열어 둔 세션들.

    조향 설정(self.steering)은 이 PC 의 config.yaml 이 아니라 **세션을 촬영할 때의 설정**을 따른다. 학생이 차(Jetson)에서
    steering.classes 를 바꿔 촬영해도 PC 의 config.yaml 은 그대로인 경우가 많고, 그러면 클래스별 장수와 데이터셋의
    meta.json 이 실제 데이터와 어긋난다. 촬영 설정이 없는 세션(옛 세션, 손으로 만든 폴더)만 config.yaml 을 쓴다.

    자르는 범위(self.crop)는 작업 공간에 하나다: 데이터셋 하나에 범위 하나. 정해 둔 범위가 있는 세션을 처음 열면 그 값으로,
    없으면 기본값으로 시작한다. 한번 정해지면(읽었거나 사람이 선을 옮겼거나) 열려 있는 세션 모두가 그 값을 따른다.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.steering = SteeringModel(cfg.steering)
        self.steering_from: Optional[str] = None        # 조향 설정을 따온 세션 이름. None 이면 config.yaml
        self.sessions: List[SessionSource] = []
        self.curations: Dict[str, Curation] = {}
        self._pre = Preprocessor.from_config(cfg.preprocess)    # 모델 입력 크기와 기본 범위
        self.default_crop: Tuple[float, float] = fit_crop(*self._pre.crop)
        self.crop: Tuple[float, float] = self.default_crop
        self.crop_chosen = False                        # 정해진 범위인가 (저장된 값을 읽었거나 사람이 정했다)
        self.crop_from: Optional[str] = None            # 범위를 읽어 온 세션 이름
        self.crop_replaced: Dict[str, Tuple[float, float]] = {}   # 세션 → 그 세션에 적혀 있던 다른 범위 (화면에 알리려고)

    def open(self, path: Path | str) -> SessionSource:
        src = SessionSource(path)
        if any(s.name == src.name for s in self.sessions):
            src.close()
            raise AlreadyOpen(src.name)
        if not src.frames:
            src.close()
            raise NoFrames(src.name)
        recorded = session_steering(src.meta)
        if recorded and self.steering_from and _steering_key(recorded) != _steering_key(self.steering.cfg):
            src.close()
            raise SteeringMismatch(
                f"조향 설정이 열려 있는 세션({self.steering_from})과 다릅니다: 클래스 {recorded.classes}, 직진 {recorded.center}°"
                f"{', 좌우 반대' if recorded.invert != self.steering.invert else ''} ↔ 클래스 {self.steering.classes}, "
                f"직진 {self.steering.center}°. 조향 설정이 같은 세션끼리만 한 데이터셋으로 묶을 수 있습니다")
        try:
            curation = Curation.load_or_init(src)
        except Exception:
            src.close()
            raise
        self.sessions.append(src)
        self.curations[src.name] = curation
        if recorded and not self.steering_from:
            self.steering = SteeringModel(recorded)
            self.steering_from = src.name
        if not self.crop_chosen:
            if curation.crop is not None:               # 이 세션에 정해 둔 범위가 있다: 그 값으로 시작한다
                self.crop, self.crop_chosen, self.crop_from = curation.crop, True, src.name
        else:                                           # 이미 정해진 범위가 있다: 이 세션도 그 값을 따른다
            if curation.crop is not None and curation.crop != self.crop:
                self.crop_replaced[src.name] = curation.crop
            curation.crop = self.crop
        return src

    @property
    def steering_differs_from_config(self) -> bool:
        """열어 둔 세션의 조향 설정이 이 PC 의 config.yaml 과 다른가 (화면에 알려 주기 위해)."""
        return bool(self.steering_from) and _steering_key(self.steering.cfg) != _steering_key(self.cfg.steering)

    # -- 자르는 범위 -------------------------------------------------------
    @property
    def preprocessor(self) -> Preprocessor:
        """지금 범위로 사진을 모델 입력으로 바꾸는 전처리기 (화면의 "모델이 보는 사진")."""
        return self._pre.with_crop(self.crop)

    def set_crop(self, top: Optional[float] = None, bottom: Optional[float] = None) -> bool:
        """자르는 선을 옮긴다 (비율 0~1, 주지 않은 쪽은 그대로). 값은 fit_crop 으로 맞춘다. 돌려주는 값: 범위가 바뀌었는가."""
        moved = "bottom" if top is None else "top"
        new = fit_crop(self.crop[0] if top is None else top, self.crop[1] if bottom is None else bottom, moved)
        changed = new != self.crop
        self.crop, self.crop_chosen, self.crop_from = new, True, None
        for c in self.curations.values():
            c.crop = new
        return changed

    def crop_state(self) -> tuple:
        """되돌리기에 쓸 지금 상태."""
        return self.crop, self.crop_chosen, self.crop_from, {name: c.crop for name, c in self.curations.items()}

    def restore_crop_state(self, state: tuple) -> None:
        self.crop, self.crop_chosen, self.crop_from, per_session = state
        for name, crop in per_session.items():
            if name in self.curations:
                self.curations[name].crop = crop

    def close_all(self) -> None:
        for s in self.sessions:
            s.close()
        self.sessions.clear()
        self.curations.clear()
        self.steering = SteeringModel(self.cfg.steering)
        self.steering_from = None
        self.crop, self.crop_chosen, self.crop_from = self.default_crop, False, None
        self.crop_replaced.clear()

    def frames(self) -> List[Frame]:
        """모든 세션의 프레임을 수집 순서(세션 순 → 시각 순)로."""
        out: List[Frame] = []
        for s in sorted(self.sessions, key=lambda s: s.name):
            out.extend(s.frames)
        return out

    def source(self, frame: Frame) -> SessionSource:
        return next(s for s in self.sessions if s.name == frame.session)

    def curation(self, frame: Frame) -> Curation:
        return self.curations[frame.session]

    def is_excluded(self, frame: Frame) -> bool:
        return self.curation(frame).is_excluded(frame.name)

    def read_image(self, frame: Frame) -> bytes:
        return self.source(frame).read_bytes(frame.name)

    def save(self) -> None:
        for s in self.sessions:
            self.curations[s.name].save(s)

    def counts(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        """(포함된 프레임의 클래스별 수, 제외된 프레임의 클래스별 수). 각도는 가장 가까운 클래스로 묶는다."""
        inc: Dict[int, int] = {c: 0 for c in self.steering.classes}
        exc: Dict[int, int] = {c: 0 for c in self.steering.classes}
        for f in self.frames():
            c = self.steering.quantize(f.info.angle)
            (exc if self.is_excluded(f) else inc)[c] += 1
        return inc, exc


# ---------------------------------------------------------------------------
# 데이터셋 zip
# ---------------------------------------------------------------------------
def build_dataset(ws: Workspace, out_dir: Path | str, now: Optional[datetime] = None,
                  progress=None) -> Path:
    """포함된 프레임을 수집 순서로 다시 번호를 매겨 dataset_YYYYMMDD_HHMM.zip 으로 쓴다. 돌려주는 값은 zip 경로."""
    now = now or datetime.now()
    stamp = make_dataset_stamp(now)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / make_dataset_zip_name(stamp)
    frames = [f for f in ws.frames() if not ws.is_excluded(f)]
    if not frames:
        raise NoFrames("포함된 사진이 없습니다")
    inc, exc = ws.counts()
    labels = io.StringIO()
    w = csv.writer(labels, lineterminator="\n")
    w.writerow(["index", "image", "servo_angle", "motor_speed", "source"])
    part_path = zip_path.with_name(zip_path.name + ".part")
    try:
        _write_dataset(ws, part_path, frames, stamp, now, inc, exc, labels, w, progress)
        os.replace(part_path, zip_path)
    except BaseException:
        part_path.unlink(missing_ok=True)
        raise
    return zip_path


def _write_dataset(ws: Workspace, path: Path, frames: List[Frame], stamp: str, now: datetime,
                   inc: Dict[int, int], exc: Dict[int, int], labels: io.StringIO, w, progress) -> None:
    cam = ws.cfg.camera
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as z:   # JPEG 은 더 안 줄어든다
        for i, f in enumerate(frames):
            ext = f.name.rsplit(".", 1)[-1].lower()             # 원본 확장자를 그대로 (내용과 이름이 어긋나지 않게)
            img_name = make_dataset_image_name(stamp, i, ext)
            data = ws.read_image(f)
            if not image_is_complete(f.name, data):
                raise BrokenImage(f)
            z.writestr(f"images/{img_name}", data)
            w.writerow([i, img_name, f.info.angle, f.info.speed, f.key])
            if progress and (i % 50 == 0 or i == len(frames) - 1):
                progress(i + 1, len(frames))
        z.writestr("labels.csv", labels.getvalue())
        meta = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "created_at": now.isoformat(timespec="seconds"),
            "tool": {"name": "curator", "version": __version__},
            # 촬영할 때의 조향 설정 (Workspace 참고). 학습기는 클래스를 여기서 읽고, center 를 좌우 반전의 대칭 축으로 쓴다
            "steering": {"mode": ws.steering.mode, "classes": list(ws.steering.classes),
                         "center": ws.steering.center, "min": ws.steering.min, "max": ws.steering.max},
            "image": {"width": cam.width, "height": cam.height, "format": "jpg"},
            # 모델에 넣을 세로 범위 (사진 높이에 대한 비율). 사진은 원본 그대로 들어 있고, 학습기가 이 값대로 자른다
            "preprocess": crop_to_dict(ws.crop),
            "sessions": [s.name for s in sorted(ws.sessions, key=lambda s: s.name)],
            "frames": len(frames),
            "counts_by_angle": {str(k): v for k, v in sorted(inc.items())},
            "excluded": sum(exc.values()),
            "excluded_by_angle": {str(k): v for k, v in sorted(exc.items())},
            "empty_files_skipped": sum(len(s.empty) for s in ws.sessions),
        }
        z.writestr("meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
