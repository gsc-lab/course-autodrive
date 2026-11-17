"""가공 도구의 핵심 로직. GUI 와 무관하고 pytest 로 검증한다.

세션 소스: 수집기가 만든 세션 폴더(dataset/YYYYMMDD_HHMM/) 또는 그 폴더를 압축한 zip.
제외 상태: 세션 옆의 curation.json 에 소프트 기록 (원본은 절대 건드리지 않는다). 처음 열 때 marks.csv 구간의
프레임은 "mark" 이유로 미리 제외된다 (Ctrl+Z 또는 복구로 되돌릴 수 있다).
데이터셋 zip: images/ 에 수집 순서대로 다시 번호를 매긴 사진, labels.csv, meta.json (CLAUDE.md 의 계약).
"""
from __future__ import annotations

import csv
import io
import json
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from common import __version__
from common.config import Config
from common.filename import (FrameInfo, make_dataset_image_name, make_dataset_stamp, make_dataset_zip_name,
                             try_parse_frame_name)
from common.steering import SteeringModel

DATASET_SCHEMA_VERSION = 1
CURATION_SCHEMA_VERSION = 1
REASON_MARK = "mark"        # marks.csv 구간 안이라 자동 제외
REASON_USER = "user"        # 사람이 제외


class AlreadyOpen(ValueError):
    """같은 이름의 세션이 이미 열려 있다. 오류라기보다 알림이다."""


class NoFrames(ValueError):
    """수집기가 만든 사진(…_aNNN_sNNN.jpg)이 하나도 없는 폴더/zip."""


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
        self.name = self.path.stem if self.is_zip else self.path.name
        self.frames: List[Frame] = self._list_frames()
        self.marks: List[MarkInterval] = self._read_marks()
        self.meta: dict = self._read_meta()

    # -- 내부 --------------------------------------------------------------
    def _entries(self) -> Iterable[str]:
        """파일 이름(경로 없이)을 돌려준다. zip 이면 안의 경로를 _members 에 기억한다."""
        if self._zip is None:
            for p in sorted(self.path.iterdir()):
                if p.is_file():
                    yield p.name
        else:
            for m in self._zip.namelist():
                if m.endswith("/"):
                    continue
                base = m.rsplit("/", 1)[-1]
                self._members[base] = m
                yield base

    def _list_frames(self) -> List[Frame]:
        out = []
        for base in self._entries():
            info = try_parse_frame_name(base)
            if info is not None:
                out.append(Frame(self.name, info))
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
            return json.loads(text)
        except ValueError:
            return {}

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
    _undo: List[Tuple[str, Dict[str, Optional[str]]]] = field(default_factory=list, repr=False)

    @classmethod
    def load_or_init(cls, src: SessionSource) -> "Curation":
        p = src.curation_path
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                return cls(excluded=dict(data.get("excluded", {})))
            except (ValueError, OSError):
                pass
        c = cls()
        for f in src.frames_in_marks():            # 처음 열 때: □ 로 표시한 구간은 미리 제외
            c.excluded[f.name] = REASON_MARK
        return c

    def save(self, src: SessionSource) -> None:
        data = {"schema_version": CURATION_SCHEMA_VERSION, "session": src.name,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "tool": {"name": "curator", "version": __version__},
                "excluded": dict(sorted(self.excluded.items()))}
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


# ---------------------------------------------------------------------------
# 작업 공간: 세션 여러 개 + 각 세션의 제외 상태
# ---------------------------------------------------------------------------
class Workspace:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.steering = SteeringModel(cfg.steering)
        self.sessions: List[SessionSource] = []
        self.curations: Dict[str, Curation] = {}

    def open(self, path: Path | str) -> SessionSource:
        src = SessionSource(path)
        if any(s.name == src.name for s in self.sessions):
            src.close()
            raise AlreadyOpen(src.name)
        if not src.frames:
            src.close()
            raise NoFrames(src.name)
        self.sessions.append(src)
        self.curations[src.name] = Curation.load_or_init(src)
        return src

    def close_all(self) -> None:
        for s in self.sessions:
            s.close()
        self.sessions.clear()
        self.curations.clear()

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
    cam = ws.cfg.camera
    labels = io.StringIO()
    w = csv.writer(labels, lineterminator="\n")
    w.writerow(["index", "image", "servo_angle", "motor_speed", "source"])
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as z:   # JPEG 은 더 안 줄어든다
        for i, f in enumerate(frames):
            img_name = make_dataset_image_name(stamp, i)
            z.writestr(f"images/{img_name}", ws.read_image(f))
            w.writerow([i, img_name, f.info.angle, f.info.speed, f.key])
            if progress and (i % 50 == 0 or i == len(frames) - 1):
                progress(i + 1, len(frames))
        z.writestr("labels.csv", labels.getvalue())
        meta = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "created_at": now.isoformat(timespec="seconds"),
            "tool": {"name": "curator", "version": __version__},
            "steering": {"mode": ws.cfg.steering.mode, "classes": list(ws.steering.classes),
                         "min": ws.cfg.steering.min, "max": ws.cfg.steering.max},
            "image": {"width": cam.width, "height": cam.height, "format": "jpg"},
            "sessions": [s.name for s in sorted(ws.sessions, key=lambda s: s.name)],
            "frames": len(frames),
            "counts_by_angle": {str(k): v for k, v in sorted(inc.items())},
            "excluded": sum(exc.values()),
            "excluded_by_angle": {str(k): v for k, v in sorted(exc.items())},
        }
        z.writestr("meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
    return zip_path
