"""세션 폴더 기록. 수집기가 쓰고 가공기가 읽는다.

dataset/YYYYMMDD_HHMM/
  YYYYMMDD_HHMMSS_mmm_aAAA_sSSS.jpg   프레임 (파일 이름이 라벨)
  session_meta.json                  설정 스냅샷, 카메라 정보, 클래스별 장수
  events.csv                         모든 입력 이벤트 (t_iso, t_mono, kind, value)
  marks.csv                          운전자가 표시한 불량 구간 (t_start_iso, t_end_iso, note)
  collector.log
"""
from __future__ import annotations

import csv
import json
import platform
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from common import __version__
from common.filename import make_frame_name, make_session_name

SCHEMA_VERSION = 1
META_FLUSH_EVERY = 50


class SessionWriter:
    def __init__(self, root: str | Path, config_snapshot: dict, jpeg_quality: int = 95,
                 extra_meta: Optional[dict] = None, started_at: Optional[datetime] = None):
        self.started_at = started_at or datetime.now()
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        base = make_session_name(self.started_at)
        self.path = root / base
        n = 2
        while self.path.exists():
            self.path = root / f"{base}_{n}"
            n += 1
        self.path.mkdir()
        self.name = self.path.name
        self.jpeg_quality = int(jpeg_quality)
        self.config_snapshot = config_snapshot
        self.extra_meta = dict(extra_meta or {})
        self.counts: Counter = Counter()
        self.saved = 0
        self.marks = 0
        self.last_frame_name: Optional[str] = None

        self._events_f = open(self.path / "events.csv", "w", newline="", encoding="utf-8")
        self._events = csv.writer(self._events_f)
        self._events.writerow(["t_iso", "t_mono", "kind", "value"])
        self._marks_f = open(self.path / "marks.csv", "w", newline="", encoding="utf-8")
        self._marks = csv.writer(self._marks_f)
        self._marks.writerow(["t_start_iso", "t_end_iso", "note"])
        self.write_meta()

    # ------------------------------------------------------------------ frames
    def save_frame(self, frame_bgr: np.ndarray, angle: int, speed: int, ts: Optional[datetime] = None) -> str:
        ts = ts or datetime.now()
        name = make_frame_name(ts, angle, speed)
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG 인코딩 실패")
        (self.path / name).write_bytes(buf.tobytes())
        self.counts[int(angle)] += 1
        self.saved += 1
        self.last_frame_name = name
        if self.saved % META_FLUSH_EVERY == 0:
            self.write_meta()
        return name

    # ------------------------------------------------------------------ logs
    def log_event(self, kind: str, value=None, t_mono: Optional[float] = None) -> None:
        self._events.writerow([datetime.now().isoformat(timespec="milliseconds"),
                               f"{t_mono if t_mono is not None else time.monotonic():.3f}", kind,
                               "" if value is None else value])
        self._events_f.flush()  # 실행 중에도 tail -f 로 볼 수 있게

    def mark(self, seconds: float, note: str = "", now: Optional[datetime] = None) -> None:
        now = now or datetime.now()
        start = now - timedelta(seconds=seconds)
        self._marks.writerow([start.isoformat(timespec="milliseconds"), now.isoformat(timespec="milliseconds"), note])
        self._marks_f.flush()
        self.marks += 1

    # ------------------------------------------------------------------ meta
    def meta(self, ended: bool = False) -> dict:
        d = {
            "schema_version": SCHEMA_VERSION,
            "session": self.name,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "ended_at": datetime.now().isoformat(timespec="seconds") if ended else None,
            "tool": {"name": "collector", "version": __version__},
            "host": {"node": platform.node(), "platform": platform.platform(), "python": platform.python_version()},
            "frames_saved": self.saved,
            "counts_by_angle": {str(k): v for k, v in sorted(self.counts.items())},
            "marks": self.marks,
            "config": self.config_snapshot,
        }
        d.update(self.extra_meta)
        return d

    def write_meta(self, ended: bool = False) -> None:
        tmp = self.path / "session_meta.json.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.meta(ended), f, ensure_ascii=False, indent=2)
        tmp.replace(self.path / "session_meta.json")
        self._events_f.flush()

    def close(self) -> None:
        self.write_meta(ended=True)
        self._events_f.close()
        self._marks_f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
