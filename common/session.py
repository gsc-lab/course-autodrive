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
import logging
import os
import platform
import threading
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from common import __version__
from common.filename import make_frame_name, make_session_name

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
META_FLUSH_EVERY = 50
SYNC_EVERY_SEC = 2.0        # 이 주기로 새 파일을 디스크에 확정한다. 전원이 끊기면 잃는 양이 이 정도로 줄어든다


class SessionWriter:
    def __init__(self, root: str | Path, config_snapshot: dict, jpeg_quality: int = 95,
                 extra_meta: Optional[dict] = None, started_at: Optional[datetime] = None,
                 sync_every_sec: float = SYNC_EVERY_SEC):
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
        # 디스크 확정(fsync)은 뒤의 스레드가 한다: SD 카드에서는 한 번에 수십~수백 ms 가 걸려 주 루프에서 하면 조종이 끊긴다.
        # 확정하지 않으면 전원이 끊길 때 마지막 십여 초의 사진이 0바이트 파일로 남는다 (2026-11-10 두 세션에서 27장씩).
        self._unsynced: list = []
        self._sync_lock = threading.Lock()
        self._sync_stop = threading.Event()
        self._sync_thread: Optional[threading.Thread] = None
        self.write_meta()
        if sync_every_sec > 0:
            self._sync_thread = threading.Thread(target=self._sync_loop, args=(float(sync_every_sec),),
                                                 name="session-sync", daemon=True)
            self._sync_thread.start()

    # ------------------------------------------------------------------ frames
    def save_frame(self, frame_bgr: np.ndarray, angle: int, speed: int, ts: Optional[datetime] = None) -> str:
        ts = ts or datetime.now()
        name = make_frame_name(ts, angle, speed)
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG 인코딩 실패")
        (self.path / name).write_bytes(buf.tobytes())
        with self._sync_lock:
            self._unsynced.append(self.path / name)
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
        with self._sync_lock:
            self._unsynced.append(self.path / "session_meta.json")

    # ------------------------------------------------------------------ 디스크 확정
    def _sync_loop(self, every: float) -> None:
        while not self._sync_stop.wait(every):
            try:
                self.sync()
            except Exception as e:                  # 확정 실패가 주행을 멈추면 안 된다
                log.warning("디스크 확정 실패: %s", e)

    def sync(self) -> int:
        """지난번 이후 쓴 파일을 디스크에 확정한다. 돌려주는 값은 확정한 파일 수."""
        with self._sync_lock:
            paths, self._unsynced = self._unsynced, []
        done = 0
        for p in paths:
            try:
                fd = os.open(p, os.O_RDWR)
                try:
                    os.fsync(fd)
                    done += 1
                finally:
                    os.close(fd)
            except OSError as e:
                log.debug("fsync 실패 %s: %s", p.name, e)
        for f in (self._events_f, self._marks_f):
            try:
                os.fsync(f.fileno())
            except (OSError, ValueError):           # 이미 닫힘
                pass
        if paths and os.name == "posix":            # 새 파일 이름(디렉터리 항목)도 확정
            try:
                dfd = os.open(self.path, os.O_RDONLY)
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)
            except OSError:
                pass
        return done

    def close(self) -> None:
        self._sync_stop.set()
        if self._sync_thread is not None:
            self._sync_thread.join(timeout=5.0)
        self.write_meta(ended=True)
        self._events_f.flush(); self._marks_f.flush()
        try:
            self.sync()
        except Exception as e:
            log.warning("마지막 디스크 확정 실패: %s", e)
        self._events_f.close()
        self._marks_f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
