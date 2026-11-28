"""세션 폴더 기록. 수집기가 쓰고 가공기가 읽는다.

dataset/YYYYMMDD_HHMM/
  YYYYMMDD_HHMMSS_mmm_aAAA_sSSS.jpg   프레임 (파일 이름이 라벨)
  session_meta.json                  설정 스냅샷, 카메라 정보, 클래스별 장수
  events.csv                         모든 입력 이벤트 (t_iso, t_mono, kind, value)
  marks.csv                          운전자가 표시한 불량 구간 (t_start_iso, t_end_iso, note)
  collector.log

파일을 쓰는 일은 모두 뒤의 스레드가 한다 (common/bgio.py). 주 루프는 무엇을 쓸지만 넘긴다: SD 카드가 바쁠 때는
write 한 번이 1초 넘게 막히기도 해서, 주 루프가 직접 쓰면 그동안 조종이 끊긴다.
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
from common.bgio import BackgroundWriter
from common.filename import make_frame_name, make_session_name

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
META_FLUSH_EVERY = 50
SYNC_EVERY_SEC = 2.0        # 이 주기로 새 파일을 디스크에 확정한다. 전원이 끊기면 잃는 양이 이 정도로 줄어든다
PENDING_FRAMES_MAX = 60     # 아직 쓰지 못한 사진이 이만큼 밀리면 새 사진을 버린다 (한 장 약 0.15 MB → 메모리 9 MB)
PENDING_TASKS_MAX = 5000    # 입력 이벤트 기록까지 합친 한도
CLOSE_WAIT_SEC = 15.0       # 끝낼 때 밀린 사진을 마저 쓰도록 기다리는 시간 (차는 이미 서 있다)


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
        self.saved = 0                      # 저장하기로 한 사진 수 (디스크에 쓰는 것은 조금 뒤다. 끝낼 때 못 쓴 것은 뺀다)
        self.dropped = 0                    # 디스크가 밀려 저장하지 못하고 버린 사진 수
        self.marks = 0
        self.last_frame_name: Optional[str] = None
        self._pending: dict = {}            # 맡겼지만 아직 쓰지 못한 사진: 파일 이름 → 각도
        self._pending_lock = threading.Lock()

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
        self.write_meta()                   # 시작할 때는 바로 쓴다 (아직 달리지 않는다)
        self._io = BackgroundWriter("session-writer", PENDING_TASKS_MAX)
        if sync_every_sec > 0:
            self._sync_thread = threading.Thread(target=self._sync_loop, args=(float(sync_every_sec),),
                                                 name="session-sync", daemon=True)
            self._sync_thread.start()

    # ------------------------------------------------------------------ frames
    def save_frame(self, frame_bgr: np.ndarray, angle: int, speed: int, ts: Optional[datetime] = None) -> Optional[str]:
        """사진 한 장을 저장하라고 맡긴다. 돌려주는 값은 파일 이름, 디스크가 밀려 버렸으면 None.

        JPEG 로 만드는 것까지만 여기서 하고(몇 ms, 디스크와 무관) 파일은 뒤의 스레드가 쓴다. 기다리지 않는다.
        """
        self._raise_if_failed()
        ts = ts or datetime.now()
        name = make_frame_name(ts, angle, speed)
        with self._pending_lock:
            backlog = len(self._pending)
        if backlog >= PENDING_FRAMES_MAX:           # 디스크가 멈춰 있다: 사진을 쌓아 두지 않고 버린다
            self.dropped += 1
            return None
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG 인코딩 실패")
        data = buf.tobytes()
        with self._pending_lock:
            self._pending[name] = int(angle)
        if not self._io.submit(lambda: self._write_frame(name, data)):
            with self._pending_lock:
                self._pending.pop(name, None)
            self.dropped += 1
            return None
        self.counts[int(angle)] += 1
        self.saved += 1
        self.last_frame_name = name
        if self.saved % META_FLUSH_EVERY == 0:
            meta = self.meta()                      # 내용은 지금의 값으로, 쓰는 것은 뒤에서
            self._io.submit(lambda: self._write_meta_file(meta))
        return name

    def _write_frame(self, name: str, data: bytes) -> None:
        """(쓰는 스레드) 사진 파일을 쓴다."""
        (self.path / name).write_bytes(data)
        with self._sync_lock:
            self._unsynced.append(self.path / name)
        with self._pending_lock:
            self._pending.pop(name, None)

    # ------------------------------------------------------------------ logs
    def log_event(self, kind: str, value=None, t_mono: Optional[float] = None) -> None:
        self._raise_if_failed()
        row = [datetime.now().isoformat(timespec="milliseconds"),
               f"{t_mono if t_mono is not None else time.monotonic():.3f}", kind, "" if value is None else value]
        self._io.submit(lambda: self._write_row(self._events, self._events_f, row))

    def mark(self, seconds: float, note: str = "", now: Optional[datetime] = None) -> None:
        self._raise_if_failed()
        now = now or datetime.now()
        start = now - timedelta(seconds=seconds)
        row = [start.isoformat(timespec="milliseconds"), now.isoformat(timespec="milliseconds"), note]
        self._io.submit(lambda: self._write_row(self._marks, self._marks_f, row))
        self.marks += 1

    @staticmethod
    def _write_row(writer, f, row: list) -> None:
        """(쓰는 스레드) CSV 한 줄. 바로 flush 한다: 실행 중에도 tail -f 로 볼 수 있게."""
        writer.writerow(row)
        f.flush()

    def _raise_if_failed(self) -> None:
        """뒤의 스레드가 쓰다가 실패했으면(디스크가 가득 참 등) 여기서 알린다. 수집기는 이 예외로 멈춘다."""
        e = self._io.failed
        if e is not None:
            raise OSError(f"세션 파일을 쓰지 못했습니다 ({self.path}): {e}") from e

    def flush(self, timeout: float = 5.0) -> bool:
        """맡긴 쓰기가 끝날 때까지 기다린다 (주 루프에서는 부르지 않는다). 시간 안에 끝났으면 True."""
        return self._io.flush(timeout)

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
        """session_meta.json 을 지금 쓴다 (시작과 끝에서만 부른다. 주행 중의 갱신은 save_frame 이 뒤의 스레드에 맡긴다)."""
        self._write_meta_file(self.meta(ended))

    def _write_meta_file(self, meta: dict) -> None:
        tmp = self.path / "session_meta.json.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
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

    def close(self, timeout: float = CLOSE_WAIT_SEC) -> None:
        drained = self._io.close(timeout)           # 밀린 사진과 기록을 마저 쓴다 (차는 이미 서 있다)
        with self._pending_lock:
            lost = dict(self._pending)
        if lost:                                    # 끝까지 쓰지 못한 사진은 장수에서 뺀다 (파일이 없다)
            for angle in lost.values():
                self.counts[angle] -= 1
                if self.counts[angle] <= 0:
                    del self.counts[angle]
            self.saved -= len(lost)
            log.warning("사진 %d장을 디스크에 쓰지 못했습니다 (%s)", len(lost),
                        self._io.failed if self._io.failed is not None else
                        ("디스크가 응답하지 않음" if not drained else "기록 중단"))
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
