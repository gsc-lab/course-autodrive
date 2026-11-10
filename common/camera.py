"""USB 웹캠. 백그라운드 스레드가 계속 읽어 최신 프레임만 보관한다 (버퍼 지연 방지).

모니터가 없으므로 어디에서도 cv2.imshow 를 호출하지 않는다.
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from common.config import CameraConfig
from common.hw.checks import check_frame_size

log = logging.getLogger(__name__)


class BaseCamera:
    def read(self) -> Tuple[Optional[np.ndarray], float]:
        """(최신 프레임 BGR, monotonic 시각). 프레임이 없으면 (None, 0)."""
        raise NotImplementedError

    @property
    def error(self) -> bool:
        return False

    def props(self) -> dict:
        return {}

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Camera(BaseCamera):
    MAX_CONSECUTIVE_FAILURES = 30

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(cfg.device, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"카메라를 열 수 없습니다: device={cfg.device!r}")
        if cfg.fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self.cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cfg.auto_exposure is not None:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, cfg.auto_exposure)
        if cfg.exposure is not None:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, cfg.exposure)

        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.cap.release()
            raise RuntimeError("카메라에서 첫 프레임을 읽지 못했습니다")
        size = check_frame_size(frame.shape, cfg.width, cfg.height)
        if not size.ok:
            self.cap.release()
            raise RuntimeError(size.message)
        self._frame = frame
        self._t = time.monotonic()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._failures = 0
        self._error = False
        self.frames_read = 1
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        log.info("카메라 준비: %s", self.props())

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    self._failures += 1
                    if self._failures >= self.MAX_CONSECUTIVE_FAILURES:
                        self._error = True
                        log.error("카메라 프레임 읽기 연속 실패 %d회", self._failures)
                    time.sleep(0.01)
                    continue
                self._failures = 0
                with self._lock:
                    self._frame = frame
                    self._t = time.monotonic()
                    self.frames_read += 1
        finally:
            # release() 는 read() 를 하는 이 스레드만 호출한다 (다른 스레드와 동시 호출 금지)
            self.cap.release()

    def read(self) -> Tuple[Optional[np.ndarray], float]:
        with self._lock:
            if self._frame is None:
                return None, 0.0
            return self._frame.copy(), self._t

    @property
    def error(self) -> bool:
        return self._error

    def props(self) -> dict:
        return {
            "device": self.cfg.device,
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(self.cap.get(cv2.CAP_PROP_FPS)),
            "fourcc": self.cfg.fourcc,
            "auto_exposure": self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
            "exposure": self.cap.get(cv2.CAP_PROP_EXPOSURE),
        }

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)
        if self._thread.is_alive():
            # cap.read() 가 멈춘 경우(USB 카메라 행). release() 를 동시에 부르면 크래시할 수 있으므로 그대로 둔다
            log.warning("카메라 스레드가 종료되지 않아 장치 해제를 건너뜁니다")


class MockCamera(BaseCamera):
    """PC 테스트용 합성 프레임 (타임스탬프와 움직이는 띠)."""

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self._t0 = time.monotonic()
        self.frames_read = 0

    def read(self) -> Tuple[Optional[np.ndarray], float]:
        t = time.monotonic()
        h, w = self.cfg.height, self.cfg.width
        frame = np.full((h, w, 3), 40, dtype=np.uint8)
        x = int(((t - self._t0) * 120) % w)
        cv2.rectangle(frame, (x, h // 2), (min(w, x + 60), h), (0, 200, 255), -1)
        cv2.putText(frame, f"MOCK {t - self._t0:7.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        self.frames_read += 1
        return frame, t

    def props(self) -> dict:
        return {"device": "mock", "width": self.cfg.width, "height": self.cfg.height, "fps": self.cfg.fps}


def create_camera(cfg: CameraConfig, mock: bool = False) -> BaseCamera:
    return MockCamera(cfg) if mock else Camera(cfg)
