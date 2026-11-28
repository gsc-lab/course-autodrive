"""파일 쓰기를 주 루프 대신 해 주는 스레드.

차를 움직이는 프로그램의 주 루프(50 Hz)는 파일을 직접 쓰면 안 된다. SD 카드는 다른 프로그램이 쓰고 있는 동안
write 한 번이 1초 넘게 막힐 수 있고, 그동안 조종이 끊겨 차는 직전 명령으로 달린다.

    2026-11-24 기준 보드: 한가할 때는 1초마다의 flush 가 최대 38 ms. 다른 프로그램이 SD 카드에 쓰는 동안에는
    73번 가운데 4번이 0.5~1.19초, 로그 파일 한 줄 쓰기가 145 ms. 같은 날 실차에서 자율주행 중 주 루프가 0.68초 멈췄다.

그래서 주 루프는 "무엇을 쓸지"만 넘기고(submit) 기다리지 않는다. 쓰는 일은 이 스레드가 순서대로 한다.
밀린 일이 한도를 넘으면 새 일을 버린다 (기다리는 것보다 낫다. 버린 수는 dropped 에 남는다).
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, Optional


class BackgroundWriter:
    """일(인자 없는 함수)을 받아 한 스레드에서 맡긴 순서대로 실행한다.

    submit()  절대 기다리지 않는다. 받았으면 True, 밀린 일이 max_pending 개라 버렸으면 False.
    failed    일을 하다 난 첫 예외. 그 뒤의 일은 하지 않고 버린다 (디스크가 가득 찬 뒤 계속 써 봐야 소용없다).
              맡긴 쪽이 주 루프에서 이 값을 보고 대응한다 (기록을 그만두거나 프로그램을 끝낸다).
    close()   밀린 일을 마저 하고 끝낸다. 정해 준 시간 안에 끝나지 않으면(디스크가 멈춤) False 를 돌려주고 기다리지 않는다.
    """

    def __init__(self, name: str, max_pending: int = 1000):
        self._q: queue.Queue = queue.Queue(maxsize=max(1, int(max_pending)))
        self.dropped = 0
        self.failed: Optional[BaseException] = None
        self._quit = False                  # stop_soon(): 밀린 일을 하지 않고 끝낸다
        self._closed = False
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    @property
    def pending(self) -> int:
        """아직 하지 못한 일의 수 (대략)."""
        return self._q.qsize()

    def submit(self, task: Callable[[], None]) -> bool:
        if self._closed or self.failed is not None:
            self.dropped += 1
            return False
        try:
            self._q.put_nowait(task)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _run(self) -> None:
        while True:
            try:
                task = self._q.get(timeout=0.5)
            except queue.Empty:
                if self._quit:              # 끝 표시를 넣을 자리가 없었던 경우(줄이 가득 참)에도 여기서 끝난다
                    return
                continue
            if task is None:                # close() 가 넣은 끝 표시
                return
            if isinstance(task, threading.Event):       # flush() 가 넣은 표시: 여기까지 왔다고 알린다 (실패한 뒤에도)
                task.set()
                continue
            if self._quit:
                continue
            if self.failed is None:
                try:
                    task()
                except Exception as e:      # 여기서 로그를 쓰지 않는다 (로그 파일을 쓰는 것도 이 스레드일 수 있다)
                    self.failed = e

    def flush(self, timeout: float = 5.0) -> bool:
        """지금까지 맡긴 일이 끝날 때까지 기다린다 (주 루프에서는 부르지 않는다). 시간 안에 끝났으면 True."""
        if not self._thread.is_alive():
            return True
        done = threading.Event()
        try:
            self._q.put(done, timeout=timeout)
        except queue.Full:
            return False
        return done.wait(timeout) or not self._thread.is_alive()

    def close(self, timeout: float = 5.0) -> bool:
        """밀린 일을 마저 하고 스레드를 끝낸다. 시간 안에 끝났으면 True. 여러 번 불러도 된다."""
        if not self._closed:
            self._closed = True
            try:
                self._q.put(None, timeout=timeout)
            except queue.Full:              # 스레드가 멈춰 있고 줄도 가득 찼다
                self._quit = True
                return False
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def stop_soon(self) -> None:
        """기다리지 않고 끝내라고만 알린다. 밀린 일은 버린다 (실패한 기록을 주 루프에서 내려놓을 때)."""
        self._closed = self._quit = True
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass                            # 줄이 가득 차 있으면 스레드가 남은 일을 건너뛰다 프로그램과 함께 끝난다


class BackgroundFileHandler(logging.Handler):
    """로그 파일 핸들러. 파일에 쓰는 일은 뒤의 스레드가 한다.

    logging.FileHandler 는 로그를 남긴 스레드에서 바로 파일을 쓴다. 주 루프가 경고 한 줄을 남기다 SD 카드에 막히면
    조종이 끊기고, 감시 스레드도 같은 핸들러의 잠금을 기다리게 된다.
    """

    def __init__(self, path, encoding: str = "utf-8", max_pending: int = 5000, level: int = logging.NOTSET):
        super().__init__(level)
        self._f = open(path, "a", encoding=encoding)
        self._io = BackgroundWriter("log-file", max_pending)

    def _write(self, text: str) -> None:
        self._f.write(text)
        self._f.flush()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = self.format(record) + "\n"       # 글로 바꾸는 것까지는 남긴 쪽에서 (예외 정보, 인자가 그때의 값이어야 한다)
            self._io.submit(lambda: self._write(text))
        except Exception:                           # 로그 때문에 주행이 영향을 받으면 안 된다
            pass

    def flush(self) -> None:
        self._io.flush(timeout=2.0)

    def close(self) -> None:
        try:
            if self._io.close(timeout=3.0):         # 끝까지 쓰지 못했으면(디스크 멈춤) 파일은 그대로 둔다: 스레드가 쓰는 중이다
                self._f.close()
        finally:
            super().close()
