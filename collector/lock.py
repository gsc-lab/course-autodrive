"""수집기 단일 실행 잠금. 패드 모드는 SSH 가 끊겨도 계속 달리므로, 모르고 하나를 더 띄워 카메라와 I2C 를 다투는 일을 막는다.

리눅스에서는 flock 을 쓴다: 프로세스가 어떻게 죽든 커널이 잠금을 풀어 주므로 묵은 pid 파일 문제가 없다.
Windows(PC 모의 실행)에서는 잠그지 않는다.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class AlreadyRunning(RuntimeError):
    def __init__(self, pid: Optional[int], since: str):
        self.pid, self.since = pid, since
        super().__init__(self.message())

    def message(self) -> str:
        who = f"PID {self.pid}" if self.pid else "다른 프로세스"
        when = f" ({self.since.replace('T', ' ')} 시작)" if self.since else ""
        lines = [f"수집기가 이미 실행 중입니다: {who}{when}.",
                 "먼저 그 수집기를 끝내세요. 패드의 Options 버튼을 누르거나, 이 터미널에서:"]
        lines.append(f"    kill {self.pid}" if self.pid else "    pkill -f 'python -m collector'")
        lines.append("몇 초 뒤에도 남아 있으면:  kill -9 " + (str(self.pid) if self.pid else "<PID>"))
        return "\n".join(lines)


@dataclass
class InstanceLock:
    path: Path
    fd: Optional[int] = None

    def acquire(self) -> "InstanceLock":
        if not sys.platform.startswith("linux"):
            return self
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            pid, since = None, ""
            try:
                text = os.read(fd, 256).decode(errors="ignore").split()
                pid = int(text[0]) if text else None
                since = text[1].replace("T", " ") if len(text) > 1 else ""
            except (ValueError, OSError):
                pass
            os.close(fd)
            raise AlreadyRunning(pid, since) from None
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n".encode())
        os.fsync(fd)
        self.fd = fd
        return self

    def release(self) -> None:
        if self.fd is not None:
            try:
                os.close(self.fd)          # flock 은 fd 가 닫히면 풀린다
            finally:
                self.fd = None
            try:
                self.path.unlink()
            except OSError:
                pass
