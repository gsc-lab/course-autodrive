"""차를 움직이는 프로그램(수집기, 추론기)의 단일 실행 잠금. 패드 모드는 SSH 가 끊겨도 계속 달리므로, 모르고 하나를 더 띄워
카메라와 I2C 를 다투는 일을 막는다. 두 프로그램이 같은 잠금 파일(dataset/collector.lock)을 쓰므로 서로도 함께 돌지 않는다.

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
        lines = [f"차를 움직이는 프로그램(수집기 또는 추론기)이 이미 실행 중입니다: {who}{when}.",
                 "먼저 그 프로그램을 끝내세요. 패드의 Options 버튼을 누르거나, 이 터미널에서:"]
        lines.append(f"    kill {self.pid}" if self.pid else "    pkill -f 'python -m (collector|inference)'")
        lines.append("몇 초 뒤에도 남아 있으면:  kill -9 " + (str(self.pid) if self.pid else "<PID>"))
        return "\n".join(lines)


LOCK_TRIES = 5


def _same_file(fd: int, path: Path) -> bool:
    try:
        opened, named = os.fstat(fd), os.stat(path)
    except OSError:
        return False
    return (opened.st_dev, opened.st_ino) == (named.st_dev, named.st_ino)


@dataclass
class InstanceLock:
    path: Path
    fd: Optional[int] = None

    def acquire(self) -> "InstanceLock":
        if not sys.platform.startswith("linux"):
            return self
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(LOCK_TRIES):
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
            # 파일을 열고 잠그는 그 사이에 앞 프로그램이 끝나며 파일을 지웠을 수 있다 (release 는 파일을 지운다). 그러면 방금
            # 잠근 것은 이름이 없는 옛 파일이고, 뒤에 오는 프로그램은 새 파일을 잠가 둘이 함께 돌게 된다. 지금 그 이름으로
            # 있는 파일과 같은 파일인지 확인하고, 아니면 다시 한다.
            if _same_file(fd, self.path):
                break
            os.close(fd)
        else:
            raise OSError(f"잠금 파일을 잡지 못했습니다: {self.path}")
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n".encode())
        os.fsync(fd)
        self.fd = fd
        return self

    def release(self) -> None:
        if self.fd is not None:
            try:
                self.path.unlink()         # 잠금을 쥔 채 지운다: 닫은 뒤 지우면 그 사이 잠근 다른 프로그램의 파일을 지울 수 있다
            except OSError:
                pass
            try:
                os.close(self.fd)          # flock 은 fd 가 닫히면 풀린다
            finally:
                self.fd = None
