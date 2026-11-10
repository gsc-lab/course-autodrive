"""키보드 입력 (SSH 터미널). 패드가 없을 때의 대체 수단.

리눅스: termios raw 모드 (객체 생성 시 켜고 close() 에서 되돌린다 — import 시 부작용 없음)
윈도우: msvcrt (PC 에서 --mock 테스트용)

매핑: ↑ 전진 토글  ↓ 후진 토글  Space/T 정지  ←→ 조향  S 중앙  A/Z 속도 ±  R 녹화  M 불량 표시
      [ ] 저장 간격 ∓  Q/ESC/Ctrl+C 종료
"""
from __future__ import annotations

import logging
import os
import sys
from typing import List

from common.events import EventKind, InputEvent
from common.inputs.base import InputDevice

log = logging.getLogger(__name__)

KEYMAP = {
    "UP": (EventKind.FORWARD_TOGGLE, None),
    "DOWN": (EventKind.REVERSE_TOGGLE, None),
    "LEFT": (EventKind.STEER_STEP, -1),
    "RIGHT": (EventKind.STEER_STEP, 1),
    " ": (EventKind.STOP, None),
    "t": (EventKind.STOP, None),
    "s": (EventKind.STEER_CENTER, None),
    "a": (EventKind.SPEED_STEP, 1),
    "z": (EventKind.SPEED_STEP, -1),
    "r": (EventKind.RECORD_TOGGLE, None),
    "m": (EventKind.MARK, None),
    "[": (EventKind.INTERVAL_STEP, -1),
    "]": (EventKind.INTERVAL_STEP, 1),
    "q": (EventKind.QUIT, None),
    "ESC": (EventKind.QUIT, None),
    "CTRL_C": (EventKind.QUIT, None),
}

_ARROWS = {"A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT"}


def keys_to_events(keys: List[str]) -> List[InputEvent]:
    out = []
    for k in keys:
        m = KEYMAP.get(k if len(k) > 1 else k.lower())
        if m:
            out.append(InputEvent(m[0], m[1]))
    return out


def parse_terminal_keys(data: str) -> List[str]:
    """터미널 raw 입력을 키 이름 목록으로.

    - ESC [ A/B/C/D, ESC O A/B/C/D → 방향키
    - 그 밖의 ESC 시퀀스(PageUp `ESC[5~`, Home `ESC[H`, F키 등)는 통째로 버린다 (ESC 로 오인해 종료되지 않도록)
    - 단독 ESC → "ESC"
    """
    keys: List[str] = []
    i, n = 0, len(data)
    while i < n:
        ch = data[i]
        if ch == "\x1b":
            if i + 1 < n and data[i + 1] in "[O":
                j = i + 2
                # CSI 파라미터(0x30–0x3F)와 중간 바이트(0x20–0x2F)를 지나 최종 바이트(0x40–0x7E)까지
                while j < n and not (0x40 <= ord(data[j]) <= 0x7E):
                    j += 1
                if j < n:
                    final = data[j]
                    if j == i + 2 and final in _ARROWS:
                        keys.append(_ARROWS[final])
                    i = j + 1
                    continue
                # 시퀀스가 잘림 → 남은 입력 전체를 버린다
                break
            keys.append("ESC")
            i += 1
            continue
        if ch == "\x03":
            keys.append("CTRL_C")
        else:
            keys.append(ch)
        i += 1
    return keys


class _PosixReader:
    def __init__(self):
        import fcntl
        import termios
        self.termios, self.fcntl = termios, fcntl
        self.fd = sys.stdin.fileno()
        self.old_term = termios.tcgetattr(self.fd)
        self.old_flags = fcntl.fcntl(self.fd, fcntl.F_GETFL)
        new = termios.tcgetattr(self.fd)
        new[3] &= ~(termios.ICANON | termios.ECHO | termios.ISIG)  # ISIG 끄면 Ctrl+C 가 키로 들어옴
        termios.tcsetattr(self.fd, termios.TCSANOW, new)
        fcntl.fcntl(self.fd, fcntl.F_SETFL, self.old_flags | os.O_NONBLOCK)

    def read_keys(self) -> List[str]:
        try:
            data = os.read(self.fd, 64).decode(errors="ignore")
        except BlockingIOError:
            return []
        return parse_terminal_keys(data)

    def close(self):
        self.termios.tcsetattr(self.fd, self.termios.TCSAFLUSH, self.old_term)
        self.fcntl.fcntl(self.fd, self.fcntl.F_SETFL, self.old_flags)


class _WindowsReader:
    ARROWS = {"H": "UP", "P": "DOWN", "K": "LEFT", "M": "RIGHT"}

    def __init__(self):
        import msvcrt
        self.msvcrt = msvcrt

    def read_keys(self) -> List[str]:
        keys: List[str] = []
        while self.msvcrt.kbhit():
            ch = self.msvcrt.getwch()
            if ch in ("\x00", "\xe0"):
                code = self.msvcrt.getwch()       # 확장 키: 방향키 외(PageUp 등)는 버림
                name = self.ARROWS.get(code)
                if name:
                    keys.append(name)
            elif ch == "\x1b":
                keys.append("ESC")
            elif ch == "\x03":
                keys.append("CTRL_C")
            else:
                keys.append(ch)
        return keys

    def close(self):
        pass


class KeyboardInput(InputDevice):
    name = "keyboard"
    requires_deadman = False

    def __init__(self):
        if not sys.stdin.isatty():
            raise RuntimeError("키보드 입력은 터미널(tty)에서만 동작합니다")
        self._reader = _WindowsReader() if os.name == "nt" else _PosixReader()
        log.info("키보드 입력 준비 (%s)", "msvcrt" if os.name == "nt" else "termios raw")

    def poll(self) -> List[InputEvent]:
        return keys_to_events(self._reader.read_keys())

    def close(self) -> None:
        self._reader.close()
