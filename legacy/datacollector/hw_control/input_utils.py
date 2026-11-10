# -*- coding: utf-8 -*-
"""
===============================================================================
File Name     : input_utils.py
Description   : 터미널 Raw 모드(ECHO OFF)에서 키 입력을 1개씩 읽어 반환하는 유틸리티.
                - 방향키, ESC, Ctrl+C, 일반 키 입력 처리
                - ECHO 플래그를 끄기 때문에, 누른 키 값이 화면에 출력되지 않음.

Author        : Youngchul Jung
Date Created  : 2025-11-13
===============================================================================
"""

import sys
import termios
import fcntl
import os
import atexit

fd = sys.stdin.fileno()

# ===== Save original terminal settings =====
old_term = termios.tcgetattr(fd)
old_flags = fcntl.fcntl(fd, fcntl.F_GETFL)


def enable_raw_mode():
    new_term = termios.tcgetattr(fd)

    # ICANON: canonical mode off (read returns immediately)
    # ECHO: do not print characters
    new_term[3] &= ~(termios.ICANON | termios.ECHO)

    termios.tcsetattr(fd, termios.TCSANOW, new_term)

    # set non-blocking
    fcntl.fcntl(fd, fcntl.F_SETFL, old_flags | os.O_NONBLOCK)


def disable_raw_mode():
    termios.tcsetattr(fd, termios.TCSAFLUSH, old_term)
    fcntl.fcntl(fd, fcntl.F_SETFL, old_flags)


# run cleanup automatically when program exits
atexit.register(disable_raw_mode)

# Enable raw mode immediately
enable_raw_mode()


# ========== Key Parser ==========
def get_key_nonblock():
    """
    Returns:
        - "UP", "DOWN", "LEFT", "RIGHT"
        - "CTRL_C", "ESC"
        - normal key: 'a', 'b', '1', ...
        - None if no input
    """
    try:
        data = os.read(fd, 8).decode(errors='ignore')  # read up to 8 bytes
    except BlockingIOError:
        return None

    if len(data) == 0:
        return None

    # --- Arrow keys ---
    if data == '\x1b[A':
        return "UP"
    if data == '\x1b[B':
        return "DOWN"
    if data == '\x1b[C':
        return "RIGHT"
    if data == '\x1b[D':
        return "LEFT"

    # --- ESC ---
    if data == '\x1b':
        return "ESC"

    # --- Ctrl+C ---
    if data == '\x03':
        return "CTRL_C"

    # --- Normal key ---
    return data[-1]  # return last valid char