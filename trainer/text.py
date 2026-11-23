"""터미널 표 맞추기. 한글은 화면에서 두 칸을 차지하므로 글자 수가 아니라 화면 폭으로 줄을 맞춘다."""
from __future__ import annotations

import unicodedata


def width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def rjust(text, cells: int) -> str:
    """오른쪽 맞춤 (f"{text:>{cells}}" 의 한글 대응판)."""
    text = str(text)
    return " " * max(0, cells - width(text)) + text
