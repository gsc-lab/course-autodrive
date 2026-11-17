"""가공 도구 화면 (PySide6). 로직은 curator/core.py 에 있고 여기서는 보여 주고 조작을 받는다.

  uv run -m curator [세션 폴더 또는 zip ...]

화면: 머리띠(로고·제목) → 도구 막대 → [사진 + 조향 화살표 | 옆 패널(세션, 구간 선택, 클래스 분포, 현재 프레임)]
      → 타임라인(세션 경계·제외·구간·현재 위치를 한 줄에) → 필름스트립.
용어: 세션 = 수집기를 한 번 실행해 생긴 폴더(파일로 남는다). 선택 = 한꺼번에 제외/복구하려고 고른 사진들(임시, 저장 안 됨).
선택은 탐색기와 같다: Shift+←/→ 또는 Shift+클릭 = 연속, Ctrl+클릭 = 떨어진 낱장, Esc = 해제.
제외 상태는 바뀔 때마다 세션 옆 curation.json 에 저장된다. 원본 사진은 건드리지 않는다. 단축키는 F1.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QObject, QPointF, QRectF, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtCore import QUrl
from PySide6.QtGui import (QAction, QBrush, QColor, QDesktopServices, QFont, QIcon, QImage, QKeySequence, QPainter, QPen,
                           QPixmap, QPolygonF)
from PySide6.QtWidgets import (QApplication, QDialog, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QProgressDialog, QPushButton,
                               QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

from common.config import Config
from curator.core import REASON_MARK, Frame, SessionSource, Workspace, build_dataset

log = logging.getLogger("curator")

APP_TITLE = "영진전문대 글로벌시스템융합과-자율주행 RC카 데이터 가공 프로그램"
LOGO_PATH = Path(__file__).parent / "assets" / "yju-symbol-white.png"

# 색 (어두운 테마)
C_BG, C_PANEL, C_CARD, C_BORDER = "#16191f", "#1d2129", "#252a34", "#363c49"
C_TEXT, C_MUTED, C_ACCENT, C_DANGER, C_OK, C_AMBER = "#e6e8ec", "#95a0b3", "#3b82f6", "#ef4444", "#22c55e", "#ffc400"
C_HEADER = "#0f2747"
SESSION_COLORS = ["#3b82f6", "#14b8a6", "#a855f7", "#f59e0b", "#ec4899", "#84cc16"]

QSS = f"""
QMainWindow, QWidget {{ background: {C_BG}; color: {C_TEXT}; font-family: "Malgun Gothic", "Segoe UI", sans-serif; font-size: 13px; }}
#header {{ background: {C_HEADER}; }}
#header QLabel {{ background: transparent; }}
#org {{ color: #b8c7e0; font-size: 12px; }}
#title {{ color: white; font-size: 17px; font-weight: 700; }}
#toolbar {{ background: {C_PANEL}; border-bottom: 1px solid {C_BORDER}; }}
#toolbar QLabel {{ background: transparent; color: {C_MUTED}; font-size: 11px; }}
QPushButton {{ background: {C_CARD}; border: 1px solid {C_BORDER}; border-radius: 6px; padding: 6px 12px; color: {C_TEXT}; }}
QPushButton:hover {{ background: #303746; border-color: #4a5263; }}
QPushButton:pressed {{ background: #1f242d; }}
QPushButton:disabled {{ color: #5d6575; }}
QPushButton#primary {{ background: {C_ACCENT}; border-color: {C_ACCENT}; color: white; font-weight: 700; }}
QPushButton#primary:hover {{ background: #5a97f8; }}
QPushButton#danger {{ color: #ff8f8f; }}
QPushButton#ghost {{ background: transparent; border-color: #5f7396; color: #dbe6f7; }}
#side {{ background: {C_PANEL}; border-left: 1px solid {C_BORDER}; }}
QFrame#card {{ background: {C_CARD}; border: 1px solid {C_BORDER}; border-radius: 8px; }}
QFrame#card QLabel, QFrame#card QWidget {{ background: transparent; }}
QLabel#cardTitle {{ font-weight: 700; font-size: 13px; }}
QLabel#hint {{ color: {C_MUTED}; font-size: 11px; }}
QLabel#value {{ font-size: 13px; }}
QListWidget {{ background: transparent; border: none; outline: none; }}
QListWidget#sessions::item {{ padding: 6px 4px; border-radius: 6px; }}
QListWidget#sessions::item:selected {{ background: #313a4d; color: {C_TEXT}; }}
QListWidget#strip {{ background: {C_PANEL}; border-top: 1px solid {C_BORDER}; }}
QListWidget#strip::item {{ color: {C_MUTED}; border: 2px solid transparent; border-radius: 4px; }}
QListWidget#strip::item:selected {{ border: 2px solid {C_AMBER}; color: {C_TEXT}; background: #2b3140; }}
QScrollArea {{ border: none; }}
QScrollBar:horizontal {{ height: 10px; background: {C_PANEL}; }}
QScrollBar::handle:horizontal {{ background: #4a5263; border-radius: 4px; min-width: 30px; }}
QScrollBar:vertical {{ width: 10px; background: {C_PANEL}; }}
QScrollBar::handle:vertical {{ background: #4a5263; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QStatusBar {{ background: {C_PANEL}; color: {C_MUTED}; border-top: 1px solid {C_BORDER}; }}
QDialog {{ background: {C_PANEL}; }}
QDialog QLabel {{ background: transparent; }}
QToolTip {{ background: #0b0d11; color: {C_TEXT}; border: 1px solid {C_BORDER}; padding: 4px; }}
"""

SHORTCUTS = [
    ("이동", [("←  /  →", "한 장 앞뒤"), ("PageUp / PageDown", "10장 앞뒤"), ("Home / End", "처음 / 끝"),
            ("Space", "자동 재생 · 정지"), ("타임라인 클릭·끌기", "그 위치로 이동")]),
    ("선택", [("Shift + ← / →", "연속으로 선택 넓히기 (PageUp/Down, Home/End 도 가능)"),
            ("Shift + 클릭", "현재 장부터 클릭한 장까지 연속 선택"), ("Ctrl + 클릭", "떨어진 사진을 낱장으로 선택·해제"),
            ("Ctrl+A", "전체 선택"), ("Esc", "선택 해제")]),
    ("제외 · 복구", [("Delete", "선택한 사진(없으면 현재 장) 제외"), ("Insert", "선택한 사진(없으면 현재 장) 복구"),
                 ("X", "현재 장 제외/복구 토글"), ("Ctrl+Z", "되돌리기")]),
    ("파일", [("Ctrl+O", "세션 폴더 열기"), ("Ctrl+Shift+O", "세션 zip 열기"), ("Ctrl+S", "지금 저장 (평소엔 자동 저장)"),
            ("Ctrl+B", "데이터셋 zip 만들기"), ("F1", "이 도움말")]),
]


def arrow_rotation_deg(side: float, max_deg: float, invert: bool) -> float:
    """조향 쪽(-1 왼쪽 … +1 오른쪽)을 화살표 기울기(시계 방향 양수)로. 화면 표시용 반전은 따로 적용한다."""
    deg = max(-1.0, min(1.0, side)) * max_deg
    return -deg if invert else deg


def runs(flags: List[bool]) -> List[Tuple[int, int]]:
    """True 가 이어진 구간들을 (시작, 끝 포함) 목록으로."""
    out, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i - 1)); start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


# ---------------------------------------------------------------------------
# 큰 그림 + 화살표
# ---------------------------------------------------------------------------
class FrameView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap: Optional[QPixmap] = None
        self.rotation = 0.0
        self.caption = ""
        self.excluded_reason: Optional[str] = None
        self.setMinimumSize(360, 270)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_frame(self, pixmap: Optional[QPixmap], rotation: float, caption: str, excluded_reason: Optional[str]) -> None:
        self.pixmap, self.rotation, self.caption, self.excluded_reason = pixmap, rotation, caption, excluded_reason
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt 이름)
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0d0f13"))
        if self.pixmap is None:
            p.setPen(QColor(C_MUTED))
            f = QFont(); f.setPointSize(12); p.setFont(f)
            p.drawText(self.rect(), Qt.AlignCenter, "세션 폴더를 여세요  (Ctrl+O)")
            return
        scaled = self.pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        p.drawPixmap(x, y, scaled)
        p.setRenderHint(QPainter.Antialiasing)
        length = scaled.height() * 0.35
        p.save()
        p.translate(x + scaled.width() / 2, y + scaled.height() - 12)
        p.rotate(self.rotation)
        p.setPen(QPen(QColor(0, 0, 0, 150), 10, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(0, 0), QPointF(0, -length))
        p.setPen(QPen(QColor(C_AMBER), 6, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(0, 0), QPointF(0, -length))
        p.setBrush(QBrush(QColor(C_AMBER))); p.setPen(Qt.NoPen)
        p.drawPolygon(QPolygonF([QPointF(0, -length - 16), QPointF(-15, -length + 8), QPointF(15, -length + 8)]))
        p.restore()
        f = QFont(); f.setPointSize(11); f.setBold(True); p.setFont(f)
        p.fillRect(x, y, scaled.width(), 30, QColor(0, 0, 0, 150))
        p.setPen(QColor("white"))
        p.drawText(QRectF(x + 10, y, scaled.width() - 20, 30), Qt.AlignVCenter | Qt.AlignLeft, self.caption)
        if self.excluded_reason:
            p.fillRect(x, y + 30, scaled.width(), scaled.height() - 30, QColor(200, 30, 30, 70))
            why = "불량 표시 구간 (□)" if self.excluded_reason == REASON_MARK else "제외됨"
            f.setPointSize(13); p.setFont(f)
            w = p.fontMetrics().horizontalAdvance("✕  " + why) + 24
            badge = QRectF(x + scaled.width() - w - 10, y + 40, w, 30)
            p.setBrush(QColor(C_DANGER)); p.setPen(Qt.NoPen); p.drawRoundedRect(badge, 6, 6)
            p.setPen(QColor("white")); p.drawText(badge, Qt.AlignCenter, "✕  " + why)


# ---------------------------------------------------------------------------
# 클래스 분포: 세로 막대 (포함 = 파랑, 제외 = 빨강을 위에 쌓음)
# ---------------------------------------------------------------------------
class BarChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data: List[Tuple[int, int, int]] = []     # (각도, 포함, 제외)
        self.current: Optional[int] = None             # 현재 프레임의 클래스
        self.setMinimumHeight(180)

    def set_data(self, data: List[Tuple[int, int, int]], current: Optional[int]) -> None:
        self.data, self.current = data, current
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if not self.data:
            return
        w, h = self.width(), self.height()
        top, bottom, left, right = 22, 40, 6, 6
        plot_h = h - top - bottom
        n = len(self.data)
        slot = (w - left - right) / n
        bar_w = min(46.0, slot * 0.62)
        peak = max(1, max(i + e for _, i, e in self.data))
        total = sum(i for _, i, _ in self.data)
        p.setPen(QPen(QColor(C_BORDER), 1, Qt.DotLine))
        for k in (0.5, 1.0):
            yy = top + plot_h * (1 - k)
            p.drawLine(QPointF(left, yy), QPointF(w - right, yy))
        p.setPen(QPen(QColor("#5d6575"), 1))
        p.drawLine(QPointF(left, top + plot_h), QPointF(w - right, top + plot_h))
        small = QFont(); small.setPointSize(8)
        bold = QFont(); bold.setPointSize(9); bold.setBold(True)
        for k, (angle, inc, exc) in enumerate(self.data):
            cx = left + slot * (k + 0.5)
            x0 = cx - bar_w / 2
            hi = plot_h * inc / peak
            he = plot_h * exc / peak
            is_cur = angle == self.current
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#6aa5ff" if is_cur else C_ACCENT))
            if inc:
                p.drawRoundedRect(QRectF(x0, top + plot_h - hi, bar_w, hi), 3, 3)
            if exc:
                p.setBrush(QColor(C_DANGER))
                p.drawRoundedRect(QRectF(x0, top + plot_h - hi - he, bar_w, max(2.0, he)), 3, 3)
            p.setFont(bold); p.setPen(QColor(C_TEXT))
            p.drawText(QRectF(cx - slot / 2, top + plot_h - hi - he - 18, slot, 16), Qt.AlignCenter, str(inc))
            p.setPen(QColor(C_AMBER if is_cur else C_TEXT))
            p.drawText(QRectF(cx - slot / 2, top + plot_h + 4, slot, 16), Qt.AlignCenter, f"{angle}°")
            p.setFont(small); p.setPen(QColor(C_MUTED))
            share = inc / total * 100 if total else 0.0
            p.drawText(QRectF(cx - slot / 2, top + plot_h + 20, slot, 14), Qt.AlignCenter, f"{share:.1f}%")


# ---------------------------------------------------------------------------
# 데이터셋 구성 요약: 포함 + 제외 = 전체 (숫자 세 개와 비율 막대)
# ---------------------------------------------------------------------------
class SummaryView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.included = self.excluded = 0
        self.setFixedHeight(102)
        self.setToolTip("포함: ‘데이터셋 만들기’ 에 들어가 학습에 쓰이는 사진\n제외: 빼기로 한 사진 (원본은 그대로, Insert 로 복구)\n전체: 연 세션의 모든 사진")

    def set_counts(self, included: int, excluded: int) -> None:
        self.included, self.excluded = included, excluded
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        total = self.included + self.excluded
        big = QFont(); big.setPointSize(17); big.setBold(True)
        lab = QFont(); lab.setPointSize(8)
        op = QFont(); op.setPointSize(13)
        cols = (("포함", self.included, C_ACCENT), ("제외", self.excluded, C_DANGER), ("전체", total, C_TEXT))
        col_w, op_w = (w - 2 * 22) / 3, 22
        x = 0.0
        for k, (name, value, color) in enumerate(cols):
            p.setFont(big); p.setPen(QColor(color))
            p.drawText(QRectF(x, 2, col_w, 30), Qt.AlignCenter, f"{value:,}")
            p.setFont(lab); p.setPen(QColor(C_MUTED))
            p.drawText(QRectF(x, 32, col_w, 16), Qt.AlignCenter, f"{name} (장)")
            x += col_w
            if k < 2:
                p.setFont(op); p.setPen(QColor(C_MUTED))
                p.drawText(QRectF(x, 2, op_w, 30), Qt.AlignCenter, "+" if k == 0 else "=")
                x += op_w
        # 비율 막대: 파랑 = 포함, 빨강 = 제외
        bar = QRectF(8, 62, w - 16, 14)
        p.setPen(Qt.NoPen); p.setBrush(QColor("#30353f")); p.drawRoundedRect(bar, 7, 7)
        if total:
            wi = bar.width() * self.included / total
            p.setBrush(QColor(C_DANGER)); p.drawRoundedRect(bar, 7, 7)
            if self.included:
                p.setBrush(QColor(C_ACCENT)); p.drawRoundedRect(QRectF(bar.x(), bar.y(), max(14.0, wi), bar.height()), 7, 7)
                if self.excluded:                       # 경계는 반듯하게
                    p.drawRect(QRectF(bar.x() + max(14.0, wi) - 7, bar.y(), 7, bar.height()))
        p.setFont(lab); p.setPen(QColor(C_MUTED))
        if total:
            text = f"전체의 {self.included / total * 100:.1f}% 를 학습에 사용  ·  {self.excluded / total * 100:.1f}% 제외"
        else:
            text = "세션을 열면 사진 수가 표시됩니다"
        p.drawText(QRectF(0, 84, w, 16), Qt.AlignCenter, text)


# ---------------------------------------------------------------------------
# 타임라인: 전체 프레임을 한 줄로. 세션 경계, 제외(빨강), 구간(파랑), 현재 위치(노랑)
# ---------------------------------------------------------------------------
class Timeline(QWidget):
    seek = Signal(int, bool)        # (프레임 번호, Shift 를 누른 채인지)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.n = 0
        self.sessions: List[Tuple[str, int, int, str]] = []    # (이름, 시작, 끝, 색)
        self.excluded: List[Tuple[int, int]] = []
        self.selected: List[Tuple[int, int]] = []
        self.cursor = 0
        self.setFixedHeight(58)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("전체 프레임 한눈에 보기 — 위: 세션, 빨강: 제외, 파랑: 선택한 사진, 노랑: 현재 위치. 클릭하면 이동 (Shift+클릭은 연속 선택)")

    def set_state(self, n, sessions, excluded, selected, cursor) -> None:
        self.n, self.sessions, self.excluded, self.selected, self.cursor = n, sessions, excluded, selected, cursor
        self.update()

    def _x(self, i: float) -> float:
        return 10 + (self.width() - 20) * (i / max(1, self.n))

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(C_PANEL))
        if not self.n:
            return
        p.setRenderHint(QPainter.Antialiasing)
        f = QFont(); f.setPointSize(8); p.setFont(f)
        label_y, track_y, track_h = 4, 24, 22
        for name, a, b, color in self.sessions:
            x0, x1 = self._x(a), self._x(b + 1)
            c = QColor(color)
            p.setPen(Qt.NoPen)
            c.setAlpha(70); p.setBrush(c)
            p.drawRect(QRectF(x0, track_y, x1 - x0, track_h))
            p.setBrush(QColor(color)); p.drawRect(QRectF(x0, label_y + 14, x1 - x0, 3))
            p.setPen(QColor(C_TEXT))
            text = f"세션 {name}  ·  {b - a + 1}장"
            if p.fontMetrics().horizontalAdvance(text) < x1 - x0 - 8:
                p.drawText(QRectF(x0 + 4, label_y, x1 - x0 - 8, 14), Qt.AlignVCenter | Qt.AlignLeft, text)
            p.setPen(QPen(QColor(C_BG), 2)); p.drawLine(QPointF(x1, label_y), QPointF(x1, track_y + track_h))
        p.setPen(Qt.NoPen); p.setBrush(QColor(C_DANGER))
        for a, b in self.excluded:
            p.drawRect(QRectF(self._x(a), track_y + 6, max(1.5, self._x(b + 1) - self._x(a)), track_h - 12))
        c = QColor(C_ACCENT); c.setAlpha(110)
        p.setBrush(c); p.setPen(QPen(QColor("#9cc2ff"), 1.5))
        for lo, hi in self.selected:
            p.drawRect(QRectF(self._x(lo), track_y - 2, max(3.0, self._x(hi + 1) - self._x(lo)), track_h + 4))
        xc = self._x(self.cursor + 0.5)
        p.setPen(QPen(QColor(C_AMBER), 2)); p.drawLine(QPointF(xc, track_y - 4), QPointF(xc, track_y + track_h + 4))
        p.setBrush(QColor(C_AMBER)); p.setPen(Qt.NoPen)
        p.drawPolygon(QPolygonF([QPointF(xc - 5, track_y - 6), QPointF(xc + 5, track_y - 6), QPointF(xc, track_y)]))

    def _seek_at(self, x: float, shift: bool) -> None:
        if self.n:
            i = int((x - 10) / max(1, self.width() - 20) * self.n)
            self.seek.emit(max(0, min(self.n - 1, i)), shift)

    def mousePressEvent(self, e) -> None:  # noqa: N802
        self._seek_at(e.position().x(), bool(e.modifiers() & Qt.ShiftModifier))

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        if e.buttons() & Qt.LeftButton:
            self._seek_at(e.position().x(), bool(e.modifiers() & Qt.ShiftModifier))


# ---------------------------------------------------------------------------
# 썸네일을 뒤에서 만든다
# ---------------------------------------------------------------------------
class ThumbWorker(QObject):
    ready = Signal(int, QImage)
    finished = Signal()

    def __init__(self, jobs: List[tuple], thumb_px: int):
        super().__init__()
        self.jobs = jobs            # (index, 세션 경로, 파일 이름)
        self.thumb_px = thumb_px
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        sources: dict = {}
        try:
            for idx, spath, name in self.jobs:
                if self._stop:
                    break
                src = sources.get(spath)
                if src is None:
                    src = sources[spath] = SessionSource(spath)     # 스레드 전용 핸들 (zip 은 스레드 안전하지 않다)
                img = QImage.fromData(src.read_bytes(name))
                if not img.isNull():
                    img = img.scaled(self.thumb_px, self.thumb_px, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.ready.emit(idx, img)
        finally:
            for s in sources.values():
                s.close()
            self.finished.emit()


class ShortcutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("단축키")
        grid = QGridLayout(self)
        grid.setContentsMargins(20, 16, 20, 16); grid.setHorizontalSpacing(18); grid.setVerticalSpacing(6)
        row = 0
        for group, items in SHORTCUTS:
            t = QLabel(group); t.setStyleSheet(f"font-weight:700; color:{C_AMBER}; padding-top:8px;")
            grid.addWidget(t, row, 0, 1, 2); row += 1
            for key, desc in items:
                k = QLabel(key)
                k.setStyleSheet(f"background:{C_CARD}; border:1px solid {C_BORDER}; border-radius:4px; padding:2px 8px;"
                                "font-family: Consolas, monospace;")
                grid.addWidget(k, row, 0, Qt.AlignLeft); grid.addWidget(QLabel(desc), row, 1); row += 1
        close = QPushButton("닫기"); close.clicked.connect(self.accept)
        grid.addWidget(close, row, 1, Qt.AlignRight)


def reveal_in_folder(path: Path) -> None:
    """파일이 든 폴더를 연다. 윈도우에서는 탐색기에서 그 파일을 선택한 상태로."""
    if sys.platform == "win32":
        subprocess.Popen(["explorer", "/select,", str(path)])
    else:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))


class BuildDoneDialog(QDialog):
    """데이터셋 작성 완료: 무엇이 어디에 만들어졌는지 보여 주고 바로 폴더를 열 수 있게 한다."""

    def __init__(self, path: Path, included: int, excluded: int, sessions: int, parent=None):
        path = Path(path).resolve()
        super().__init__(parent)
        self.path = path
        self.setWindowTitle("데이터셋 작성 완료")
        self.setMinimumWidth(520)
        lay = QVBoxLayout(self); lay.setContentsMargins(24, 20, 24, 18); lay.setSpacing(14)
        head = QHBoxLayout(); head.setSpacing(12)
        badge = QLabel("✓"); badge.setFixedSize(40, 40); badge.setAlignment(Qt.AlignCenter)
        badge.setStyleSheet(f"background:{C_OK}; color:white; border-radius:20px; font-size:22px; font-weight:700;")
        title = QLabel("데이터셋 작성 완료"); title.setStyleSheet("font-size:17px; font-weight:700;")
        head.addWidget(badge); head.addWidget(title); head.addStretch(1)
        lay.addLayout(head)
        size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0.0
        stats = QHBoxLayout(); stats.setSpacing(10)
        for value, label, color in ((f"{included:,}", "포함된 사진", C_ACCENT), (f"{excluded:,}", "제외된 사진", C_DANGER),
                                    (str(sessions), "세션", C_TEXT), (f"{size_mb:.1f} MB", "파일 크기", C_TEXT)):
            box = QFrame(); box.setObjectName("card")
            bl = QVBoxLayout(box); bl.setContentsMargins(10, 8, 10, 8); bl.setSpacing(0)
            v = QLabel(value); v.setAlignment(Qt.AlignCenter); v.setStyleSheet(f"font-size:18px; font-weight:700; color:{color};")
            n = QLabel(label); n.setAlignment(Qt.AlignCenter); n.setObjectName("hint")
            bl.addWidget(v); bl.addWidget(n); stats.addWidget(box)
        lay.addLayout(stats)
        info = QLabel(f"<table cellspacing='4'><tr><td style='color:{C_MUTED}'>파일</td><td><b>{path.name}</b></td></tr>"
                      f"<tr><td style='color:{C_MUTED}'>위치</td><td>{path.parent}</td></tr></table>")
        info.setWordWrap(True); info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(info)
        hint = QLabel("이 zip 을 학습(trainer)에 그대로 넣습니다. 원본 세션 폴더는 바뀌지 않았습니다."); hint.setObjectName("hint")
        lay.addWidget(hint)
        row = QHBoxLayout(); row.addStretch(1)
        close = QPushButton("닫기"); close.clicked.connect(self.accept)
        self.open_btn = QPushButton("폴더 열기"); self.open_btn.setObjectName("primary"); self.open_btn.setDefault(True)
        self.open_btn.clicked.connect(self._open)
        row.addWidget(close); row.addWidget(self.open_btn); lay.addLayout(row)

    def _open(self) -> None:
        reveal_in_folder(self.path)
        self.accept()


def card(title: str, hint: str = "") -> Tuple[QFrame, QVBoxLayout]:
    f = QFrame(); f.setObjectName("card")
    lay = QVBoxLayout(f); lay.setContentsMargins(12, 10, 12, 12); lay.setSpacing(6)
    t = QLabel(title); t.setObjectName("cardTitle"); lay.addWidget(t)
    if hint:
        h = QLabel(hint); h.setObjectName("hint"); h.setWordWrap(True); lay.addWidget(h)
    return f, lay


# ---------------------------------------------------------------------------
# 메인 창
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ws = Workspace(cfg)
        self.frames: List[Frame] = []
        self.index = 0
        self.selected: set = set()                          # 선택한 프레임 번호 (임시, 저장 안 됨)
        self.anchor = 0                                     # Shift 연속 선택의 기준점
        self._history: List[List[str]] = []                 # 되돌리기 순서: 한 번의 조작이 건드린 세션들
        self._thumbs: Dict[int, QPixmap] = {}
        self._flag_state: List[Optional[Tuple[bool, bool]]] = []
        self._session_span: Dict[str, Tuple[int, int]] = {}
        self._thumb_thread: Optional[QThread] = None
        self._thumb_worker: Optional[ThumbWorker] = None
        self._play = QTimer(self)
        self._play.timeout.connect(self._play_tick)
        self.setWindowTitle(APP_TITLE)
        self.setStyleSheet(QSS)
        self.resize(1360, 900)
        self.setMinimumSize(1000, 680)
        self._logo = QPixmap(str(LOGO_PATH)) if LOGO_PATH.exists() else QPixmap()
        if not self._logo.isNull():
            self.setWindowIcon(self._make_icon())
        self._build_ui()
        self._build_shortcuts()
        self._refresh_all()

    def _make_icon(self) -> QIcon:
        pm = QPixmap(64, 64); pm.fill(Qt.transparent)
        p = QPainter(pm); p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor(C_HEADER)); p.setPen(Qt.NoPen); p.drawRoundedRect(0, 0, 64, 64, 12, 12)
        p.drawPixmap(8, 8, self._logo.scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation)); p.end()
        return QIcon(pm)

    # -- UI ----------------------------------------------------------------
    def _btn(self, text: str, tip: str, fn, name: str = "") -> QPushButton:
        b = QPushButton(text)
        b.setToolTip(tip); b.setFocusPolicy(Qt.NoFocus); b.clicked.connect(fn)
        if name:
            b.setObjectName(name)
        return b

    def _build_ui(self) -> None:
        # 머리띠
        header = QWidget(); header.setObjectName("header"); header.setFixedHeight(60)
        hl = QHBoxLayout(header); hl.setContentsMargins(16, 6, 16, 6); hl.setSpacing(12)
        if not self._logo.isNull():
            lg = QLabel(); lg.setPixmap(self._logo.scaled(44, 44, Qt.KeepAspectRatio, Qt.SmoothTransformation)); hl.addWidget(lg)
        titles = QVBoxLayout(); titles.setSpacing(0)
        org = QLabel("영진전문대 글로벌시스템융합과"); org.setObjectName("org")
        ttl = QLabel("자율주행 RC카 데이터 가공 프로그램"); ttl.setObjectName("title")
        titles.addWidget(org); titles.addWidget(ttl)
        hl.addLayout(titles); hl.addStretch(1)
        hl.addWidget(self._btn("단축키  F1", "단축키 목록 보기", self.show_shortcuts, "ghost"))

        # 도구 막대: 기능별 묶음
        bar = QWidget(); bar.setObjectName("toolbar")
        bl = QHBoxLayout(bar); bl.setContentsMargins(12, 8, 12, 8); bl.setSpacing(6)

        def group(label: str, buttons: List[QPushButton]) -> None:
            box = QVBoxLayout(); box.setSpacing(3)
            box.addWidget(QLabel(label))
            row = QHBoxLayout(); row.setSpacing(6)
            for b in buttons:
                row.addWidget(b)
            box.addLayout(row); bl.addLayout(box)
            sep = QFrame(); sep.setFrameShape(QFrame.VLine); sep.setStyleSheet(f"color:{C_BORDER};"); bl.addWidget(sep)
        group("세션", [self._btn("폴더 열기", "세션 폴더 열기 (Ctrl+O). 여러 번 열면 세션이 추가된다", self.open_folder_dialog),
                     self._btn("zip 열기", "세션 zip 열기 (Ctrl+Shift+O)", self.open_zip_dialog)])
        group("제외 · 복구", [self._btn("제외", "선택한 사진(없으면 현재 장)을 데이터셋에서 제외 (Delete)", self.exclude_selection, "danger"),
                          self._btn("복구", "선택한 사진(없으면 현재 장)을 다시 포함 (Insert)", self.restore_selection),
                          self._btn("되돌리기", "마지막 제외/복구 되돌리기 (Ctrl+Z)", self.undo)])
        self.play_btn = self._btn("▶ 재생", "자동 재생 · 정지 (Space)", self.toggle_play)
        group("보기", [self.play_btn])
        bl.addStretch(1)
        self.build_btn = self._btn("데이터셋 만들기", "포함된 사진으로 dataset_*.zip 생성 (Ctrl+B)", self.build_dataset_dialog, "primary")
        self.build_btn.setMinimumHeight(38)
        bl.addWidget(self.build_btn, 0, Qt.AlignBottom)

        # 가운데: 사진 + 옆 패널
        self.view = FrameView()
        side_inner = QWidget(); side_inner.setObjectName("side")
        sl = QVBoxLayout(side_inner); sl.setContentsMargins(12, 10, 12, 10); sl.setSpacing(8)

        c, lay = card("세션")
        c.setToolTip("세션 = 수집기를 한 번 실행해 찍은 폴더. 파일로 남으며, 여러 세션을 열어 한 데이터셋으로 합칠 수 있다.\n"
                     "클릭하면 그 세션의 첫 사진으로 이동")
        self.sessions_list = QListWidget(); self.sessions_list.setObjectName("sessions")
        self.sessions_list.setFocusPolicy(Qt.NoFocus); self.sessions_list.setFixedHeight(50)
        self.sessions_list.itemClicked.connect(self._on_session_clicked)
        lay.addWidget(self.sessions_list); sl.addWidget(c)

        c, lay = card("데이터셋 구성")
        self.summary = SummaryView(); lay.addWidget(self.summary)
        sl.addWidget(c)

        c, lay = card("클래스 분포", "조향 각도별 포함 장수와 비율. 빨강은 제외된 장수.")
        self.chart = BarChart(); lay.addWidget(self.chart)
        sl.addWidget(c)

        c, lay = card("선택", "한꺼번에 제외·복구할 사진들. 임시 선택이라 저장되지 않는다.")
        self.range_label = QLabel(); self.range_label.setObjectName("value"); self.range_label.setWordWrap(True)
        lay.addWidget(self.range_label); sl.addWidget(c)

        c, lay = card("현재 프레임")
        self.info = QLabel(); self.info.setObjectName("value"); self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.info); sl.addWidget(c)
        sl.addStretch(1)
        side = QScrollArea(); side.setWidget(side_inner); side.setWidgetResizable(True)
        side.setFixedWidth(372); side.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        mid = QHBoxLayout(); mid.setContentsMargins(0, 0, 0, 0); mid.setSpacing(0)
        mid.addWidget(self.view, 1); mid.addWidget(side)

        # 아래: 타임라인 + 필름스트립
        self.timeline = Timeline(); self.timeline.seek.connect(lambda i, shift: self.goto(i, extend=shift))
        self.strip = QListWidget(); self.strip.setObjectName("strip")
        self.strip.setFlow(QListWidget.LeftToRight); self.strip.setViewMode(QListWidget.IconMode)
        self.strip.setWrapping(False); self.strip.setMovement(QListWidget.Static); self.strip.setFocusPolicy(Qt.NoFocus)
        px = self.cfg.curator.thumb_px
        self.strip.setIconSize(QSize(px, int(px * 0.75))); self.strip.setFixedHeight(int(px * 0.75) + 54); self.strip.setUniformItemSizes(True)
        self.strip.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.strip.itemClicked.connect(self._on_strip_clicked)

        central = QWidget()
        cl = QVBoxLayout(central); cl.setContentsMargins(0, 0, 0, 0); cl.setSpacing(0)
        cl.addWidget(header); cl.addWidget(bar); cl.addLayout(mid, 1); cl.addWidget(self.timeline); cl.addWidget(self.strip)
        self.setCentralWidget(central)
        self.statusBar().showMessage("준비")

    def _build_shortcuts(self) -> None:
        for key, fn in (("Ctrl+O", self.open_folder_dialog), ("Ctrl+Shift+O", self.open_zip_dialog), ("Ctrl+S", self.save),
                        ("Ctrl+B", self.build_dataset_dialog), ("Delete", self.exclude_selection),
                        ("Insert", self.restore_selection), ("X", self.toggle_current), ("Ctrl+Z", self.undo),
                        ("Space", self.toggle_play), ("F1", self.show_shortcuts),
                        ("Ctrl+A", self.select_all), ("Escape", self.clear_selection)):
            a = QAction(self); a.setShortcut(QKeySequence(key)); a.triggered.connect(fn); self.addAction(a)
        moves = (("Left", lambda: self.index - 1), ("Right", lambda: self.index + 1), ("PgUp", lambda: self.index - 10),
                 ("PgDown", lambda: self.index + 10), ("Home", lambda: 0), ("End", lambda: len(self.frames) - 1))
        for key, target in moves:                       # 그냥 누르면 이동, Shift 와 함께면 연속 선택
            for prefix, extend in (("", False), ("Shift+", True)):
                a = QAction(self); a.setShortcut(QKeySequence(prefix + key))
                a.triggered.connect(lambda _=False, t=target, e=extend: self.goto(t(), extend=e)); self.addAction(a)

    def show_shortcuts(self) -> None:
        ShortcutDialog(self).exec()

    # -- 세션 ------------------------------------------------------------
    def open_paths(self, paths: List[str]) -> None:
        for p in paths:
            try:
                self.ws.open(p)
            except (ValueError, FileNotFoundError, OSError) as e:
                QMessageBox.warning(self, "열기 실패", f"{p}\n{e}")
        self._refresh_all()

    def open_folder_dialog(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "세션 폴더 선택", str(Path(self.cfg.recording.root)))
        if d:
            self.open_paths([d])

    def open_zip_dialog(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "세션 zip 선택", str(Path(self.cfg.recording.root)), "zip (*.zip)")
        if f:
            self.open_paths([f])

    def save(self) -> None:
        self.ws.save()
        self.statusBar().showMessage("curation.json 저장", 2000)

    def _session_color(self, name: str) -> str:
        names = sorted(s.name for s in self.ws.sessions)
        return SESSION_COLORS[names.index(name) % len(SESSION_COLORS)]

    def _on_session_clicked(self, item: QListWidgetItem) -> None:
        span = self._session_span.get(item.data(Qt.UserRole))
        if span:
            self.goto(span[0])

    # -- 표시 ------------------------------------------------------------
    def _refresh_all(self) -> None:
        self.frames = self.ws.frames()
        self.index = min(self.index, max(0, len(self.frames) - 1))
        self.selected = set(); self.anchor = self.index
        self._session_span = {}
        for i, f in enumerate(self.frames):
            a, _ = self._session_span.get(f.session, (i, i))
            self._session_span[f.session] = (a, i)
        self._rebuild_strip()
        self._refresh_sessions()
        self._show_current()
        self._refresh_counts()
        has = bool(self.frames)
        self.build_btn.setEnabled(has)

    def _refresh_sessions(self) -> None:
        self.sessions_list.clear()
        for s in sorted(self.ws.sessions, key=lambda s: s.name):
            exc = len(self.ws.curations[s.name].excluded)
            chip = QPixmap(12, 28); chip.fill(QColor(self._session_color(s.name)))
            it = QListWidgetItem(QIcon(chip), f"{s.name}{'  (zip)' if s.is_zip else ''}\n"
                                              f"{len(s.frames)}장 · 포함 {len(s.frames) - exc} · 제외 {exc} · □ 표시 {len(s.marks)}회")
            it.setData(Qt.UserRole, s.name)
            self.sessions_list.addItem(it)
        self.sessions_list.setFixedHeight(min(3, max(1, len(self.ws.sessions))) * 48 + 4)   # 내용만큼만 (3개 넘으면 스크롤)
        if not self.ws.sessions:
            it = QListWidgetItem("열린 세션이 없습니다. ‘폴더 열기’ 로 추가하세요."); it.setFlags(Qt.NoItemFlags)
            self.sessions_list.addItem(it)

    def _rebuild_strip(self) -> None:
        self._stop_thumbs()
        self._thumbs = {}
        self._flag_state = [None] * len(self.frames)
        self.strip.blockSignals(True)
        self.strip.clear()
        px = self.cfg.curator.thumb_px
        for f in self.frames:
            it = QListWidgetItem(f"{f.info.angle}°")
            it.setSizeHint(QSize(px + 10, int(px * 0.75) + 30)); it.setTextAlignment(Qt.AlignHCenter)
            self.strip.addItem(it)
        self.strip.blockSignals(False)
        self._paint_strip_flags()
        if self.frames:
            self._start_thumbs()

    def _start_thumbs(self) -> None:
        jobs = [(i, str(self.ws.source(f).path), f.name) for i, f in enumerate(self.frames)]
        self._thumb_thread = QThread(self)
        self._thumb_worker = ThumbWorker(jobs, self.cfg.curator.thumb_px)
        self._thumb_worker.moveToThread(self._thumb_thread)
        self._thumb_thread.started.connect(self._thumb_worker.run)
        self._thumb_worker.ready.connect(self._on_thumb)
        self._thumb_worker.finished.connect(self._thumb_thread.quit)
        self._thumb_thread.start()

    def _stop_thumbs(self) -> None:
        if self._thumb_worker is not None:
            self._thumb_worker.stop()
        if self._thumb_thread is not None:
            self._thumb_thread.quit()
            self._thumb_thread.wait(2000)
        self._thumb_thread = self._thumb_worker = None

    def _on_thumb(self, idx: int, img: QImage) -> None:
        if idx < len(self.frames) and not img.isNull():
            self._thumbs[idx] = QPixmap.fromImage(img)
            self._flag_state[idx] = None
            self._paint_strip_item(idx)

    def _strip_icon(self, idx: int, excluded: bool, in_range: bool) -> QIcon:
        px = self.cfg.curator.thumb_px
        base = self._thumbs.get(idx)
        pm = QPixmap(px, int(px * 0.75)); pm.fill(QColor("#30353f"))
        p = QPainter(pm)
        if base is not None:
            p.drawPixmap(0, 0, base.scaled(pm.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
        if excluded:
            p.fillRect(pm.rect(), QColor(210, 30, 30, 120))
            p.setPen(QPen(QColor("white"), 3)); m = pm.height() * 0.3
            cx, cy = pm.width() / 2, pm.height() / 2
            p.drawLine(QPointF(cx - m, cy - m), QPointF(cx + m, cy + m)); p.drawLine(QPointF(cx - m, cy + m), QPointF(cx + m, cy - m))
        if in_range:
            p.fillRect(0, 0, pm.width(), 5, QColor(C_ACCENT)); p.fillRect(0, pm.height() - 5, pm.width(), 5, QColor(C_ACCENT))
        p.end()
        return QIcon(pm)

    def _paint_strip_item(self, i: int) -> None:
        it = self.strip.item(i)
        if it is None:
            return
        state = (self.ws.is_excluded(self.frames[i]), i in self.selected)
        if self._flag_state[i] == state:
            return
        self._flag_state[i] = state
        it.setIcon(self._strip_icon(i, *state))
        it.setForeground(QBrush(QColor("#ff8f8f" if state[0] else C_MUTED)))

    def _paint_strip_flags(self) -> None:
        for i in range(len(self.frames)):
            self._paint_strip_item(i)

    def _refresh_timeline(self) -> None:
        sessions = [(name, a, b, self._session_color(name)) for name, (a, b) in sorted(self._session_span.items())]
        excluded = runs([self.ws.is_excluded(f) for f in self.frames])
        self.timeline.set_state(len(self.frames), sessions, excluded,
                                runs([i in self.selected for i in range(len(self.frames))]), self.index)

    def _show_current(self) -> None:
        self._refresh_timeline()
        self._refresh_range_label()
        if not self.frames:
            self.view.set_frame(None, 0.0, "", None)
            self.info.setText(f"<span style='color:{C_MUTED}'>세션을 열면 사진 정보가 여기에 나옵니다.</span>")
            self.statusBar().showMessage("세션을 여세요  ·  단축키는 F1")
            return
        f = self.frames[self.index]
        img = QImage.fromData(self.ws.read_image(f))
        pm = QPixmap.fromImage(img) if not img.isNull() else None
        side = self.ws.steering.logical_side(f.info.angle)
        rot = arrow_rotation_deg(side, self.cfg.curator.arrow_max_deg, self.cfg.curator.arrow_invert)
        reason = self.ws.curation(f).excluded.get(f.name)
        ts = f"{f.info.timestamp:%H:%M:%S}.{f.info.timestamp.microsecond // 1000:03d}"
        self.view.set_frame(pm, rot, f"{self.index + 1} / {len(self.frames)}     조향 {f.info.angle}°     속도 {f.info.speed}%     {ts}", reason)
        a, b = self._session_span[f.session]
        state = (f"<span style='color:{C_DANGER}'>제외 — {'□ 불량 표시 구간' if reason == REASON_MARK else '직접 제외'}</span>"
                 if reason else f"<span style='color:{C_OK}'>포함</span>")
        self.info.setText(
            f"<table cellspacing='3'>"
            f"<tr><td style='color:{C_MUTED}'>파일</td><td>{f.name}</td></tr>"
            f"<tr><td style='color:{C_MUTED}'>세션</td><td>{f.session} &nbsp;({self.index - a + 1} / {b - a + 1})</td></tr>"
            f"<tr><td style='color:{C_MUTED}'>라벨</td><td>조향 {f.info.angle}° · 속도 {f.info.speed}%</td></tr>"
            f"<tr><td style='color:{C_MUTED}'>상태</td><td>{state}</td></tr></table>")
        self.statusBar().showMessage(f"{self.index + 1} / {len(self.frames)}   {f.key}   ·   제외는 자동 저장됩니다 (curation.json)")
        self.chart.set_data(self.chart.data, self.ws.steering.quantize(f.info.angle))
        for r in range(self.sessions_list.count()):
            it = self.sessions_list.item(r)
            if it.data(Qt.UserRole) == f.session:
                self.sessions_list.setCurrentRow(r)
        self.strip.blockSignals(True)
        self.strip.setCurrentRow(self.index)
        self.strip.scrollToItem(self.strip.item(self.index), QListWidget.PositionAtCenter)
        self.strip.blockSignals(False)

    def _refresh_range_label(self) -> None:
        n = len(self.selected)
        if n:
            groups = runs([i in self.selected for i in range(len(self.frames))])
            where = (f"{groups[0][0] + 1} ~ {groups[0][1] + 1}번" if len(groups) == 1 and n > 1
                     else (f"{groups[0][0] + 1}번" if n == 1 else f"떨어진 {len(groups)}곳"))
            sess = {self.frames[i].session for i in self.selected}
            span = f" · 세션 {len(sess)}개에 걸침" if len(sess) > 1 else ""
            self.range_label.setText(f"<b style='color:#9cc2ff'>{n}장 선택</b> &nbsp;({where}{span})<br>"
                                     f"<span style='color:{C_MUTED}'>Delete 로 제외, Insert 로 복구, Esc 로 해제</span>")
        else:
            self.range_label.setText(f"<span style='color:{C_MUTED}'>선택 없음 — 제외·복구는 현재 한 장에 적용<br>"
                                     "<b>Shift+←/→</b> 또는 <b>Shift+클릭</b>: 연속 선택<br>"
                                     "<b>Ctrl+클릭</b>: 떨어진 낱장 선택</span>")

    def _refresh_counts(self) -> None:
        inc, exc = self.ws.counts()
        cur = self.ws.steering.quantize(self.frames[self.index].info.angle) if self.frames else None
        self.chart.set_data([(c, inc[c], exc[c]) for c in self.ws.steering.classes], cur)
        self.summary.set_counts(sum(inc.values()), sum(exc.values()))

    # -- 이동 ------------------------------------------------------------
    def goto(self, i: int, extend: bool = False) -> None:
        """i 번 사진으로 이동. extend(Shift) 면 기준점부터 거기까지 연속 선택, 아니면 선택을 지우고 기준점을 옮긴다."""
        if not self.frames:
            return
        self.index = max(0, min(len(self.frames) - 1, i))
        if extend:
            lo, hi = min(self.anchor, self.index), max(self.anchor, self.index)
            self.selected = set(range(lo, hi + 1))
        else:
            self.selected = set()
            self.anchor = self.index
        self._paint_strip_flags()
        self._show_current()

    def toggle_select(self, i: int) -> None:
        """Ctrl+클릭: i 번 사진만 선택에 넣거나 뺀다 (떨어진 낱장 선택)."""
        if not (0 <= i < len(self.frames)):
            return
        if not self.selected and self.index != i:
            self.selected.add(self.index)               # 탐색기처럼: 지금 보던 사진도 선택에 포함해 시작
        self.selected.symmetric_difference_update({i})
        self.index = self.anchor = i
        self._paint_strip_flags()
        self._show_current()

    def select_all(self) -> None:
        self.selected = set(range(len(self.frames)))
        self._paint_strip_flags(); self._show_current()

    def clear_selection(self) -> None:
        self.selected = set(); self.anchor = self.index
        self._paint_strip_flags(); self._show_current()

    def _on_strip_clicked(self, item: QListWidgetItem) -> None:
        row = self.strip.row(item)
        mods = QApplication.keyboardModifiers()
        if mods & Qt.ControlModifier:
            self.toggle_select(row)
        else:
            self.goto(row, extend=bool(mods & Qt.ShiftModifier))

    def toggle_play(self) -> None:
        if self._play.isActive():
            self._play.stop(); self.play_btn.setText("▶ 재생")
        elif self.frames:
            self._play.start(int(1000 / self.cfg.curator.autoplay_fps)); self.play_btn.setText("■ 정지")

    def _play_tick(self) -> None:
        if self.index >= len(self.frames) - 1:
            self._play.stop(); self.play_btn.setText("▶ 재생")
        else:
            self.goto(self.index + 1)

    # -- 제외·복구 ---------------------------------------------------------
    def _selection(self) -> List[Frame]:
        if self.selected:
            return [self.frames[i] for i in sorted(self.selected)]
        return [self.frames[self.index]] if self.frames else []

    def _after_change(self, msg: str) -> None:
        self.ws.save()                                  # 바뀔 때마다 저장: 원본이 아니라 curation.json 만
        self._paint_strip_flags(); self._refresh_sessions(); self._show_current(); self._refresh_counts()
        self.statusBar().showMessage(msg, 3000)

    def _change(self, sel: List[Frame], exclude: bool) -> None:
        touched = []
        for s in sorted({f.session for f in sel}):
            cur = self.ws.curations[s]
            depth = len(cur._undo)
            names = [f.name for f in sel if f.session == s]
            cur.exclude(names) if exclude else cur.restore(names)
            if len(cur._undo) > depth:
                touched.append(s)
        if touched:
            self._history.append(touched)

    def exclude_selection(self) -> None:
        sel = self._selection()
        if not sel:
            return
        self._change(sel, True)
        self.selected = set(); self.anchor = self.index
        self._after_change(f"{len(sel)}장 제외")

    def restore_selection(self) -> None:
        sel = self._selection()
        if not sel:
            return
        self._change(sel, False)
        self.selected = set(); self.anchor = self.index
        self._after_change(f"{len(sel)}장 복구")

    def toggle_current(self) -> None:
        if not self.frames:
            return
        f = self.frames[self.index]
        self._change([f], not self.ws.is_excluded(f))
        self._after_change("제외" if self.ws.is_excluded(f) else "복구")

    def undo(self) -> None:
        if not self._history:
            self.statusBar().showMessage("되돌릴 것이 없습니다", 2000)
            return
        for s in self._history.pop():
            if s in self.ws.curations:
                self.ws.curations[s].undo()
        self._after_change("되돌리기")

    # -- 데이터셋 ----------------------------------------------------------
    def build_dataset_dialog(self) -> None:
        if not self.frames:
            return
        out = QFileDialog.getExistingDirectory(self, "데이터셋 zip 을 저장할 폴더", str(Path(self.cfg.recording.root)))
        if not out:
            return
        inc, exc = self.ws.counts()
        dlg = QProgressDialog("데이터셋 zip 쓰는 중…", None, 0, sum(inc.values()), self)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.show()
        try:
            path = build_dataset(self.ws, out, progress=lambda i, n: (dlg.setValue(i), QApplication.processEvents()))
        except ValueError as e:
            dlg.close()
            QMessageBox.warning(self, "데이터셋", str(e))
            return
        dlg.close()
        self.statusBar().showMessage(f"데이터셋 작성 완료: {path}", 8000)
        BuildDoneDialog(path, sum(inc.values()), sum(exc.values()), len(self.ws.sessions), self).exec()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._play.stop()
        self._stop_thumbs()
        self.ws.save()
        self.ws.close_all()
        super().closeEvent(event)


def main(argv=None) -> int:
    import argparse
    from common.config import ConfigError, load_config
    p = argparse.ArgumentParser(prog="curator", description="세션 검토·데이터셋 생성 (PC)")
    p.add_argument("paths", nargs="*", help="세션 폴더 또는 zip")
    p.add_argument("--config")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"[설정 오류] {e}", file=sys.stderr)
        return 1
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow(cfg)
    if args.paths:
        win.open_paths(args.paths)
    win.show()
    return app.exec()
