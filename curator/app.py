"""가공 도구 화면 (PySide6). 로직은 curator/core.py 에 있고 여기서는 보여 주고 조작을 받는다.

  uv run -m curator [세션 폴더 또는 zip ...]

화면: 머리띠(로고·제목) → 도구 막대 → [왼쪽(세션, 현재 프레임) | 사진 + 조향 화살표 + 자르는 선, 그 아래 모델이 보는 사진 |
      오른쪽(데이터셋 구성, 클래스 분포, 선택)] → 타임라인(세션 경계·제외·선택·현재 위치를 한 줄에) → 필름스트립.
용어: 세션 = 수집기를 한 번 실행해 생긴 폴더(파일로 남는다). 선택 = 한꺼번에 제외/복구하려고 고른 사진들(임시, 저장 안 됨).
선택은 탐색기와 같다: Shift+←/→ 또는 Shift+클릭 = 연속, Ctrl+클릭 = 떨어진 낱장, Esc = 해제.
자르는 범위: 사진 위의 가로선 두 개를 끌어 모델에 넣을 세로 범위를 정한다. 어둡게 덮인 부분이 잘려 나가고, 사진 아래의
"모델이 보는 사진"이 그 결과다. 범위는 데이터셋에 하나이고 데이터셋 zip 에 적혀 학습기로 간다.
제외 상태와 자르는 범위는 바뀔 때마다 세션 옆 curation.json 에 저장된다. 원본 사진은 건드리지 않는다. 단축키는 F1.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QObject, QPointF, QRectF, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtCore import QUrl
from PySide6.QtGui import (QAction, QBrush, QColor, QDesktopServices, QFont, QIcon, QImage, QKeySequence, QPainter, QPen,
                           QPixmap, QPolygonF)
from PySide6.QtWidgets import (QApplication, QDialog, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QProgressDialog, QPushButton,
                               QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

import zipfile

from common.config import Config
from common.preprocess import describe_crop
from curator.core import (REASON_MARK, AlreadyOpen, BrokenImage, Frame, NoFrames, SessionSource, Workspace,
                          build_dataset, image_is_complete, model_view)

log = logging.getLogger("curator")

APP_TITLE = "영진전문대 글로벌시스템융합과-자율주행 RC카 데이터 가공 프로그램"
LOGO_PATH = Path(__file__).parent / "assets" / "yju-symbol-white.png"

# 색 (어두운 테마)
C_BG, C_PANEL, C_CARD, C_BORDER = "#16191f", "#1d2129", "#252a34", "#363c49"
C_TEXT, C_MUTED, C_ACCENT, C_DANGER, C_OK, C_AMBER = "#e6e8ec", "#95a0b3", "#3b82f6", "#ef4444", "#22c55e", "#ffc400"
C_HEADER = "#0f2747"
C_CROP = "#22d3ee"          # 자르는 선
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
QFrame#seg {{ background: #171b22; border: 1px solid {C_BORDER}; border-radius: 8px; }}
QPushButton#segBtn {{ background: transparent; border: none; border-radius: 6px; padding: 3px 0; min-width: 40px;
                     color: #8f99aa; font-weight: 600; }}
QPushButton#segBtn:hover {{ background: #2a303c; color: {C_TEXT}; }}
QPushButton#segBtn:checked {{ background: {C_ACCENT}; color: white; font-weight: 700; }}
QPushButton#ghost {{ background: transparent; border-color: #5f7396; color: #dbe6f7; }}
#side {{ background: {C_PANEL}; border-left: 1px solid {C_BORDER}; }}
#sideLeft {{ background: {C_PANEL}; border-right: 1px solid {C_BORDER}; }}
QFrame#card {{ background: {C_CARD}; border: 1px solid {C_BORDER}; border-radius: 8px; }}
QFrame#card QLabel, QFrame#card QWidget {{ background: transparent; }}
QLabel#cardTitle {{ font-weight: 700; font-size: 13px; }}
QLabel#hint {{ color: {C_MUTED}; font-size: 11px; }}
QLabel#value {{ font-size: 13px; }}
QListWidget {{ background: transparent; border: none; outline: none; }}
QListWidget#sessions::item {{ border: 1px solid {C_BORDER}; border-radius: 8px; margin-bottom: 6px; background: #20242d; }}
QListWidget#sessions::item:hover {{ border-color: #55607a; }}
QListWidget#sessions::item:selected {{ background: #2c3446; border: 1px solid {C_ACCENT}; }}
QLabel#sessName {{ font-weight: 700; font-size: 13px; }}
QLabel#statValue {{ font-weight: 700; font-size: 14px; }}
#modelbar {{ background: {C_PANEL}; border-top: 1px solid {C_BORDER}; }}
#modelbar QLabel {{ background: transparent; }}
QLabel#modelView {{ background: #0d0f13; border: 1px solid {C_BORDER}; color: {C_MUTED}; font-size: 11px; }}
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
            ("Space", "자동 재생 · 정지"), ("+  /  -", "재생 배속 올리기 · 내리기 (1 · 2 · 5 · 10배, 1배 = 실제 주행 속도)"), ("타임라인 클릭·끌기", "그 위치로 이동")]),
    ("선택", [("Shift + ← / →", "연속으로 선택 넓히기 (PageUp/Down, Home/End 도 가능)"),
            ("Shift + 클릭", "현재 장부터 클릭한 장까지 연속 선택"), ("Ctrl + 클릭", "떨어진 사진을 낱장으로 선택·해제"),
            ("Ctrl+A", "전체 선택"), ("Esc", "선택 해제")]),
    ("제외 · 복구", [("Delete", "선택한 사진(없으면 현재 장) 제외"), ("Insert", "선택한 사진(없으면 현재 장) 복구"),
                 ("X", "현재 장 제외/복구 토글"), ("Ctrl+Z", "되돌리기")]),
    ("자르는 범위", [("사진 위의 선 끌기", "모델에 넣을 범위(위 · 아래) 정하기. 어두운 부분이 잘려 나간다"),
                 ("C", "자르는 선 보이기 · 숨기기")]),
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
    """현재 사진 + 조향 화살표 + 자르는 선 두 개.

    자르는 선(위, 아래)은 마우스로 끌 수 있다. 이 위젯은 "어느 선을 어디로 끌었다"만 알리고(crop_dragged),
    값을 맞추고 저장하는 일은 창(MainWindow)이 한 뒤 set_crop 으로 돌려준다.
    """
    crop_dragged = Signal(str, float)       # 끄는 중: ("top" | "bottom", 사진 높이에 대한 비율)
    crop_released = Signal()                # 끌기를 마쳤다
    GRAB_PX = 9                             # 선에서 이만큼 안쪽을 누르면 그 선을 잡는다

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap: Optional[QPixmap] = None
        self.rotation = 0.0
        self.caption = ""
        self.excluded_reason: Optional[str] = None
        self.message = ""               # 사진이 없을 때 대신 보일 글 (비면 '세션 폴더를 여세요')
        self.crop: Optional[Tuple[float, float]] = None     # 자르는 범위 (위, 아래). None 이면 선을 그리지 않는다
        self._drag: Optional[str] = None                    # 끌고 있는 선
        self._hover: Optional[str] = None                   # 마우스가 올라가 있는 선
        self.setMinimumSize(360, 270)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)     # 누르지 않아도 선 위에서 커서 모양을 바꾸려고

    def set_frame(self, pixmap: Optional[QPixmap], rotation: float, caption: str, excluded_reason: Optional[str],
                  message: str = "") -> None:
        self.pixmap, self.rotation, self.caption, self.excluded_reason = pixmap, rotation, caption, excluded_reason
        self.message = message
        self.update()

    def set_crop(self, crop: Optional[Tuple[float, float]]) -> None:
        self.crop = crop
        if crop is None:
            self._drag = self._hover = None
            self.unsetCursor()
        self.update()

    # -- 자르는 선 ---------------------------------------------------------
    def image_rect(self) -> Optional[QRectF]:
        """사진이 그려지는 자리 (위젯 좌표). paintEvent 와 마우스 처리가 같은 자리를 쓴다."""
        if self.pixmap is None or self.pixmap.isNull():
            return None
        s = self.pixmap.size().scaled(self.size(), Qt.KeepAspectRatio)
        return QRectF((self.width() - s.width()) // 2, (self.height() - s.height()) // 2, s.width(), s.height())

    def line_y(self, which: str) -> Optional[float]:
        r = self.image_rect()
        if r is None or self.crop is None:
            return None
        return r.top() + r.height() * self.crop[0 if which == "top" else 1]

    def _line_at(self, pos: QPointF) -> Optional[str]:
        r = self.image_rect()
        if r is None or self.crop is None or not (r.left() <= pos.x() <= r.right()):
            return None
        near = min(("top", "bottom"), key=lambda w: abs(pos.y() - self.line_y(w)))
        return near if abs(pos.y() - self.line_y(near)) <= self.GRAB_PX else None

    def mousePressEvent(self, e) -> None:  # noqa: N802
        if e.button() == Qt.LeftButton:
            self._drag = self._line_at(e.position())
            if self._drag:
                self.update()
                return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        if self._drag:
            r = self.image_rect()
            if r is not None and r.height() > 0:
                self.crop_dragged.emit(self._drag, (e.position().y() - r.top()) / r.height())
            return
        hover = self._line_at(e.position())
        if hover != self._hover:
            self._hover = hover
            if hover:
                self.setCursor(Qt.SizeVerCursor)
            else:
                self.unsetCursor()
            self.update()

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if self._drag and e.button() == Qt.LeftButton:
            self._drag = None
            self._hover = self._line_at(e.position())
            if not self._hover:
                self.unsetCursor()
            self.update()
            self.crop_released.emit()
            return
        super().mouseReleaseEvent(e)

    def leaveEvent(self, e) -> None:  # noqa: N802
        if self._hover and not self._drag:
            self._hover = None
            self.unsetCursor()
            self.update()
        super().leaveEvent(e)

    def _paint_crop_lines(self, p: QPainter, r: QRectF) -> None:
        """선 두 개와 꼬리표. 꼬리표는 잘려 나가는 쪽에 두고, 자리가 없으면 선의 반대쪽에 둔다."""
        top, bottom = self.crop
        f = QFont(); f.setPointSize(9); f.setBold(True); p.setFont(f)
        fm = p.fontMetrics()
        for which, ratio, text in (("top", top, f"위쪽 {top * 100:.0f}% 잘라 냄"),
                                   ("bottom", bottom, f"아래쪽 {(1 - bottom) * 100:.0f}% 잘라 냄")):
            ly = min(r.top() + r.height() * ratio, r.bottom() - 1)
            active = which in (self._drag, self._hover)
            p.setPen(QPen(QColor(0, 0, 0, 170), 4 if active else 3))
            p.drawLine(QPointF(r.left(), ly), QPointF(r.right(), ly))
            p.setPen(QPen(QColor("white" if active else C_CROP), 2))
            p.drawLine(QPointF(r.left(), ly), QPointF(r.right(), ly))
            w, h = fm.horizontalAdvance(text) + 16, 22
            if which == "top":                          # 잘려 나가는 쪽은 선의 위. 사진 맨 위의 글 띠(30 px)와 겹치면 아래로
                ty = ly - h - 3 if ly - h - 3 >= r.top() + 32 else ly + 3
            else:                                       # 잘려 나가는 쪽은 선의 아래
                ty = ly + 3 if ly + 3 + h <= r.bottom() else ly - h - 3
            tag = QRectF(r.left() + 10, ty, w, h)
            p.setPen(Qt.NoPen); p.setBrush(QColor("white" if active else C_CROP))
            p.drawRoundedRect(tag, 4, 4)
            p.setPen(QColor("#06262c")); p.drawText(tag, Qt.AlignCenter, text)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt 이름)
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0d0f13"))
        if self.pixmap is None:
            p.setPen(QColor(C_MUTED))
            f = QFont(); f.setPointSize(12); p.setFont(f)
            p.drawText(self.rect(), Qt.AlignCenter, self.message or "세션 폴더를 여세요  (Ctrl+O)")
            return
        scaled = self.pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        p.drawPixmap(x, y, scaled)
        p.setRenderHint(QPainter.Antialiasing)
        if self.crop is not None:                       # 잘려 나가는 부분(모델이 보지 않는 부분)을 어둡게 덮는다.
            shade = QColor(6, 8, 12, 125)               # 사진을 검토할 때 그 부분도 알아볼 수 있을 만큼만
            cut_top, cut_bottom = scaled.height() * self.crop[0], scaled.height() * self.crop[1]
            p.fillRect(QRectF(x, y, scaled.width(), cut_top), shade)
            p.fillRect(QRectF(x, y + cut_bottom, scaled.width(), scaled.height() - cut_bottom), shade)
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
        if self.crop is not None:                       # 선은 맨 위에 그린다 (글 띠나 제외 표시에 가리지 않게)
            self._paint_crop_lines(p, QRectF(x, y, scaled.width(), scaled.height()))


# ---------------------------------------------------------------------------
# 클래스 분포: 가로 막대. 한 줄 = 각도 | 막대(포함 파랑 + 제외 빨강) | 장수 | 비율. 점선 = 각도별 평균(균형 기준)
# 카드 높이에 맞춰 줄 높이와 글자가 커지고, 맨 아래에 균형 판정 한 줄이 붙는다.
# ---------------------------------------------------------------------------
IMBALANCE_RATIO = 3.0        # 가장 많은 각도가 가장 적은 각도의 몇 배 이상이면 "불균형" 으로 알릴지


def balance_note(data: List[Tuple[int, int, int]]) -> Tuple[str, bool]:
    """(각도, 포함, 제외) 목록에서 균형 판정 한 줄과 경고 여부를 만든다."""
    if not data or not sum(i for _, i, _ in data):
        return "", False
    lo = min(data, key=lambda r: r[1])
    hi = max(data, key=lambda r: r[1])
    if lo[1] == 0:
        return f"{lo[0]}° 사진이 없습니다", True
    ratio = hi[1] / lo[1]
    if ratio >= IMBALANCE_RATIO:
        return f"불균형 · {hi[0]}° 가 {lo[0]}° 의 {ratio:.1f}배", True
    return f"균형 양호 · 최다 {hi[0]}° 가 최소 {lo[0]}° 의 {ratio:.1f}배", False


class BarChart(QWidget):
    ROW_MIN, ROW_MAX, TOP, BOTTOM = 32, 64, 6, 46

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data: List[Tuple[int, int, int]] = []     # (각도, 포함, 제외)
        self.current: Optional[int] = None             # 현재 프레임의 클래스
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(self.TOP + 5 * self.ROW_MIN + self.BOTTOM)

    def set_data(self, data: List[Tuple[int, int, int]], current: Optional[int]) -> None:
        self.data, self.current = data, current
        n = max(1, len(data))
        self.setMinimumHeight(self.TOP + n * self.ROW_MIN + self.BOTTOM)
        self.setMaximumHeight(self.TOP + n * self.ROW_MAX + self.BOTTOM)
        self.update()

    def row_height(self) -> float:
        n = max(1, len(self.data))
        return min(float(self.ROW_MAX), max(float(self.ROW_MIN), (self.height() - self.TOP - self.BOTTOM) / n))

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if not self.data:
            return
        w = self.width()
        rh = self.row_height()
        big = rh >= 34                                  # 자리가 넉넉하면 글자와 막대를 키운다
        label_w, count_w, pct_w, gap = (50, 56, 62, 10) if big else (42, 46, 52, 8)
        x0 = label_w + gap
        track_w = w - x0 - gap - count_w - pct_w
        peak = max(1, max(i + e for _, i, e in self.data))
        total = sum(i for _, i, _ in self.data)
        f_label = QFont(); f_label.setPointSize(13 if big else 10); f_label.setBold(True)
        f_count = QFont(); f_count.setPointSize(13 if big else 10); f_count.setBold(True)
        f_pct = QFont(); f_pct.setPointSize(11 if big else 10)
        f_small = QFont(); f_small.setPointSize(8)
        f_note = QFont(); f_note.setPointSize(9); f_note.setBold(True)
        bar_h = min(26.0, max(14.0, rh * 0.46))
        for k, (angle, inc, exc) in enumerate(self.data):
            y = self.TOP + k * rh
            is_cur = angle == self.current
            if is_cur:                                   # 지금 보고 있는 사진의 각도: 줄 전체를 살짝 밝게
                p.setPen(Qt.NoPen); p.setBrush(QColor(255, 196, 0, 26))
                p.drawRoundedRect(QRectF(0, y + 2, w, rh - 4), 6, 6)
            p.setFont(f_label); p.setPen(QColor(C_AMBER if is_cur else C_TEXT))
            p.drawText(QRectF(4, y, label_w - 4, rh), Qt.AlignVCenter | Qt.AlignRight, f"{angle}°")
            bar = QRectF(x0, y + (rh - bar_h) / 2, track_w, bar_h)
            p.setPen(Qt.NoPen); p.setBrush(QColor("#30353f")); p.drawRoundedRect(bar, 4, 4)
            wi = track_w * inc / peak
            we = track_w * exc / peak
            if inc:
                p.setBrush(QColor("#6aa5ff" if is_cur else C_ACCENT))
                p.drawRoundedRect(QRectF(bar.x(), bar.y(), max(4.0, wi), bar.height()), 4, 4)
            if exc:
                p.setBrush(QColor(C_DANGER))
                p.drawRoundedRect(QRectF(bar.x() + wi, bar.y(), max(3.0, we), bar.height()), 3, 3)
            p.setFont(f_count); p.setPen(QColor(C_TEXT))
            p.drawText(QRectF(x0 + track_w + gap, y, count_w, rh), Qt.AlignVCenter | Qt.AlignRight, f"{inc:,}")
            share = inc / total * 100 if total else 0.0
            p.setFont(f_pct); p.setPen(QColor("#c3cad6"))
            p.drawText(QRectF(w - pct_w, y, pct_w - 4, rh), Qt.AlignVCenter | Qt.AlignRight, f"{share:.1f}%")
        rows_h = len(self.data) * rh
        if not total:
            return
        # 균형 기준선: 각도별 평균 장수 (모든 막대가 이 선 근처면 균형이 맞는 것)
        mean = total / len(self.data)
        xm = x0 + track_w * mean / peak
        p.setPen(QPen(QColor("#aab3c2"), 1, Qt.DashLine))
        p.drawLine(QPointF(xm, self.TOP + 2), QPointF(xm, self.TOP + rows_h - 2))
        p.setFont(f_small); p.setPen(QColor(C_MUTED))
        text = f"평균 {mean:,.0f}장"
        tw = p.fontMetrics().horizontalAdvance(text)
        p.drawText(QRectF(min(max(x0, xm - tw / 2), x0 + track_w - tw), self.TOP + rows_h + 2, tw + 2, 16), Qt.AlignCenter, text)
        # 균형 판정 한 줄
        note, warn = balance_note(self.data)
        c = QColor(C_AMBER if warn else C_OK)
        box = QRectF(0, self.TOP + rows_h + 22, w, 22)
        bg = QColor(c); bg.setAlpha(30)
        p.setPen(Qt.NoPen); p.setBrush(bg); p.drawRoundedRect(box, 6, 6)
        p.setFont(f_note); p.setPen(c)
        p.drawText(box, Qt.AlignCenter, note)


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
# 현재 프레임: 위치(세션 안 몇 번째) · 상태 알약 · 조향 계기 · 속도 단계 · 촬영 시각 · 파일
# ---------------------------------------------------------------------------
class FrameInfoView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.d: Optional[dict] = None
        self.setFixedHeight(214)

    def set_info(self, d: Optional[dict]) -> None:
        self.d = d
        self.setToolTip(d["name"] if d else "")
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        small = QFont(); small.setPointSize(8)
        if not self.d:
            p.setFont(small); p.setPen(QColor(C_MUTED))
            p.drawText(self.rect(), Qt.AlignCenter, "세션을 열면 사진 정보가 여기에 나옵니다")
            return
        d = self.d
        big = QFont(); big.setPointSize(15); big.setBold(True)
        mid = QFont(); mid.setPointSize(10)
        val = QFont(); val.setPointSize(14); val.setBold(True)
        bold = QFont(); bold.setPointSize(9); bold.setBold(True)

        # 1) 위치 + 상태 알약
        p.setFont(big); p.setPen(QColor(C_TEXT))
        pos = f"{d['in_session']:,}"
        p.drawText(QRectF(0, 0, w, 28), Qt.AlignVCenter | Qt.AlignLeft, pos)
        x = p.fontMetrics().horizontalAdvance(pos) + 5
        p.setFont(mid); p.setPen(QColor(C_MUTED))
        p.drawText(QRectF(x, 4, w, 24), Qt.AlignVCenter | Qt.AlignLeft, f"/ {d['session_len']:,} 번째")
        excluded = d["reason"] is not None
        text = ("제외 · 불량 표시 □" if d["reason"] == REASON_MARK else "제외") if excluded else "포함"
        p.setFont(bold)
        pw = p.fontMetrics().horizontalAdvance(text) + 26
        pill = QRectF(w - pw, 3, pw, 22)
        c = QColor(C_DANGER if excluded else C_OK)
        bg = QColor(c); bg.setAlpha(40)
        p.setBrush(bg); p.setPen(QPen(c, 1)); p.drawRoundedRect(pill, 11, 11)
        p.setBrush(c); p.setPen(Qt.NoPen); p.drawEllipse(QPointF(pill.x() + 11, pill.center().y()), 3.5, 3.5)
        p.setPen(QColor("#ff9c9c" if excluded else "#7ee2a0"))
        p.drawText(QRectF(pill.x() + 18, pill.y(), pw - 22, 22), Qt.AlignVCenter | Qt.AlignLeft, text)

        # 2) 세션 안 진행 막대 + 세션 이름 / 전체 위치
        track = QRectF(0, 34, w, 5)
        p.setPen(Qt.NoPen); p.setBrush(QColor("#30353f")); p.drawRoundedRect(track, 2.5, 2.5)
        frac = d["in_session"] / max(1, d["session_len"])
        p.setBrush(QColor(d["color"])); p.drawRoundedRect(QRectF(0, 34, max(5.0, w * frac), 5), 2.5, 2.5)
        p.setFont(small); p.setPen(QColor(C_MUTED))
        p.drawText(QRectF(0, 42, w, 16), Qt.AlignVCenter | Qt.AlignLeft, f"세션 {d['session']}")
        p.drawText(QRectF(0, 42, w, 16), Qt.AlignVCenter | Qt.AlignRight, f"전체 {d['overall']:,} / {d['total']:,}")

        # 3) 라벨 두 칸: 조향 계기, 속도 단계
        tw, ty, th = (w - 8) / 2, 66, 88
        for k in range(2):
            p.setBrush(QColor("#20242d")); p.setPen(QPen(QColor(C_BORDER), 1))
            p.drawRoundedRect(QRectF(k * (tw + 8), ty, tw, th), 8, 8)
        p.setFont(small); p.setPen(QColor(C_MUTED))
        p.drawText(QRectF(10, ty + 6, tw, 14), Qt.AlignLeft | Qt.AlignVCenter, "조향 라벨")
        p.drawText(QRectF(tw + 18, ty + 6, tw, 14), Qt.AlignLeft | Qt.AlignVCenter, "속도 라벨")
        # 조향: 반원 눈금 + 바늘 (왼쪽 끝 = 가장 왼쪽 조향)
        cx, cy, r = tw / 2, ty + 58, 26
        p.setPen(QPen(QColor("#4a5263"), 4, Qt.SolidLine, Qt.RoundCap)); p.setBrush(Qt.NoBrush)
        p.drawArc(QRectF(cx - r, cy - r, 2 * r, 2 * r), 30 * 16, 120 * 16)
        p.save(); p.translate(cx, cy); p.rotate(d["side"] * 60)
        p.setPen(QPen(QColor(C_AMBER), 3, Qt.SolidLine, Qt.RoundCap)); p.drawLine(QPointF(0, 0), QPointF(0, -r + 3))
        p.restore()
        p.setBrush(QColor(C_AMBER)); p.setPen(Qt.NoPen); p.drawEllipse(QPointF(cx, cy), 3.5, 3.5)
        p.setFont(val); p.setPen(QColor(C_TEXT))
        p.drawText(QRectF(0, ty + 60, tw, 24), Qt.AlignCenter, f"{d['angle']}°")
        side_text = "직진" if abs(d["side"]) < 0.05 else ("왼쪽" if d["side"] < 0 else "오른쪽")
        p.setFont(small); p.setPen(QColor(C_AMBER))
        p.drawText(QRectF(0, ty + 6, tw - 10, 14), Qt.AlignRight | Qt.AlignVCenter, side_text)
        # 속도: 단계 막대 (설정의 throttle.levels 중 몇 단계인지)
        levels = d["levels"]
        ox = tw + 8
        if levels:
            n = len(levels)
            level = min(range(n), key=lambda i: abs(levels[i] - d["speed"]))
            bw, gap, base = 12, 5, ty + 56
            start = ox + (tw - (n * bw + (n - 1) * gap)) / 2
            for i in range(n):
                h = 12 + 22 * (i + 1) / n
                p.setBrush(QColor(C_ACCENT if i <= level else "#3a404c")); p.setPen(Qt.NoPen)
                p.drawRoundedRect(QRectF(start + i * (bw + gap), base - h, bw, h), 2, 2)
            p.setFont(small); p.setPen(QColor(C_ACCENT))
            p.drawText(QRectF(ox, ty + 6, tw - 10, 14), Qt.AlignRight | Qt.AlignVCenter, f"{level + 1}단계 / {n}")
        p.setFont(val); p.setPen(QColor(C_TEXT))
        p.drawText(QRectF(ox, ty + 60, tw, 24), Qt.AlignCenter, f"{d['speed']}%")

        # 4) 촬영 시각, 파일
        y = ty + th + 10
        for label, value in (("촬영", d["time"]), ("파일", d["name"])):
            p.setFont(small); p.setPen(QColor(C_MUTED))
            p.drawText(QRectF(0, y, 30, 18), Qt.AlignVCenter | Qt.AlignLeft, label)
            p.setFont(mid if label == "촬영" else small); p.setPen(QColor(C_TEXT if label == "촬영" else "#c3cad6"))
            text = p.fontMetrics().elidedText(value, Qt.ElideMiddle, int(w - 34))
            p.drawText(QRectF(34, y, w - 34, 18), Qt.AlignVCenter | Qt.AlignLeft, text)
            y += 20


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
    ready = Signal(int, int, QImage)        # (세대, 프레임 번호, 그림)
    finished = Signal()

    def __init__(self, jobs: List[tuple], thumb_px: int, gen: int = 0):
        super().__init__()
        self.jobs = jobs            # (index, 세션 경로, 파일 이름)
        self.thumb_px = thumb_px
        self.gen = gen              # 프레임 목록이 바뀌면 번호가 밀리므로, 받는 쪽이 지난 세대의 결과를 버린다
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        sources: dict = {}
        try:
            for idx, spath, name in self.jobs:
                if self._stop:
                    break
                try:
                    src = sources.get(spath)
                    if src is None:
                        src = sources[spath] = SessionSource(spath)     # 스레드 전용 핸들 (zip 은 스레드 안전하지 않다)
                    img = QImage.fromData(src.read_bytes(name))
                except (OSError, KeyError, zipfile.BadZipFile):     # 한 장을 못 읽어도 나머지는 계속
                    continue
                if not img.isNull():
                    img = img.scaled(self.thumb_px, self.thumb_px, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.ready.emit(self.gen, idx, img)
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

    def __init__(self, path: Path, included: int, excluded: int, sessions: int, parent=None,
                 crop: Optional[Tuple[float, float]] = None):
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
        key = f"color:{C_MUTED}; white-space:nowrap; padding-right:8px"
        crop_row = f"<tr><td style='{key}'>자르는 범위</td><td>{describe_crop(crop)}</td></tr>" if crop else ""
        info = QLabel(f"<table cellspacing='4'><tr><td style='{key}'>파일</td><td><b>{path.name}</b></td></tr>"
                      f"<tr><td style='{key}'>위치</td><td>{path.parent}</td></tr>{crop_row}</table>")
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


def session_when(name: str, meta: dict) -> str:
    """세션 이름(YYYYMMDD_HHMM)과 메타에서 '11월 10일 19:06 · 16분 주행' 같은 한 줄을 만든다."""
    try:
        t = datetime.strptime(name[:13], "%Y%m%d_%H%M")
        text = f"{t.month}월 {t.day}일 {t:%H:%M} 촬영"
    except ValueError:
        return ""
    try:
        sec = (datetime.fromisoformat(meta["ended_at"]) - datetime.fromisoformat(meta["started_at"])).total_seconds()
        if sec >= 60:
            text += f" · {round(sec / 60)}분 주행"
    except (KeyError, TypeError, ValueError):
        pass
    return text


class SessionItem(QWidget):
    """세션 한 개의 카드: 색 띠(타임라인과 같은 색), 이름, 촬영 시각, 숫자 세 개(사진·제외·주행 중 불량 표시)."""

    def __init__(self, name: str, when: str, color: str, frames: int, excluded: int, marks: int, is_zip: bool,
                 empty: int = 0, parent=None):
        super().__init__(parent)
        self.setToolTip("사진: 이 세션에 저장된 장수\n제외: 데이터셋에서 빼기로 한 장수\n"
                        "불량 표시: 주행 중 패드의 □ 버튼을 누른 횟수 (누를 때마다 직전 3초가 자동으로 제외된다)"
                        + (f"\n\n빈 사진 {empty}장은 목록에서 뺐습니다 (전원이 끊겨 저장되지 못한 사진)" if empty else ""))
        row = QHBoxLayout(self); row.setContentsMargins(0, 0, 10, 0); row.setSpacing(10)
        bar = QFrame(); bar.setFixedWidth(5)
        bar.setStyleSheet(f"background:{color}; border-top-left-radius:7px; border-bottom-left-radius:7px;")
        row.addWidget(bar)
        col = QVBoxLayout(); col.setContentsMargins(0, 8, 0, 8); col.setSpacing(2)
        title = QLabel(name + ("  ·  zip" if is_zip else "")); title.setObjectName("sessName")
        col.addWidget(title)
        if when:
            w = QLabel(when); w.setObjectName("hint"); col.addWidget(w)
        col.addStretch(1)
        stats = QHBoxLayout(); stats.setContentsMargins(0, 4, 0, 0); stats.setSpacing(0)
        for value, label, c in ((f"{frames:,}", "사진", C_TEXT), (f"{excluded:,}", "제외", C_DANGER if excluded else C_MUTED),
                                (f"{marks}회", "불량 표시 □", C_AMBER if marks else C_MUTED)):
            box = QVBoxLayout(); box.setSpacing(0)
            v = QLabel(value); v.setObjectName("statValue"); v.setStyleSheet(f"color:{c};"); v.setAlignment(Qt.AlignCenter)
            n = QLabel(label); n.setObjectName("hint"); n.setAlignment(Qt.AlignCenter)
            box.addWidget(v); box.addWidget(n); stats.addLayout(box, 1)
        col.addLayout(stats)
        row.addLayout(col, 1)


PLAY_SPEEDS = (1, 2, 5, 10)          # 재생 배속. 1 = 실제 촬영 시간 그대로
PLAY_GAP_MAX_SEC = 1.0               # 멈췄다 다시 달린 곳·세션 경계의 긴 공백은 이만큼만 기다린다


def play_delay_ms(dt_sec: float, speed: float) -> int:
    """다음 사진까지 기다릴 시간(ms): 두 사진의 실제 촬영 간격을 배속으로 나눈다."""
    dt = min(PLAY_GAP_MAX_SEC, max(0.05, dt_sec))
    return max(15, round(dt / speed * 1000))


def _reason(e: Exception) -> str:
    """알림에 넣을 짧은 이유. OSError 의 기본 글에는 전체 경로가 붙으므로 설명만 꺼낸다."""
    if isinstance(e, OSError) and e.strerror:
        return e.strerror
    return str(e) or type(e).__name__


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
        self._history: list = []                            # 되돌리기 순서. 제외·복구 = 건드린 세션 이름들,
                                                            #   자르는 범위 = ("crop", 바꾸기 전 상태)
        self.show_crop = True                               # 사진 위에 자르는 선을 보이는가
        self._crop_before: Optional[tuple] = None           # 선을 끌기 시작할 때의 상태 (끌기 한 번 = 되돌리기 한 번)
        self._cur_data: Optional[bytes] = None              # 지금 보고 있는 사진 파일의 내용 ("모델이 보는 사진"을 다시 그릴 때)
        self._thumbs: Dict[int, QPixmap] = {}
        self._flag_state: List[Optional[Tuple[bool, bool]]] = []
        self._session_span: Dict[str, Tuple[int, int]] = {}
        self._thumb_thread: Optional[QThread] = None
        self._thumb_worker: Optional[ThumbWorker] = None
        self._thumb_gen = 0                                 # 썸네일 작업의 세대 (프레임 목록이 바뀔 때마다 +1)
        self._retired: List[tuple] = []                     # 제때 멈추지 못한 (스레드, 작업자). 끝날 때까지 참조를 쥔다
        self._play = QTimer(self)
        self._play.timeout.connect(self._play_tick)
        self.setWindowTitle(APP_TITLE)
        self.setStyleSheet(QSS)
        self.resize(1480, 920)
        self.setMinimumSize(1180, 700)
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
                          self._btn("되돌리기", "마지막 제외/복구, 자르는 범위 변경 되돌리기 (Ctrl+Z)", self.undo)])
        self.play_btn = self._btn("▶ 재생", "자동 재생 · 정지 (Space)", self.toggle_play)
        self.play_btn.setMinimumWidth(78)
        seg = QFrame(); seg.setObjectName("seg")       # 배속: 네 칸 중 하나가 켜져 있는 선택 버튼
        sr = QHBoxLayout(seg); sr.setContentsMargins(3, 3, 3, 3); sr.setSpacing(2)
        self.speed_btns: Dict[int, QPushButton] = {}
        for v in PLAY_SPEEDS:
            b = self._btn(f"{v}×", f"{v}배속" + (" · 실제 주행 속도" if v == 1 else "") + "  (+ / - 키로도 바꿀 수 있다)",
                          lambda _=False, v=v: self.set_speed(v), "segBtn")
            b.setCheckable(True); b.setAutoExclusive(True)
            self.speed_btns[v] = b; sr.addWidget(b)
        self.set_speed(self.cfg.curator.autoplay_speed)
        group("재생 · 배속", [self.play_btn, seg])
        bl.addStretch(1)
        self.build_btn = self._btn("데이터셋 만들기", "포함된 사진으로 dataset_*.zip 생성 (Ctrl+B)", self.build_dataset_dialog, "primary")
        self.build_btn.setMinimumHeight(38)
        bl.addWidget(self.build_btn, 0, Qt.AlignBottom)

        # 가운데: [왼쪽 = 무엇을 보고 있나 (세션, 현재 프레임)] [사진] [오른쪽 = 무엇이 데이터셋에 들어가나 (구성, 분포, 선택)]
        self.view = FrameView()
        self.view.crop_dragged.connect(self._on_crop_dragged)
        self.view.crop_released.connect(self._on_crop_released)

        def panel(name: str, width: int) -> Tuple[QScrollArea, QVBoxLayout]:
            inner = QWidget(); inner.setObjectName(name)
            lay_ = QVBoxLayout(inner); lay_.setContentsMargins(12, 10, 12, 10); lay_.setSpacing(8)
            area = QScrollArea(); area.setWidget(inner); area.setWidgetResizable(True)
            area.setFixedWidth(width); area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            return area, lay_

        left, ll = panel("sideLeft", 300)
        c, lay = card("세션", "수집기를 한 번 실행해 저장한 사진 폴더입니다.<br>클릭하면 해당 폴더의 첫 사진으로 이동합니다.")
        self.sessions_list = QListWidget(); self.sessions_list.setObjectName("sessions")
        self.sessions_list.setFocusPolicy(Qt.NoFocus); self.sessions_list.setMinimumHeight(60)
        self.sessions_list.itemClicked.connect(self._on_session_clicked)
        lay.addWidget(self.sessions_list, 1); ll.addWidget(c, 1)            # 남는 높이는 세션 목록이 쓴다

        c, lay = card("현재 프레임")
        self.info = FrameInfoView()
        lay.addWidget(self.info); ll.addWidget(c)


        right, sl = panel("side", 348)
        c, lay = card("데이터셋 구성")
        self.summary = SummaryView(); lay.addWidget(self.summary)
        sl.addWidget(c)

        c, lay = card("클래스 분포", "파랑 포함 · 빨강 제외 · 점선은 각도별 평균")
        self.chart = BarChart(); lay.addWidget(self.chart, 1); lay.addStretch(0)
        sl.addWidget(c, 1)

        c, lay = card("선택")
        self.range_label = QLabel(); self.range_label.setObjectName("value"); self.range_label.setWordWrap(True)
        lay.addWidget(self.range_label); sl.addWidget(c)

        # 사진 바로 아래: 모델이 보는 사진 (위의 사진을 두 선 사이만 남기고 줄인 것). 옆 패널은 세로 자리가 모자라
        # 스크롤해야 보이므로, 선을 끄는 동안 언제나 보이는 이 자리에 둔다
        pc = self.cfg.preprocess
        scale = min(1.0, self.MODEL_VIEW_MAX_H / pc.out_height)
        self._model_view_size = QSize(max(1, round(pc.out_width * scale)), max(1, round(pc.out_height * scale)))
        self.model_bar = QWidget(); self.model_bar.setObjectName("modelbar")
        ml = QHBoxLayout(self.model_bar); ml.setContentsMargins(12, 8, 12, 8); ml.setSpacing(12)
        self.model_view = QLabel(); self.model_view.setAlignment(Qt.AlignCenter); self.model_view.setObjectName("modelView")
        self.model_view.setFixedSize(self._model_view_size.width() + 2, self._model_view_size.height() + 2)
        ml.addWidget(self.model_view)
        t = QLabel("모델이 보는 사진"); t.setObjectName("cardTitle")
        self.crop_label = QLabel(); self.crop_label.setObjectName("value")
        h = QLabel("사진 위의 두 선을 끌어 범위를 정합니다. 어두운 부분은 잘려 나갑니다."); h.setObjectName("hint")
        for lab in (self.crop_label, h):                # 창이 좁으면 글이 잘릴 뿐, 옆의 사진이나 버튼을 덮지 않게
            lab.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred); lab.setMinimumWidth(0)
        self.crop_show_btn = self._btn("선 숨기기", "사진 위의 자르는 선 보이기 · 숨기기 (C)", self.toggle_crop_lines)
        self.crop_reset_btn = self._btn("기본값", "자르는 범위를 처음 값으로 되돌린다", self.reset_crop)
        for b in (self.crop_show_btn, self.crop_reset_btn):
            b.setStyleSheet("padding: 3px 10px;")
        head = QHBoxLayout(); head.setSpacing(6)
        head.addWidget(t); head.addStretch(1); head.addWidget(self.crop_show_btn); head.addWidget(self.crop_reset_btn)
        text = QVBoxLayout(); text.setSpacing(2)
        text.addStretch(1); text.addLayout(head); text.addWidget(self.crop_label); text.addWidget(h); text.addStretch(1)
        ml.addLayout(text, 1)
        self.model_bar.setToolTip("위의 사진에서 두 선 사이만 남기고 줄인 것입니다. 모델은 이 사진만 보고 조향을 정합니다.")
        center = QVBoxLayout(); center.setContentsMargins(0, 0, 0, 0); center.setSpacing(0)
        center.addWidget(self.view, 1); center.addWidget(self.model_bar)

        mid = QHBoxLayout(); mid.setContentsMargins(0, 0, 0, 0); mid.setSpacing(0)
        mid.addWidget(left); mid.addLayout(center, 1); mid.addWidget(right)

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
                        ("Space", self.toggle_play), ("F1", self.show_shortcuts), ("C", self.toggle_crop_lines),
                        ("+", lambda: self.step_speed(1)), ("=", lambda: self.step_speed(1)), ("-", lambda: self.step_speed(-1)),
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
    def _notice(self, title: str, text: str, detail: str = "") -> None:
        """짧은 알림 창: 굵은 한 줄 + (있으면) 작은 설명 한 줄. 전체 경로는 보여 주지 않는다."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning); box.setWindowTitle(title); box.setText(f"<b>{text}</b>")
        if detail:
            box.setInformativeText(detail)
        box.exec()

    def open_paths(self, paths: List[str]) -> None:
        opened, already, problems = [], [], []
        for p in paths:
            name = Path(p).name
            try:
                opened.append(self.ws.open(p))
            except AlreadyOpen as e:
                already.append(str(e))
            except NoFrames:
                problems.append(f"{name} — 수집한 사진이 없습니다")
            except FileNotFoundError:
                problems.append(f"{name} — 찾을 수 없습니다")
            except zipfile.BadZipFile:
                problems.append(f"{name} — 열 수 없는 zip 입니다")
            except (ValueError, OSError) as e:
                problems.append(f"{name} — {_reason(e)}")
        self._refresh_all()
        if opened:
            self.goto(self._session_span[opened[0].name][0])
            empty = sum(len(s.empty) for s in opened)
            extra = f"  ·  저장되지 못한 빈 사진 {empty:,}장은 뺐습니다" if empty else ""
            if self.ws.steering_differs_from_config:        # 차에서 조향 클래스를 바꿔 촬영한 세션: 촬영 설정을 따른다
                extra += f"  ·  조향 클래스는 촬영할 때의 설정 {self.ws.steering.classes} 을 따릅니다"
            if any(s.name in self.ws.crop_replaced for s in opened):    # 다른 범위가 적혀 있던 세션: 먼저 정해진 쪽을 따른다
                extra += f"  ·  자르는 범위는 먼저 정한 값({describe_crop(self.ws.crop)})을 따릅니다"
            self.statusBar().showMessage(f"세션 {len(opened)}개를 열었습니다 (사진 {sum(len(s.frames) for s in opened):,}장){extra}",
                                         10000 if extra else 4000)
        elif already and not problems:                      # 오류가 아니다: 그 세션으로 데려다주기만 한다
            self.goto(self._session_span[already[0]][0])
            self.statusBar().showMessage(f"이미 열려 있는 세션입니다: {', '.join(already)}", 4000)
        if problems:
            self._notice("세션 열기", "열지 못한 항목이 있습니다" if len(problems) > 1 else "세션을 열지 못했습니다",
                         "<br>".join(problems))

    def open_folder_dialog(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "세션 폴더 선택", str(Path(self.cfg.recording.root)))
        if d:
            self.open_paths([d])

    def open_zip_dialog(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "세션 zip 선택", str(Path(self.cfg.recording.root)), "zip (*.zip)")
        if f:
            self.open_paths([f])

    SAVE_FAILED = "제외 목록을 저장하지 못했습니다 (폴더가 읽기 전용인지 확인하세요). 변경은 화면에만 남아 있습니다"
    MODEL_VIEW_MAX_H = 96       # "모델이 보는 사진"은 실제 크기(preprocess.out_width × out_height)로 그린다. 이보다 크면 줄인다

    def _save(self) -> bool:
        """curation.json 저장. 실패해도 프로그램은 계속 쓴다 (읽기 전용 폴더, 동기화 프로그램이 잡고 있는 파일)."""
        try:
            self.ws.save()
            return True
        except OSError as e:
            log.warning("제외 목록 저장 실패: %s", e)
            return False

    def save(self) -> None:
        self.statusBar().showMessage("제외 목록을 저장했습니다" if self._save() else self.SAVE_FAILED, 4000)

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
        self.crop_show_btn.setEnabled(has); self.crop_reset_btn.setEnabled(has)
        self._refresh_crop()

    def _refresh_sessions(self) -> None:
        self.sessions_list.clear()
        for s in sorted(self.ws.sessions, key=lambda s: s.name):
            exc = sum(1 for f in s.frames if self.ws.is_excluded(f))
            w = SessionItem(s.name, session_when(s.name, s.meta), self._session_color(s.name),
                            len(s.frames), exc, len(s.marks), s.is_zip, len(s.empty))
            it = QListWidgetItem()
            it.setData(Qt.UserRole, s.name)
            it.setSizeHint(QSize(0, 98))                  # 이름 + 촬영 시각 + 숫자 줄 + 아래 여백
            self.sessions_list.addItem(it)
            self.sessions_list.setItemWidget(it, w)
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
        self._thumb_gen += 1
        self._thumb_thread = QThread(self)
        self._thumb_worker = ThumbWorker(jobs, self.cfg.curator.thumb_px, self._thumb_gen)
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
            if not self._thumb_thread.wait(2000):           # 느린 디스크에서 아직 읽는 중: 돌고 있는 객체를 버리지 않는다
                self._retired.append((self._thumb_thread, self._thumb_worker))
        self._retired = [(t, w) for t, w in self._retired if t.isRunning()]
        self._thumb_thread = self._thumb_worker = None
        self._thumb_gen += 1                                # 이미 보낸 신호가 뒤늦게 도착해도 버려진다

    def _on_thumb(self, gen: int, idx: int, img: QImage) -> None:
        if gen != self._thumb_gen:                          # 프레임 목록이 바뀌기 전의 결과: 번호가 맞지 않는다
            return
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
            self.info.set_info(None)
            self._cur_data = None
            self._refresh_model_view()
            self.statusBar().showMessage("세션을 여세요  ·  단축키는 F1")
            return
        f = self.frames[self.index]
        try:
            data = self.ws.read_image(f)
            img = QImage.fromData(data) if image_is_complete(f.name, data) else QImage()
        except (OSError, KeyError, zipfile.BadZipFile) as e:
            log.warning("사진을 읽지 못했습니다: %s (%s)", f.key, e)
            data, img = b"", QImage()
        pm = QPixmap.fromImage(img) if not img.isNull() else None
        self._cur_data = data if pm is not None else None
        self._refresh_model_view()
        unreadable = "" if pm is not None else "사진을 읽을 수 없습니다  (파일이 지워졌거나 저장이 덜 됐습니다. Delete 로 제외하세요)"
        side = self.ws.steering.logical_side(f.info.angle)
        rot = arrow_rotation_deg(side, self.cfg.curator.arrow_max_deg, self.cfg.curator.arrow_invert)
        reason = self.ws.curation(f).excluded.get(f.name)
        ts = f"{f.info.timestamp:%H:%M:%S}.{f.info.timestamp.microsecond // 1000:03d}"
        self.view.set_frame(pm, rot, f"{self.index + 1} / {len(self.frames)}     조향 {f.info.angle}°     속도 {f.info.speed}%     {ts}", reason, unreadable)
        a, b = self._session_span[f.session]
        self.info.set_info({
            "name": f.name, "session": f.session, "color": self._session_color(f.session),
            "in_session": self.index - a + 1, "session_len": b - a + 1, "overall": self.index + 1, "total": len(self.frames),
            "angle": f.info.angle, "side": side, "speed": f.info.speed, "levels": list(self.cfg.throttle.levels),
            "time": ts, "reason": reason})
        self.statusBar().showMessage(f"{self.index + 1} / {len(self.frames):,}   ·   세션 {f.session}   ·   제외 목록은 자동 저장됩니다")
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
                                     f"<span style='color:{C_MUTED}'><b>Delete</b> 제외 · <b>Insert</b> 복구 · <b>Esc</b> 해제</span>")
        else:
            rows = (("Shift + ← / →", "연속 선택"), ("Shift + 클릭", "클릭한 곳까지 연속 선택"),
                    ("Ctrl + 클릭", "떨어진 낱장 선택"))
            self.range_label.setText("<table cellspacing='0' cellpadding='0'>" + "".join(
                f"<tr><td style='padding:2px 14px 2px 0; color:#c3cad6'><b>{k}</b></td>"
                f"<td style='padding:2px 0; color:{C_MUTED}'>{v}</td></tr>" for k, v in rows) + "</table>")

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
            self._play.start(self._play_delay()); self.play_btn.setText("■ 정지")

    def _play_delay(self) -> int:
        """지금 사진에서 다음 사진까지의 실제 촬영 간격을 배속으로 나눈 시간(ms)."""
        if self.index >= len(self.frames) - 1:
            return play_delay_ms(PLAY_GAP_MAX_SEC, self.speed)
        a, b = self.frames[self.index], self.frames[self.index + 1]
        return play_delay_ms((b.info.timestamp - a.info.timestamp).total_seconds(), self.speed)

    def _play_tick(self) -> None:
        if self.index >= len(self.frames) - 1:
            self._play.stop(); self.play_btn.setText("▶ 재생")
        else:
            self.goto(self.index + 1)
            self._play.setInterval(self._play_delay())

    def set_speed(self, speed: float) -> None:
        self.speed = min(PLAY_SPEEDS, key=lambda v: abs(v - speed))
        self.speed_btns[self.speed].setChecked(True)
        if self._play.isActive():
            self._play.setInterval(self._play_delay())

    def step_speed(self, step: int) -> None:
        k = min(len(PLAY_SPEEDS) - 1, max(0, PLAY_SPEEDS.index(self.speed) + step))
        self.set_speed(PLAY_SPEEDS[k])
        self.statusBar().showMessage(f"재생 배속 {self.speed}× (1× = 실제 주행 속도)", 2500)

    # -- 제외·복구 ---------------------------------------------------------
    def _selection(self) -> List[Frame]:
        if self.selected:
            return [self.frames[i] for i in sorted(self.selected)]
        return [self.frames[self.index]] if self.frames else []

    def _after_change(self, msg: str) -> None:
        saved = self._save()                            # 바뀔 때마다 저장: 원본이 아니라 curation.json 만
        self._paint_strip_flags(); self._refresh_sessions(); self._show_current(); self._refresh_counts()
        self.statusBar().showMessage(msg if saved else self.SAVE_FAILED, 4000 if saved else 10000)

    def _change(self, sel: List[Frame], exclude: bool) -> int:
        """돌려주는 값 = 실제로 상태가 바뀐 장수 (이미 제외된 사진을 또 제외하면 0)."""
        changed = sum(1 for f in sel if self.ws.is_excluded(f) != exclude)
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
        return changed

    def _change_selection(self, exclude: bool) -> None:
        sel = self._selection()
        if not sel:
            return
        n = self._change(sel, exclude)
        self.selected = set(); self.anchor = self.index
        if n:
            self._after_change(f"{n:,}장을 {'제외' if exclude else '복구'}했습니다  ·  Ctrl+Z 로 되돌리기")
        else:                                           # 바뀐 것이 없다: 저장도, 되돌리기 기록도 없다
            self._paint_strip_flags(); self._show_current()
            self.statusBar().showMessage("이미 제외된 사진입니다" if exclude else "이미 포함된 사진입니다", 4000)

    def exclude_selection(self) -> None:
        self._change_selection(True)

    def restore_selection(self) -> None:
        self._change_selection(False)

    def toggle_current(self) -> None:
        if not self.frames:
            return
        f = self.frames[self.index]
        self._change([f], not self.ws.is_excluded(f))
        self._after_change("이 사진을 제외했습니다" if self.ws.is_excluded(f) else "이 사진을 복구했습니다")

    def undo(self) -> None:
        if not self._history:
            self.statusBar().showMessage("되돌릴 것이 없습니다", 2000)
            return
        entry = self._history.pop()
        if isinstance(entry, tuple):                    # ("crop", 바꾸기 전 상태): 자르는 범위를 되돌린다
            self.ws.restore_crop_state(entry[1])
            self._crop_changed(f"자르는 범위를 되돌렸습니다: {describe_crop(self.ws.crop)}")
            return
        for s in entry:
            if s in self.ws.curations:
                self.ws.curations[s].undo()
        self._after_change("마지막 변경을 되돌렸습니다")

    # -- 자르는 범위 -------------------------------------------------------
    def _refresh_crop(self) -> None:
        """자르는 범위나 선 표시가 바뀌었을 때: 사진 위의 선, 글, 모델이 보는 사진."""
        has = bool(self.frames)
        self.view.set_crop(self.ws.crop if (self.show_crop and has) else None)
        self.crop_show_btn.setText("선 숨기기" if self.show_crop else "선 보이기")
        pre = self.ws.preprocessor
        self.crop_label.setText(f"{describe_crop(self.ws.crop)}  →  {pre.out_width}×{pre.out_height} 로 줄임"
                                if has else "세션을 열면 여기에 보입니다")
        self._refresh_model_view()

    def _refresh_model_view(self) -> None:
        """지금 사진을 지금 범위로 자르고 줄인 것. 학습기가 쓰는 전처리를 그대로 쓴다 (curator.core.model_view)."""
        view = model_view(self._cur_data, self.ws.preprocessor) if self._cur_data else None
        if view is None:
            self.model_view.setPixmap(QPixmap())
            self.model_view.setText("사진을 읽을 수 없습니다" if self.frames else "")
            return
        h, w, _ = view.shape
        img = QImage(view.data, w, h, 3 * w, QImage.Format_RGB888).copy()       # copy: numpy 배열이 사라져도 남게
        pm = QPixmap.fromImage(img)
        if pm.size() != self._model_view_size:
            pm = pm.scaled(self._model_view_size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.model_view.setPixmap(pm)

    def _crop_changed(self, msg: str = "") -> None:
        saved = self._save()                            # 제외 목록과 같은 curation.json 에 적힌다
        self._refresh_crop()
        msg = msg or f"자르는 범위: {describe_crop(self.ws.crop)}  ·  Ctrl+Z 로 되돌리기"
        self.statusBar().showMessage(msg if saved else self.SAVE_FAILED, 4000 if saved else 10000)

    def _on_crop_dragged(self, which: str, ratio: float) -> None:
        if not self.frames:
            return
        if self._crop_before is None:                   # 끌기 시작: 되돌리기는 끌기 한 번을 한 단위로 한다
            self._crop_before = self.ws.crop_state()
        if self.ws.set_crop(**{which: ratio}):
            self._refresh_crop()

    def _on_crop_released(self) -> None:
        before, self._crop_before = self._crop_before, None
        if before is None:                              # 누르기만 하고 움직이지 않았다
            return
        if self.ws.crop == before[0]:                   # 끌었다가 제자리에 놓았다: 바뀐 것이 없다
            self.ws.restore_crop_state(before)
            return
        self._history.append(("crop", before))
        self._crop_changed()

    def reset_crop(self) -> None:
        if not self.frames:
            return
        before = self.ws.crop_state()
        if not self.ws.set_crop(*self.ws.default_crop):
            self.ws.restore_crop_state(before)
            self.statusBar().showMessage("이미 기본 범위입니다", 3000)
            return
        self._history.append(("crop", before))
        self._crop_changed()

    def toggle_crop_lines(self) -> None:
        self.show_crop = not self.show_crop
        self._refresh_crop()

    # -- 데이터셋 ----------------------------------------------------------
    def build_dataset_dialog(self) -> None:
        if not self.frames:
            return
        inc, exc = self.ws.counts()
        if not sum(inc.values()):
            self._notice("데이터셋 만들기", "포함된 사진이 없습니다", "제외한 사진을 복구(Insert)한 뒤 다시 만드세요.")
            return
        out = QFileDialog.getExistingDirectory(self, "데이터셋을 저장할 폴더", str(Path(self.cfg.recording.root)))
        if not out:
            return
        dlg = QProgressDialog(f"데이터셋 만드는 중…  ({sum(inc.values()):,}장)", None, 0, sum(inc.values()), self)
        dlg.setWindowTitle("데이터셋 만들기")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.show()
        try:
            path = build_dataset(self.ws, out, progress=lambda i, n: (dlg.setValue(i), QApplication.processEvents()))
        except BrokenImage as e:                        # 그 사진으로 데려가서 제외하게 한다
            dlg.close()
            if e.frame in self.frames:
                self.goto(self.frames.index(e.frame))
            self._notice("데이터셋 만들기", "저장이 덜 된 사진이 있습니다",
                         "그 사진으로 이동했습니다. Delete 로 제외한 뒤 다시 만드세요.")
            return
        except Exception as e:                          # 무엇이 실패하든 진행 창을 닫고 이유를 알린다 (반쪽 zip 은 남지 않는다)
            log.exception("데이터셋 생성 실패")
            self._notice("데이터셋 만들기", "데이터셋을 만들지 못했습니다", _reason(e))
            return
        finally:
            dlg.close()
        self.statusBar().showMessage(f"데이터셋 작성 완료: {path.name}", 8000)
        BuildDoneDialog(path, sum(inc.values()), sum(exc.values()), len(self.ws.sessions), self, crop=self.ws.crop).exec()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._play.stop()
        self._stop_thumbs()
        self._save()                                    # 저장에 실패해도 창은 닫혀야 한다
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
        cfg = load_config(args.config, program="curator")
    except ConfigError as e:
        print(f"[설정 오류] {e}", file=sys.stderr)
        return 1
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow(cfg)
    if args.paths:
        win.open_paths(args.paths)
    win.show()
    return app.exec()
