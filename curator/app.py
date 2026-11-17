"""가공 도구 화면 (PySide6). 로직은 curator/core.py 에 있고 여기서는 보여 주고 키를 받는다.

  uv run -m curator [세션 폴더 또는 zip ...]

키: ←/→ 한 장, PageUp/PageDown 10장, Home/End 처음/끝, Space 자동 재생, [ ] 구간 시작/끝, Delete 구간(없으면 현재) 제외,
    Insert 구간(없으면 현재) 복구, X 현재 프레임 제외/복구, Ctrl+Z 되돌리기, Ctrl+S 저장, Ctrl+O 폴더 열기, Esc 구간 해제.
제외 상태는 바뀔 때마다 세션 옆 curation.json 에 저장된다. 원본 사진은 건드리지 않는다.
"""
from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QObject, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (QAction, QBrush, QColor, QFont, QIcon, QImage, QKeySequence, QPainter, QPen, QPixmap,
                           QPolygonF)
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow,
                               QMessageBox, QProgressDialog, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from common.config import Config
from curator.core import REASON_MARK, Frame, SessionSource, Workspace, build_dataset

log = logging.getLogger("curator")


def arrow_rotation_deg(side: float, max_deg: float, invert: bool) -> float:
    """조향 쪽(-1 왼쪽 … +1 오른쪽)을 화살표 기울기(시계 방향 양수)로. 화면 표시용 반전은 따로 적용한다."""
    deg = max(-1.0, min(1.0, side)) * max_deg
    return -deg if invert else deg


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
        self.setMinimumSize(320, 240)

    def set_frame(self, pixmap: Optional[QPixmap], rotation: float, caption: str, excluded_reason: Optional[str]) -> None:
        self.pixmap, self.rotation, self.caption, self.excluded_reason = pixmap, rotation, caption, excluded_reason
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt 이름)
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(24, 26, 30))
        if self.pixmap is None:
            p.setPen(QColor(160, 160, 160))
            p.drawText(self.rect(), Qt.AlignCenter, "세션 폴더를 여세요 (Ctrl+O)")
            return
        scaled = self.pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        p.drawPixmap(x, y, scaled)
        # 화살표: 사진 아래 가운데에서 위로, 조향만큼 기울임
        p.setRenderHint(QPainter.Antialiasing)
        length = scaled.height() * 0.35
        p.save()
        p.translate(x + scaled.width() / 2, y + scaled.height() - 10)
        p.rotate(self.rotation)
        pen = QPen(QColor(255, 196, 0), 6, Qt.SolidLine, Qt.RoundCap)
        p.setPen(pen)
        p.drawLine(QPointF(0, 0), QPointF(0, -length))
        head = QPolygonF([QPointF(0, -length - 14), QPointF(-14, -length + 8), QPointF(14, -length + 8)])
        p.setBrush(QBrush(QColor(255, 196, 0)))
        p.setPen(Qt.NoPen)
        p.drawPolygon(head)
        p.restore()
        # 설명 글자
        p.setPen(QColor(255, 255, 255))
        f = QFont(); f.setPointSize(11); f.setBold(True); p.setFont(f)
        p.fillRect(x, y, scaled.width(), 26, QColor(0, 0, 0, 140))
        p.drawText(x + 8, y + 19, self.caption)
        if self.excluded_reason:
            p.fillRect(x, y, scaled.width(), scaled.height(), QColor(180, 0, 0, 70))
            f.setPointSize(20); p.setFont(f)
            p.setPen(QColor(255, 80, 80))
            why = "불량 표시 구간 (□)" if self.excluded_reason == REASON_MARK else "제외됨"
            p.drawText(QPointF(x + 12, y + scaled.height() - 16), f"✕ {why}")


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
        self.range_start: Optional[int] = None
        self.range_end: Optional[int] = None
        self._thumb_thread: Optional[QThread] = None
        self._thumb_worker: Optional[ThumbWorker] = None
        self._play = QTimer(self)
        self._play.timeout.connect(self._play_tick)
        self.setWindowTitle("자율주행 RC카 데이터 가공")
        self.resize(1280, 820)
        self._build_ui()
        self._build_actions()
        self._refresh_all()

    # -- UI ----------------------------------------------------------------
    def _build_ui(self) -> None:
        self.view = FrameView()
        self.sessions_list = QListWidget()
        self.sessions_list.setMaximumHeight(120)
        self.counts = QTableWidget(0, 4)
        self.counts.setHorizontalHeaderLabels(["각도", "포함", "제외", "비율"])
        self.counts.verticalHeader().setVisible(False)
        self.counts.setEditTriggers(QTableWidget.NoEditTriggers)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.help = QLabel("←/→ 한 장 · PgUp/PgDn 10장 · Home/End · Space 재생\n"
                           "[ ] 구간 · Delete 제외 · Insert 복구 · X 토글 · Ctrl+Z 되돌리기\n"
                           "Ctrl+O 폴더 열기 · Ctrl+S 저장 · Ctrl+B 데이터셋 만들기")
        self.help.setStyleSheet("color: #888;")
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addWidget(QLabel("세션"))
        rl.addWidget(self.sessions_list)
        rl.addWidget(QLabel("클래스별 장수 (포함 기준)"))
        rl.addWidget(self.counts)
        rl.addWidget(self.info)
        rl.addStretch(1)
        rl.addWidget(self.help)
        right.setMinimumWidth(300)
        self.strip = QListWidget()
        self.strip.setFlow(QListWidget.LeftToRight)
        self.strip.setViewMode(QListWidget.IconMode)
        self.strip.setWrapping(False)
        self.strip.setMovement(QListWidget.Static)
        px = self.cfg.curator.thumb_px
        self.strip.setIconSize(QSize(px, px))
        self.strip.setFixedHeight(px + 44)
        self.strip.setUniformItemSizes(True)
        self.strip.currentRowChanged.connect(self._on_strip_row)
        top = QSplitter(Qt.Horizontal)
        top.addWidget(self.view)
        top.addWidget(right)
        top.setStretchFactor(0, 1)
        central = QWidget()
        cl = QVBoxLayout(central)
        cl.addWidget(top, 1)
        cl.addWidget(self.strip)
        self.setCentralWidget(central)
        self.statusBar().showMessage("준비")

    def _build_actions(self) -> None:
        tb = self.addToolBar("도구")
        tb.setMovable(False)

        def act(text, shortcut, fn):
            a = QAction(text, self)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            a.triggered.connect(fn)
            self.addAction(a)
            tb.addAction(a)
            return a
        act("폴더 열기", "Ctrl+O", self.open_folder_dialog)
        act("zip 열기", "Ctrl+Shift+O", self.open_zip_dialog)
        act("저장", "Ctrl+S", self.save)
        act("데이터셋 만들기", "Ctrl+B", self.build_dataset_dialog)
        tb.addSeparator()
        act("제외", "Delete", self.exclude_selection)
        act("복구", "Insert", self.restore_selection)
        act("토글", "X", self.toggle_current)
        act("되돌리기", "Ctrl+Z", self.undo)
        tb.addSeparator()
        act("재생/정지", "Space", self.toggle_play)
        # 이동 키 (툴바에는 안 보임)
        for key, fn in (("Left", lambda: self.goto(self.index - 1)), ("Right", lambda: self.goto(self.index + 1)),
                        ("PgUp", lambda: self.goto(self.index - 10)), ("PgDown", lambda: self.goto(self.index + 10)),
                        ("Home", lambda: self.goto(0)), ("End", lambda: self.goto(len(self.frames) - 1)),
                        ("[", self.mark_range_start), ("]", self.mark_range_end), ("Escape", self.clear_range)):
            a = QAction(self); a.setShortcut(QKeySequence(key)); a.triggered.connect(fn); self.addAction(a)

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

    # -- 표시 ------------------------------------------------------------
    def _refresh_all(self) -> None:
        self.frames = self.ws.frames()
        self.index = min(self.index, max(0, len(self.frames) - 1))
        self.range_start = self.range_end = None
        self.sessions_list.clear()
        for s in sorted(self.ws.sessions, key=lambda s: s.name):
            self.sessions_list.addItem(f"{s.name}  ({len(s.frames)}장, 불량 표시 {len(s.marks)}회)")
        self._rebuild_strip()
        self._show_current()
        self._refresh_counts()

    def _rebuild_strip(self) -> None:
        self._stop_thumbs()
        self.strip.blockSignals(True)
        self.strip.clear()
        px = self.cfg.curator.thumb_px
        for f in self.frames:
            it = QListWidgetItem(f"a{f.info.angle:03d}")
            it.setSizeHint(QSize(px + 8, px + 36))
            it.setTextAlignment(Qt.AlignHCenter)
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
        it = self.strip.item(idx)
        if it is not None and not img.isNull():
            it.setIcon(QIcon(QPixmap.fromImage(img)))

    def _paint_strip_flags(self) -> None:
        lo, hi = self._range()
        for i, f in enumerate(self.frames):
            it = self.strip.item(i)
            if it is None:
                continue
            ex = self.ws.is_excluded(f)
            in_range = lo is not None and lo <= i <= hi
            it.setText(("✕ " if ex else "") + f"a{f.info.angle:03d}")
            it.setBackground(QBrush(QColor(255, 220, 220)) if ex else (QBrush(QColor(220, 235, 255)) if in_range else QBrush()))
            it.setForeground(QBrush(QColor(170, 0, 0)) if ex else QBrush(QColor(30, 30, 30)))

    def _show_current(self) -> None:
        if not self.frames:
            self.view.set_frame(None, 0.0, "", None)
            self.info.setText("")
            self.statusBar().showMessage("세션을 여세요")
            return
        f = self.frames[self.index]
        img = QImage.fromData(self.ws.read_image(f))
        pm = QPixmap.fromImage(img) if not img.isNull() else None
        side = self.ws.steering.logical_side(f.info.angle)
        rot = arrow_rotation_deg(side, self.cfg.curator.arrow_max_deg, self.cfg.curator.arrow_invert)
        reason = self.ws.curation(f).excluded.get(f.name)
        cap = f"{self.index + 1}/{len(self.frames)}   {f.info.angle:3d}°   {f.info.speed}%   {f.info.timestamp:%H:%M:%S.%f}"[:-3]
        self.view.set_frame(pm, rot, cap, reason)
        lo, hi = self._range()
        rng = f"구간 {lo + 1}~{hi + 1} ({hi - lo + 1}장)" if lo is not None else ("구간 시작 " + str(self.range_start + 1) if self.range_start is not None else "구간 없음")
        self.info.setText(f"<b>{f.name}</b><br>세션 {f.session}<br>{'제외됨: ' + reason if reason else '포함'}<br>{rng}")
        self.statusBar().showMessage(f"{self.index + 1}/{len(self.frames)}  {f.key}")
        self.strip.blockSignals(True)
        self.strip.setCurrentRow(self.index)
        self.strip.scrollToItem(self.strip.item(self.index), QListWidget.PositionAtCenter)
        self.strip.blockSignals(False)

    def _refresh_counts(self) -> None:
        inc, exc = self.ws.counts()
        total = sum(inc.values())
        classes = list(self.ws.steering.classes)
        self.counts.setRowCount(len(classes) + 1)
        for r, c in enumerate(classes):
            share = inc[c] / total * 100 if total else 0.0
            for col, text in enumerate((f"{c}°", str(inc[c]), str(exc[c]), f"{share:4.1f} %")):
                self.counts.setItem(r, col, QTableWidgetItem(text))
        for col, text in enumerate(("합계", str(total), str(sum(exc.values())), "100 %" if total else "0 %")):
            it = QTableWidgetItem(text); it.setFont(QFont("", -1, QFont.Bold))
            self.counts.setItem(len(classes), col, it)
        self.counts.resizeColumnsToContents()

    # -- 이동 ------------------------------------------------------------
    def goto(self, i: int) -> None:
        if not self.frames:
            return
        self.index = max(0, min(len(self.frames) - 1, i))
        self._show_current()

    def _on_strip_row(self, row: int) -> None:
        if 0 <= row < len(self.frames):
            self.index = row
            self._show_current()

    def toggle_play(self) -> None:
        if self._play.isActive():
            self._play.stop()
        else:
            self._play.start(int(1000 / self.cfg.curator.autoplay_fps))

    def _play_tick(self) -> None:
        if self.index >= len(self.frames) - 1:
            self._play.stop()
        else:
            self.goto(self.index + 1)

    # -- 구간·제외 -----------------------------------------------------------
    def _range(self):
        if self.range_start is None or self.range_end is None:
            return None, None
        return min(self.range_start, self.range_end), max(self.range_start, self.range_end)

    def mark_range_start(self) -> None:
        self.range_start, self.range_end = self.index, None
        self._paint_strip_flags(); self._show_current()

    def mark_range_end(self) -> None:
        if self.range_start is None:
            self.range_start = self.index
        self.range_end = self.index
        self._paint_strip_flags(); self._show_current()

    def clear_range(self) -> None:
        self.range_start = self.range_end = None
        self._paint_strip_flags(); self._show_current()

    def _selection(self) -> List[Frame]:
        lo, hi = self._range()
        if lo is None:
            return [self.frames[self.index]] if self.frames else []
        return self.frames[lo:hi + 1]

    def _after_change(self, msg: str) -> None:
        self.ws.save()                                  # 바뀔 때마다 저장: 원본이 아니라 curation.json 만
        self._paint_strip_flags(); self._show_current(); self._refresh_counts()
        self.statusBar().showMessage(msg, 3000)

    def exclude_selection(self) -> None:
        sel = self._selection()
        for s in {f.session for f in sel}:
            self.ws.curations[s].exclude([f.name for f in sel if f.session == s])
        self.clear_range()
        self._after_change(f"{len(sel)}장 제외")

    def restore_selection(self) -> None:
        sel = self._selection()
        for s in {f.session for f in sel}:
            self.ws.curations[s].restore([f.name for f in sel if f.session == s])
        self.clear_range()
        self._after_change(f"{len(sel)}장 복구")

    def toggle_current(self) -> None:
        if not self.frames:
            return
        f = self.frames[self.index]
        self.ws.curation(f).toggle(f.name)
        self._after_change("제외" if self.ws.is_excluded(f) else "복구")

    def undo(self) -> None:
        for s in sorted(self.ws.sessions, key=lambda s: s.name):       # 마지막으로 바꾼 세션부터는 알 수 없으므로 순서대로 시도
            c = self.ws.curations[s.name]
            if c.can_undo:
                label = c.undo()
                self._after_change(f"되돌리기: {label}")
                return
        self.statusBar().showMessage("되돌릴 것이 없습니다", 2000)

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
        QMessageBox.information(self, "데이터셋", f"{path}\n{sum(inc.values())}장 (제외 {sum(exc.values())}장)")

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
