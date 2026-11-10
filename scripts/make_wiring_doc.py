"""배선 문서 생성기. 배선 목록(NET) 하나에서 그림과 표를 함께 만들어 docs/wiring.html 을 쓴다.

  uv run python scripts/make_wiring_doc.py        HTML 생성 + 겹침 검사 (문제가 있으면 쓰지 않고 종료 코드 1)

그림의 모든 선은 NET 의 한 줄이고, 양 끝은 그림에 등록한 단자 이름으로만 지정한다. 그래서 그림과 표가 서로 다를 수 없다.
lint() 가 글자끼리, 선과 글자, 선과 상자, 선끼리의 겹침과 교차를 검사한다. PDF 는 만들어진 HTML 을 브라우저로 인쇄한다.
배선을 바꿀 때는 NET 과 그림 함수를 고친 뒤 이 스크립트와 tests/test_wiring_doc.py 를 다시 돌린다.
"""
from __future__ import annotations

import html
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "wiring.html"

# ---------------------------------------------------------------------------
# 배선 목록: (ID, 출발 단자, 도착 단자, 선, 역할). 이 차의 모든 전선.
# ---------------------------------------------------------------------------
NET: List[Tuple[str, str, str, str, str]] = [
    ("P1", "BAT+", "F1.in", "14AWG", "주전원 +"),
    ("P2", "F1.out", "J2.in", "14AWG", "주전원 +"),
    ("P3", "J2.a", "MDD.B+", "14AWG", "모터 드라이버 전원 +"),
    ("P4", "J2.b", "UBEC.IN+", "16AWG", "UBEC 입력 +"),
    ("P5", "UBEC.OUT+", "PCA1.TV+", "UBEC 출력선", "서보 전원 6 V"),
    ("G1", "BAT-", "J3.in", "14AWG", "배터리 −"),
    ("G2", "J3.a", "MDD.B-", "14AWG", "모터 전류가 돌아오는 선"),
    ("G3", "J3.b", "UBEC.IN-", "16AWG", "UBEC 입력 −"),
    ("G4", "UBEC.OUT-", "PCA1.TGND", "UBEC 출력선", "서보 전류가 돌아오는 선"),
    ("M1", "MDD.M1B", "ML.1", "모터선", "왼쪽 모터"),
    ("M2", "MDD.M1A", "ML.2", "모터선", "왼쪽 모터"),
    ("M3", "MDD.M2A", "MR.1", "모터선", "오른쪽 모터"),
    ("M4", "MDD.M2B", "MR.2", "모터선", "오른쪽 모터"),
    ("I1", "JET.1", "PCA1.VCC", "22AWG", "두 보드의 논리 전원 3.3 V"),
    ("I2", "JET.3", "PCA1.SDA", "22AWG", "I2C 데이터"),
    ("I3", "JET.5", "PCA1.SCL", "22AWG", "I2C 클럭"),
    ("I4", "JET.6", "PCA1.GND", "22AWG", "Jetson 의 유일한 접지선"),
    ("C1", "PCA1.VCC2", "PCA2.VCC", "22AWG", "보드 사이 연결"),
    ("C2", "PCA1.SDA2", "PCA2.SDA", "22AWG", "보드 사이 연결"),
    ("C3", "PCA1.SCL2", "PCA2.SCL", "22AWG", "보드 사이 연결"),
    ("C4", "PCA1.GND2", "PCA2.GND", "22AWG", "보드 사이 연결"),
    ("S1", "PCA1.CH0P", "SV.SIG", "서보선 흰색", "조향 신호"),
    ("S2", "PCA1.CH0V", "SV.V+", "서보선 빨강", "서보 전원 6 V"),
    ("S3", "PCA1.CH0G", "SV.GND", "서보선 검정", "서보 −"),
    # 모터 신호: 속도는 J4, 방향은 J5 (WAGO 221 3구) 에서 MDD10A 의 두 채널로 나뉜다
    ("D1", "PCA2.CH0P", "J4.in", "22AWG", "두 모터의 속도"),
    ("D2", "J4.a", "MDDC.PWM1", "22AWG", "속도 → 1번 채널"),
    ("D3", "J4.b", "MDDC.PWM2", "22AWG", "속도 → 2번 채널"),
    ("D4", "PCA2.CH1P", "J5.in", "22AWG", "두 모터의 방향"),
    ("D5", "J5.a", "MDDC.DIR1", "22AWG", "방향 → 1번 채널"),
    ("D6", "J5.b", "MDDC.DIR2", "22AWG", "방향 → 2번 채널"),
    ("D7", "PCA2.CH0G", "MDDC.GND", "22AWG", "모터 신호의 기준 접지"),
    # 상태 표시용 도트 매트릭스 (선택). Jetson 전원 영역 안에서만 이어지고 차체와는 무관하다
    ("X1", "JET.2", "DM.VCC", "듀폰", "도트 매트릭스 전원 5 V"),
    ("X2", "JET.20", "DM.GND", "듀폰", "도트 매트릭스 접지"),
    ("X3", "JET.19", "DM.DIN", "듀폰", "SPI 데이터"),
    ("X4", "JET.23", "DM.CLK", "듀폰", "SPI 클럭"),
    ("X5", "JET.24", "DM.CS", "듀폰", "SPI 선택"),
]
WIRE_IDS = [r[0] for r in NET]

# 단자 이름 → 표에 쓰는 사람용 이름
PORT_LABEL: Dict[str, str] = {
    "BAT+": "주행 배터리 +", "BAT-": "주행 배터리 −",
    "F1.in": "퓨즈 F1 입력", "F1.out": "퓨즈 F1 출력",
    "J2.in": "J2 1번", "J2.a": "J2 2번", "J2.b": "J2 3번",
    "J3.in": "J3 1번", "J3.a": "J3 2번", "J3.b": "J3 3번",
    "J4.in": "J4 1번", "J4.a": "J4 2번", "J4.b": "J4 3번",
    "J5.in": "J5 1번", "J5.a": "J5 2번", "J5.b": "J5 3번",
    "MDD.B+": "MDD10A B+", "MDD.B-": "MDD10A B−",
    "MDD.M1A": "MDD10A M1A", "MDD.M1B": "MDD10A M1B", "MDD.M2A": "MDD10A M2A", "MDD.M2B": "MDD10A M2B",
    "ML.1": "왼쪽 모터 전력선", "ML.2": "왼쪽 모터 나머지 전력선",
    "MR.1": "오른쪽 모터 전력선", "MR.2": "오른쪽 모터 나머지 전력선",
    "UBEC.IN+": "UBEC 입력 +", "UBEC.IN-": "UBEC 입력 −", "UBEC.OUT+": "UBEC 출력 +", "UBEC.OUT-": "UBEC 출력 −",
    "PCA1.TV+": "서보용 보드 단자대 V+", "PCA1.TGND": "서보용 보드 단자대 GND",
    "JET.1": "Jetson 핀 1 (3.3 V)", "JET.3": "Jetson 핀 3 (SDA)", "JET.5": "Jetson 핀 5 (SCL)", "JET.6": "Jetson 핀 6 (GND)",
    "JET.2": "Jetson 핀 2 (5 V)", "JET.20": "Jetson 핀 20 (GND)", "JET.19": "Jetson 핀 19 (SPI MOSI)",
    "JET.23": "Jetson 핀 23 (SPI SCK)", "JET.24": "Jetson 핀 24 (SPI CS0)",
    "DM.VCC": "도트 매트릭스 VCC", "DM.GND": "도트 매트릭스 GND", "DM.DIN": "도트 매트릭스 DIN",
    "DM.CLK": "도트 매트릭스 CLK", "DM.CS": "도트 매트릭스 CS",
    "PCA1.VCC": "서보용 보드 6핀 헤더 VCC", "PCA1.SDA": "서보용 보드 6핀 헤더 SDA",
    "PCA1.SCL": "서보용 보드 6핀 헤더 SCL", "PCA1.GND": "서보용 보드 6핀 헤더 GND",
    "PCA1.VCC2": "서보용 보드 반대쪽 헤더 VCC", "PCA1.SDA2": "서보용 보드 반대쪽 헤더 SDA",
    "PCA1.SCL2": "서보용 보드 반대쪽 헤더 SCL", "PCA1.GND2": "서보용 보드 반대쪽 헤더 GND",
    "PCA2.VCC": "모터용 보드 6핀 헤더 VCC", "PCA2.SDA": "모터용 보드 6핀 헤더 SDA",
    "PCA2.SCL": "모터용 보드 6핀 헤더 SCL", "PCA2.GND": "모터용 보드 6핀 헤더 GND",
    "PCA1.CH0P": "서보용 보드 채널 0 PWM 핀", "PCA1.CH0V": "서보용 보드 채널 0 V+ 핀", "PCA1.CH0G": "서보용 보드 채널 0 GND 핀",
    "SV.SIG": "서보 신호", "SV.V+": "서보 +", "SV.GND": "서보 −",
    "PCA2.CH0P": "모터용 보드 채널 0 PWM 핀", "PCA2.CH1P": "모터용 보드 채널 1 PWM 핀",
    "PCA2.CH0G": "모터용 보드 채널 0 GND 핀",
    "MDDC.PWM1": "MDD10A 제어 헤더 PWM1", "MDDC.PWM2": "MDD10A 제어 헤더 PWM2",
    "MDDC.DIR1": "MDD10A 제어 헤더 DIR1", "MDDC.DIR2": "MDD10A 제어 헤더 DIR2", "MDDC.GND": "MDD10A 제어 헤더 GND",
}

GROUND_IDS = ["G1", "G2", "G3", "G4", "I4", "C4", "D7"]
POWER_IDS = ["P1", "P2", "P3", "P4", "P5", "G1", "G2", "G3", "G4", "M1", "M2", "M3", "M4"]
SIGNAL_IDS = ["I1", "I2", "I3", "I4", "C1", "C2", "C3", "C4", "S1", "S2", "S3", "D1", "D2", "D3", "D4", "D5", "D6", "D7"]
DISPLAY_IDS = ["X1", "X2", "X3", "X4", "X5"]

# 선 종류별 색 (CSS 변수)과 굵기
KIND = {"P": ("--w-plus", 4.0), "G": ("--w-gnd", 4.0), "M": ("--w-motor", 3.0),
        "I": ("--w-sig", 2.2), "C": ("--w-sig", 2.2), "S": ("--w-sig", 2.2), "D": ("--w-sig", 2.2), "X": ("--w-sig", 2.2)}
OVERRIDE = {"P5": ("--w-servo", 4.0), "S2": ("--w-servo", 3.0), "S3": ("--w-gnd", 3.0),
            "I1": ("--w-vcc", 2.2), "C1": ("--w-vcc", 2.2),
            "I4": ("--w-gnd", 2.2), "C4": ("--w-gnd", 2.2), "D7": ("--w-gnd", 2.2),
            "X1": ("--w-plus", 2.2), "X2": ("--w-gnd", 2.2)}


def wire_style(wid: str) -> Tuple[str, float]:
    return OVERRIDE.get(wid, KIND[wid[0]])


# ---------------------------------------------------------------------------
# 그림 엔진
# ---------------------------------------------------------------------------
def text_width(s: str, size: float, mono: bool = False, bold: bool = False) -> float:
    """글자 폭 어림 (겹침 검사용, 실제보다 조금 넓게 잡는다)."""
    w = 0.0
    for ch in s:
        if ord(ch) >= 0x1100:      # 한글·전각·화살표
            w += 1.0
        elif ch == " ":
            w += 0.60 if mono else 0.32
        else:
            w += 0.60
    return w * size * (1.05 if bold else 1.0)


@dataclass
class Text:
    x: float; y: float; s: str; size: float = 12; anchor: str = "start"; color: str = "--ink"
    bold: bool = False; mono: bool = False

    def bbox(self) -> Tuple[float, float, float, float]:
        w = text_width(self.s, self.size, self.mono, self.bold)
        x0 = self.x if self.anchor == "start" else (self.x - w if self.anchor == "end" else self.x - w / 2)
        return (x0, self.y - 0.86 * self.size, x0 + w, self.y + 0.24 * self.size)


@dataclass
class Box:
    x: float; y: float; w: float; h: float; color: str = "--ink"; dashed: bool = False

    def rect(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)


@dataclass
class Wire:
    wid: str; pts: List[Tuple[float, float]]; color: str; width: float


@dataclass
class Diagram:
    w: int; h: int; name: str
    boxes: List[Box] = field(default_factory=list)
    texts: List[Text] = field(default_factory=list)
    wires: List[Wire] = field(default_factory=list)
    ports: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    extra: List[str] = field(default_factory=list)      # 검사 대상이 아닌 장식 (핀 네모 등)

    # -- 그리기 --------------------------------------------------------------
    def box(self, x, y, w, h, color="--ink", dashed=False) -> Box:
        b = Box(x, y, w, h, color, dashed); self.boxes.append(b); return b

    def t(self, x, y, s, size=12, anchor="start", color="--ink", bold=False, mono=False) -> None:
        self.texts.append(Text(x, y, s, size, anchor, color, bold, mono))

    def port(self, name: str, x: float, y: float) -> None:
        assert name not in self.ports, name
        self.ports[name] = (x, y)

    def pin(self, x: float, y: float, color: str = "--ink", filled: bool = False, r: float = 5) -> None:
        fill = f"var({color})" if filled else "var(--paper)"
        self.extra.append(f'<rect x="{x - r:g}" y="{y - r:g}" width="{2 * r:g}" height="{2 * r:g}" '
                          f'fill="{fill}" stroke="var({color})" stroke-width="1.2"/>')

    def wire(self, wid: str, via: Optional[List[Tuple[float, float]]] = None) -> None:
        """NET 의 선 wid 를 그린다. 양 끝은 NET 에 적힌 단자의 좌표이고 via 는 중간 꺾임점."""
        row = next(r for r in NET if r[0] == wid)
        a, b = self.ports[row[1]], self.ports[row[2]]
        pts = [a] + list(via or []) + [b]
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            assert x1 == x2 or y1 == y2, f"{self.name} {wid}: 직각이 아닌 선분 {(x1, y1)} → {(x2, y2)}"
        color, width = wire_style(wid)
        self.wires.append(Wire(wid, pts, color, width))

    # -- 검사 ----------------------------------------------------------------
    def lint(self) -> List[str]:
        errs: List[str] = []
        pad = 1.5
        tb = [(t, t.bbox()) for t in self.texts]
        for i, (t1, b1) in enumerate(tb):
            if b1[0] < 0 or b1[1] < 0 or b1[2] > self.w or b1[3] > self.h:
                errs.append(f"글자가 그림 밖으로 나감: {t1.s!r}")
            for t2, b2 in tb[i + 1:]:
                if b1[0] < b2[2] + pad and b2[0] < b1[2] + pad and b1[1] < b2[3] and b2[1] < b1[3]:
                    errs.append(f"글자 겹침: {t1.s!r} / {t2.s!r}")
        segs = [(wr.wid, p, q) for wr in self.wires for p, q in zip(wr.pts, wr.pts[1:])]

        def hits(seg, rect, m=0.0) -> bool:
            (x1, y1), (x2, y2) = seg
            rx0, ry0, rx1, ry1 = rect[0] - m, rect[1] - m, rect[2] + m, rect[3] + m
            return min(x1, x2) < rx1 and max(x1, x2) > rx0 and min(y1, y2) < ry1 and max(y1, y2) > ry0

        for wid, p, q in segs:
            for t, b in tb:
                if hits((p, q), b, 2.0):
                    errs.append(f"선 {wid} 이 글자를 지남: {t.s!r}")
            for bx in self.boxes:
                r = bx.rect()
                if hits((p, q), (r[0] + 1, r[1] + 1, r[2] - 1, r[3] - 1)):
                    errs.append(f"선 {wid} 이 상자 안을 지남: {r}")
        for i, (w1, p1, q1) in enumerate(segs):
            for w2, p2, q2 in segs[i + 1:]:
                if w1 == w2:
                    continue
                v1, v2 = p1[0] == q1[0], p2[0] == q2[0]
                if v1 == v2:                                    # 평행: 같은 직선 위에서 겹치면 오류
                    if v1 and abs(p1[0] - p2[0]) < 6:
                        if min(p1[1], q1[1]) < max(p2[1], q2[1]) and min(p2[1], q2[1]) < max(p1[1], q1[1]):
                            errs.append(f"선 겹침(세로): {w1} / {w2}")
                    if (not v1) and abs(p1[1] - p2[1]) < 6:
                        if min(p1[0], q1[0]) < max(p2[0], q2[0]) and min(p2[0], q2[0]) < max(p1[0], q1[0]):
                            errs.append(f"선 겹침(가로): {w1} / {w2}")
                else:
                    (vx, vy0), (_, vy1) = (p1, q1) if v1 else (p2, q2)
                    (hx0, hy), (hx1, _) = (p2, q2) if v1 else (p1, q1)
                    if min(hx0, hx1) < vx < max(hx0, hx1) and min(vy0, vy1) < hy < max(vy0, vy1):
                        errs.append(f"선 교차: {w1} × {w2}")
        for i, b1 in enumerate(self.boxes):
            for b2 in self.boxes[i + 1:]:
                r1, r2 = b1.rect(), b2.rect()
                if r1[0] < r2[2] and r2[0] < r1[2] and r1[1] < r2[3] and r2[1] < r1[3]:
                    errs.append(f"상자 겹침: {r1} / {r2}")
        return [f"[{self.name}] {e}" for e in errs]

    # -- 출력 ----------------------------------------------------------------
    def svg(self, label: str) -> str:
        o = [f'<svg viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{html.escape(label)}">']
        for b in self.boxes:
            dash = ' stroke-dasharray="6 4"' if b.dashed else ""
            o.append(f'<rect x="{b.x:g}" y="{b.y:g}" width="{b.w:g}" height="{b.h:g}" rx="5" fill="var(--paper)" '
                     f'stroke="var({b.color})" stroke-width="1.6"{dash}/>')
        o.extend(self.extra)
        for wr in self.wires:
            d = "M" + " L".join(f"{x:g} {y:g}" for x, y in wr.pts)
            o.append(f'<path d="{d}" fill="none" stroke="var({wr.color})" stroke-width="{wr.width:g}" '
                     f'stroke-linejoin="round" stroke-linecap="round"/>')
            for x, y in (wr.pts[0], wr.pts[-1]):
                o.append(f'<circle cx="{x:g}" cy="{y:g}" r="3.4" fill="var({wr.color})"/>')
        for t in self.texts:
            fam = "var(--font-mono)" if t.mono else "var(--font-body)"
            wt = ' font-weight="700"' if t.bold else ""
            o.append(f'<text x="{t.x:g}" y="{t.y:g}" font-size="{t.size:g}" text-anchor="{t.anchor}" '
                     f'fill="var({t.color})" font-family="{fam}"{wt}>{html.escape(t.s)}</text>')
        o.append("</svg>")
        return "\n".join(o)


# ---------------------------------------------------------------------------
# 그림 1: 전원 배선
# ---------------------------------------------------------------------------
def fig_power() -> Diagram:
    d = Diagram(1200, 600, "그림 1")
    wl = dict(size=11, mono=True, color="--muted")

    # 주행 배터리
    d.box(40, 240, 150, 130)
    d.t(115, 300, "주행 배터리", 14, "middle", bold=True)
    d.t(115, 320, "NiMH 7.2 V", 12, "middle", "--muted")
    d.t(115, 262, "+", 15, "middle", "--w-plus", bold=True)
    d.t(115, 360, "−", 15, "middle", bold=True)
    d.port("BAT+", 115, 240); d.port("BAT-", 115, 370)

    # 퓨즈, J2 (주전원 스위치는 없다. 배터리 커넥터를 뽑는 것이 전원 차단이다)
    d.box(260, 62, 100, 46)
    d.t(310, 82, "퓨즈 F1", 13, "middle", bold=True); d.t(310, 99, "10 A", 12, "middle", "--muted")
    d.port("F1.in", 260, 85); d.port("F1.out", 360, 85)
    d.box(480, 55, 420, 60, "--w-plus")
    d.t(690, 80, "J2 · + 분배", 14, "middle", "--w-plus", bold=True)
    d.t(690, 98, "WAGO 221 3구", 11, "middle", "--muted")
    d.port("J2.in", 480, 85); d.port("J2.a", 560, 115); d.port("J2.b", 810, 115)

    # J3
    d.box(480, 505, 420, 60)
    d.t(690, 530, "J3 · 공통 접지", 14, "middle", bold=True)
    d.t(690, 548, "WAGO 221 3구", 11, "middle", "--muted")
    d.port("J3.in", 480, 535); d.port("J3.a", 560, 505); d.port("J3.b", 810, 505)

    # MDD10A 와 모터 (단자 순서는 보드 인쇄대로 M1B, M1A, …, M2A, M2B)
    d.box(450, 215, 220, 190)
    d.t(580, 300, "MDD10A", 15, "middle", bold=True)
    d.t(580, 320, "모터 드라이버", 12, "middle", "--muted")
    d.t(560, 236, "B+", 12, "middle", mono=True); d.t(560, 396, "B−", 12, "middle", mono=True)
    for name, y in (("M1B", 255), ("M1A", 285), ("M2A", 335), ("M2B", 365)):
        d.t(459, y + 4, name, 12, mono=True); d.port("MDD." + name, 450, y)
    d.port("MDD.B+", 560, 215); d.port("MDD.B-", 560, 405)
    d.box(235, 235, 140, 70)
    d.t(305, 266, "왼쪽 모터", 13, "middle", bold=True); d.t(305, 284, "전력선 2가닥", 11, "middle", "--muted")
    d.port("ML.1", 375, 255); d.port("ML.2", 375, 285)
    d.box(235, 315, 140, 70)
    d.t(305, 346, "오른쪽 모터", 13, "middle", bold=True); d.t(305, 364, "전력선 2가닥", 11, "middle", "--muted")
    d.port("MR.1", 375, 335); d.port("MR.2", 375, 365)

    # UBEC 와 서보용 보드 단자대
    d.box(730, 250, 160, 120)
    d.t(810, 305, "UBEC", 15, "middle", bold=True); d.t(810, 324, "6 V 설정", 12, "middle", "--muted")
    d.t(810, 270, "IN+", 12, "middle", mono=True); d.t(810, 362, "IN−", 12, "middle", mono=True)
    d.t(882, 284, "OUT+", 11, "end", mono=True); d.t(882, 346, "OUT−", 11, "end", mono=True)
    d.port("UBEC.IN+", 810, 250); d.port("UBEC.IN-", 810, 370)
    d.port("UBEC.OUT+", 890, 292); d.port("UBEC.OUT-", 890, 332)
    d.box(1000, 235, 170, 150, "--w-sig")
    d.t(1085, 262, "서보용 PCA9685", 13, "middle", "--w-sig", bold=True)
    d.t(1085, 372, "단자대 (그림 2 로 이어짐)", 10.5, "middle", "--muted")
    d.t(1010, 296, "V+", 12, mono=True); d.t(1010, 336, "GND", 12, mono=True)
    d.port("PCA1.TV+", 1000, 292); d.port("PCA1.TGND", 1000, 332)

    # 선
    d.wire("P1", [(115, 85)]); d.t(124, 165, "P1", **wl)
    d.wire("P2"); d.t(420, 77, "P2", anchor="middle", **wl)
    d.wire("P3"); d.t(569, 170, "P3", **wl)
    d.wire("P4"); d.t(819, 185, "P4", **wl)
    d.wire("G1", [(115, 535)]); d.t(124, 455, "G1", **wl)
    d.wire("G2"); d.t(569, 460, "G2", **wl)
    d.wire("G3"); d.t(819, 440, "G3", **wl)
    d.wire("P5"); d.t(945, 283, "P5", anchor="middle", **wl)
    d.wire("G4"); d.t(945, 350, "G4", anchor="middle", **wl)
    for wid, y in (("M1", 255), ("M2", 285), ("M3", 335), ("M4", 365)):
        d.wire(wid); d.t(412, y - 7, wid, anchor="middle", **wl)

    # Jetson 전원 (차체 전원과 분리)
    d.t(930, 428, "Jetson 전원 (차체 전원과 분리)", 12, color="--muted", bold=True)
    d.box(930, 446, 90, 44, "--muted"); d.t(975, 473, "보조배터리", 12, "middle")
    d.box(1118, 446, 62, 44, "--muted"); d.t(1149, 473, "Jetson", 12, "middle")
    d.extra.append('<path d="M1020 468 L1118 468" stroke="var(--muted)" stroke-width="2" fill="none"/>')
    d.t(1069, 460, "USB-C PD", 10.5, "middle", "--muted")
    return d


# ---------------------------------------------------------------------------
# 그림 2: 신호 배선
# ---------------------------------------------------------------------------
def fig_signal() -> Diagram:
    d = Diagram(1200, 580, "그림 2")
    wl = dict(size=11, mono=True, color="--muted")
    ys = {"VCC": 200, "SDA": 240, "SCL": 280, "GND": 320}

    # Jetson
    d.box(40, 140, 220, 220)
    d.t(52, 166, "Jetson Orin Nano", 14, bold=True)
    for (pin, name), y in zip((("1", "3.3 V"), ("3", "SDA"), ("5", "SCL"), ("6", "GND")), ys.values()):
        d.t(250, y + 4, f"핀 {pin} · {name}", 12, "end", mono=True); d.port("JET." + pin, 260, y)
    for pin, x in (("2", 70), ("20", 110), ("19", 150), ("23", 190), ("24", 230)):
        d.t(x, 350, f"핀 {pin}", 10, "middle", mono=True); d.port("JET." + pin, x, 360)

    # 도트 매트릭스 (선택)
    d.box(40, 440, 220, 80, "--muted")
    d.t(150, 492, "도트 매트릭스 MAX7219", 13, "middle", bold=True)
    d.t(150, 510, "상태 표시 (선택). 없어도 주행에는 지장 없음", 10, "middle", "--muted")
    for n, x in (("VCC", 70), ("GND", 110), ("DIN", 150), ("CLK", 190), ("CS", 230)):
        d.t(x, 460, n, 10, "middle", mono=True); d.port("DM." + n, x, 440)

    # USB 카메라
    d.box(290, 440, 110, 56, "--muted"); d.t(345, 473, "USB 카메라", 12, "middle")
    d.extra.append('<path d="M260 345 L345 345 L345 440" stroke="var(--muted)" stroke-width="2" fill="none"/>')
    d.t(300, 338, "USB", 10, color="--muted", mono=True)

    # 서보용 보드
    d.box(410, 140, 280, 220, "--w-sig")
    d.t(550, 236, "서보용 PCA9685", 15, "middle", "--w-sig", bold=True)
    d.t(550, 256, "주소 0x40 · 50 Hz", 12, "middle", "--muted")
    d.t(550, 286, "6핀 헤더의 OE, V+ 는 비움", 11, "middle", "--muted")
    for n, y in ys.items():
        d.t(419, y + 4, n, 12, mono=True); d.port("PCA1." + n, 410, y)
        d.t(681, y + 4, n, 12, "end", mono=True); d.port(f"PCA1.{n}2", 690, y)
    d.t(550, 160, "단자대", 11, "middle", "--muted")
    d.t(500, 160, "V+", 12, "middle", mono=True); d.t(600, 160, "GND", 12, "middle", mono=True)
    d.port("PCA1.TV+", 500, 140); d.port("PCA1.TGND", 600, 140)
    d.t(550, 328, "채널 0", 11, "middle", "--muted")
    for n, x, lab in (("CH0P", 480, "PWM"), ("CH0V", 550, "V+"), ("CH0G", 620, "GND")):
        d.t(x, 350, lab, 12, "middle", mono=True); d.port("PCA1." + n, x, 360)

    # UBEC (그림 1 에서 옴)
    d.box(470, 28, 160, 44)
    d.t(550, 47, "UBEC 6 V 출력", 13, "middle", bold=True); d.t(550, 63, "그림 1 에서 옴", 10.5, "middle", "--muted")
    d.port("UBEC.OUT+", 500, 72); d.port("UBEC.OUT-", 600, 72)
    d.t(510, 90, "+", 13, color="--w-servo", bold=True); d.t(610, 90, "−", 13, bold=True)

    # 서보
    d.box(440, 450, 220, 96)
    d.t(550, 506, "조향 서보", 14, "middle", bold=True); d.t(550, 526, "LD-1501MG", 12, "middle", "--muted")
    for n, x, lab in (("SIG", 480, "흰"), ("V+", 550, "빨"), ("GND", 620, "검")):
        d.t(x, 471, lab, 12, "middle"); d.port("SV." + n, x, 450)

    # 모터용 보드
    d.box(840, 140, 300, 220, "--w-sig")
    d.t(990, 236, "모터용 PCA9685", 15, "middle", "--w-sig", bold=True)
    d.t(990, 256, "주소 0x41 · 1.5 kHz", 12, "middle", "--muted")
    d.t(990, 286, "단자대, 채널 V+ 줄, OE 는 비움", 11, "middle", "--muted")
    for n, y in ys.items():
        d.t(849, y + 4, n, 12, mono=True); d.port("PCA2." + n, 840, y)
    d.t(990, 316, "채널 2~15 는 비움", 10.5, "middle", "--muted")
    cols = (("CH0P", 895, "채널0", "PWM"), ("CH1P", 1005, "채널1", "PWM"), ("CH0G", 1115, "채널0", "GND"))
    for n, x, top, bot in cols:
        d.t(x, 336, top, 10.5, "middle", "--muted"); d.t(x, 351, bot, 11, "middle", mono=True)
        d.port("PCA2." + n, x, 360)

    # 분기 블록 J4 (속도), J5 (방향): WAGO 221 3구. 1번에 보드에서 온 선, 2·3번에 MDD10A 로 가는 선
    for name, x0, cx in (("J4", 845, 895), ("J5", 955, 1005)):
        d.box(x0, 405, 100, 40, "--w-sig")
        d.t(cx, 429, f"{name} · WAGO 3구", 10.5, "middle", "--w-sig", bold=True)
        d.port(f"{name}.in", cx, 405); d.port(f"{name}.a", cx - 30, 445); d.port(f"{name}.b", cx + 30, 445)

    # MDD10A 제어 헤더 (그림의 핀 순서는 선이 엇갈리지 않게 놓은 것. 실제 핀 위치는 그림 3)
    d.box(840, 490, 300, 80)
    d.t(990, 540, "MDD10A 제어 헤더", 13, "middle", bold=True)
    d.t(990, 558, "실제 핀 위치는 그림 3", 10.5, "middle", "--muted")
    for n, x in (("PWM1", 865), ("PWM2", 925), ("DIR1", 975), ("DIR2", 1035), ("GND", 1115)):
        d.t(x, 511, n, 11, "middle", mono=True); d.port("MDDC." + n, x, 490)

    # 선
    for wid, n in (("I1", "VCC"), ("I2", "SDA"), ("I3", "SCL"), ("I4", "GND")):
        d.wire(wid); d.t(335, ys[n] - 7, wid, anchor="middle", **wl)
    for wid, n in (("C1", "VCC"), ("C2", "SDA"), ("C3", "SCL"), ("C4", "GND")):
        d.wire(wid); d.t(765, ys[n] - 7, wid, anchor="middle", **wl)
    d.wire("P5"); d.t(491, 112, "P5", anchor="end", **wl)
    d.wire("G4"); d.t(591, 112, "G4", anchor="end", **wl)
    d.wire("S1"); d.t(471, 410, "S1", anchor="end", **wl)
    d.wire("S2"); d.t(541, 410, "S2", anchor="end", **wl)
    d.wire("S3"); d.t(629, 410, "S3", **wl)
    d.wire("D1"); d.t(886, 388, "D1", anchor="end", **wl)
    d.wire("D4"); d.t(996, 388, "D4", anchor="end", **wl)
    for wid, x in (("D2", 865), ("D3", 925), ("D5", 975), ("D6", 1035)):
        d.wire(wid); d.t(x - 8, 472, wid, anchor="end", **wl)
    d.wire("D7"); d.t(1124, 430, "D7", **wl)
    for wid, x in (("X1", 70), ("X2", 110), ("X3", 150), ("X4", 190), ("X5", 230)):
        d.wire(wid); d.t(x - 5, 405, wid, anchor="end", **wl)
    return d


# ---------------------------------------------------------------------------
# 그림 3: 핀 위치 (선 없음)
# ---------------------------------------------------------------------------
def _pca_board(d: Diagram, ox: float, oy: float, servo: bool) -> None:
    """PCA9685 보드 한 장. 쓰는 핀만 색을 칠하고, 비우는 핀은 테두리만 그린다."""
    mu = "--muted"
    d.t(ox, oy - 10, "서보용 보드 (주소 0x40)" if servo else "모터용 보드 (주소 0x41)", 14, bold=True)
    d.box(ox, oy, 570, 290, "--w-sig")
    tc = "--w-servo" if servo else mu
    dash = "" if servo else ' stroke-dasharray="5 4"'
    d.extra.append(f'<rect x="{ox + 235:g}" y="{oy + 10:g}" width="100" height="38" rx="3" fill="var(--paper)" '
                   f'stroke="var({tc})" stroke-width="1.6"{dash}/>')
    d.t(ox + 262, oy + 34, "V+", 12, "middle", "--ink" if servo else mu, mono=True)
    d.t(ox + 308, oy + 34, "GND", 12, "middle", "--ink" if servo else mu, mono=True)
    d.t(ox + 285, oy + 63, "단자대: UBEC 6 V (P5, G4)" if servo else "단자대: 비움", 11, "middle", tc)
    left = "Jetson 에서 (I1~I4)" if servo else "서보용 보드에서 (C1~C4)"
    right = "모터용 보드로 (C1~C4)" if servo else "비움"
    d.t(ox + 8, oy + 68, left, 10, color="--w-sig")
    d.t(ox + 562, oy + 68, right, 10, "end", "--w-sig" if servo else mu)
    for k, n in enumerate(("GND", "OE", "SCL", "SDA", "VCC", "V+")):
        y = oy + 86 + 24 * k
        used = n in ("GND", "SCL", "SDA", "VCC")
        d.pin(ox + 16, y, "--w-sig" if used else mu, used)
        d.t(ox + 30, y + 4, n, 11.5, mono=True, color="--ink" if used else mu)
        ru = used and servo
        d.pin(ox + 554, y, "--w-sig" if ru else mu, ru)
        d.t(ox + 540, y + 4, n, 11.5, "end", mono=True, color="--ink" if ru else mu)
    d.t(ox + 405, oy + 96, "주소 점퍼", 10.5, "middle", mu)
    for k, n in enumerate(("A5", "A4", "A3", "A2", "A1", "A0")):
        x = ox + 340 + 26 * k
        closed = (n == "A0") and not servo
        col = "--w-motor" if closed else mu
        fill = "var(--w-motor)" if closed else "var(--paper)"
        d.extra.append(f'<rect x="{x - 7:g}" y="{oy + 102:g}" width="14" height="8" rx="1.5" fill="{fill}" stroke="var({col})"/>')
        d.t(x, oy + 124, n, 9.5, "middle", col, mono=True, bold=closed)
    d.t(ox + 405, oy + 142, "모두 열림" if servo else "A0 만 납땜 (맨 오른쪽, 그림 4)", 10.5, "middle", mu if servo else "--w-motor")
    for k in range(16):
        x = ox + 66 + 30 * k
        pwm = (k == 0) if servo else (k < 2)
        d.pin(x, oy + 222, "--w-sig" if pwm else mu, pwm, r=4.5)
        vp = servo and k == 0
        d.pin(x, oy + 242, "--w-servo" if vp else mu, vp, r=4.5)
        gn = k == 0
        d.pin(x, oy + 262, "--w-gnd" if gn else mu, gn, r=4.5)
        hot = pwm or gn
        d.t(x, oy + 282, str(k), 9.5, "middle", "--ink" if hot else mu, mono=True, bold=hot)
    d.t(ox + 530, oy + 226, "PWM", 10.5, mono=True); d.t(ox + 530, oy + 246, "V+", 10.5, mono=True)
    d.t(ox + 530, oy + 266, "GND", 10.5, mono=True)
    if servo:
        d.t(ox, oy + 310, "쓰는 핀: 채널 0 의 PWM, V+, GND. 서보 커넥터(S1~S3)를 그대로 꽂는다.", 11.5)
    else:
        d.t(ox, oy + 310, "쓰는 핀: 채널 0 의 PWM (D1), 채널 1 의 PWM (D4), 채널 0 의 GND (D7).", 11.5)


def fig_pins() -> Diagram:
    d = Diagram(1200, 640, "그림 3")
    mu = "--muted"
    _pca_board(d, 20, 50, servo=True)
    _pca_board(d, 610, 50, servo=False)

    # MDD10A
    d.t(20, 418, "MDD10A (인쇄된 이름이 기준)", 14, bold=True)
    d.box(20, 428, 520, 196)
    d.t(520, 446, "제어 헤더", 10, "end", mu)
    for k, (n, wid) in enumerate((("DIR1", "D5"), ("PWM1", "D2"), ("DIR2", "D6"), ("PWM2", "D3"), ("GND", "D7"))):
        y = 462 + 26 * k
        d.pin(510, y, "--w-gnd" if n == "GND" else "--w-sig", True)
        d.t(494, y + 4, f"{n}  {wid}", 11.5, "end", mono=True)
    d.t(520, 592, "D2, D3 은 J4 에서,", 10, "end", mu)
    d.t(520, 606, "D5, D6 은 J5 에서 온다", 10, "end", mu)
    d.t(220, 468, "시험 버튼 4개", 13, "middle", bold=True)
    d.t(220, 486, "M1A · M1B · M2A · M2B", 11, "middle", mu, mono=True)
    d.t(220, 504, "누르면 신호 없이 모터가 돈다", 11, "middle", mu)
    d.t(36, 540, "전력 단자대", 10, color=mu)
    for k, (n, wid) in enumerate((("M1B", "M1"), ("M1A", "M2"), ("B+", "P3"), ("B−", "G2"), ("M2A", "M3"), ("M2B", "M4"))):
        x = 60 + 62 * k
        col = "--w-plus" if n == "B+" else ("--w-gnd" if n == "B−" else "--w-motor")
        d.pin(x, 600, col, True, r=6)
        d.t(x, 583, n, 12, "middle", mono=True); d.t(x, 566, wid, 10, "middle", mu, mono=True)

    # 읽는 법
    d.t(600, 440, "읽는 법", 13, bold=True)
    notes = (
        "색을 칠한 핀만 쓴다. 테두리만 있는 핀은 비운다.",
        "핀 옆의 번호(I1, D2 …)는 그 핀에 꽂는 전선의 번호다.",
        "채널 0 은 단자대를 위로 두었을 때 맨 왼쪽 세로 3핀이다.",
        "세로 3핀은 위가 PWM, 가운데가 V+, 아래(보드 가장자리 쪽)가 GND.",
        "6핀 헤더는 양쪽이 같은 신호다. 위에서부터 GND, OE, SCL, SDA, VCC, V+.",
        "주소 점퍼는 A5 … A0 순서로 인쇄되어 A0 가 맨 오른쪽이다.",
        "MDD10A 의 1번 채널 단자는 B, A 순서다 (M1B, M1A).",
        "단자대의 V+ 와 GND 좌우는 보드 인쇄를 보고 확인한다.",
    )
    for k, line in enumerate(notes):
        d.t(600, 464 + 22 * k, line, 11.5, color=mu if k else "--ink")
    return d


# ---------------------------------------------------------------------------
# 그림 4: 모터용 보드의 주소 납땜 (확대)
# ---------------------------------------------------------------------------
def fig_solder() -> Diagram:
    d = Diagram(820, 530, "그림 4")        # 세로 쪽에 넣는 그림이라 폭을 줄였다
    mu = "--muted"
    # 보드 위쪽 가장자리 확대
    d.t(20, 30, "모터용 보드의 위쪽 가장자리를 확대한 모습 (단자대를 위로 두고 본다)", 12, color=mu)
    d.extra.append('<rect x="20" y="45" width="780" height="190" rx="5" fill="var(--paper)" stroke="var(--w-sig)" '
                   'stroke-width="1.6" stroke-dasharray="6 4"/>')      # 보드 가장자리 (장식)
    d.box(40, 62, 130, 60, mu)
    d.t(105, 88, "단자대", 12, "middle", bold=True); d.t(105, 108, "V+  GND", 11, "middle", mu, mono=True)
    d.t(475, 70, "주소 패드 (인쇄 순서대로 A5 … A0)", 11, "middle", mu)
    for k, n in enumerate(("A5", "A4", "A3", "A2", "A1", "A0")):
        x = 300 + 70 * k
        closed = n == "A0"
        col = "--w-motor" if closed else mu
        for dx in (-19, 3):
            d.extra.append(f'<rect x="{x + dx:g}" y="88" width="16" height="24" rx="2" fill="var(--paper)" '
                           f'stroke="var({col})" stroke-width="1.4"/>')
        if closed:
            d.extra.append(f'<ellipse cx="{x:g}" cy="100" rx="15" ry="9" fill="var(--w-motor)" opacity="0.9"/>')
        d.t(x, 134, n, 11.5, "middle", col, mono=True, bold=closed)
    d.t(676, 104, "← 납", 11, color="--w-motor", bold=True)
    d.t(40, 192, "A0 의 두 패드 사이를 납으로 이어 한 덩어리로 만든다. 주소가 0x40 에서 0x41 이 된다.", 11.5)
    d.t(40, 216, "A1~A5 는 그대로 둔다. 서보용 보드는 여섯 개 모두 그대로 둔다 (0x40).", 11.5, color=mu)

    # 순서
    d.t(20, 272, "순서", 13, bold=True)
    steps = (
        "1. 보드를 전원과 케이블에서 모두 뺀 상태에서 한다.",
        "2. 인쇄된 글자 A0 를 찾는다 (위치가 아니라 글자가 기준이다. 호환 보드는 자리가 다를 수 있다).",
        "3. A0 의 두 패드에 납을 조금 올려 사이를 잇는다. 실물은 패드 간격이 1~2 mm 라 옆 패드 A1 로 번지기 쉽다.",
        "4. 식은 뒤 멀티미터 도통 모드로 A0 두 패드가 이어졌는지 본다.",
        "5. 보드에 '모터 0x41' 이라고 써 붙인다. 두 보드는 이 납땜 말고는 똑같다.",
        "6. 배선을 끝낸 뒤 4절 1번(pin_test.py scan)으로 주소를 확인한다.",
        "40 과 41 이 보여야 한다. 70 은 모든 PCA9685 가 함께 듣는 주소라 같이 보여도 정상이다.",
        "",
        "주소 = 0x40 + 닫은 패드의 값. A0 = 1, A1 = 2, A2 = 4, A3 = 8, A4 = 16, A5 = 32.",
        "A0 만 닫으면 0x41. 41 대신 다른 번호가 보이면 납이 옆 패드까지 번진 것이다.",
    )
    for k, line in enumerate(steps):
        if line:
            d.t(38 if k == 6 else 20, 300 + 22 * k, line, 11.5, color="--ink" if k < 7 else mu)
    return d


def all_figures() -> List[Diagram]:
    return [fig_power(), fig_signal(), fig_pins(), fig_solder()]


def check_coverage(figs: List[Diagram]) -> List[str]:
    """NET 의 모든 선이 그림에 한 번 이상 있고, 단자마다 전선이 하나뿐인지."""
    errs = []
    drawn = {w.wid for f in figs for w in f.wires}
    for wid, a, b, *_ in NET:
        if wid not in drawn:
            errs.append(f"그림에 없는 선: {wid}")
        for p in (a, b):
            if p not in PORT_LABEL:
                errs.append(f"이름표가 없는 단자: {p}")
    ids = [r[0] for r in NET]
    if len(ids) != len(set(ids)):
        errs.append("선 ID 중복")
    if sorted(POWER_IDS + SIGNAL_IDS + DISPLAY_IDS) != sorted(ids):
        errs.append("표에 빠지거나 두 번 들어간 선이 있음")
    ends = [p for r in NET for p in (r[1], r[2])]
    dup = sorted({p for p in ends if ends.count(p) > 1})
    if dup:
        errs.append(f"한 단자에 전선이 두 개 이상: {dup}")
    return errs


# ---------------------------------------------------------------------------
# 문서
# ---------------------------------------------------------------------------
CSS = """
:root {
  --bg: #f6f5f1; --paper: #ffffff; --ink: #1f2430; --muted: #5c6370; --rule: #d8d5cc; --accent: #1d5c8f; --warn: #b3261e;
  --w-plus: #c0392b; --w-gnd: #1f2430; --w-servo: #d35400; --w-motor: #a86a12; --w-sig: #1a7f4b; --w-vcc: #7d3c98;
  --font-body: "Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans KR", system-ui, sans-serif;
  --font-mono: Consolas, "D2Coding", ui-monospace, "Malgun Gothic", monospace;
}
@media screen and (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #15181e; --paper: #1d2129; --ink: #e6e4dd; --muted: #a2a7b1; --rule: #353b46; --accent: #7fb3e0; --warn: #ff7b72;
    --w-plus: #ff7b72; --w-gnd: #e6e4dd; --w-servo: #f29a4f; --w-motor: #e3b341; --w-sig: #5ccc8f; --w-vcc: #c39bd3;
  }
}
@media screen {
  :root[data-theme="dark"] {
    color-scheme: dark;
    --bg: #15181e; --paper: #1d2129; --ink: #e6e4dd; --muted: #a2a7b1; --rule: #353b46; --accent: #7fb3e0; --warn: #ff7b72;
    --w-plus: #ff7b72; --w-gnd: #e6e4dd; --w-servo: #f29a4f; --w-motor: #e3b341; --w-sig: #5ccc8f; --w-vcc: #c39bd3;
  }
}
* { box-sizing: border-box; }
html, body { margin: 0; }
body { background: var(--bg); color: var(--ink); font-family: var(--font-body); font-size: 15px; line-height: 1.6;
       word-break: keep-all; overflow-wrap: break-word; }
main { max-width: 1080px; margin: 0 auto; padding: 28px 16px 60px; }
h1 { font-size: 26px; margin: 0 0 4px; text-wrap: balance; }
.meta { color: var(--muted); font-size: 13px; border-bottom: 2px solid var(--ink); padding-bottom: 10px; margin-bottom: 18px; }
h2 { font-size: 19px; margin: 34px 0 8px; color: var(--accent); text-wrap: balance; }
h3 { font-size: 15px; margin: 18px 0 6px; }
p { margin: 8px 0; max-width: 76ch; }
ul, ol { margin: 8px 0; padding-left: 22px; max-width: 80ch; } li { margin: 4px 0; }
table { border-collapse: collapse; width: 100%; font-size: 14px; margin: 10px 0 14px; }
th, td { text-align: left; padding: 6px 9px; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { color: var(--muted); font-weight: 600; font-size: 13px; border-bottom: 1.5px solid var(--ink); white-space: nowrap; }
td.id { font-family: var(--font-mono); font-weight: 700; white-space: nowrap; }
td.k { font-weight: 600; }
.parts td.k { white-space: nowrap; }
code { font-family: var(--font-mono); font-size: 0.93em; }
pre { font-family: var(--font-mono); font-size: 13px; background: var(--paper); border: 1px solid var(--rule); border-radius: 6px;
      padding: 10px 12px; overflow-x: auto; margin: 8px 0; }
.note { border-left: 3px solid var(--w-motor); background: var(--paper); padding: 8px 12px; margin: 12px 0; max-width: 80ch; }
.danger { border-left-color: var(--warn); }
.scroll { overflow-x: auto; background: var(--paper); border: 1px solid var(--rule); border-radius: 6px; padding: 10px; }
.scroll svg { display: block; min-width: 860px; width: 100%; height: auto; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 13px; color: var(--muted); margin: 6px 0; }
.legend span::before { content: ""; display: inline-block; width: 24px; height: 0; border-top: 4px solid currentColor;
                       margin-right: 6px; vertical-align: middle; }
.l-plus { color: var(--w-plus); } .l-gnd { color: var(--w-gnd); } .l-servo { color: var(--w-servo); }
.l-motor { color: var(--w-motor); } .l-sig { color: var(--w-sig); } .l-vcc { color: var(--w-vcc); }
.cap { color: var(--muted); font-size: 13px; margin: 6px 0 0; max-width: 90ch; }
.tablewrap { overflow-x: auto; }

@page { size: A4 portrait; margin: 15mm 15mm; }
@page wide { size: A4 landscape; margin: 11mm 12mm; }
@media print {
  :root, :root[data-theme="dark"] {
    --bg: #ffffff; --paper: #ffffff; --ink: #1f2430; --muted: #5c6370; --rule: #cfccc3; --accent: #1d5c8f; --warn: #b3261e;
    --w-plus: #c0392b; --w-gnd: #1f2430; --w-servo: #d35400; --w-motor: #a86a12; --w-sig: #1a7f4b; --w-vcc: #7d3c98;
  }
  body { font-size: 10.5pt; line-height: 1.5; }
  main { max-width: none; padding: 0; }
  h1 { font-size: 19pt; } h2 { font-size: 13pt; margin: 6mm 0 2mm; break-after: avoid; } h3 { font-size: 11pt; break-after: avoid; }
  .meta { font-size: 9.5pt; }
  table { font-size: 9.2pt; } th { font-size: 8.8pt; } th, td { padding: 1.15mm 2mm; }
  tr, .note, pre { break-inside: avoid; }
  .wires th, .wires td { padding: 0.65mm 2mm; }
  .wires table { margin: 1mm 0 1.5mm; }
  h3 { margin: 4mm 0 1mm; }
  pre { font-size: 9pt; }
  .wide { page: wide; break-before: page; break-after: page; }
  .wide h2 { margin-top: 0; }
  .scroll { border: none; padding: 0; overflow: visible; }
  .scroll svg { min-width: 0; width: 100%; height: auto; max-height: 140mm; margin: 0 auto; }
  .pagebreak { break-before: page; }
}
"""


def rows(ids: List[str]) -> str:
    out = []
    for wid in ids:
        _, a, b, gauge, role = next(r for r in NET if r[0] == wid)
        out.append(f'<tr><td class="id">{wid}</td><td>{html.escape(PORT_LABEL[a])}</td>'
                   f'<td>{html.escape(PORT_LABEL[b])}</td><td>{html.escape(gauge)}</td><td>{html.escape(role)}</td></tr>')
    return "\n".join(out)


def wire_table(ids: List[str]) -> str:
    return ('<div class="tablewrap wires"><table><thead><tr><th>번호</th><th>출발</th><th>도착</th><th>선</th><th>역할</th></tr></thead>'
            f"<tbody>\n{rows(ids)}\n</tbody></table></div>")


def build_html(figs: List[Diagram]) -> str:
    power, signal, pins, solder = figs
    n = len(WIRE_IDS)
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>자율주행자동차 배선도</title>
<style>{CSS}</style>
</head>
<body>
<main>

<h1>자율주행자동차 배선도</h1>
<div class="meta">Version 1.0</div>

<h2>1. 구성</h2>
<div class="tablewrap parts"><table>
<thead><tr><th>부품</th><th>하는 일</th><th>설정과 주의</th></tr></thead>
<tbody>
<tr><td class="k">Jetson Orin Nano</td><td>카메라 영상을 처리하고 조향·속도 명령을 낸다</td><td>헤더는 핀 1, 3, 5, 6 네 개만 쓴다. 전원은 보조배터리</td></tr>
<tr><td class="k">서보용 PCA9685</td><td>조향 서보 신호를 만든다 (50 Hz)</td><td>주소 0x40. 단자대에 UBEC 6 V 를 넣는다</td></tr>
<tr><td class="k">모터용 PCA9685</td><td>모터 속도·방향 신호를 만든다 (1.5 kHz)</td><td>주소 0x41 (A0 패드 납땜, 그림 4). 채널 0 과 1 만 쓰고 단자대는 비운다</td></tr>
<tr><td class="k">MDD10A</td><td>모터 두 개를 구동한다</td><td>왼쪽 모터는 1번 채널(M1A·M1B), 오른쪽 모터는 2번 채널(M2A·M2B). 두 채널은 J4, J5 에서 나뉜 같은 속도·방향 신호를 받는다</td></tr>
<tr><td class="k">UBEC</td><td>주행 배터리 7.2 V 를 서보 전원 6 V 로 바꾼다</td><td>6 V 로 설정 (점퍼). 출력 + 는 단자대 V+, − 는 GND 에</td></tr>
<tr><td class="k">조향 서보 LD-1501MG</td><td>앞바퀴 조향</td><td>검정 GND, 빨강 6 V, 흰색 신호</td></tr>
<tr><td class="k">퓨즈 F1</td><td>과전류 차단</td><td>10 A. 주전원 스위치는 없다. 전원을 끊을 때는 주행 배터리 커넥터를 뽑는다</td></tr>
<tr><td class="k">J2, J3</td><td>+ 분배, 공통 접지</td><td>WAGO 221 3구. 굵은 전원선만 물린다</td></tr>
<tr><td class="k">도트 매트릭스 MAX7219</td><td>주행 상태 표시 (선택)</td><td>Jetson SPI 에 직결, 5 V. 없어도 주행에는 지장 없다 (<code>display.enabled: false</code>)</td></tr>
<tr><td class="k">J4, J5</td><td>모터 속도 신호(J4)와 방향 신호(J5)를 MDD10A 의 두 채널로 나눈다</td><td>WAGO 221 3구. 1번에 보드에서 온 선, 2·3번에 MDD10A 로 가는 선. 22AWG 또는 더 굵은 선만 물린다</td></tr>
</tbody></table></div>
<p>Jetson 헤더는 출력 구동력이 약해 서보를 직접 움직이지 못했으므로 신호는 모두 PCA9685 에서 냅니다. PCA9685 는 보드 하나가 한 주파수만 내므로 서보용(50 Hz)과 모터용(1.5 kHz)을 따로 둡니다.</p>

<section class="wide">
<h2>그림 1. 전원 배선</h2>
<div class="legend"><span class="l-plus">배터리 + 7.2 V</span><span class="l-gnd">접지</span><span class="l-servo">서보 전원 6 V</span><span class="l-motor">모터 전력선</span></div>
<div class="scroll">{power.svg("전원 배선도. 주행 배터리의 + 는 퓨즈를 지나 J2 에서 MDD10A 와 UBEC 로 나뉜다. 배터리 − 는 J3 에서 MDD10A 와 UBEC 로 나뉜다. UBEC 출력 6 V 는 서보용 보드 단자대로 간다. 왼쪽 모터는 M1A·M1B 에, 오른쪽 모터는 M2A·M2B 에 물린다.")}</div>
<p class="cap">위가 +, 아래가 접지입니다. 배터리 전류는 퓨즈를 지나 J2 에서 모터 드라이버와 UBEC 로 나뉘고 J3 에서 모입니다. Jetson 은 보조배터리로 따로 움직입니다.</p>
</section>

<section class="wide">
<h2>그림 2. 신호 배선</h2>
<div class="legend"><span class="l-vcc">논리 전원 3.3 V</span><span class="l-plus">5 V (Jetson)</span><span class="l-sig">신호</span><span class="l-gnd">접지</span><span class="l-servo">서보 전원 6 V</span></div>
<div class="scroll">{signal.svg("신호 배선도. Jetson 핀 1, 3, 5, 6 이 서보용 보드의 VCC, SDA, SCL, GND 로 간다. 서보용 보드 반대쪽 헤더에서 같은 네 신호가 모터용 보드로 이어진다. 서보용 보드 채널 0 의 세 핀에 서보를 꽂는다. 모터용 보드 채널 0 의 PWM 핀은 J4 로, 채널 1 의 PWM 핀은 J5 로 가고, J4 에서 PWM1 과 PWM2 로, J5 에서 DIR1 과 DIR2 로 나뉜다. 채널 0 의 GND 핀이 MDD10A 제어 헤더 GND 로 간다. Jetson 아래쪽 핀 2, 20, 19, 23, 24 는 상태 표시용 도트 매트릭스의 VCC, GND, DIN, CLK, CS 로 간다.")}</div>
<p class="cap">Jetson 의 네 가닥이 서보용 보드를 거쳐 모터용 보드로 이어집니다. 속도는 J4, 방향은 J5 에서 MDD10A 의 두 채널로 나뉘어 두 모터가 항상 같은 신호를 받습니다. MDD10A 핀과 모터용 보드 GND 핀의 실제 위치는 그림 3 에 있습니다. 도트 매트릭스(X1~X5)는 Jetson 전원 영역 안에서만 이어지는 선택 부품입니다.</p>
</section>

<h2 class="pagebreak">2. 배선 목록</h2>
<p>번호는 그림의 선 번호와 같습니다. 선을 꽂을 때마다 표에서 지워 나가면 빠뜨리지 않습니다. WAGO 블록의 1~3번은 구분용 번호이고 구멍은 전기적으로 모두 같습니다.</p>
<h3>전원과 모터 ({len(POWER_IDS)}가닥)</h3>
{wire_table(POWER_IDS)}
<h3>신호 ({len(SIGNAL_IDS)}가닥)</h3>
{wire_table(SIGNAL_IDS)}
<h3>상태 표시 도트 매트릭스 ({len(DISPLAY_IDS)}가닥, 선택)</h3>
{wire_table(DISPLAY_IDS)}
<ul>
<li>I1~I4 와 C1~C4 는 각각 한 묶음의 케이블로 만듭니다. 6핀 헤더의 OE 와 V+ 는 비웁니다.</li>
<li>S1~S3 는 서보의 3핀 커넥터 그대로입니다. 검정이 보드 가장자리 쪽입니다.</li>
<li>D1~D7 은 한 다발로 묶어 모터 전력선과 떨어뜨립니다. <strong>WAGO 221 은 24AWG 보다 가는 선을 물지 못하므로</strong> 듀폰 점퍼선(26~28AWG) 대신 22AWG 선을 쓰고, 한쪽은 피복을 벗겨 WAGO 에, 다른 쪽은 듀폰 암 단자를 압착해 보드와 MDD10A 핀에 꽂습니다.</li>
<li>PCA9685 출력 하나가 MDD10A 입력 둘을 구동해도 입력 전류가 µA 수준이라 문제없습니다.</li>
<li>X1~X5 는 듀폰 암-암 점퍼선 그대로 꽂습니다. Jetson 핀 2(5 V)와 핀 1(3.3 V)을 헷갈리지 않게 봅니다.</li>
<li>USB 카메라는 Jetson 의 USB 포트에 꽂습니다. 표에는 넣지 않았습니다.</li>
</ul>

<h2>3. 접지</h2>
<p>서보 커넥터 안의 검정선(S3)을 빼면 접지선은 일곱 가닥입니다. 전류가 돌아오는 선 넷과 신호의 기준이 되는 가는 선 셋입니다.</p>
{wire_table(GROUND_IDS)}
<ul>
<li><strong>차체로 가는 Jetson 의 접지선은 I4 하나뿐입니다.</strong> (X2 는 Jetson 과 도트 매트릭스 사이의 선이라 차체와 무관합니다.) 그래서 모터나 서보 전류가 Jetson 을 지나갈 길이 없습니다. Jetson 은 플라스틱 기둥으로 고정합니다. 금속 기둥은 접지선을 하나 더 만듭니다.</li>
<li><strong>모터 신호의 기준 접지는 D7 입니다.</strong> 신호선과 같은 다발로 다녀야 모터 전류의 영향이 신호에 실리지 않습니다.</li>
<li><strong>굵은 접지선 G2, G3, G4 는 확실히 조입니다.</strong> 하나가 빠져도 전류가 가는 접지선(D7, C4)으로 돌아 흘러 겉으로는 동작하지만, 가는 선이 뜨거워지고 퓨즈는 이를 막지 못합니다.</li>
</ul>

<section class="wide">
<h2>그림 3. 핀 위치</h2>
<div class="scroll">{pins.svg("핀 위치. 서보용 보드는 한쪽 6핀 헤더의 GND, SCL, SDA, VCC 에 Jetson 케이블을, 반대쪽 같은 네 핀에 모터용 보드로 가는 케이블을 꽂고, 단자대에 UBEC 6 V 를 넣고, 채널 0 의 세 핀에 서보를 꽂는다. 모터용 보드는 한쪽 6핀 헤더의 네 핀만 쓰고 A0 를 납땜하며, 채널 0 과 1 의 PWM 핀과 채널 0 의 GND 핀을 쓴다. MDD10A 제어 헤더는 위에서부터 DIR1, PWM1, DIR2, PWM2, GND 이고, PWM1 과 PWM2 는 J4 에서, DIR1 과 DIR2 는 J5 에서 온다. 전력 단자대는 M1B, M1A, B+, B−, M2A, M2B 순서다.")}</div>
<p class="cap">두 보드는 같은 제품이고 A0 납땜 여부만 다르므로 모터용 보드에 표시를 해 둡니다. 호환 보드는 인쇄가 다를 수 있으니 보드에 적힌 이름을 기준으로 꽂습니다.</p>
</section>

<h2 class="pagebreak">그림 4. 모터용 보드의 주소 납땜</h2>
<p>두 보드는 기본 주소가 같은 0x40 이라 한 버스에서 함께 쓸 수 없습니다. 모터용 보드의 A0 패드를 납으로 이어 0x41 로 바꿉니다.</p>
<div class="scroll">{solder.svg("모터용 보드의 주소 납땜. 보드 위쪽 가장자리에 A5 부터 A0 까지 패드 여섯 쌍이 있다. 맨 오른쪽 A0 의 두 패드 사이만 납으로 잇는다. 그러면 주소가 0x40 에서 0x41 이 된다. 서보용 보드는 아무 패드도 잇지 않는다. 납땜 뒤 pin_test.py scan 으로 40 과 41 이 보이는지 확인한다.")}</div>
<p class="cap">이미 납땜된 보드를 받았다면 4절 1번의 scan 으로 확인만 하면 됩니다.</p>

<h2 class="pagebreak">4. 조립 후 동작 시험</h2>
<div class="note danger"><strong>PCA9685 는 통신이 끊겨도 마지막 출력을 유지합니다.</strong> I2C 선이 빠지거나 프로그램이 멈추면 모터는 그 속도로 계속 돌고, 세울 방법은 배터리 커넥터를 뽑는 것뿐입니다. 커넥터는 손이 바로 닿는 곳에 두고, 시험은 바퀴를 띄우고 합니다.</div>
<p>배선을 2절의 표대로 모두 꽂은 뒤, 바퀴를 띄우고 주행 배터리를 연결하고 Jetson 을 켭니다. 아래 명령을 순서대로 실행해 서보와 모터가 동작하는지 봅니다.</p>
<div class="tablewrap"><table>
<thead><tr><th>순서</th><th>명령</th><th>정상</th><th>정상이 아니면</th></tr></thead>
<tbody>
<tr><td class="id">1</td><td><code>uv run python scripts/pin_test.py scan</code></td><td>0x40 서보용 보드, 0x41 모터용 보드가 모두 "응답"</td><td>둘 다 없으면 Jetson 케이블(I1~I4), 0x41 만 없으면 보드 사이 케이블(C1~C4)과 A0 납땜(그림 4)</td></tr>
<tr><td class="id">2</td><td><code>uv run python scripts/pin_test.py servo</code></td><td>서보가 60° → 120° → 90° 로 움직인다</td><td>안 움직이면 UBEC 출력(P5, G4)과 서보 커넥터(S1~S3) 방향. 5절 참조</td></tr>
<tr><td class="id">3</td><td><code>uv run python scripts/pin_test.py motor</code></td><td>두 바퀴가 같은 방향으로 10초 돌고, 멈춘 뒤 반대 방향으로 10초 돈다</td><td>한 방향만 돌거나 정지 명령에 빨라지면 속도·방향 선이 바뀐 것(D1 과 D4). 한쪽만 안 돌면 그쪽 모터 전력선과 J4·J5 레버. 두 바퀴가 서로 반대로 돌면 한쪽 모터의 전력선 두 가닥을 맞바꾼다</td></tr>
<tr><td class="id">4</td><td><code>uv run python scripts/dotmatrix_test.py status</code> (도트 매트릭스를 달았을 때)</td><td>정지 ■ → 전진 ▲ 와 속도 막대 → 후진 ▼ → 조향 점 좌우 → 녹화 모서리 점멸 → X 점멸</td><td>아무것도 안 켜지면 <code>bash scripts/jetson_setup.sh verify</code> 의 SPI 항목과 X1~X5. 화살표가 돌아가 있으면 <code>display.rotate</code></td></tr>
<tr><td class="id">5</td><td><code>uv run python -m collector --check</code></td><td>"점검 완료" 와 함께 서보가 중앙으로 한 번 움직인다</td><td>실패한 점검 이름(i2c_bus, pca9685:서보, pca9685:모터 …)과 메시지를 따라 고친다</td></tr>
</tbody></table></div>
<p>3번에서 전진 명령에 차가 뒤로 가면 선은 그대로 두고 <code>config.yaml</code> 의 <code>hardware.dir_forward_high</code> 를 <code>false</code> 로 바꿉니다.</p>

<h2 class="pagebreak">5. 잘못 꽂기 쉬운 곳</h2>
<div class="tablewrap"><table>
<thead><tr><th>실수</th><th>결과</th><th>막는 방법</th></tr></thead>
<tbody>
<tr><td class="k">I1 을 VCC 가 아니라 옆의 V+ 에 꽂음</td><td>Jetson 의 3.3 V 에 6 V 가 들어가 파손된다</td><td>VCC 와 V+ 는 이웃한 핀이다. 인쇄를 읽고 꽂는다</td></tr>
<tr><td class="k">Jetson 쪽에서 핀 1 대신 핀 2, 핀 6 대신 핀 4 에 꽂음</td><td>3.3 V 버스에 5 V 가 걸린다</td><td>핀 2, 4 는 5 V 다. 바깥쪽 줄이 짝수 핀이다</td></tr>
<tr><td class="k">두 PCA9685 보드가 뒤바뀜</td><td>서보에 모터 신호가, 모터에 서보 신호가 간다</td><td>모터용 보드에 표시를 한다. A0 를 납땜한 쪽이 모터용이다</td></tr>
<tr><td class="k">속도 선과 방향 선이 뒤바뀜 (D1 과 D4, 또는 J4 와 J5 의 출력)</td><td>한 방향으로만 돌고, 정지 명령에 전속으로 돈다</td><td>채널 0 이 PWM, 채널 1 이 DIR 이다. J4 에서 나온 선은 PWM1, PWM2 에만, J5 에서 나온 선은 DIR1, DIR2 에만 꽂는다. MDD10A 인쇄(DIR1, PWM1 …)를 보고 맞춘다</td></tr>
<tr><td class="k">J4, J5 에 가는 듀폰 점퍼선(26~28AWG)을 물림</td><td>달리다 흔들려 빠지거나 접촉이 끊겨 모터가 멈추거나 멋대로 돈다</td><td>22AWG 또는 더 굵은 선만 물리고, 레버를 닫은 뒤 당겨 본다</td></tr>
<tr><td class="k">서보 커넥터를 뒤집어 꽂음</td><td>서보가 움직이지 않는다</td><td>검정이 보드 가장자리 쪽이다</td></tr>
<tr><td class="k">MDD10A 의 B+ 와 B− 를 반대로 물림</td><td>드라이버가 파손될 수 있다</td><td>판매처 자료에 역극성 보호가 없다고 되어 있다. 배터리를 연결하기 전에 B+ 가 J2, B− 가 J3 에서 오는지 다시 본다</td></tr>
</tbody></table></div>

</main>
</body>
</html>
"""


if __name__ == "__main__":
    figs = all_figures()
    problems = check_coverage(figs) + [e for f in figs for e in f.lint()]
    for p in problems:
        print("문제:", p)
    print(f"전선 {len(WIRE_IDS)}가닥, 그림 {len(figs)}장, 문제 {len(problems)}건")
    if problems:
        sys.exit(1)
    OUT.write_text(build_html(figs), encoding="utf-8")
    print("작성:", OUT)
