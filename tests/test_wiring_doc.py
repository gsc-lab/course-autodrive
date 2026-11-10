"""배선 문서 생성기(scripts/make_wiring_doc.py)의 검사 기능과 실제 그림을 확인한다."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_wiring_doc as m  # noqa: E402


def _dia():
    d = m.Diagram(400, 300, "시험")
    d.port("JET.1", 20, 100); d.port("PCA1.VCC", 300, 100)      # I1
    d.port("JET.3", 150, 20); d.port("PCA1.SDA", 150, 250)      # I2
    return d


def test_lint_reports_each_kind_of_overlap():
    d = _dia()
    d.wire("I1"); d.wire("I2")                                   # 가로선과 세로선이 (150, 100) 에서 교차
    assert any("선 교차" in e for e in d.lint())
    d = _dia(); d.wire("I1"); d.t(100, 104, "겹치는 글자")
    assert any("글자를 지남" in e for e in d.lint())
    d = _dia(); d.t(50, 50, "가나다라"); d.t(60, 52, "마바사아")
    assert any("글자 겹침" in e for e in d.lint())
    d = _dia(); d.wire("I1"); d.box(120, 60, 60, 80)
    assert any("상자 안을 지남" in e for e in d.lint())
    d = _dia(); d.t(380, 50, "그림 밖으로 나가는 긴 글자")
    assert any("그림 밖" in e for e in d.lint())
    d = _dia(); d.wire("I1"); d.t(100, 80, "떨어진 글자")
    assert d.lint() == []


def test_parallel_wires_on_the_same_line_are_reported():
    d = m.Diagram(400, 300, "시험")
    d.port("JET.1", 20, 100); d.port("PCA1.VCC", 300, 100)
    d.port("JET.3", 100, 102); d.port("PCA1.SDA", 350, 102)
    d.wire("I1"); d.wire("I2")
    assert any("선 겹침" in e for e in d.lint())


def test_real_figures_are_clean_and_cover_the_netlist():
    figs = m.all_figures()
    assert m.check_coverage(figs) == []
    for f in figs:
        assert f.lint() == [], f.name
    ids = [r[0] for r in m.NET]
    assert len(ids) == len(set(ids)) == len(m.WIRE_IDS) == 36
    assert set(m.GROUND_IDS) <= set(ids)


def test_every_wire_endpoint_is_a_distinct_named_terminal():
    ends = [p for r in m.NET for p in (r[1], r[2])]
    assert len(ends) == len(set(ends)), "한 단자에는 전선이 하나만 꽂힌다"
    assert all(p in m.PORT_LABEL for p in ends)


def test_motor_signals_split_at_j4_and_j5():
    """속도는 채널 0 → J4 → PWM1·PWM2, 방향은 채널 1 → J5 → DIR1·DIR2. 두 블록이 섞이면 안 된다."""
    rows = {r[0]: (r[1], r[2]) for r in m.NET}
    assert rows["D1"] == ("PCA2.CH0P", "J4.in") and rows["D4"] == ("PCA2.CH1P", "J5.in")
    assert {rows["D2"][1], rows["D3"][1]} == {"MDDC.PWM1", "MDDC.PWM2"} and all(rows[w][0].startswith("J4.") for w in ("D2", "D3"))
    assert {rows["D5"][1], rows["D6"][1]} == {"MDDC.DIR1", "MDDC.DIR2"} and all(rows[w][0].startswith("J5.") for w in ("D5", "D6"))
    assert rows["D7"] == ("PCA2.CH0G", "MDDC.GND")
    assert not any("SW1" in p for r in m.NET for p in r[1:3]), "주전원 스위치는 없다"
    assert [rows[w][0] for w in m.DISPLAY_IDS] == ["JET.2", "JET.20", "JET.19", "JET.23", "JET.24"]   # 도트 매트릭스: 5 V, GND, MOSI, SCK, CS0
