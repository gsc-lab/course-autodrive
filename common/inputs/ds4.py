"""DualShock 4 (CUH-ZCT2G) — python-evdev 로 읽는다.

커널 드라이버에 따라 같은 패드가 다른 evdev 코드를 낸다:
- sony     : hid-sony(커널 ≤6.1) / hid-playstation(≥6.2). 라이트바·진동 가능.
- generic  : hid-generic (JetPack 6.2.1 커널에는 hid-sony 가 없음 → 이 경우). 스틱·버튼만 동작.
드라이버는 /sys/class/input/eventN/device/device/driver 로 감지하고, config 의 input.ds4.layout 으로 강제할 수 있다.

매핑 (B1 확정):
  왼쪽 스틱 X  조향            D패드 ←→  조향 한 단계
  R2           전진 속도        L2        후진 속도 (녹화 안 됨)
  L1 (누름)    데드맨           ○         녹화 토글
  □            직전 N초 불량    △         조향 중앙
  Options      종료

보드에서 실제 코드를 확인하려면:  uv run python -m common.inputs.ds4
"""
from __future__ import annotations

import glob
import logging
import os
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

from common.config import DS4Config
from common.events import EventKind, InputEvent
from common.hw.checks import find_ds4_lightbar_dirs
from common.inputs.base import STATUS_ERROR, STATUS_IDLE, STATUS_RECORDING, InputDevice

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 리눅스 input 상수 (evdev 없이도 테스트할 수 있도록 숫자로 둔다)
# ---------------------------------------------------------------------------
EV_KEY, EV_ABS = 0x01, 0x03
ABS_X, ABS_Y, ABS_Z, ABS_RX, ABS_RY, ABS_RZ = 0x00, 0x01, 0x02, 0x03, 0x04, 0x05
ABS_HAT0X, ABS_HAT0Y = 0x10, 0x11
BTN_SOUTH, BTN_EAST, BTN_C, BTN_NORTH, BTN_WEST, BTN_Z = 0x130, 0x131, 0x132, 0x133, 0x134, 0x135
BTN_TL, BTN_TR, BTN_TL2, BTN_TR2, BTN_SELECT, BTN_START, BTN_MODE, BTN_THUMBL = 0x136, 0x137, 0x138, 0x139, 0x13A, 0x13B, 0x13C, 0x13D

SONY_VENDOR = 0x054C
DS4_PRODUCTS = (0x09CC, 0x05C4)   # 2세대(CUH-ZCT2), 1세대(CUH-ZCT1)

# 역할 → evdev 코드. 키 이름은 config input.ds4.codes 로 덮어쓸 수 있다.
LAYOUTS: Dict[str, Dict[str, int]] = {
    "sony": {
        "steer": ABS_X, "throttle": ABS_RZ, "reverse": ABS_Z, "dpad": ABS_HAT0X,
        "deadman": BTN_TL, "record": BTN_EAST, "mark": BTN_WEST, "center": BTN_NORTH, "quit": BTN_START,
    },
    # hid-generic: DS4 HID 디스크립터 순서 그대로. 축: X,Y,Z(RX스틱),RZ(RY스틱),Hat,buttons,RX(L2),RY(R2)
    # 버튼 1..14 = □ × ○ △ L1 R1 L2 R2 Share Options L3 R3 PS Touch → BTN_GAMEPAD(0x130)+n-1
    "generic": {
        "steer": ABS_X, "throttle": ABS_RY, "reverse": ABS_RX, "dpad": ABS_HAT0X,
        "deadman": BTN_WEST, "record": BTN_C, "mark": BTN_SOUTH, "center": BTN_NORTH, "quit": BTN_TR2,
    },
}
COLORS = {STATUS_IDLE: (0, 0, 255), STATUS_RECORDING: (255, 0, 0), STATUS_ERROR: (255, 0, 0)}

# 표시용: 레이아웃별로 각 코드가 패드의 어떤 입력인지 (쓰지 않는 것도 포함). 2026-11-04 실기 확인: generic 의 ○□△L1 Options R2.
BUTTON_NAMES = {
    "generic": {BTN_SOUTH: "□", BTN_EAST: "×", BTN_C: "○", BTN_NORTH: "△", BTN_WEST: "L1", BTN_Z: "R1",
                BTN_TL: "L2클릭", BTN_TR: "R2클릭", BTN_TL2: "Share", BTN_TR2: "Options", BTN_SELECT: "L3",
                BTN_START: "R3", BTN_MODE: "PS", BTN_THUMBL: "터치패드",
                ABS_X: "왼스틱X", ABS_Y: "왼스틱Y", ABS_Z: "오른스틱X", ABS_RX: "L2", ABS_RY: "R2", ABS_RZ: "오른스틱Y",
                ABS_HAT0X: "D패드X", ABS_HAT0Y: "D패드Y"},
    "sony": {BTN_SOUTH: "×", BTN_EAST: "○", BTN_NORTH: "△", BTN_WEST: "□", BTN_TL: "L1", BTN_TR: "R1",
             BTN_TL2: "L2클릭", BTN_TR2: "R2클릭", BTN_SELECT: "Share", BTN_START: "Options", BTN_MODE: "PS",
             BTN_THUMBL: "L3", 0x13E: "R3",
             ABS_X: "왼스틱X", ABS_Y: "왼스틱Y", ABS_Z: "L2", ABS_RX: "오른스틱X", ABS_RY: "오른스틱Y", ABS_RZ: "R2",
             ABS_HAT0X: "D패드X", ABS_HAT0Y: "D패드Y"},
}


def layout_for_driver(driver: str) -> str:
    return "sony" if driver in ("sony", "playstation") else "generic"


def resolve_codes(layout: str, overrides: Optional[dict] = None) -> Dict[str, int]:
    if layout not in LAYOUTS:
        raise ValueError(f"알 수 없는 DS4 layout: {layout!r} (sony | generic)")
    codes = dict(LAYOUTS[layout])
    for k, v in (overrides or {}).items():
        if k not in codes:
            raise ValueError(f"input.ds4.codes 에 알 수 없는 키: {k!r}. 허용: {sorted(codes)}")
        codes[k] = int(v, 0) if isinstance(v, str) else int(v)
    return codes


class DS4Mapper:
    """evdev (type, code, value) → InputEvent. 순수 함수라 PC 에서 테스트한다."""

    def __init__(self, codes: Dict[str, int], absinfo: Dict[int, Tuple[int, int]], deadzone: float = 0.08):
        self.codes = codes
        self.absinfo = absinfo          # code -> (min, max)
        self.deadzone = float(deadzone)
        self._by_abs = {codes["steer"]: "steer", codes["throttle"]: "throttle",
                        codes["reverse"]: "reverse", codes["dpad"]: "dpad"}
        self._by_key = {codes["deadman"]: "deadman", codes["record"]: "record", codes["mark"]: "mark",
                        codes["center"]: "center", codes["quit"]: "quit"}

    def _range(self, code: int) -> Tuple[int, int]:
        return self.absinfo.get(code, (0, 255))

    def norm_stick(self, code: int, value: int) -> float:
        lo, hi = self._range(code)
        mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0
        x = (value - mid) / half if half else 0.0
        dz = self.deadzone
        if abs(x) < dz:
            return 0.0
        x = (abs(x) - dz) / (1.0 - dz) * (1 if x > 0 else -1)
        return max(-1.0, min(1.0, x))

    def norm_trigger(self, code: int, value: int) -> float:
        lo, hi = self._range(code)
        return max(0.0, min(1.0, (value - lo) / float(hi - lo))) if hi > lo else 0.0

    def translate(self, ev_type: int, code: int, value: int) -> Optional[InputEvent]:
        if ev_type == EV_ABS:
            role = self._by_abs.get(code)
            if role == "steer":
                return InputEvent(EventKind.STEER_AXIS, self.norm_stick(code, value))
            if role == "throttle":
                return InputEvent(EventKind.THROTTLE_AXIS, self.norm_trigger(code, value))
            if role == "reverse":
                return InputEvent(EventKind.REVERSE_AXIS, self.norm_trigger(code, value))
            if role == "dpad" and value != 0:
                return InputEvent(EventKind.STEER_STEP, 1 if value > 0 else -1)
            return None
        if ev_type == EV_KEY:
            role = self._by_key.get(code)
            if role is None or value == 2:        # 2 = 자동 반복
                return None
            pressed = value == 1
            if role == "deadman":
                return InputEvent(EventKind.DEADMAN, pressed)
            if not pressed:
                return None
            return InputEvent({"record": EventKind.RECORD_TOGGLE, "mark": EventKind.MARK,
                               "center": EventKind.STEER_CENTER, "quit": EventKind.QUIT}[role])
        return None


# ---------------------------------------------------------------------------
# 장치 탐색
# ---------------------------------------------------------------------------
def detect_driver(event_path: str) -> str:
    """'/dev/input/eventN' 을 잡고 있는 HID 드라이버 이름: hid-generic | sony | playstation | unknown"""
    event = os.path.basename(event_path)
    for rel in ("device/device/driver", "device/driver"):
        p = f"/sys/class/input/{event}/{rel}"
        try:
            return os.path.basename(os.readlink(p))
        except OSError:
            continue
    return "unknown"


def _is_ds4(dev, fallback_name: str) -> bool:
    name = dev.name or ""
    if "Touchpad" in name or "Motion" in name:
        return False
    try:
        if dev.info.vendor == SONY_VENDOR and dev.info.product in DS4_PRODUCTS:
            return True
    except AttributeError:
        pass
    return name == fallback_name


def find_ds4_device(fallback_name: str = "Wireless Controller"):
    import evdev
    for path in sorted(evdev.list_devices()):
        try:
            dev = evdev.InputDevice(path)
        except OSError:
            continue
        if _is_ds4(dev, fallback_name):
            return dev
        dev.close()
    return None


# ---------------------------------------------------------------------------
# 링크 감시: 패드는 가만히 있어도 초당 수백 개의 HID 보고를 보낸다 (hid-generic, 블루투스, 2026-11-10 실측 약 450/s).
# evdev 는 값이 바뀔 때만 이벤트를 주므로 "조용함"과 "끊김"을 구분하지 못한다. hidraw 를 같이 읽어 보고가 끊기면
# 링크가 죽은 것으로 보고 즉시 정지시킨다 (커널이 장치를 지우는 데는 블루투스 감시 시간 ~20초가 걸린다).
# ---------------------------------------------------------------------------
def find_ds4_hidraw(uniq: Optional[str], name: str, sys_root: str = "/sys/class/hidraw") -> Optional[str]:
    """evdev 장치와 같은 패드(HID_UNIQ = 패드 MAC)의 hidraw 경로. 없으면 이름으로, 그래도 없으면 None."""
    by_name = None
    for d in sorted(glob.glob(f"{sys_root}/hidraw*")):
        try:
            with open(f"{d}/device/uevent") as f:
                info = dict(line.strip().split("=", 1) for line in f if "=" in line)
        except OSError:
            continue
        path = f"/dev/{os.path.basename(d)}"
        if uniq and info.get("HID_UNIQ", "").lower() == uniq.lower():
            return path
        if name and name in info.get("HID_NAME", "") and by_name is None:
            by_name = path
    return by_name


class HidrawLink:
    """hidraw 보고를 세어 링크 생사를 판단한다. reader() 는 보고 한 개(bytes) 또는 None(지금은 없음)을 돌려준다."""

    def __init__(self, reader, timeout_sec: float, now: Optional[float] = None, close=None):
        self._reader, self._close = reader, close
        self.timeout = float(timeout_sec)
        self.last_report = time.monotonic() if now is None else now
        self.reports = 0

    @classmethod
    def open(cls, path: str, timeout_sec: float) -> "HidrawLink":
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)

        def reader():
            try:
                return os.read(fd, 128)
            except BlockingIOError:
                return None
        return cls(reader, timeout_sec, close=lambda: os.close(fd))

    def poll(self, now: float) -> bool:
        """쌓인 보고를 비우고 살아 있으면 True. timeout 동안 보고가 없으면 False."""
        for _ in range(2000):               # 한 번에 너무 오래 붙들지 않게
            data = self._reader()
            if not data:
                break
            self.reports += 1
            self.last_report = now
        return (now - self.last_report) <= self.timeout

    def close(self) -> None:
        if self._close:
            self._close()
            self._close = None


class DS4Input(InputDevice):
    name = "ds4"
    requires_deadman = True

    def __init__(self, cfg: DS4Config):
        self.cfg = cfg
        self.requires_deadman = bool(cfg.deadman)
        self.dev = find_ds4_device(cfg.device_name)
        if self.dev is None:
            raise RuntimeError("DualShock 4 (evdev) 를 찾지 못했습니다. 페어링/연결 상태와 input 그룹 권한을 확인하세요")
        self.driver = detect_driver(self.dev.path)
        self.layout = cfg.layout if cfg.layout != "auto" else layout_for_driver(self.driver)
        codes = resolve_codes(self.layout, cfg.codes)
        caps = self.dev.capabilities(absinfo=True)
        absinfo = {code: (info.min, info.max) for code, info in caps.get(EV_ABS, [])}
        self.mapper = DS4Mapper(codes, absinfo, cfg.stick_deadzone)
        self.dev.grab()  # 다른 프로세스가 같은 입력을 받지 않도록

        self.features = self.layout == "sony"   # 라이트바·진동은 sony 계열 드라이버에서만
        self._leds = find_ds4_lightbar_dirs() if (cfg.lightbar and self.features) else []
        self._status: Optional[str] = None
        self._blink_on = True
        self._last_blink = time.monotonic()
        self._rumble_thread: Optional[threading.Thread] = None
        self.link: Optional[HidrawLink] = None
        self._link_lost = False
        if cfg.link_timeout_sec > 0:
            path = find_ds4_hidraw(getattr(self.dev, "uniq", None), cfg.device_name)
            try:
                if path is None:
                    raise OSError("hidraw 장치를 찾지 못함")
                self.link = HidrawLink.open(path, cfg.link_timeout_sec)
            except OSError as e:
                log.warning("패드 링크 감시를 쓸 수 없습니다 (%s). 연결이 조용히 끊기면 커널이 알아챌 때까지(~20초) "
                            "마지막 명령이 유지됩니다. udev 규칙 91-ds4-hidraw.rules 와 input 그룹을 확인하세요", e)
        log.info("DS4 연결: %s (%s) driver=%s layout=%s lightbar=%s 링크감시=%s",
                 self.dev.name, self.dev.path, self.driver, self.layout, bool(self._leds),
                 f"{cfg.link_timeout_sec:g}s" if self.link else "없음")
        self.set_status(STATUS_IDLE)

    # ------------------------------------------------------------------ API
    def poll(self) -> List[InputEvent]:
        events: List[InputEvent] = []
        try:
            while True:
                ev = self.dev.read_one()
                if ev is None:
                    break
                out = self.mapper.translate(ev.type, ev.code, ev.value)
                if out is not None:
                    events.append(out)
        except OSError as e:  # ENODEV: 블루투스 끊김
            log.error("DS4 장치 소실: %s", e)
            events.append(InputEvent(EventKind.DEVICE_LOST))
        if self.link is not None and not self._link_lost:
            try:
                alive = self.link.poll(time.monotonic())
            except OSError as e:
                log.error("DS4 hidraw 읽기 실패: %s", e)
                alive = False
            if not alive:
                self._link_lost = True
                log.error("DS4 보고가 %.1f초 동안 없음 → 링크 끊김으로 보고 정지", self.link.timeout)
                events.append(InputEvent(EventKind.DEVICE_LOST))
        self._tick_blink()
        return events

    def _write_leds(self, rgb) -> None:
        if not self._leds:
            return
        for d, v in zip(self._leds, rgb):
            try:
                with open(f"{d}/brightness", "w") as f:
                    f.write(str(int(v)))
            except OSError as e:
                log.warning("라이트바 쓰기 실패 (%s) — 비활성화. udev 규칙 또는 sudo 필요", e)
                self._leds = []
                return

    def _tick_blink(self) -> None:
        if self._status != STATUS_ERROR or not self._leds:
            return
        now = time.monotonic()
        if now - self._last_blink >= 0.25:
            self._last_blink = now
            self._blink_on = not self._blink_on
            self._write_leds(COLORS[STATUS_ERROR] if self._blink_on else (0, 0, 0))

    def set_status(self, status: str) -> None:
        if status == self._status:
            return
        self._status = status
        self._write_leds(COLORS.get(status, (0, 0, 0)))

    def notify(self, kind: str) -> None:
        """진동 알림. 제어 루프를 막지 않도록 별도 스레드에서 재생한다. generic 드라이버에서는 불가."""
        if not (self.cfg.rumble and self.features):
            return
        if self._rumble_thread is not None and self._rumble_thread.is_alive():
            return
        pattern = {"record_on": [250], "record_off": [120, 120], "mark": [60]}.get(kind, [100])
        self._rumble_thread = threading.Thread(target=self._rumble, args=(pattern,), name="ds4-rumble", daemon=True)
        self._rumble_thread.start()

    def _rumble(self, pattern: list) -> None:
        try:
            from evdev import ecodes, ff
            for i, dur in enumerate(pattern):
                effect = ff.Effect(
                    ecodes.FF_RUMBLE, -1, 0, ff.Trigger(0, 0), ff.Replay(dur, 0),
                    ff.EffectType(ff_rumble_effect=ff.Rumble(strong_magnitude=0x8000, weak_magnitude=0xFFFF)),
                )
                eid = self.dev.upload_effect(effect)
                self.dev.write(ecodes.EV_FF, eid, 1)
                time.sleep(dur / 1000.0 + (0.1 if i < len(pattern) - 1 else 0))
                self.dev.erase_effect(eid)
        except Exception as e:  # 진동은 보조 기능 — 실패해도 주행에 영향 없음
            log.debug("진동 실패: %s", e)

    def battery(self) -> int | None:
        for path in glob.glob("/sys/class/power_supply/*sony*controller*/capacity") + \
                glob.glob("/sys/class/power_supply/ps-controller-battery-*/capacity"):
            try:
                with open(path) as f:
                    return int(f.read().strip())
            except (OSError, ValueError):
                pass
        return None

    def close(self) -> None:
        try:
            self._write_leds((0, 0, 0))
            self.dev.ungrab()
        except Exception:
            pass
        if self.link is not None:
            self.link.close()
        self.dev.close()


# ---------------------------------------------------------------------------
# 보드에서 매핑 확인용:  uv run python -m common.inputs.ds4 [초]   (초를 생략하면 Options 또는 Ctrl+C 까지)
# ---------------------------------------------------------------------------
def _main(argv: List[str]) -> int:
    import evdev
    seconds = float(argv[1]) if len(argv) > 1 else None
    dev = find_ds4_device()
    if dev is None:
        print("DS4 를 찾지 못했습니다. 보이는 장치:")
        for p in evdev.list_devices():
            d = evdev.InputDevice(p)
            print(f"  {p}  {d.name}  vendor={d.info.vendor:04x} product={d.info.product:04x} driver={detect_driver(p)}")
        return 1
    driver = detect_driver(dev.path)
    layout = layout_for_driver(driver)
    caps = dev.capabilities(absinfo=True)
    absinfo = {code: (i.min, i.max) for code, i in caps.get(EV_ABS, [])}
    mapper = DS4Mapper(resolve_codes(layout), absinfo)
    codes = resolve_codes(layout)
    role_of_key = {v: k for k, v in codes.items() if k in ("deadman", "record", "mark", "center", "quit")}
    role_of_abs = {v: k for k, v in codes.items() if k in ("steer", "throttle", "reverse", "dpad")}
    print(f"장치: {dev.name} ({dev.path})  driver={driver}  → layout={layout}")
    print(f"ABS 축: {sorted(absinfo)}   KEY: {[hex(k) for k in sorted(caps.get(EV_KEY, []))]}")
    print("기대 매핑:", ", ".join(f"{r}=0x{c:03x}" for c, r in sorted(role_of_key.items())),
          "|", ", ".join(f"{r}=0x{c:02x}" for c, r in sorted(role_of_abs.items())))
    names = BUTTON_NAMES.get(layout, {})
    limit = f"{seconds:.0f}초 동안" if seconds else "Options 버튼(종료) 또는 Ctrl+C 까지"
    print(f"{limit} ○ □ △ L1 Options R2 L2 D패드 왼쪽스틱 순서로 눌러 보세요. "
          "스틱 중앙 떨림은 숨기고 버튼과 실제 움직임만 출력합니다.\n", flush=True)
    seen_abs, seen_key = {}, set()
    last_norm: Dict[int, float] = {}
    t_end = time.monotonic() + seconds if seconds else None
    try:
        while t_end is None or time.monotonic() < t_end:
            ev = dev.read_one()
            if ev is None:
                time.sleep(0.005)
                continue
            if ev.type == EV_ABS:
                seen_abs[ev.code] = ev.value
                lo, hi = absinfo.get(ev.code, (0, 255))
                norm = (ev.value - lo) / float(hi - lo) if hi > lo else 0.0
                if ev.code in (ABS_HAT0X, ABS_HAT0Y):
                    changed = True
                else:
                    changed = abs(norm - last_norm.get(ev.code, 0.5)) >= 0.05
                if not changed:
                    continue
                last_norm[ev.code] = norm
                role = role_of_abs.get(ev.code, "(미사용)")
            elif ev.type == EV_KEY:
                if ev.value == 2:
                    continue
                seen_key.add(ev.code)
                role = role_of_key.get(ev.code, "(미사용)")
            else:
                continue
            out = mapper.translate(ev.type, ev.code, ev.value)
            tag = "ABS" if ev.type == EV_ABS else "KEY"
            print(f"{tag} 0x{ev.code:03x} {names.get(ev.code, '?'):6s} value={ev.value:4d}  역할={role:10s} → {out if out else '-'}",
                  flush=True)
            if out is not None and out.kind == EventKind.QUIT and seconds is None:
                print("\nOptions → 종료")
                break
    except KeyboardInterrupt:
        print("\nCtrl+C → 종료")
    print("\n관측된 ABS 코드:", {f'0x{c:02x}': v for c, v in sorted(seen_abs.items())})
    print("관측된 KEY 코드:", [f"0x{c:03x}" for c in sorted(seen_key)])
    print("각 버튼이 기대한 역할로 해석되지 않으면 config.yaml 의 input.ds4.layout 또는 codes 를 조정하세요.")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
