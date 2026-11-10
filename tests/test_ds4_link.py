"""패드 링크 감시: HID 보고가 끊기면 DEVICE_LOST 를 낸다."""
from common.inputs.ds4 import HidrawLink, find_ds4_hidraw


def test_link_alive_while_reports_flow_and_dead_after_timeout():
    queue = [b"\x01" * 10] * 5
    link = HidrawLink(lambda: queue.pop() if queue else None, timeout_sec=0.5, now=0.0)
    assert link.poll(0.1) is True and link.reports == 5          # 보고 5개를 비움
    assert link.poll(0.5) is True                                 # 아직 0.5초 안
    assert link.poll(0.61) is False                               # 보고 없이 0.5초 초과 → 끊김
    queue.append(b"\x01")
    assert link.poll(5.0) is True                                 # 보고가 다시 오면 살아 있음


def test_find_hidraw_by_uniq_then_by_name(tmp_path):
    for n, (uniq, name) in enumerate((("aa:bb", "Logitech USB Receiver"), ("88:03:4c:28:fa:93", "Wireless Controller"),
                                      ("11:22", "Wireless Controller"))):
        d = tmp_path / f"hidraw{n}" / "device"
        d.mkdir(parents=True)
        (d / "uevent").write_text(f"DRIVER=hid-generic\nHID_NAME={name}\nHID_UNIQ={uniq}\n")
    root = str(tmp_path)
    assert find_ds4_hidraw("88:03:4C:28:FA:93", "Wireless Controller", root) == "/dev/hidraw1"   # MAC 우선 (대소문자 무시)
    assert find_ds4_hidraw("ff:ff", "Wireless Controller", root) == "/dev/hidraw1"                # 없으면 이름으로 첫 것
    assert find_ds4_hidraw(None, "Nothing", root) is None
