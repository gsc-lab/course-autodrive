"""입력 장치. 모두 InputDevice 인터페이스로 InputEvent 를 만들어 낸다."""
from common.inputs.base import InputDevice
from common.inputs.factory import create_input

__all__ = ["InputDevice", "create_input"]
