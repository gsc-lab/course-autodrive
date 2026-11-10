"""모든 프로그램(collector / curator / trainer / inference)이 공유하는 코드.

규칙:
- 이 패키지는 PC(Windows)에서도 부작용 없이 import 되어야 한다.
  리눅스 전용 모듈(fcntl, evdev, termios)은 함수/클래스 안에서 지연 import 한다.
- 조향 클래스 목록, 파일 이름 규약, 전처리 파라미터는 여기(와 config.yaml)에만 존재한다.
"""

__version__ = "0.1.0"
