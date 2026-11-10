# legacy/

Jetson Nano 시절(2024~2025) 코드입니다. 2026-11-04에 전면 재작성을 시작하면서 참고용으로만 옮겨 두었습니다.

- 실행하지 않습니다. 경로가 바뀌어 그대로는 동작하지 않으며, 하드웨어 코드(L298N, devmem PWM)는 Orin Nano에서 쓸 수 없습니다.
- 새 파이프라인이 끝까지 동작하면 이 디렉토리는 삭제합니다.
- 참고할 만한 부분: `preprocessor/RCPreprocessor.py`(전처리 파라미터), `training/model.py`(PilotNet 구조), `training/train_pilotnet.py`(학습 루프).

새 코드의 구조와 규약은 저장소 루트의 CLAUDE.md를 보세요.
