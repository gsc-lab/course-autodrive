# docs

| 파일 | 내용 |
|---|---|
| [jetson-setup.md](jetson-setup.md) | Jetson Orin Nano 설정 절차, 골든 이미지 배포, I2C 연결 확인, DualShock 4 페어링 |
| [wiring.pdf](wiring.pdf) | 자율주행자동차 배선도 v1.0 (전선 31가닥, 그림 4장, 조립 후 동작 시험). 원본은 `wiring.html` |

배선도는 `scripts/make_wiring_doc.py` 가 배선 목록 하나에서 만든다. 배선을 바꿀 때는 HTML 을 손으로 고치지 말고 그 스크립트의
`NET` 과 그림 함수를 고친 뒤 다시 생성한다 (`uv run python scripts/make_wiring_doc.py`, `tests/test_wiring_doc.py` 가 검사).
