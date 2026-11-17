# docs

| 파일 | 내용 |
|---|---|
| [jetson-setup.md](jetson-setup.md) | Jetson Orin Nano 설정, SD 카드 복제 배포, I2C 연결 확인, DualShock 4 페어링, 문제 해결 |
| [wiring.pdf](wiring.pdf) | 자율주행자동차 배선도 v1.0 (배선 목록, 그림, 조립 후 동작 시험) |
| [system-brief.md](system-brief.md) | 시스템 전체 요약 (강의 자료 작성용) |

배선도는 직접 고치지 않습니다. `scripts/make_wiring_doc.py` 의 배선 목록(`NET`)과 그림 함수를 고친 뒤 다시 만듭니다.

```bash
uv run python scripts/make_wiring_doc.py
```
