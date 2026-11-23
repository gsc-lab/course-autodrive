"""학습기 (PC). 데이터셋 zip → 학습 → 시험 분석 → ONNX 모델 묶음.

    uv run -m trainer train dataset_YYYYMMDD_HHMM.zip

읽는 순서 (학습 한 번이 거치는 순서와 같다):
    split.py     데이터 나누기 (train / val / test)
    data.py      Dataset 과 DataLoader
    models.py    모델 (PilotNet 등)과 입력 스케일링
    losses.py    손실 함수
    optim.py     옵티마이저와 학습률 스케줄
    engine.py    학습 루프 (한 에폭 학습, 검증, 조기 종료)
    run.py       위의 것을 순서대로 엮어 학습 한 번을 실행
    evaluate.py  시험 데이터 분석 (혼동 행렬, 클래스별 지표)
    export.py    ONNX 내보내기와 model_meta.json
"""
__version__ = "0.1.0"
