"""모델 번들 읽기: 학습기가 만든 models/<실행 이름>/ 의 model.onnx + model_meta.json.

추론기는 전처리(자르는 범위, 입력 크기)와 조향 단계를 config.yaml 이 아니라 model_meta.json 에서 읽는다.
그래야 모델이 학습할 때 본 것과 똑같이 자른 사진을 받는다 (CLAUDE.md, "Contracts between stages").
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from common.preprocess import CROP_MIN_HEIGHT, Preprocessor, crop_from_dict

SUPPORTED_SCHEMA = 1


class BundleError(RuntimeError):
    """모델 폴더를 쓸 수 없다. 문구는 그대로 화면에 한 줄로 나간다."""


@dataclass(frozen=True)
class ModelBundle:
    path: Path                  # models/<실행 이름>/
    run_id: str
    classes: List[int]          # 출력 i 번째 값 = classes[i] 도(서보 각도)의 점수
    pre: Preprocessor           # 모델이 학습할 때 쓴 전처리
    onnx_path: Path
    onnx_sha256: str
    meta: dict

    @property
    def input_shape(self) -> tuple:
        return (1, *self.pre.input_shape)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bundle(path: str | Path) -> ModelBundle:
    """모델 폴더(또는 그 안의 model.onnx / model_meta.json)를 읽어 검사한다. 쓸 수 없으면 BundleError."""
    path = Path(path)
    folder = path.parent if path.is_file() else path
    if not folder.is_dir():
        raise BundleError(f"모델 폴더가 없습니다: {path}  (PC 의 models/<실행 이름>/ 폴더를 차로 복사했는지 확인하세요)")
    meta_path, onnx_path = folder / "model_meta.json", folder / "model.onnx"
    for p, what in ((meta_path, "모델 설명서"), (onnx_path, "모델")):
        if not p.is_file():
            raise BundleError(f"{folder} 에 {p.name}({what}) 이 없습니다. 학습 결과 폴더를 통째로 복사하세요 "
                              "(PC 에서 `uv run -m trainer export models/<실행 이름>` 로 다시 만들 수 있습니다)")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        if not isinstance(meta, dict):
            raise ValueError("JSON 의 맨 바깥이 묶음({ })이 아닙니다")
    except (OSError, ValueError) as e:
        raise BundleError(f"{meta_path} 를 읽을 수 없습니다: {e}") from None

    if meta.get("schema_version") != SUPPORTED_SCHEMA:
        raise BundleError(f"model_meta.json 의 형식 번호가 {meta.get('schema_version')!r} 입니다 (이 프로그램은 {SUPPORTED_SCHEMA}). "
                          "차의 코드와 PC 의 코드를 같은 판으로 맞추세요 (git pull)")
    if meta.get("task") != "classification":
        raise BundleError(f"이 프로그램은 분류(classification) 모델만 다룹니다: task = {meta.get('task')!r}")

    classes = meta.get("classes")
    if (not isinstance(classes, list) or len(classes) < 2
            or any(isinstance(c, bool) or not isinstance(c, int) for c in classes) or len(set(classes)) != len(classes)):
        raise BundleError(f"model_meta.json 의 classes 가 올바르지 않습니다: {classes!r}")

    try:
        pre = Preprocessor.from_dict(meta["preprocess"])
        crop = crop_from_dict(meta["preprocess"])
    except (KeyError, TypeError, ValueError) as e:
        raise BundleError(f"model_meta.json 의 preprocess 를 읽을 수 없습니다: {e}") from None
    if crop[1] - crop[0] < CROP_MIN_HEIGHT - 1e-9 or pre.out_width < 1 or pre.out_height < 1:
        raise BundleError(f"model_meta.json 의 preprocess 값이 올바르지 않습니다: {meta['preprocess']!r}")

    want = [1, *pre.input_shape]
    try:
        shape, out_shape = list(meta["input"]["shape"]), list(meta["output"]["shape"])
    except (KeyError, TypeError):
        raise BundleError("model_meta.json 에 input / output 의 shape 가 없습니다") from None
    if shape != want or out_shape != [1, len(classes)]:
        raise BundleError(f"model_meta.json 의 입출력 모양이 맞지 않습니다: 입력 {shape} (전처리로는 {want}), "
                          f"출력 {out_shape} (클래스 {len(classes)}개)")

    digest = sha256_of(onnx_path)
    recorded = (meta.get("onnx") or {}).get("sha256")
    if recorded and recorded != digest:
        raise BundleError(f"{folder.name} 의 model.onnx 와 model_meta.json 이 서로 다른 학습에서 나온 파일입니다. "
                          "두 파일을 따로 복사하지 말고 폴더를 통째로 다시 복사하세요")
    run_id = str((meta.get("training") or {}).get("run_id") or folder.name)
    return ModelBundle(path=folder, run_id=run_id, classes=[int(c) for c in classes], pre=pre,
                       onnx_path=onnx_path, onnx_sha256=digest, meta=meta)
