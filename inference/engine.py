"""모델을 실행하는 부분 (백엔드). Jetson 에서는 TensorRT, PC 의 모의 실행과 테스트에서는 onnxruntime.

TensorRT 엔진은 보드와 TensorRT 판에 맞춰 만들어야 해서 PC 에서 만들어 올 수 없다. 그래서 차가 model.onnx 로 직접 만든다
(처음 한 번, 5~25초). 만든 엔진은 모델 폴더에 model.engine 으로 두고 다음부터는 바로 읽는다 (0.01초).
model.onnx 가 바뀌었거나 TensorRT 판이 달라지면 다시 만든다 (model.engine.json 에 적어 둔 값으로 판단한다).

장치(GPU) 메모리는 libcudart 를 ctypes 로 불러서 다룬다: JetPack 6.2.1 에는 cuda-python 도 pycuda 도 없고, 필요한 것은
메모리 할당과 복사뿐이다 (2026-11-23 보드에서 확인). 모듈을 가져오는 것만으로는 아무 장치도 열지 않는다.

엔진만 따로 확인하려면 (카메라·차량 없이):  uv run python -m inference.engine models/<실행 이름>
"""
from __future__ import annotations

import ctypes
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from inference.bundle import BundleError, ModelBundle, load_bundle

log = logging.getLogger(__name__)

INPUT_NAME, OUTPUT_NAME = "input", "output"       # 학습기와의 약속 (trainer/export.py)
ENGINE_FILE, ENGINE_INFO = "model.engine", "model.engine.json"
WORKSPACE_BYTES = 1 << 28
H2D, D2H = 1, 2                                   # cudaMemcpyKind


class EngineError(RuntimeError):
    """모델을 실행할 수 없다. 문구는 그대로 화면에 한 줄로 나간다."""


class Backend:
    """사진 한 장(전처리한 것)을 넣으면 클래스별 점수를 돌려준다."""

    name = "backend"
    note = ""                                     # 어떻게 열었는지 (화면에 한 번 보여 준다)

    def infer(self, x: np.ndarray) -> np.ndarray:
        """x: (1, 3, H, W) float32, RGB 0~1 → (클래스 수,) float32 점수(logit)."""
        raise NotImplementedError

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# onnxruntime (PC)
# ---------------------------------------------------------------------------
class OnnxRuntimeBackend(Backend):
    name = "onnxruntime"

    def __init__(self, bundle: ModelBundle):
        try:
            import onnxruntime as ort
        except ImportError:
            raise EngineError("onnxruntime 이 없습니다. PC 에서는 `uv sync --extra gui --extra train` (GPU 가 없으면 train-cpu) "
                              "로 설치합니다. Jetson 에서는 TensorRT 를 씁니다") from None
        try:
            self.session = ort.InferenceSession(bundle.onnx_path.read_bytes(), providers=["CPUExecutionProvider"])
        except Exception as e:
            raise EngineError(f"model.onnx 를 열 수 없습니다: {e}") from None
        self.note = "onnxruntime (CPU)"

    def infer(self, x: np.ndarray) -> np.ndarray:
        return self.session.run([OUTPUT_NAME], {INPUT_NAME: x})[0][0]


# ---------------------------------------------------------------------------
# TensorRT (Jetson)
# ---------------------------------------------------------------------------
def _load_cudart():
    last = None
    for name in ("libcudart.so.12", "libcudart.so"):
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError as e:
            last = e
    else:
        raise EngineError(f"CUDA 실행 라이브러리(libcudart)를 찾지 못했습니다: {last}")
    lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.cudaFree.argtypes = [ctypes.c_void_p]
    lib.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
    lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
    return lib


def _cuda(code: int, what: str) -> None:
    if code != 0:
        raise EngineError(f"GPU 호출 실패 ({what}, CUDA 오류 {code})")


def _trt_logger(trt):
    class Logger(trt.ILogger):
        """TensorRT 가 하는 말 가운데 경고 이상만 모아 둔다 (화면에 쏟아 내지 않는다)."""

        def __init__(self):
            trt.ILogger.__init__(self)
            self.messages = []

        def log(self, severity, msg):
            if severity <= trt.ILogger.Severity.WARNING:
                self.messages.append(f"{severity.name}: {msg}")
    return Logger()


def build_engine(trt, onnx_bytes: bytes, precision: str, logger) -> bytes:
    """ONNX → 이 보드용 엔진(바이트). TensorRT 10 의 방식으로 만든다."""
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_bytes):
        errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise EngineError(f"TensorRT 가 model.onnx 를 읽지 못했습니다: {errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_BYTES)
    # TF32(곱셈의 가수를 10비트로 줄여 빠르게 하는 기능)는 이 GPU 에서 기본으로 켜져 있다. 끄면 PC 와의 출력 차이가
    # 0.008 에서 0.000001 로 줄고 추론은 0.3 ms 느려진다 (2026-11-23 실측). 차는 PC 에서 본 답을 그대로 내는 쪽을 고른다.
    if hasattr(trt.BuilderFlag, "TF32"):
        config.clear_flag(trt.BuilderFlag.TF32)
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise EngineError("TensorRT 가 엔진을 만들지 못했습니다: " + "; ".join(logger.messages[-3:]))
    return bytes(blob)


class TensorRTBackend(Backend):
    name = "tensorrt"

    def __init__(self, bundle: ModelBundle, precision: str = "fp32", say: Callable[[str], None] = lambda _m: None):
        try:
            import tensorrt as trt
        except ImportError:
            raise EngineError("tensorrt 모듈이 없습니다. Jetson 의 가상 환경을 `uv venv --system-site-packages` 로 만들었는지 "
                              "확인하세요 (PC 에서 돌려 보려면 --mock)") from None
        self._trt = trt
        self._cudart = _load_cudart()
        self._logger = _trt_logger(trt)
        self._buffers: dict = {}
        self._stream = ctypes.c_void_p()
        self._context = self._engine = self._runtime = None

        key = {"onnx_sha256": bundle.onnx_sha256, "tensorrt": trt.__version__, "precision": precision, "tf32": False}
        blob, self.note = self._cached_engine(bundle.path, key)
        if blob is not None:
            self._runtime = trt.Runtime(self._logger)
            self._engine = self._runtime.deserialize_cuda_engine(blob)
            if self._engine is None:                 # 다른 보드에서 만든 엔진을 복사해 왔을 때 등
                blob = None
        if blob is None:
            say("이 보드에 맞는 엔진을 만듭니다. 처음 한 번만 하며 5~25초 걸립니다 …")
            t0 = time.monotonic()
            blob = build_engine(trt, bundle.onnx_path.read_bytes(), precision, self._logger)
            self.note = f"엔진을 새로 만듦 ({time.monotonic() - t0:.0f}초)"
            self._save_engine(bundle.path, blob, key)
            self._runtime = trt.Runtime(self._logger)
            self._engine = self._runtime.deserialize_cuda_engine(blob)
            if self._engine is None:
                raise EngineError("방금 만든 엔진을 다시 읽지 못했습니다: " + "; ".join(self._logger.messages[-3:]))
        try:
            self._open_io(bundle)
        except Exception:
            self.close()
            raise
        self.note = f"TensorRT {trt.__version__} {precision}, {self.note}"

    # ------------------------------------------------------------------ 엔진 파일
    @staticmethod
    def _cached_engine(folder: Path, key: dict):
        """저장해 둔 엔진이 지금의 모델·TensorRT 와 맞으면 (바이트, 설명). 아니면 (None, 이유)."""
        engine, info = folder / ENGINE_FILE, folder / ENGINE_INFO
        if not engine.is_file() or not info.is_file():
            return None, "저장된 엔진 없음"
        try:
            saved = json.loads(info.read_text(encoding="utf-8"))
            if saved != key:
                return None, "모델 또는 TensorRT 가 바뀜"
            return engine.read_bytes(), "저장된 엔진을 읽음"
        except (OSError, ValueError):
            return None, "저장된 엔진을 읽을 수 없음"

    @staticmethod
    def _save_engine(folder: Path, blob: bytes, key: dict) -> None:
        """엔진을 모델 폴더에 둔다. 쓸 수 없는 폴더라도 달리는 데는 지장이 없다 (다음에 다시 만들 뿐이다)."""
        part = folder / (ENGINE_FILE + ".part")
        try:
            (folder / ENGINE_INFO).unlink(missing_ok=True)      # 엔진만 바뀌고 설명이 옛것인 순간이 없게: 설명을 먼저 지운다
            part.write_bytes(blob)
            os.replace(part, folder / ENGINE_FILE)
            (folder / ENGINE_INFO).write_text(json.dumps(key, indent=2), encoding="utf-8")
        except OSError as e:
            try:
                part.unlink(missing_ok=True)
            except OSError:                                 # 읽기 전용 폴더에서는 지우는 것도 실패한다
                pass
            log.warning("엔진을 %s 에 저장하지 못했습니다 (다음 실행에서 다시 만듭니다): %s", folder, e)

    # ------------------------------------------------------------------ 입출력 버퍼
    def _open_io(self, bundle: ModelBundle) -> None:
        trt, engine, cudart = self._trt, self._engine, self._cudart
        self._context = engine.create_execution_context()
        if self._context is None:
            raise EngineError("TensorRT 실행 문맥을 만들지 못했습니다 (GPU 메모리 부족일 수 있습니다)")
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        if sorted(names) != sorted((INPUT_NAME, OUTPUT_NAME)):
            raise EngineError(f"모델의 입출력 이름이 약속과 다릅니다: {names} (기대: {INPUT_NAME}, {OUTPUT_NAME})")
        want = {INPUT_NAME: bundle.input_shape, OUTPUT_NAME: (1, len(bundle.classes))}
        for name in names:
            shape = tuple(engine.get_tensor_shape(name))
            dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
            if shape != tuple(want[name]) or dtype != np.float32:
                raise EngineError(f"모델의 {name} 모양이 model_meta.json 과 다릅니다: {shape} {dtype} (기대 {tuple(want[name])} float32)")
            ptr = ctypes.c_void_p()
            nbytes = int(np.prod(shape)) * dtype.itemsize
            _cuda(cudart.cudaMalloc(ctypes.byref(ptr), nbytes), "cudaMalloc")
            self._buffers[name] = (ptr, nbytes)
            if not self._context.set_tensor_address(name, ptr.value):
                raise EngineError(f"TensorRT 에 {name} 버퍼를 연결하지 못했습니다")
        # 전용 스트림을 쓴다: 기본 스트림(0)으로도 돌지만 TensorRT 가 불필요한 동기화를 한다고 경고한다
        _cuda(cudart.cudaStreamCreate(ctypes.byref(self._stream)), "cudaStreamCreate")
        self._out = np.empty((1, len(bundle.classes)), dtype=np.float32)

    def infer(self, x: np.ndarray) -> np.ndarray:
        cudart, stream = self._cudart, self._stream
        (pin, nin), (pout, nout) = self._buffers[INPUT_NAME], self._buffers[OUTPUT_NAME]
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.nbytes != nin:
            raise EngineError(f"입력 크기가 모델과 다릅니다: {x.shape}")
        _cuda(cudart.cudaMemcpyAsync(pin, x.ctypes.data_as(ctypes.c_void_p), nin, H2D, stream), "입력 복사")
        if not self._context.execute_async_v3(stream.value):
            raise EngineError("TensorRT 실행 실패 (execute_async_v3)")
        _cuda(cudart.cudaMemcpyAsync(self._out.ctypes.data_as(ctypes.c_void_p), pout, nout, D2H, stream), "출력 복사")
        _cuda(cudart.cudaStreamSynchronize(stream), "동기화")
        return self._out[0].copy()

    def close(self) -> None:
        cudart = self._cudart
        for ptr, _ in self._buffers.values():
            cudart.cudaFree(ptr)
        self._buffers = {}
        if self._stream:
            cudart.cudaStreamDestroy(self._stream)
            self._stream = ctypes.c_void_p()
        self._context = self._engine = self._runtime = None      # 실행 문맥 → 엔진 → 런타임 순서로 놓는다


def open_backend(bundle: ModelBundle, kind: str = "auto", precision: str = "fp32",
                 say: Callable[[str], None] = lambda _m: None) -> Backend:
    """kind: auto(tensorrt 가 있으면 그것, 없으면 onnxruntime) | tensorrt | onnxruntime"""
    if kind == "auto":
        try:
            import tensorrt  # noqa: F401
            kind = "tensorrt"
        except ImportError:
            kind = "onnxruntime"
    if kind == "tensorrt":
        return TensorRTBackend(bundle, precision, say)
    if kind == "onnxruntime":
        return OnnxRuntimeBackend(bundle)
    raise EngineError(f"알 수 없는 백엔드: {kind!r} (auto | tensorrt | onnxruntime)")


# ---------------------------------------------------------------------------
# 엔진만 확인:  uv run python -m inference.engine models/<실행 이름> [--backend ...] [--precision ...]
# 카메라도 차량도 열지 않는다. 엔진을 만들고(또는 읽고) 무작위 입력으로 추론 시간을 잰다.
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="python -m inference.engine", description="모델 폴더로 엔진을 만들고 추론 시간을 잰다 (차는 움직이지 않는다)")
    p.add_argument("model", help="models/<실행 이름>")
    p.add_argument("--backend", choices=["auto", "tensorrt", "onnxruntime"], default="auto")
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    p.add_argument("--runs", type=int, default=300)
    args = p.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    backend: Optional[Backend] = None
    try:
        bundle = load_bundle(args.model)
        print(f"모델 {bundle.run_id}: 조향 단계 {bundle.classes}, 입력 {list(bundle.input_shape)}")
        backend = open_backend(bundle, args.backend, args.precision, say=print)
        print(f"백엔드: {backend.note}")
        rng = np.random.default_rng(0)
        x = rng.random(bundle.input_shape, dtype=np.float32)
        first = backend.infer(x)
        if first.shape != (len(bundle.classes),) or not np.isfinite(first).all():
            raise EngineError(f"출력이 올바르지 않습니다: {first!r}")
        times = []
        for _ in range(max(1, args.runs)):
            t0 = time.perf_counter()
            backend.infer(x)
            times.append((time.perf_counter() - t0) * 1e3)
        times.sort()
        print(f"출력 {[round(float(v), 4) for v in first]} → {bundle.classes[int(first.argmax())]}°")
        print(f"추론 시간 ({len(times)}회): 가운데 값 {times[len(times) // 2]:.2f} ms, 가장 느린 것 {times[-1]:.2f} ms")
        return 0
    except (BundleError, EngineError) as e:
        print(f"오류: {e}", file=sys.stderr)
        return 1
    finally:
        if backend is not None:
            backend.close()


if __name__ == "__main__":
    sys.exit(main())
