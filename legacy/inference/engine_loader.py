# inference/engine_loader.py

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # CUDA context 자동 생성
import numpy as np


class TRTInferenceEngine:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.INFO)

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        self.input_idx = self.engine.get_binding_index("input")
        self.output_idx = self.engine.get_binding_index("output")

        input_shape = self.engine.get_binding_shape(self.input_idx)
        output_shape = self.engine.get_binding_shape(self.output_idx)

        self.input_size = int(np.prod(input_shape))
        self.output_size = int(np.prod(output_shape))

        self.d_input = cuda.mem_alloc(self.input_size * np.float32().nbytes)
        self.d_output = cuda.mem_alloc(self.output_size * np.float32().nbytes)

        self.h_output = np.empty(self.output_size, dtype=np.float32)

        self.stream = cuda.Stream()

    def infer(self, input_np):
        """
        input_np: (1, 3, 66, 200) float32
        """
        # ★ dtype 및 contiguous 보장
        input_np = np.asarray(input_np, dtype=np.float32)
        if not input_np.flags["C_CONTIGUOUS"]:
            input_np = np.ascontiguousarray(input_np)

        cuda.memcpy_htod_async(self.d_input, input_np, self.stream)

        self.context.execute_async_v2(
            bindings=[int(self.d_input), int(self.d_output)],
            stream_handle=self.stream.handle,
        )

        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

        return self.h_output.reshape(1, -1)  # (1, num_classes)
