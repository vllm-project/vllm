from __future__ import annotations

import argparse
from math import prod
from pathlib import Path
import ctypes
import logging
import numpy as np


import gguf
from gguf.constants import GGMLQuantizationType


logger = logging.getLogger(__name__)


c_float_p = ctypes.POINTER(ctypes.c_float)


class ggml_init_params(ctypes.Structure):
    _fields_ = [
        ("mem_size", ctypes.c_size_t),
        ("mem_buffer", ctypes.c_void_p),
        ("no_alloc", ctypes.c_bool),
    ]


class GGMLQuants:
    libggml: ctypes.CDLL

    def __init__(self, libggml: Path):
        self.libggml = ctypes.CDLL(str(libggml), winmode=0)
        # self.libggml = ctypes.WinDLL(str(libggml), winmode=0)
        self.libggml.ggml_quantize_chunk.restype = ctypes.c_size_t
        # enum ggml_type   type,
        #    const float * src,
        #           void * dst,
        #        int64_t   start,
        #        int64_t   nrows,
        #        int64_t   n_per_row,
        #    const float * imatrix) {
        self.libggml.ggml_quantize_chunk.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
        )

        self.libggml.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
        self.libggml.ggml_quantize_requires_imatrix.argtypes = (ctypes.c_int,)

        for t in (
            "q4_0", "q4_1", "q5_0", "q5_1", "q8_0",
            "q2_K", "q3_K", "q4_K", "q5_K", "q6_K"
        ):
            dequant_func: ctypes._NamedFuncPointer = getattr(self.libggml, "dequantize_row_" + t)
            dequant_func.restype = None
            dequant_func.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64)

        self.libggml.ggml_fp16_to_fp32_row.restype = None
        self.libggml.ggml_fp16_to_fp32_row.argtypes = (ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_float), ctypes.c_int64)
        self.libggml.ggml_bf16_to_fp32_row.restype = None
        self.libggml.ggml_bf16_to_fp32_row.argtypes = (ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_float), ctypes.c_int64)

        self.libggml.ggml_init.argtypes = (ggml_init_params,)

        self.libggml.ggml_init(ggml_init_params(1 * 1024 * 1024, 0, False))

    def dequantize(self, tensor: np.ndarray, qtype: GGMLQuantizationType) -> np.ndarray:
        result = np.zeros(gguf.quant_shape_from_byte_shape(tensor.shape, qtype), dtype=np.float32, order="C")
        if qtype == GGMLQuantizationType.F32:
            # no-op
            result = tensor.view(np.float32)
        elif qtype == GGMLQuantizationType.F16:
            self.libggml.ggml_fp16_to_fp32_row(tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), result.ctypes.data_as(c_float_p), result.size)
        elif qtype == GGMLQuantizationType.BF16:
            self.libggml.ggml_bf16_to_fp32_row(tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), result.ctypes.data_as(c_float_p), result.size)
        else:
            lw_qname = qtype.name.lower()
            if lw_qname[-1] == "k":
                lw_qname = lw_qname[:-1] + "K"
            dequant_func: ctypes._NamedFuncPointer = getattr(self.libggml, "dequantize_row_" + lw_qname)
            dequant_func(tensor.ctypes.data_as(ctypes.c_void_p), result.ctypes.data_as(c_float_p), result.size)
        return result

    def quantize(self, data: np.ndarray, qtype: GGMLQuantizationType) -> np.ndarray:
        result = np.zeros(gguf.quant_shape_to_byte_shape(data.shape, qtype), dtype=np.uint8, order="C")
        if self.libggml.ggml_quantize_requires_imatrix(qtype.value):
            # TODO: is a column-wise sum of squares appropriate?
            qw = np.sum((data * data).reshape((-1, data.shape[-1])), axis=0).ctypes.data_as(c_float_p)
        else:
            qw = ctypes.cast(0, c_float_p)
        result_size = self.libggml.ggml_quantize_chunk(qtype.value, data.ctypes.data_as(c_float_p), result.ctypes.data_as(ctypes.c_void_p), 0, prod(data.shape[:-1]), data.shape[-1], qw)
        assert result.size == result_size
        return result


def create_sample(ggml_quants: GGMLQuants, hidden_size, qtype: GGMLQuantizationType) -> np.ndarray:
    gguf_writer = gguf.GGUFWriter(f"Quant_{qtype.name}_{hidden_size}.gguf", "llama")

    # Create a sample tensor
    size = 1024
    tensor = np.random.normal(0, 0.5,(256, hidden_size*2, size)).astype(np.float32)
    tensor = np.clip(tensor, -1, 1)
    print("generated up proj", tensor)
    shape_str = "x".join(map(str, tensor.shape))
    gguf_writer.add_tensor(f"tensor_{qtype.name}_{shape_str}", ggml_quants.quantize(tensor, qtype), raw_dtype=qtype)

    # tensor = np.random.randn(256, size, hidden_size).astype(np.float32)
    tensor = np.random.normal(0, 0.5, (256, size, hidden_size)).astype(np.float32)
    print(tensor.max())
    print("generated down proj", tensor)
    shape_str = "x".join(map(str, tensor.shape))
    gguf_writer.add_tensor(f"tensor_{qtype.name}_{shape_str}", ggml_quants.quantize(tensor, qtype), raw_dtype=qtype)

    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()

    gguf_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Python (de)quantization against the reference C implementation")
    parser.add_argument("--libggml", type=Path, default="libggml.so", help="The path to libggml.so")
    parser.add_argument("--hidden_size", type=int, default=512, help="The hidden size of the sample tensor")
    parser.add_argument("--seed", type=int, default=0, help="The hidden size of the sample tensor")

    np.random.seed(0)

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    ggml_quants = GGMLQuants(args.libggml)

    qtypes = [
        # GGMLQuantizationType.Q2_K,
        # GGMLQuantizationType.Q3_K,
        # GGMLQuantizationType.Q4_K,
        # GGMLQuantizationType.Q5_K,
        # GGMLQuantizationType.Q6_K,
        # GGMLQuantizationType.Q4_0,
        # GGMLQuantizationType.Q5_0,
        GGMLQuantizationType.Q8_0,
    ]

    for qtype in qtypes:
        create_sample(ggml_quants, args.hidden_size, qtype)
