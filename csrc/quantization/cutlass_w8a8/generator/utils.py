"""
Utils used in generating cutlass kernels.
"""

import os
from pathlib import Path
from typing import Tuple

## Utilities ####


def to_torch_dtype_str(dtype_str):
    if dtype_str == "int8":
        return "torch::kInt8"
    if dtype_str == "fp8":
        return "torch::kFloat8_e4m3fn"
    raise ValueError("unknown type")


def to_cutlass_dtype_str(dtype_str):
    if dtype_str == "int8":
        return "int8_t"
    if dtype_str == "fp8":
        return "cutlass::float_e4m3_t"
    raise ValueError("unknown type")


def get_script_dir() -> Path:
    return Path(os.path.dirname(os.path.realpath(__file__)))


def get_as_cutlass_gemm_shape(shape: Tuple[int, int, int]):
    return f'cutlass::gemm::GemmShape<{shape[0]}, {shape[1]}, {shape[2]}>'


def get_as_cutlass3x_gemm_shape(shape: Tuple[int, int, int]):
    return f'Shape<_{shape[0]}, _{shape[1]}, _{shape[2]}>'


def file_contents_same(filepath, contents):
    if not Path(filepath).exists():
        return

    f_contents = None
    with open(filepath, "r") as f:
        f_contents = f.read()

    return f_contents == contents
