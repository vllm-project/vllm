# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build DeepGEMM's stable-ABI extension for vendoring into vLLM."""

import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import torch
from torch.utils import cpp_extension

if len(sys.argv) != 3:
    sys.exit(f"usage: {sys.argv[0]} <SRC> <OUT>")

src = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
out.mkdir(parents=True, exist_ok=True)

cuda_home = cpp_extension.CUDA_HOME
if cuda_home is None:
    sys.exit("CUDA_HOME not found; cannot build DeepGEMM _C")
# CCCL lives outside the standard CUDAToolkit search (mirrors DeepGEMM's setup.py).
includes = [
    sysconfig.get_path("include"),
    f"{cuda_home}/include",
    f"{cuda_home}/include/cccl",
    str(src / "csrc"),
    str(src / "deep_gemm/include"),
    str(src / "third-party/cutlass/include"),
    str(src / "third-party/cutlass/tools/util/include"),
    str(src / "third-party/deep_jit/include"),
    *cpp_extension.include_paths(device_type="cuda"),
]

cmd = [
    os.environ.get("CXX", "g++"),
    "-shared",
    "-fPIC",
    "-std=c++20",
    "-O3",
    "-g0",
    "-Wno-psabi",
    "-Wno-deprecated-declarations",
    "-DPy_LIMITED_API=0x030a0000",
    "-DTORCH_TARGET_VERSION=0x020b000000000000",
    "-DDJ_DISABLE_GIL=1",
    "-DUSE_CUDA",
    f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}",
    *(f"-I{p}" for p in includes),
    str(src / "csrc/python_api.cpp"),
    *(f"-L{p}" for p in cpp_extension.library_paths(device_type="cuda")),
    f"-L{cuda_home}/lib64",
    "-ltorch",
    "-ltorch_cpu",
    "-ltorch_cuda",
    "-lc10",
    "-lc10_cuda",
    "-lcudart",
    "-lnvrtc",
    "-o",
    str(out / "_C_extension.abi3.so"),
]
print("[build_deepgemm_C] " + " ".join(cmd), flush=True)
subprocess.check_call(cmd)
