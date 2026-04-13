# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import shutil
from pathlib import Path


def find_cuda_home_0():
    from torch.utils.cpp_extension import CUDA_HOME

    return CUDA_HOME


def find_cuda_home_1():
    """Find CUDA_HOME directory.

    Returns:
        Path to CUDA_HOME if found, None otherwise.
    """
    # First check CUDA_HOME environment variable
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home and Path(cuda_home).exists():
        return Path(cuda_home)

    # Try to find nvcc on PATH
    nvcc_path = shutil.which("nvcc")
    if nvcc_path is None:
        return None

    # nvcc is typically at CUDA_HOME/bin/nvcc
    # So we go up two directories
    nvcc_path = Path(nvcc_path).resolve()
    cuda_home = nvcc_path.parent.parent

    return cuda_home


def find_cublaslt_header(cuda_home):
    if cuda_home is None:
        return None

    cuda_home = Path(cuda_home)

    # Search recursively for cublasLt.h
    for path in cuda_home.rglob("cublasLt.h"):
        return path

    return None


def findit():
    cuda_home_0 = find_cuda_home_0()
    cuda_home_1 = find_cuda_home_1()

    print(f"CUDA_HOME (torch): {cuda_home_0}")
    print(f"CUDA_HOME (nvcc): {cuda_home_1}")

    # Find cublasLt.h
    for name, cuda_home in [("torch", cuda_home_0), ("nvcc", cuda_home_1)]:
        cublaslt_path = find_cublaslt_header(cuda_home)
        if cublaslt_path:
            print(f"cublasLt.h ({name}): {cublaslt_path} FOUND")
        else:
            print(f"cublasLt.h ({name}): NOT FOUND")
