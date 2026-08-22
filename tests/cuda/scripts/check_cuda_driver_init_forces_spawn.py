#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that driver-level CUDA init is detected before fork."""

import ctypes
import os

import torch

os.environ.pop("VLLM_WORKER_MULTIPROC_METHOD", None)

from vllm.utils.platform_utils import cuda_is_initialized  # noqa: E402

assert not torch.cuda.is_initialized(), "CUDA initialized before driver probe"
assert not cuda_is_initialized(), "driver probe should not initialize CUDA"
assert not torch.cuda.is_initialized(), "driver probe initialized PyTorch CUDA"

libcuda = ctypes.CDLL("libcuda.so.1")
libcuda.cuInit.argtypes = [ctypes.c_uint]
libcuda.cuInit.restype = ctypes.c_int
assert libcuda.cuInit(0) == 0, "cuInit failed"

assert not torch.cuda.is_initialized(), "cuInit initialized PyTorch CUDA"
assert cuda_is_initialized(), "driver-level CUDA init was not detected"

from vllm.utils.system_utils import _maybe_force_spawn  # noqa: E402

_maybe_force_spawn()
assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"

print("OK")
