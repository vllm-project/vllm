# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import multiprocessing
from collections.abc import Sequence
from concurrent.futures.process import ProcessPoolExecutor
from functools import cache
from typing import Any

import torch

# CUDA Driver API CUresult value for CUDA_ERROR_NOT_INITIALIZED.
CUDA_ERROR_NOT_INITIALIZED = 3


@cache
def _load_cuda_driver() -> Any | None:
    """Load libcuda and declare the driver probe signature."""
    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return None

    libcuda.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    libcuda.cuDeviceGetCount.restype = ctypes.c_int
    return libcuda


def _cuda_driver_is_initialized() -> bool:
    """Check if the CUDA Driver API has already been initialized."""
    libcuda = _load_cuda_driver()
    if libcuda is None:
        return False

    device_count = ctypes.c_int()
    result = libcuda.cuDeviceGetCount(ctypes.byref(device_count))
    # cuDeviceGetCount returns CUDA_ERROR_NOT_INITIALIZED before cuInit()
    # without initializing the driver. Any other return means the driver
    # state is initialized or ambiguous, so avoid forking after it.
    return result != CUDA_ERROR_NOT_INITIALIZED


def cuda_is_initialized() -> bool:
    """Check if CUDA is initialized at the runtime or driver level."""
    if not torch.cuda._is_compiled():
        return False
    return torch.cuda.is_initialized() or _cuda_driver_is_initialized()


def xpu_is_initialized() -> bool:
    """Check if XPU is initialized."""
    if not torch.xpu._is_compiled():
        return False
    return torch.xpu.is_initialized()


def cuda_get_device_properties(
    device, names: Sequence[str], init_cuda=False
) -> tuple[Any, ...]:
    """Get CUDA device properties without initializing CUDA when possible."""
    if init_cuda or cuda_is_initialized():
        props = torch.cuda.get_device_properties(device)
        return tuple(getattr(props, name) for name in names)

    # Run in subprocess to avoid initializing CUDA as a side effect.
    mp_ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=1, mp_context=mp_ctx) as executor:
        return executor.submit(cuda_get_device_properties, device, names, True).result()


@cache
def is_pin_memory_available() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_pin_memory_available()


@cache
def is_uva_available() -> bool:
    """Check if Unified Virtual Addressing (UVA) is available."""
    # UVA requires pinned memory.
    from vllm.platforms import current_platform

    # TODO: Add more requirements for UVA if needed.
    return is_pin_memory_available() or current_platform.is_cpu()


@cache
def num_compute_units(device_id: int = 0) -> int:
    """Get the number of compute units of the current device."""
    from vllm.platforms import current_platform

    return current_platform.num_compute_units(device_id)
