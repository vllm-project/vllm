# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import multiprocessing
from collections.abc import Sequence
from concurrent.futures.process import ProcessPoolExecutor
from functools import cache
from typing import Any

import regex as re
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def cuda_is_initialized() -> bool:
    """Check if CUDA is initialized."""
    if not torch.cuda._is_compiled():
        return False
    return torch.cuda.is_initialized()


def xpu_is_initialized() -> bool:
    """Check if XPU is initialized."""
    if not torch.xpu._is_compiled():
        return False
    return torch.xpu.is_initialized()


def cuda_get_device_properties(
    device, names: Sequence[str], init_cuda=False
) -> tuple[Any, ...]:
    """Get specified CUDA device property values without initializing CUDA in
    the current process."""
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


@cache
def get_device_name_as_file_name(device_id: int = 0) -> str:
    from vllm.platforms import current_platform

    name = current_platform.get_device_name(device_id)
    name = re.sub(r"[\s/]+", "_", name)
    return name


def parse_cuda_version(version: str | None) -> tuple[int, int] | None:
    """Parse a CUDA version string such as ``"12.9"`` or ``"12.9.86"`` into
    ``(major, minor)``. Returns None if the string cannot be parsed."""
    if not version:
        return None
    try:
        major, minor, *_ = version.split(".")
        return int(major), int(minor)
    except (ValueError, AttributeError):
        return None


@cache
def get_cuda_driver_version() -> tuple[int, int] | None:
    """Return the CUDA version supported by the loaded CUDA driver
    (``cuDriverGetVersion``) as ``(major, minor)``, or None if unavailable.

    This queries the user-mode driver library that CUDA actually uses, so it
    reflects CUDA forward-compatibility libraries when they are active. It
    does not create a CUDA context.
    """
    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
        version = ctypes.c_int(0)
        if libcuda.cuDriverGetVersion(ctypes.byref(version)) != 0:
            return None
    except (OSError, AttributeError):
        return None
    if version.value <= 0:
        return None
    # Encoded as 1000 * major + 10 * minor, e.g. 12080 -> (12, 8).
    return version.value // 1000, (version.value % 1000) // 10


@cache
def warn_if_cuda_driver_cannot_jit_ptx(kernel_name: str) -> bool:
    """Warn once per kernel family if the CUDA driver is older than the CUDA
    toolkit vLLM and PyTorch were built with.

    Kernels that ship as PTX for the current GPU (rather than native SASS) are
    JIT-compiled by the driver at load time. CUDA minor version compatibility
    does not cover PTX JIT, so a driver older than the build toolkit fails
    with ``cudaErrorUnsupportedPtxVersion`` ("the provided PTX was compiled
    with an unsupported toolchain"). Call this right before such kernels are
    first used so the user gets an actionable message instead of only the raw
    CUDA error.

    Returns True if a mismatch was detected (and a warning was logged).
    """
    toolkit_str = torch.version.cuda
    toolkit = parse_cuda_version(toolkit_str)
    driver = get_cuda_driver_version()
    if toolkit is None or driver is None or driver >= toolkit:
        return False
    driver_str = f"{driver[0]}.{driver[1]}"
    logger.warning(
        "The CUDA driver supports CUDA %s, but vLLM/PyTorch were built with "
        "CUDA %s. %s kernels that are compiled to PTX for this GPU must be "
        "JIT-compiled by the driver, which needs CUDA >= %s, and will fail "
        "with 'the provided PTX was compiled with an unsupported toolchain'. "
        "To keep using %s, do one of: upgrade the NVIDIA driver to one that "
        "supports CUDA >= %s; install a vLLM/PyTorch build for CUDA %s; or, "
        "on datacenter/professional GPUs, enable CUDA forward compatibility "
        "with VLLM_ENABLE_CUDA_COMPATIBILITY=1. See "
        "https://docs.vllm.ai/en/latest/usage/troubleshooting.html"
        "#cuda-error-the-provided-ptx-was-compiled-with-an-unsupported-toolchain",
        driver_str,
        toolkit_str,
        kernel_name,
        toolkit_str,
        kernel_name,
        toolkit_str,
        driver_str,
    )
    return True
