# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
from collections.abc import Sequence
from concurrent.futures.process import ProcessPoolExecutor
from functools import cache
from typing import Any

import torch


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
    # TODO: Add more requirements for UVA if needed.
    return is_pin_memory_available()
