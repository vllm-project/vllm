# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
from collections.abc import Sequence
from concurrent.futures.process import ProcessPoolExecutor
from functools import cache
from typing import Any

import regex as re
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


def verify_uva_coherence() -> None:
    """Verify that UVA write-after-map coherence works on this platform.

    Allocates a small pinned tensor, obtains a device view via
    ``get_accelerator_view_from_cpu_tensor``, writes a known pattern on
    the CPU side, and checks that the GPU view reflects the update.

    Raises:
        RuntimeError: If the GPU view does not reflect the CPU write,
            indicating the UVA zero-copy alias is broken (e.g. under GPU
            Confidential Computing with a stale driver classification).
    """
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    probe = torch.zeros(8, dtype=torch.int32, device="cpu", pin_memory=True)
    gpu_view = get_accelerator_view_from_cpu_tensor(probe)

    pattern = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int32)
    probe[:] = pattern

    readback = gpu_view.cpu()
    if not torch.equal(readback, pattern):
        raise RuntimeError(
            "UVA coherence check failed: a CPU write to pinned memory is not "
            "visible through the device view.  This typically occurs under GPU "
            "Confidential Computing when the CUDA runtime misclassifies the "
            "host pointer.  As a workaround, set VLLM_USE_V2_MODEL_RUNNER=0.  "
            f"Expected {pattern.tolist()}, got {readback.tolist()}."
        )
