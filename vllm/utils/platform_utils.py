# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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


_uva_available: bool | None = None


def is_uva_available() -> bool:
    """Check if Unified Virtual Addressing (UVA) is available."""
    global _uva_available
    if _uva_available is not None:
        return _uva_available

    from vllm.platforms import current_platform

    if current_platform.is_cpu():
        _uva_available = True
    # UVA requires pinned memory.
    elif not is_pin_memory_available():
        _uva_available = False
    elif _accelerator_is_initialized():
        _uva_available = _uva_alias_is_coherent()
    else:
        # No accelerator context to probe with yet. Answer optimistically
        # without caching; the worker-side call resolves it, and every consumer
        # falls back safely if it says no.
        return True
    return _uva_available


def _accelerator_is_initialized() -> bool:
    from vllm.platforms import current_platform

    if current_platform.is_xpu():
        return xpu_is_initialized()
    if current_platform.is_cuda_alike():
        return cuda_is_initialized()
    return False


def _uva_alias_is_coherent() -> bool:
    """Check that a device view of pinned host memory tracks later host writes.

    Pinned memory is necessary but not sufficient. Under GPU Confidential
    Computing `pin_memory=True` can silently yield an unpinned tensor, so the
    device view is a detached copy rather than a live alias: host writes made
    after the view is created never reach the device and kernels read stale
    zeros. Fails closed, since the fallback paths are correct but slower.

    Returns:
        True if the device view reflects a host write made after its creation.

    """
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    expected = torch.arange(1, 9, dtype=torch.int32, device="cpu")
    try:
        host = torch.zeros(
            expected.shape, dtype=torch.int32, device="cpu", pin_memory=True
        )
        view = get_accelerator_view_from_cpu_tensor(host)
        host.copy_(expected)
        torch.accelerator.synchronize()
        actual = view.cpu()
    except Exception:
        logger.exception("UVA coherence probe failed; treating UVA as unavailable.")
        return False

    if torch.equal(actual, expected):
        return True
    logger.warning(
        "UVA reports available but the device view is not a live alias of host "
        "memory: wrote %s, device read %s. Falling back to explicit copies. "
        "This is expected under GPU Confidential Computing.",
        expected.tolist(),
        actual.tolist(),
    )
    return False


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
