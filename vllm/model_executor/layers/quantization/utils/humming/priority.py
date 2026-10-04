# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Humming kernel selection priority."""

from enum import Enum
from typing import TypeVar

_KernelT = TypeVar("_KernelT", bound=type | Enum)


def prioritize_humming(
    kernels: list[_KernelT],
    compute_capability: int | None = None,
) -> list[_KernelT]:
    """Swap Humming ahead of Marlin on SM90 without modifying the input list.

    Match kernel class names or MoE backend enum names. A missing compute
    capability uses the current device.
    """
    if compute_capability is None:
        from vllm.platforms import current_platform

        if current_platform.is_cuda():
            cc = current_platform.get_device_capability()
            compute_capability = cc.to_int() if cc is not None else None
    if compute_capability != 90:
        return kernels

    names = [
        (kernel.name if isinstance(kernel, Enum) else kernel.__name__).lower()
        for kernel in kernels
    ]
    humming = next((i for i, name in enumerate(names) if "humming" in name), None)
    marlin = next((i for i, name in enumerate(names) if "marlin" in name), None)
    if humming is not None and marlin is not None and humming > marlin:
        kernels = kernels.copy()
        kernels[humming], kernels[marlin] = kernels[marlin], kernels[humming]
    return kernels
