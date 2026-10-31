# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Humming kernel selection priority."""

from enum import Enum
from typing import TypeVar

_KernelT = TypeVar("_KernelT", bound=type | Enum)

# Compute capabilities where Humming outranks Marlin.
_HUMMING_PREFERRED_CAPABILITIES = frozenset({90})


def prefers_humming(
    compute_capability: int | None = None,
    capabilities: frozenset[int] = _HUMMING_PREFERRED_CAPABILITIES,
) -> bool:
    """Whether Humming outranks Marlin. A missing compute capability uses the
    current device. ``capabilities`` lets a caller widen the preferred set for
    one kernel family without changing the global default."""
    if compute_capability is None:
        from vllm.platforms import current_platform

        if current_platform.is_cuda():
            cc = current_platform.get_device_capability()
            compute_capability = cc.to_int() if cc is not None else None
    return compute_capability in capabilities


def prioritize_humming(
    kernels: list[_KernelT],
    compute_capability: int | None = None,
    capabilities: frozenset[int] = _HUMMING_PREFERRED_CAPABILITIES,
) -> list[_KernelT]:
    """Move Humming directly ahead of Marlin where Humming is preferred.

    Every other entry keeps its relative order and the input list is not
    modified. Match kernel class names or MoE backend enum names.
    """
    if not prefers_humming(compute_capability, capabilities):
        return kernels

    names = [
        (kernel.name if isinstance(kernel, Enum) else kernel.__name__).lower()
        for kernel in kernels
    ]
    humming = next((i for i, name in enumerate(names) if "humming" in name), None)
    marlin = next((i for i, name in enumerate(names) if "marlin" in name), None)
    if humming is None or marlin is None or humming < marlin:
        return kernels
    kernels = kernels.copy()
    kernels.insert(marlin, kernels.pop(humming))
    return kernels
