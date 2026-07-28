# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model parameter offloading infrastructure."""

from vllm.model_executor.offloader.base import (
    BaseOffloader,
    NoopOffloader,
    create_offloader,
    get_offloader,
    set_offloader,
    should_pin_memory,
)
from vllm.model_executor.offloader.prefetch import PrefetchOffloader
from vllm.model_executor.offloader.uva import UVAOffloader

# HierarchicalOffloader is imported lazily via create_offloader to avoid
# circular imports through models.utils at package import time.

__all__ = [
    "BaseOffloader",
    "NoopOffloader",
    "UVAOffloader",
    "PrefetchOffloader",
    "HierarchicalOffloader",
    "create_offloader",
    "get_offloader",
    "set_offloader",
    "should_pin_memory",
]


def __getattr__(name: str):
    if name == "HierarchicalOffloader":
        from vllm.model_executor.offloader.hierarchical_offloader import (
            HierarchicalOffloader,
        )

        return HierarchicalOffloader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
