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

__all__ = [
    "BaseOffloader",
    "NoopOffloader",
    "UVAOffloader",
    "PrefetchOffloader",
    "create_offloader",
    "get_offloader",
    "set_offloader",
    "should_pin_memory",
]
