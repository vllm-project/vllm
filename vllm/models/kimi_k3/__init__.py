# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 model entry point with lazily selected optimized backends."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .nvidia.model import KimiK3ForConditionalGeneration, KimiLinearForCausalLM
    from .nvidia.mtp import KimiK3MTP

__all__ = [
    "KimiK3ForConditionalGeneration",
    "KimiK3MTP",
    "KimiLinearForCausalLM",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)

    from vllm.platforms import current_platform

    if current_platform.is_rocm():
        from .amd.linear import KimiLinearForCausalLM
        from .amd.model import KimiK3ForConditionalGeneration
        from .amd.mtp import KimiK3MTP
    else:
        from .nvidia.model import (
            KimiK3ForConditionalGeneration,
            KimiLinearForCausalLM,
        )
        from .nvidia.mtp import KimiK3MTP

    exported = {
        "KimiK3ForConditionalGeneration": KimiK3ForConditionalGeneration,
        "KimiK3MTP": KimiK3MTP,
        "KimiLinearForCausalLM": KimiLinearForCausalLM,
    }
    globals().update(exported)
    return exported[name]
