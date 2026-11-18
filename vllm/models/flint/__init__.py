# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flint model entry point."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .nvidia.model import FlintForCausalLM

__all__ = ["FlintForCausalLM"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from vllm.platforms import current_platform

    if current_platform.is_rocm():
        from .amd.model import FlintForCausalLM
    else:
        from .nvidia.model import FlintForCausalLM

    return FlintForCausalLM
