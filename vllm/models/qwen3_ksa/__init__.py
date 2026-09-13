# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3 KSA model package."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .nvidia.model import Qwen3KSAForCausalLM


def __getattr__(name: str) -> Any:
    if name == "Qwen3KSAForCausalLM":
        from vllm.platforms import current_platform

        if current_platform.is_rocm():
            raise NotImplementedError("Qwen3 KSA does not yet support ROCm")
        if current_platform.is_xpu():
            raise NotImplementedError("Qwen3 KSA does not yet support XPU")

        from .nvidia.model import Qwen3KSAForCausalLM

        return Qwen3KSAForCausalLM
    raise AttributeError(name)


__all__ = ["Qwen3KSAForCausalLM"]
