# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .nvidia.model import MiniCPM5MoEForCausalLM

__all__ = ["MiniCPM5MoEForCausalLM"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from vllm.platforms import current_platform

    if current_platform.is_rocm():
        raise NotImplementedError("MiniCPM5 MoE does not yet support AMD/ROCm.")

    from .nvidia.model import MiniCPM5MoEForCausalLM

    globals()[name] = MiniCPM5MoEForCausalLM
    return MiniCPM5MoEForCausalLM
