# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 hardware-isolated model entry point."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .nvidia.dspark import DSparkDeepseekV4ForCausalLM
    from .nvidia.vl_model import DeepseekV41ForCausalLM
    from .quant_config import DeepseekV4FP8Config

__all__ = [
    "DSparkDeepseekV4ForCausalLM",
    "DeepseekV4FP8Config",
    "DeepseekV41ForCausalLM",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name == "DeepseekV4FP8Config":
        from .quant_config import DeepseekV4FP8Config

        return DeepseekV4FP8Config

    from vllm.platforms import current_platform

    backend = "amd" if current_platform.is_rocm() else "nvidia"
    module = "dspark" if name == "DSparkDeepseekV4ForCausalLM" else "vl_model"
    return getattr(import_module(f".{backend}.{module}", __name__), name)
