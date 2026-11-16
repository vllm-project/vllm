# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 hardware-isolated model entry point."""

import importlib
from typing import TYPE_CHECKING, Any

from .quant_config import DeepseekV4FP8Config

if TYPE_CHECKING:
    from .nvidia.dspark import DSparkDeepseekV4ForCausalLM
    from .nvidia.vl_model import DeepseekV41ForCausalLM

__all__ = [
    "DSparkDeepseekV4ForCausalLM",
    "DeepseekV4FP8Config",
    "DeepseekV41ForCausalLM",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from vllm.platforms import current_platform

    impl = "amd" if current_platform.is_rocm() else "nvidia"
    module = {
        "DSparkDeepseekV4ForCausalLM": f".{impl}.dspark",
        "DeepseekV41ForCausalLM": f".{impl}.vl_model",
    }[name]
    return getattr(importlib.import_module(module, __name__), name)
