# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 model — hardware-isolated entry point.

The actual implementation lives under `nvidia/` and `amd/`; this module
picks the right one for the current platform on first attribute access, so
that importing `quant_config` does not pull in the model code.
"""

import importlib
from typing import TYPE_CHECKING, Any

from .quant_config import DeepseekV4FP8Config

if TYPE_CHECKING:
    from .common.vl_model import DeepseekV4ForConditionalGeneration
    from .nvidia.dspark import DSparkDeepseekV4ForCausalLM
    from .nvidia.model import DeepseekV4ForCausalLM
    from .nvidia.mtp import DeepSeekV4MTP

__all__ = [
    "DSparkDeepseekV4ForCausalLM",
    "DeepSeekV4MTP",
    "DeepseekV4FP8Config",
    "DeepseekV4ForCausalLM",
    "DeepseekV4ForConditionalGeneration",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from vllm.platforms import current_platform

    if current_platform.is_rocm():
        impl, vl = "amd", "common.vl_model"
    elif current_platform.is_xpu():
        impl, vl = "xpu", "vl_stub"
    elif current_platform.is_cpu():
        impl, vl = "cpu", "vl_stub"
    else:
        impl, vl = "nvidia", "common.vl_model"
    module = {
        "DSparkDeepseekV4ForCausalLM": f".{impl}.dspark",
        "DeepSeekV4MTP": f".{impl}.mtp",
        "DeepseekV4ForCausalLM": f".{impl}.model",
        "DeepseekV4ForConditionalGeneration": f".{vl}",
    }[name]
    return getattr(importlib.import_module(module, __name__), name)
