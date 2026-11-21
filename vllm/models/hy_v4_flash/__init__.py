# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HYV4 Flash model entry points; configuration imports remain GPU-independent.

NVIDIA and ROCm implementations live in independent `nvidia/` and `amd/`
subpackages; this module dispatches on the current platform.
"""

from importlib import import_module
from typing import Any

__all__ = ["HYV4FlashForCausalLM", "HYV4FlashVLForConditionalGeneration"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from vllm.platforms import current_platform

    platform = "amd" if current_platform.is_rocm() else "nvidia"
    modules = {
        "HYV4FlashForCausalLM": f".{platform}.model",
        "HYV4FlashVLForConditionalGeneration": f".{platform}.vl_model",
    }
    model_cls = getattr(import_module(modules[name], __name__), name)
    globals()[name] = model_cls
    return model_cls
