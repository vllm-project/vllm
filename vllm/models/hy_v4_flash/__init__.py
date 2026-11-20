# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HYV4 Flash model entry points; configuration imports remain GPU-independent."""

from importlib import import_module
from typing import Any

__all__ = ["HYV4FlashForCausalLM", "HYV4FlashVLForConditionalGeneration"]


def __getattr__(name: str) -> Any:
    modules = {
        "HYV4FlashForCausalLM": ".nvidia.model",
        "HYV4FlashVLForConditionalGeneration": ".nvidia.vl_model",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    model_cls = getattr(import_module(modules[name], __name__), name)
    globals()[name] = model_cls
    return model_cls
