# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HYV4 Flash model entry points; configuration imports remain GPU-independent.

NVIDIA and ROCm implementations live in `nvidia/` and `amd/`, with
vendor-neutral code in `common/`; this module dispatches on the current
platform.
"""

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .common.vl_model import HYV4FlashVLForConditionalGeneration
    from .nvidia.model import HYV4FlashForCausalLM

__all__ = ["HYV4FlashForCausalLM", "HYV4FlashVLForConditionalGeneration"]


def platform_module(name: str) -> ModuleType:
    """Import a module from the current platform's subpackage.

    Args:
        name: Module name inside `amd/` or `nvidia/`, e.g. `"model"`.

    Returns:
        The imported module.

    """
    from vllm.platforms import current_platform

    platform = "amd" if current_platform.is_rocm() else "nvidia"
    return import_module(f".{platform}.{name}", __name__)


def __getattr__(name: str) -> Any:
    """Load model code only when requested, keeping config-only imports light."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name == "HYV4FlashVLForConditionalGeneration":
        module = import_module(".common.vl_model", __name__)
    else:
        module = platform_module("model")
    model_cls = getattr(module, name)
    globals()[name] = model_cls
    return model_cls
