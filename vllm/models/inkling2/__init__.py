# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2: the model and its shared Triton kernels (with their CUDA fast
paths) live under `common/`; `nvidia/` holds the CUDA-only features and
kernels (Blackwell / SM12x / Humming kernels, fused all-reduce, vision
tower, DSpark draft, LoRA) and the CUDA entry modules `nvidia.model` /
`nvidia.loop` (aliases of the `common/` modules). Configs, processing,
quantization metadata and the DSpark speculator are shared.

The model classes load on first access from the platform package, as in
`muse_spark`: `_MODULES` holds platform-relative module names."""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .common.loop import Inkling2LoopForConditionalGeneration
    from .common.model import Inkling2ForConditionalGeneration
    from .nvidia.dspark import Inkling2DSparkModel

__all__ = [
    "Inkling2DSparkModel",
    "Inkling2ForConditionalGeneration",
    "Inkling2LoopForConditionalGeneration",
]

_MODULES = {
    "Inkling2DSparkModel": "dspark",
    "Inkling2ForConditionalGeneration": "model",
    "Inkling2LoopForConditionalGeneration": "loop",
}


def _platform_package() -> str:
    """The package of this platform's entry modules: `nvidia` (a ROCm
    `amd` package, vllm-project/vllm#67685, would be selected here)."""
    return "nvidia"


def __getattr__(name: str):
    module = _MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    package = _platform_package()
    return getattr(importlib.import_module(f".{package}.{module}", __name__), name)
