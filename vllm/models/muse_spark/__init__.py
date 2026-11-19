# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark HF model entry point.

The model classes load on first access, so importing a submodule such as
`common.vision` doesn't import the NVIDIA model. Its kernels import FlashInfer,
which queries the GPU at import and so initializes CUDA in the importing
process; processes forked from it afterwards can't use CUDA.
"""

import importlib
from typing import TYPE_CHECKING

from vllm.platforms import current_platform

if TYPE_CHECKING:
    from .nvidia.dspark import MuseSparkDSparkForCausalLM
    from .nvidia.model import MuseSparkForCausalLM
    from .nvidia.multimodal import MuseSparkForConditionalGeneration

__all__ = [
    "MuseSparkForCausalLM",
    "MuseSparkForConditionalGeneration",
    "MuseSparkDSparkForCausalLM",
]

_MODULES = {
    "MuseSparkForCausalLM": "model",
    "MuseSparkForConditionalGeneration": "multimodal",
    "MuseSparkDSparkForCausalLM": "dspark",
}


def __getattr__(name: str):
    module = _MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    platform = "amd" if current_platform.is_rocm() else "nvidia"
    return getattr(importlib.import_module(f".{platform}.{module}", __name__), name)
