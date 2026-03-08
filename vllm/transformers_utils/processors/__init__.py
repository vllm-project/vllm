# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Multi-modal processors may be defined in this directory for the following
reasons:

- There is no processing file defined by HF Hub or Transformers library.
- There is a need to override the existing processor to support vLLM.
"""

import importlib

from vllm.transformers_utils.processors.bagel import BagelProcessor
from vllm.transformers_utils.processors.deepseek_vl2 import DeepseekVLV2Processor
from vllm.transformers_utils.processors.fireredasr2_processor import (
    FireRedASR2Processor,
)
from vllm.transformers_utils.processors.funasr_processor import FunASRProcessor
from vllm.transformers_utils.processors.hunyuan_vl import HunYuanVLProcessor
from vllm.transformers_utils.processors.hunyuan_vl_image import HunYuanVLImageProcessor
from vllm.transformers_utils.processors.kimi_audio import KimiAudioProcessor
from vllm.transformers_utils.processors.ovis import OvisProcessor
from vllm.transformers_utils.processors.ovis2_5 import Ovis2_5Processor

__all__ = [
    "BagelProcessor",
    "DeepseekVLV2Processor",
    "FireRedASR2Processor",
    "FunASRProcessor",
    "HunYuanVLProcessor",
    "HunYuanVLImageProcessor",
    "KimiAudioProcessor",
    "OvisProcessor",
    "Ovis2_5Processor",
    "Qwen3ASRProcessor",
]

_CLASS_TO_MODULE: dict[str, str] = {
    "KimiAudioProcessor": "vllm.transformers_utils.processors.kimi_audio",
}


def __getattr__(name: str):
    if name in _CLASS_TO_MODULE:
        module_name = _CLASS_TO_MODULE[name]
        module = importlib.import_module(module_name)
        return getattr(module, name)

    raise AttributeError(f"module 'processors' has no attribute '{name}'")


def __dir__():
    return sorted(list(__all__))
