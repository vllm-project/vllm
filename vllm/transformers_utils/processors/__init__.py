# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Multi-modal processors may be defined in this directory for the following
reasons:

- There is no processing file defined by HF Hub or Transformers library.
- There is a need to override the existing processor to support vLLM.
"""

import importlib

_CLASS_TO_MODULE: dict[str, str] = {
    "BagelProcessor": "vllm.transformers_utils.processors.bagel",
    "DeepseekVLV2Processor": "vllm.transformers_utils.processors.deepseek_vl2",
    "FireRedASR2Processor": "vllm.transformers_utils.processors.fireredasr2",
    "FunASRProcessor": "vllm.transformers_utils.processors.funasr",
    "HunYuanVLProcessor": "vllm.transformers_utils.processors.hunyuan_vl",
    "HunYuanVLImageProcessor": "vllm.transformers_utils.processors.hunyuan_vl_image",
    "OvisProcessor": "vllm.transformers_utils.processors.ovis",
    "Ovis2_5Processor": "vllm.transformers_utils.processors.ovis2_5",
    "Qwen3ASRProcessor": "vllm.transformers_utils.processors.qwen3_asr",
}


__all__ = [
    "BagelProcessor",
    "DeepseekVLV2Processor",
    "FireRedASR2Processor",
    "FunASRProcessor",
    "HunYuanVLProcessor",
    "HunYuanVLImageProcessor",
    "OvisProcessor",
    "Ovis2_5Processor",
    "Qwen3ASRProcessor",
]


def __getattr__(name: str):
    if name in _CLASS_TO_MODULE:
        module_name = _CLASS_TO_MODULE[name]
        module = importlib.import_module(module_name)
        return getattr(module, name)

    raise AttributeError(f"module 'processors' has no attribute '{name}'")


def __dir__():
    return sorted(list(__all__))
