# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Multi-modal processors may be defined in this directory for the following
reasons:

- There is no processing file defined by HF Hub or Transformers library.
- There is a need to override the existing processor to support vLLM.
"""

import importlib

__all__ = [
    "BagelProcessor",
    "DeepseekVLV2Processor",
    "Eagle2_5_VLProcessor",
    "FireRedASR2Processor",
    "FunASRProcessor",
    "GLM4VProcessor",
    "H2OVLProcessor",
    "HunYuanVLProcessor",
    "HunYuanVLImageProcessor",
    "InternVLProcessor",
    "KimiAudioProcessor",
    "MistralCommonPixtralProcessor",
    "MistralCommonVoxtralProcessor",
    "NanoNemotronVLProcessor",
    "NemotronParseProcessor",
    "NemotronVLProcessor",
    "LlamaNemotronVLEmbedProcessor",
    "NVLMProcessor",
    "OvisProcessor",
    "Ovis2_5Processor",
    "QwenVLProcessor",
    "Qwen3ASRProcessor",
    "SkyworkR1VProcessor",
]

_CLASS_TO_MODULE: dict[str, str] = {
    "BagelProcessor": "vllm.transformers_utils.processors.bagel",
    "DeepseekVLV2Processor": "vllm.transformers_utils.processors.deepseek_vl2",
    "Eagle2_5_VLProcessor": "vllm.transformers_utils.processors.eagle2_5_vl",
    "FireRedASR2Processor": "vllm.transformers_utils.processors.fireredasr2",
    "FunASRProcessor": "vllm.transformers_utils.processors.funasr",
    "GLM4VProcessor": "vllm.transformers_utils.processors.glm4v",
    "H2OVLProcessor": "vllm.transformers_utils.processors.h2ovl",
    "HunYuanVLProcessor": "vllm.transformers_utils.processors.hunyuan_vl",
    "HunYuanVLImageProcessor": "vllm.transformers_utils.processors.hunyuan_vl_image",
    "InternVLProcessor": "vllm.transformers_utils.processors.internvl",
    "KimiAudioProcessor": "vllm.transformers_utils.processors.kimi_audio",
    "MistralCommonPixtralProcessor": "vllm.transformers_utils.processors.pixtral",
    "MistralCommonVoxtralProcessor": "vllm.transformers_utils.processors.voxtral",
    "NanoNemotronVLProcessor": "vllm.transformers_utils.processors.nano_nemotron_vl",
    "NemotronParseProcessor": "vllm.transformers_utils.processors.nemotron_parse",
    "NemotronVLProcessor": "vllm.transformers_utils.processors.nemotron_vl",
    "LlamaNemotronVLEmbedProcessor": "vllm.transformers_utils.processors.nemotron_vl",
    "NVLMProcessor": "vllm.transformers_utils.processors.nvlm_d",
    "OvisProcessor": "vllm.transformers_utils.processors.ovis",
    "Ovis2_5Processor": "vllm.transformers_utils.processors.ovis2_5",
    "QwenVLProcessor": "vllm.transformers_utils.processors.qwen_vl",
    "Qwen3ASRProcessor": "vllm.transformers_utils.processors.qwen3_asr",
    "SkyworkR1VProcessor": "vllm.transformers_utils.processors.skyworkr1v",
}


def __getattr__(name: str):
    if name in _CLASS_TO_MODULE:
        module_name = _CLASS_TO_MODULE[name]
        module = importlib.import_module(module_name)
        return getattr(module, name)

    raise AttributeError(f"module 'processors' has no attribute '{name}'")


def __dir__():
    return sorted(list(__all__))
