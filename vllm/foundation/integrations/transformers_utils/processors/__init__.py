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
    "CohereASRProcessor",
    "Cosmos3EdgeProcessor",
    "DeepseekVLV2Processor",
    "FireRedASR2Processor",
    "FunASRProcessor",
    "GLM4VProcessor",
    "Granite4VisionProcessor",
    "H2OVLProcessor",
    "Moondream3Processor",
    "InternVLProcessor",
    "IsaacProcessor",
    "KimiAudioProcessor",
    "KimiK25Processor",
    "KimiK3Processor",
    "MiMoOmniProcessor",
    "MiniCPMOProcessor",
    "MiniCPMVProcessor",
    "MiniMaxM3VLImageProcessor",
    "MiniMaxM3VLVideoProcessor",
    "MiniMaxVLProcessor",
    "MistralCommonPixtralProcessor",
    "MistralCommonVoxtralProcessor",
    "NanoNemotronVLProcessor",
    "NemotronVLProcessor",
    "LlamaNemotronVLEmbedProcessor",
    "NVLMProcessor",
    "OpenVLAProcessor",
    "OvisProcessor",
    "Ovis2_5Processor",
    "Qwen3ASRProcessor",
    "Step3VLProcessor",
    "InklingProcessor",
    "InklingImageProcessor",
    "InklingAudioFeatureExtractor",
]

_CLASS_TO_MODULE: dict[str, str] = {
    "BagelProcessor": "vllm.foundation.integrations.transformers_utils.processors.bagel",
    "CohereASRProcessor": "vllm.foundation.integrations.transformers_utils.processors.cohere_asr",
    "Cosmos3EdgeProcessor": "vllm.foundation.integrations.transformers_utils.processors.cosmos3_edge",
    "DeepseekVLV2Processor": "vllm.foundation.integrations.transformers_utils.processors.deepseek_vl2",
    "FireRedASR2Processor": "vllm.foundation.integrations.transformers_utils.processors.fireredasr2",
    "FunASRProcessor": "vllm.foundation.integrations.transformers_utils.processors.funasr",
    "GLM4VProcessor": "vllm.foundation.integrations.transformers_utils.processors.glm4v",
    "Granite4VisionProcessor": "vllm.foundation.integrations.transformers_utils.processors.granite4_vision",
    "H2OVLProcessor": "vllm.foundation.integrations.transformers_utils.processors.h2ovl",
    "InternVLProcessor": "vllm.foundation.integrations.transformers_utils.processors.internvl",
    "IsaacProcessor": "vllm.foundation.integrations.transformers_utils.processors.isaac",
    "KimiAudioProcessor": "vllm.foundation.integrations.transformers_utils.processors.kimi_audio",
    "KimiK25Processor": "vllm.foundation.integrations.transformers_utils.processors.kimi_k25",
    "KimiK3Processor": "vllm.foundation.integrations.transformers_utils.processors.kimi_k3",
    "MiMoOmniProcessor": "vllm.foundation.integrations.transformers_utils.processors.mimo_v2_omni",
    "MiniCPMOProcessor": "vllm.foundation.integrations.transformers_utils.processors.minicpmo",
    "MiniCPMVProcessor": "vllm.foundation.integrations.transformers_utils.processors.minicpmv",
    "MiniMaxM3VLImageProcessor": "vllm.foundation.integrations.transformers_utils.processors.minimax_m3",
    "MiniMaxM3VLVideoProcessor": "vllm.foundation.integrations.transformers_utils.processors.minimax_m3",
    "MiniMaxVLProcessor": "vllm.foundation.integrations.transformers_utils.processors.minimax_m3",
    "MistralCommonPixtralProcessor": "vllm.foundation.integrations.transformers_utils.processors.pixtral",
    "MistralCommonVoxtralProcessor": "vllm.foundation.integrations.transformers_utils.processors.voxtral",
    "Moondream3Processor": "vllm.foundation.integrations.transformers_utils.processors.moondream3",
    "NanoNemotronVLProcessor": "vllm.foundation.integrations.transformers_utils.processors.nano_nemotron_vl",
    "NemotronVLProcessor": "vllm.foundation.integrations.transformers_utils.processors.nemotron_vl",
    "LlamaNemotronVLEmbedProcessor": "vllm.foundation.integrations.transformers_utils.processors.nemotron_vl",
    "NVLMProcessor": "vllm.foundation.integrations.transformers_utils.processors.nvlm_d",
    "OpenVLAProcessor": "vllm.foundation.integrations.transformers_utils.processors.openvla",
    "OvisProcessor": "vllm.foundation.integrations.transformers_utils.processors.ovis",
    "Ovis2_5Processor": "vllm.foundation.integrations.transformers_utils.processors.ovis2_5",
    "Qwen3ASRProcessor": "vllm.foundation.integrations.transformers_utils.processors.qwen3_asr",
    "Step3VLProcessor": "vllm.foundation.integrations.transformers_utils.processors.step3_vl",
    "InklingProcessor": "vllm.foundation.integrations.transformers_utils.processors.inkling",
    "InklingImageProcessor": "vllm.foundation.integrations.transformers_utils.processors.inkling",
    "InklingAudioFeatureExtractor": "vllm.foundation.integrations.transformers_utils.processors.inkling",
}


def __getattr__(name: str):
    if name in _CLASS_TO_MODULE:
        module_name = _CLASS_TO_MODULE[name]
        module = importlib.import_module(module_name)
        return getattr(module, name)

    raise AttributeError(f"module 'processors' has no attribute '{name}'")


def __dir__():
    return sorted(list(__all__))
