# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Model configs may be defined in this directory for the following reasons:

- There is no configuration file defined by HF Hub or Transformers library.
- There is a need to override the existing config to support vLLM.
- The HF model_type isn't recognized by the Transformers library but can
  be mapped to an existing Transformers config, such as
  deepseek-ai/DeepSeek-V3.2-Exp.
"""

from __future__ import annotations

import importlib

_CLASS_TO_MODULE: dict[str, str] = {
    "AfmoeConfig": "vllm.transformers_utils.configs.afmoe",
    "BagelConfig": "vllm.transformers_utils.configs.bagel",
    "ChatGLMConfig": "vllm.transformers_utils.configs.chatglm",
    "DeepseekVLV2Config": "vllm.transformers_utils.configs.deepseek_vl2",
    "DotsOCRConfig": "vllm.transformers_utils.configs.dotsocr",
    "EAGLEConfig": "vllm.transformers_utils.configs.eagle",
    "FlexOlmoConfig": "vllm.transformers_utils.configs.flex_olmo",
    "FunAudioChatConfig": "vllm.transformers_utils.configs.funaudiochat",
    "FunAudioChatAudioEncoderConfig": "vllm.transformers_utils.configs.funaudiochat",
    "HunYuanVLConfig": "vllm.transformers_utils.configs.hunyuan_vl",
    "HunYuanVLTextConfig": "vllm.transformers_utils.configs.hunyuan_vl",
    "HunYuanVLVisionConfig": "vllm.transformers_utils.configs.hunyuan_vl",
    "IsaacConfig": "vllm.transformers_utils.configs.isaac",
    # RWConfig is for the original tiiuae/falcon-40b(-instruct) and
    # tiiuae/falcon-7b(-instruct) models. Newer Falcon models will use the
    # `FalconConfig` class from the official HuggingFace transformers library.
    "RWConfig": "vllm.transformers_utils.configs.falcon",
    "JAISConfig": "vllm.transformers_utils.configs.jais",
    "Lfm2MoeConfig": "vllm.transformers_utils.configs.lfm2_moe",
    "MedusaConfig": "vllm.transformers_utils.configs.medusa",
    "MiDashengLMConfig": "vllm.transformers_utils.configs.midashenglm",
    "MLPSpeculatorConfig": "vllm.transformers_utils.configs.mlp_speculator",
    "MoonViTConfig": "vllm.transformers_utils.configs.moonvit",
    "KimiLinearConfig": "vllm.transformers_utils.configs.kimi_linear",
    "KimiVLConfig": "vllm.transformers_utils.configs.kimi_vl",
    "KimiK25Config": "vllm.transformers_utils.configs.kimi_k25",
    "NemotronConfig": "vllm.transformers_utils.configs.nemotron",
    "NemotronHConfig": "vllm.transformers_utils.configs.nemotron_h",
    "Olmo3Config": "vllm.transformers_utils.configs.olmo3",
    "OvisConfig": "vllm.transformers_utils.configs.ovis",
    "PixelShuffleSiglip2VisionConfig": "vllm.transformers_utils.configs.isaac",
    "RadioConfig": "vllm.transformers_utils.configs.radio",
    "SpeculatorsConfig": "vllm.transformers_utils.configs.speculators.base",
    "UltravoxConfig": "vllm.transformers_utils.configs.ultravox",
    "VibeVoiceASRConfig": "vllm.transformers_utils.configs.vibevoice",
    "Step3VLConfig": "vllm.transformers_utils.configs.step3_vl",
    "Step3VisionEncoderConfig": "vllm.transformers_utils.configs.step3_vl",
    "Step3TextConfig": "vllm.transformers_utils.configs.step3_vl",
    "Qwen3ASRConfig": "vllm.transformers_utils.configs.qwen3_asr",
    "Qwen3NextConfig": "vllm.transformers_utils.configs.qwen3_next",
    "Tarsier2Config": "vllm.transformers_utils.configs.tarsier2",
    # Special case: DeepseekV3Config is from HuggingFace Transformers
    "DeepseekV3Config": "transformers",
}

__all__ = [
    "AfmoeConfig",
    "BagelConfig",
    "ChatGLMConfig",
    "DeepseekVLV2Config",
    "DeepseekV3Config",
    "DotsOCRConfig",
    "EAGLEConfig",
    "FlexOlmoConfig",
    "FunAudioChatConfig",
    "FunAudioChatAudioEncoderConfig",
    "HunYuanVLConfig",
    "HunYuanVLTextConfig",
    "HunYuanVLVisionConfig",
    "IsaacConfig",
    "RWConfig",
    "JAISConfig",
    "Lfm2MoeConfig",
    "MedusaConfig",
    "MiDashengLMConfig",
    "MLPSpeculatorConfig",
    "MoonViTConfig",
    "KimiLinearConfig",
    "KimiVLConfig",
    "KimiK25Config",
    "NemotronConfig",
    "NemotronHConfig",
    "Olmo3Config",
    "OvisConfig",
    "PixelShuffleSiglip2VisionConfig",
    "RadioConfig",
    "SpeculatorsConfig",
    "UltravoxConfig",
    "VibeVoiceASRConfig",
    "Step3VLConfig",
    "Step3VisionEncoderConfig",
    "Step3TextConfig",
    "Qwen3ASRConfig",
    "Qwen3NextConfig",
    "Tarsier2Config",
]


def __getattr__(name: str):
    if name in _CLASS_TO_MODULE:
        module_name = _CLASS_TO_MODULE[name]
        module = importlib.import_module(module_name)
        return getattr(module, name)

    raise AttributeError(f"module 'configs' has no attribute '{name}'")


def __dir__():
    return sorted(list(__all__))
