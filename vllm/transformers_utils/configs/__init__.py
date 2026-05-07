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
    "AXK1Config": "vllm.transformers_utils.configs.AXK1",
    "BagelConfig": "vllm.transformers_utils.configs.bagel",
    "CheersConfig": "vllm.transformers_utils.configs.cheers",
    "ChatGLMConfig": "vllm.transformers_utils.configs.chatglm",
    "ColModernVBertConfig": "vllm.transformers_utils.configs.colmodernvbert",
    "ColPaliConfig": "vllm.transformers_utils.configs.colpali",
    "ColQwen3Config": "vllm.transformers_utils.configs.colqwen3",
    "OpsColQwen3Config": "vllm.transformers_utils.configs.colqwen3",
    "Qwen3VLNemotronEmbedConfig": "vllm.transformers_utils.configs.colqwen3",
    "Cosmos3Config": "vllm.transformers_utils.configs.cosmos3",
    "DiffusionGemmaConfig": "vllm.transformers_utils.configs.diffusion_gemma",
    "DiffusionGemmaTextConfig": "vllm.transformers_utils.configs.diffusion_gemma",
    "DeepseekVLV2Config": "vllm.transformers_utils.configs.deepseek_vl2",
    "DeepseekV4Config": "vllm.transformers_utils.configs.deepseek_v4",
    "DotsOCRConfig": "vllm.transformers_utils.configs.dotsocr",
    "EAGLEConfig": "vllm.transformers_utils.configs.eagle",
    "FireRedLIDConfig": "vllm.transformers_utils.configs.fireredlid",
    "FlexOlmoConfig": "vllm.transformers_utils.configs.flex_olmo",
    "FunAudioChatConfig": "vllm.transformers_utils.configs.funaudiochat",
    "FunAudioChatAudioEncoderConfig": "vllm.transformers_utils.configs.funaudiochat",
    "Granite4VisionConfig": "vllm.transformers_utils.configs.granite4_vision",
    "HunYuanVLConfig": "vllm.transformers_utils.configs.hunyuan_vl",
    "HunYuanVLTextConfig": "vllm.transformers_utils.configs.hunyuan_vl",
    "HunYuanVLVisionConfig": "vllm.transformers_utils.configs.hunyuan_vl",
    "HCXVisionConfig": "vllm.transformers_utils.configs.hyperclovax",
    "HYV3Config": "vllm.transformers_utils.configs.hy_v3",
    "HyperCLOVAXConfig": "vllm.transformers_utils.configs.hyperclovax",
    "IsaacConfig": "vllm.transformers_utils.configs.isaac",
    # RWConfig is for the original tiiuae/falcon-40b(-instruct) and
    # tiiuae/falcon-7b(-instruct) models. Newer Falcon models will use the
    # `FalconConfig` class from the official HuggingFace transformers library.
    "RWConfig": "vllm.transformers_utils.configs.falcon",
    "LagunaConfig": "vllm.transformers_utils.configs.laguna",
    "Lfm2MoeConfig": "vllm.transformers_utils.configs.lfm2_moe",
    "MedusaConfig": "vllm.transformers_utils.configs.medusa",
    "MellumConfig": "vllm.transformers_utils.configs.mellum",
    "MiDashengLMConfig": "vllm.transformers_utils.configs.midashenglm",
    "MiniMaxM3Config": "vllm.transformers_utils.configs.minimax_m3",
    "MiniMaxM3MTPConfig": "vllm.transformers_utils.configs.minimax_m3",
    "MiniMaxM3TextConfig": "vllm.transformers_utils.configs.minimax_m3",
    "MLPSpeculatorConfig": "vllm.transformers_utils.configs.mlp_speculator",
    "Moondream3Config": "vllm.transformers_utils.configs.moondream3",
    "Moondream3TextConfig": "vllm.transformers_utils.configs.moondream3",
    "Moondream3VisionConfig": "vllm.transformers_utils.configs.moondream3",
    "MoonViTConfig": "vllm.transformers_utils.configs.moonvit",
    "KimiLinearConfig": "vllm.transformers_utils.configs.kimi_linear",
    "KimiVLConfig": "vllm.transformers_utils.configs.kimi_vl",
    "KimiK25Config": "vllm.transformers_utils.configs.kimi_k25",
    "NemotronConfig": "vllm.transformers_utils.configs.nemotron",
    "NemotronHConfig": "vllm.transformers_utils.configs.nemotron_h",
    "OlmoHybridConfig": "vllm.transformers_utils.configs.olmo_hybrid",
    "OpenVLAConfig": "vllm.transformers_utils.configs.openvla",
    "OvisConfig": "vllm.transformers_utils.configs.ovis",
    "PixelShuffleSiglip2VisionConfig": "vllm.transformers_utils.configs.isaac",
    "RadioConfig": "vllm.transformers_utils.configs.radio",
    "SpeculatorsConfig": "vllm.transformers_utils.configs.speculators",
    "UltravoxConfig": "vllm.transformers_utils.configs.ultravox",
    "UnlimitedOCRConfig": "vllm.transformers_utils.configs.unlimited_ocr",
    "Step3VLConfig": "vllm.transformers_utils.configs.step3_vl",
    "Step3VisionEncoderConfig": "vllm.transformers_utils.configs.step3_vl",
    "Step3TextConfig": "vllm.transformers_utils.configs.step3_vl",
    "Step3p5Config": "vllm.transformers_utils.configs.step3p5",
    "QianfanOCRConfig": "vllm.transformers_utils.configs.qianfan_ocr",
    "QianfanOCRVisionConfig": "vllm.transformers_utils.configs.qianfan_ocr",
    "Qwen3ASRConfig": "vllm.transformers_utils.configs.qwen3_asr",
    "Qwen3NextConfig": "vllm.transformers_utils.configs.qwen3_next",
    "Qwen3_5Config": "vllm.transformers_utils.configs.qwen3_5",
    "Qwen3_5TextConfig": "vllm.transformers_utils.configs.qwen3_5",
    "Qwen3_5MoeConfig": "vllm.transformers_utils.configs.qwen3_5_moe",
    "Qwen3_5MoeTextConfig": "vllm.transformers_utils.configs.qwen3_5_moe",
    # Special case: DeepseekV3Config is from HuggingFace Transformers
    "DeepseekV3Config": "transformers",
    "ZayaConfig": "vllm.transformers_utils.configs.zaya",
}

__all__ = [
    "AfmoeConfig",
    "AXK1Config",
    "BagelConfig",
    "CheersConfig",
    "ChatGLMConfig",
    "ColModernVBertConfig",
    "ColPaliConfig",
    "ColQwen3Config",
    "OpsColQwen3Config",
    "Qwen3VLNemotronEmbedConfig",
    "Cosmos3Config",
    "DiffusionGemmaConfig",
    "DiffusionGemmaTextConfig",
    "DeepseekVLV2Config",
    "DeepseekV3Config",
    "DeepseekV4Config",
    "DotsOCRConfig",
    "EAGLEConfig",
    "FlexOlmoConfig",
    "FireRedLIDConfig",
    "FunAudioChatConfig",
    "FunAudioChatAudioEncoderConfig",
    "Granite4VisionConfig",
    "HunYuanVLConfig",
    "HunYuanVLTextConfig",
    "HunYuanVLVisionConfig",
    "HCXVisionConfig",
    "HYV3Config",
    "HyperCLOVAXConfig",
    "IsaacConfig",
    "RWConfig",
    "LagunaConfig",
    "Lfm2MoeConfig",
    "MedusaConfig",
    "MellumConfig",
    "MiDashengLMConfig",
    "MiniMaxM3Config",
    "MiniMaxM3MTPConfig",
    "MiniMaxM3TextConfig",
    "MLPSpeculatorConfig",
    "Moondream3Config",
    "Moondream3TextConfig",
    "Moondream3VisionConfig",
    "MoonViTConfig",
    "KimiLinearConfig",
    "KimiVLConfig",
    "KimiK25Config",
    "NemotronConfig",
    "NemotronHConfig",
    "OlmoHybridConfig",
    "OpenVLAConfig",
    "OvisConfig",
    "PixelShuffleSiglip2VisionConfig",
    "RadioConfig",
    "SpeculatorsConfig",
    "UltravoxConfig",
    "UnlimitedOCRConfig",
    "Step3VLConfig",
    "Step3VisionEncoderConfig",
    "Step3TextConfig",
    "Step3p5Config",
    "QianfanOCRConfig",
    "QianfanOCRVisionConfig",
    "Qwen3ASRConfig",
    "Qwen3NextConfig",
    "Qwen3_5Config",
    "Qwen3_5TextConfig",
    "Qwen3_5MoeConfig",
    "Qwen3_5MoeTextConfig",
    "Tarsier2Config",
    "ZayaConfig",
]


def __getattr__(name: str):
    if name in _CLASS_TO_MODULE:
        module_name = _CLASS_TO_MODULE[name]
        module = importlib.import_module(module_name)
        return getattr(module, name)

    raise AttributeError(f"module 'configs' has no attribute '{name}'")


def __dir__():
    return sorted(list(__all__))
