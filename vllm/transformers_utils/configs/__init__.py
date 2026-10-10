# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model configs may be defined in this directory for the following reasons:

- There is no configuration file defined by HF Hub or Transformers library.
- There is a need to override the existing config to support vLLM.
- The HF model_type isn't recognized by the Transformers library but can
  be mapped to an existing Transformers config, such as
  deepseek-ai/DeepSeek-V3.2-Exp.
"""

from __future__ import annotations

import importlib

_CLASS_TO_MODULE: dict[str, str] = {
    "BagelConfig": "vllm.transformers_utils.configs.bagel",
    "BailingMoeV3TextConfig": "vllm.transformers_utils.configs.bailing_moe_v3_vl",
    "BailingMoeV3VisionConfig": "vllm.transformers_utils.configs.bailing_moe_v3_vl",
    "BailingMoeV3VLConfig": "vllm.transformers_utils.configs.bailing_moe_v3_vl",
    "ChatGLMConfig": "vllm.transformers_utils.configs.chatglm",
    "ColModernVBertConfig": "vllm.transformers_utils.configs.colmodernvbert",
    "ColPaliConfig": "vllm.transformers_utils.configs.colpali",
    "ColQwen3Config": "vllm.transformers_utils.configs.colqwen3",
    "OpsColQwen3Config": "vllm.transformers_utils.configs.colqwen3",
    "Qwen3VLNemotronEmbedConfig": "vllm.transformers_utils.configs.colqwen3",
    "Cosmos3EdgeConfig": "vllm.transformers_utils.configs.cosmos3_edge",
    "Cosmos3EdgeProjectorConfig": "vllm.transformers_utils.configs.cosmos3_edge",
    "Cosmos3EdgeTextConfig": "vllm.transformers_utils.configs.cosmos3_edge",
    "Cosmos3EdgeVisionConfig": "vllm.transformers_utils.configs.cosmos3_edge",
    "DeepseekVLV2Config": "vllm.transformers_utils.configs.deepseek_vl2",
    "DeepseekV4Config": "vllm.transformers_utils.configs.deepseek_v4",
    "DeepseekV41Config": "vllm.transformers_utils.configs.deepseek_v41",
    "Dots3NoteConfig": "vllm.transformers_utils.configs.dots3_note",
    "K3DSparkConfig": "vllm.transformers_utils.configs.k3_dspark",
    "DotsOCRConfig": "vllm.transformers_utils.configs.dotsocr",
    "EAGLEConfig": "vllm.transformers_utils.configs.eagle",
    "FunAudioChatConfig": "vllm.transformers_utils.configs.funaudiochat",
    "FunAudioChatAudioEncoderConfig": "vllm.transformers_utils.configs.funaudiochat",
    "Glm4MoeLitePlusPlusConfig": "vllm.transformers_utils.configs.glm4_moe_lite_plus_plus",
    "Granite4VisionConfig": "vllm.transformers_utils.configs.granite4_vision",
    "Glm5NextConfig": "vllm.transformers_utils.configs.glm5_next",
    "Glm5NextTextConfig": "vllm.transformers_utils.configs.glm5_next",
    "Glm5NextVisionConfig": "vllm.transformers_utils.configs.glm5_next",
    "HYV3Config": "vllm.transformers_utils.configs.hy_v3",
    "HYV4Config": "vllm.transformers_utils.configs.hy_v4",
    "IsaacConfig": "vllm.transformers_utils.configs.isaac",
    # RWConfig is for the original tiiuae/falcon-40b(-instruct) and
    # tiiuae/falcon-7b(-instruct) models. Newer Falcon models will use the
    # `FalconConfig` class from the official HuggingFace transformers library.
    "RWConfig": "vllm.transformers_utils.configs.falcon",
    "MedusaConfig": "vllm.transformers_utils.configs.medusa",
    "MiDashengLMConfig": "vllm.transformers_utils.configs.midashenglm",
    "MiniMaxM3Config": "vllm.transformers_utils.configs.minimax_m3",
    "MiniMaxM3MTPConfig": "vllm.transformers_utils.configs.minimax_m3",
    "MiniMaxM3TextConfig": "vllm.transformers_utils.configs.minimax_m3",
    "MLPSpeculatorConfig": "vllm.transformers_utils.configs.mlp_speculator",
    "Moondream3Config": "vllm.transformers_utils.configs.moondream3",
    "Moondream3TextConfig": "vllm.transformers_utils.configs.moondream3",
    "Moondream3VisionConfig": "vllm.transformers_utils.configs.moondream3",
    "MossTranscribeDiarizeConfig": (
        "vllm.transformers_utils.configs.moss_transcribe_diarize"
    ),
    "MoonViTConfig": "vllm.transformers_utils.configs.moonvit",
    "KimiLinearConfig": "vllm.transformers_utils.configs.kimi_linear",
    "KimiVLConfig": "vllm.transformers_utils.configs.kimi_vl",
    "MuseGlimmerConfig": "vllm.transformers_utils.configs.muse_glimmer",
    "MuseGlimmerTextConfig": "vllm.transformers_utils.configs.muse_glimmer",
    "MuseGlimmerVisionConfig": "vllm.transformers_utils.configs.muse_glimmer",
    "MuseGlimmerAssistantConfig": "vllm.transformers_utils.configs.muse_glimmer",
    "KimiK3Config": "vllm.transformers_utils.configs.kimi_k3",
    "KimiK3VisionConfig": "vllm.transformers_utils.configs.kimi_k3",
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
    "InklingModelConfig": "vllm.models.inkling.configs",
    "InklingAudioConfig": "vllm.models.inkling.configs",
    "InklingVisionConfig": "vllm.models.inkling.configs",
    "InklingMMConfig": "vllm.models.inkling.configs",
    # Upstream Transformers classes registered in _CONFIG_REGISTRY
    "DeepseekV3Config": "transformers",
    "HyperCLOVAXConfig": "transformers",
    "Kimi_K25Config": "transformers",
}

__all__ = [
    "BagelConfig",
    "BailingMoeV3TextConfig",
    "BailingMoeV3VisionConfig",
    "BailingMoeV3VLConfig",
    "ChatGLMConfig",
    "ColModernVBertConfig",
    "ColPaliConfig",
    "ColQwen3Config",
    "OpsColQwen3Config",
    "Qwen3VLNemotronEmbedConfig",
    "Cosmos3EdgeConfig",
    "Cosmos3EdgeProjectorConfig",
    "Cosmos3EdgeTextConfig",
    "Cosmos3EdgeVisionConfig",
    "DeepseekVLV2Config",
    "DeepseekV3Config",
    "DeepseekV4Config",
    "Dots3NoteConfig",
    "K3DSparkConfig",
    "DotsOCRConfig",
    "EAGLEConfig",
    "FunAudioChatConfig",
    "FunAudioChatAudioEncoderConfig",
    "Glm4MoeLitePlusPlusConfig",
    "Granite4VisionConfig",
    "Glm5NextConfig",
    "Glm5NextTextConfig",
    "Glm5NextVisionConfig",
    "HYV3Config",
    "HYV4Config",
    "HyperCLOVAXConfig",
    "IsaacConfig",
    "RWConfig",
    "MedusaConfig",
    "MiDashengLMConfig",
    "MiniMaxM3Config",
    "MiniMaxM3MTPConfig",
    "MiniMaxM3TextConfig",
    "MLPSpeculatorConfig",
    "Moondream3Config",
    "Moondream3TextConfig",
    "Moondream3VisionConfig",
    "MossTranscribeDiarizeConfig",
    "MoonViTConfig",
    "KimiLinearConfig",
    "KimiVLConfig",
    "Kimi_K25Config",
    "MuseGlimmerConfig",
    "MuseGlimmerTextConfig",
    "MuseGlimmerVisionConfig",
    "MuseGlimmerAssistantConfig",
    "KimiK3Config",
    "KimiK3VisionConfig",
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
    "InklingModelConfig",
    "InklingAudioConfig",
    "InklingVisionConfig",
    "InklingMMConfig",
]


def __getattr__(name: str):
    if name in _CLASS_TO_MODULE:
        module_name = _CLASS_TO_MODULE[name]
        module = importlib.import_module(module_name)
        return getattr(module, name)

    raise AttributeError(f"module 'configs' has no attribute '{name}'")


def __dir__():
    return sorted(list(__all__))
