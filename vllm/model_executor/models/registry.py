# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Whenever you add an architecture to this page, please also update
`tests/models/registry.py` with example HuggingFace models for it.
"""

import importlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Callable, Set
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import torch.nn as nn
import transformers

from vllm import envs
from vllm.config import (
    ModelConfig,
    iter_architecture_defaults,
    try_match_architecture_defaults,
)
from vllm.logger import init_logger
from vllm.logging_utils import logtime
from vllm.transformers_utils.dynamic_module import try_get_class_from_dynamic_module
from vllm.utils.hashing import safe_hash

if TYPE_CHECKING:
    from vllm.config.model import AttnTypeStr
    from vllm.config.pooler import SequencePoolingType, TokenPoolingType
else:
    AttnTypeStr = Any
    SequencePoolingType = Any
    TokenPoolingType = Any


from .interfaces import (
    has_inner_state,
    has_noops,
    is_attention_free,
    is_hybrid,
    requires_raw_input_tokens,
    supports_cross_encoding,
    supports_mamba_prefix_caching,
    supports_multimodal,
    supports_multimodal_encoder_tp_data,
    supports_multimodal_raw_input_only,
    supports_pp,
    supports_transcription,
)
from .interfaces_base import (
    get_attn_type,
    get_default_seq_pooling_type,
    get_default_tok_pooling_type,
    is_pooling_model,
    is_text_generation_model,
)

logger = init_logger(__name__)

_TEXT_GENERATION_MODELS = {
    # [Decoder-only]
    "AfmoeForCausalLM": ("afmoe", "AfmoeForCausalLM"),
    "ApertusForCausalLM": ("apertus", "ApertusForCausalLM"),
    "AquilaModel": ("llama", "LlamaForCausalLM"),
    "AquilaForCausalLM": ("llama", "LlamaForCausalLM"),  # AquilaChat2
    "ArceeForCausalLM": ("arcee", "ArceeForCausalLM"),
    "ArcticForCausalLM": ("arctic", "ArcticForCausalLM"),
    # baichuan-7b, upper case 'C' in the class name
    "BaiChuanForCausalLM": ("baichuan", "BaiChuanForCausalLM"),
    # baichuan-13b, lower case 'c' in the class name
    "BaichuanForCausalLM": ("baichuan", "BaichuanForCausalLM"),
    "BailingMoeForCausalLM": ("bailing_moe", "BailingMoeForCausalLM"),
    "BailingMoeV2ForCausalLM": ("bailing_moe", "BailingMoeV2ForCausalLM"),
    "BambaForCausalLM": ("bamba", "BambaForCausalLM"),
    "BloomForCausalLM": ("bloom", "BloomForCausalLM"),
    "ChatGLMModel": ("chatglm", "ChatGLMForCausalLM"),
    "ChatGLMForConditionalGeneration": ("chatglm", "ChatGLMForCausalLM"),
    "CohereForCausalLM": ("commandr", "CohereForCausalLM"),
    "Cohere2ForCausalLM": ("commandr", "CohereForCausalLM"),
    "CwmForCausalLM": ("llama", "LlamaForCausalLM"),
    "DbrxForCausalLM": ("dbrx", "DbrxForCausalLM"),
    "DeciLMForCausalLM": ("nemotron_nas", "DeciLMForCausalLM"),
    "DeepseekForCausalLM": ("deepseek_v2", "DeepseekForCausalLM"),
    "DeepseekV2ForCausalLM": ("deepseek_v2", "DeepseekV2ForCausalLM"),
    "DeepseekV3ForCausalLM": ("deepseek_v2", "DeepseekV3ForCausalLM"),
    "DeepseekV32ForCausalLM": ("deepseek_v2", "DeepseekV3ForCausalLM"),
    "Dots1ForCausalLM": ("dots1", "Dots1ForCausalLM"),
    "Ernie4_5ForCausalLM": ("ernie45", "Ernie4_5ForCausalLM"),
    "Ernie4_5_MoeForCausalLM": ("ernie45_moe", "Ernie4_5_MoeForCausalLM"),
    "ExaoneForCausalLM": ("exaone", "ExaoneForCausalLM"),
    "Exaone4ForCausalLM": ("exaone4", "Exaone4ForCausalLM"),
    "ExaoneMoEForCausalLM": ("exaone_moe", "ExaoneMoeForCausalLM"),
    "Fairseq2LlamaForCausalLM": ("fairseq2_llama", "Fairseq2LlamaForCausalLM"),
    "FalconForCausalLM": ("falcon", "FalconForCausalLM"),
    "FalconMambaForCausalLM": ("mamba", "MambaForCausalLM"),
    "FalconH1ForCausalLM": ("falcon_h1", "FalconH1ForCausalLM"),
    "FlexOlmoForCausalLM": ("flex_olmo", "FlexOlmoForCausalLM"),
    "GemmaForCausalLM": ("gemma", "GemmaForCausalLM"),
    "Gemma2ForCausalLM": ("gemma2", "Gemma2ForCausalLM"),
    "Gemma3ForCausalLM": ("gemma3", "Gemma3ForCausalLM"),
    "Gemma3nForCausalLM": ("gemma3n", "Gemma3nForCausalLM"),
    "Qwen3NextForCausalLM": ("qwen3_next", "Qwen3NextForCausalLM"),
    "GlmForCausalLM": ("glm", "GlmForCausalLM"),
    "Glm4ForCausalLM": ("glm4", "Glm4ForCausalLM"),
    "Glm4MoeForCausalLM": ("glm4_moe", "Glm4MoeForCausalLM"),
    "GptOssForCausalLM": ("gpt_oss", "GptOssForCausalLM"),
    "GPT2LMHeadModel": ("gpt2", "GPT2LMHeadModel"),
    "GPTBigCodeForCausalLM": ("gpt_bigcode", "GPTBigCodeForCausalLM"),
    "GPTJForCausalLM": ("gpt_j", "GPTJForCausalLM"),
    "GPTNeoXForCausalLM": ("gpt_neox", "GPTNeoXForCausalLM"),
    "GraniteForCausalLM": ("granite", "GraniteForCausalLM"),
    "GraniteMoeForCausalLM": ("granitemoe", "GraniteMoeForCausalLM"),
    "GraniteMoeHybridForCausalLM": ("granitemoehybrid", "GraniteMoeHybridForCausalLM"),  # noqa: E501
    "GraniteMoeSharedForCausalLM": ("granitemoeshared", "GraniteMoeSharedForCausalLM"),  # noqa: E501
    "GritLM": ("gritlm", "GritLM"),
    "Grok1ModelForCausalLM": ("grok1", "GrokForCausalLM"),
    "Grok1ForCausalLM": ("grok1", "GrokForCausalLM"),
    "HunYuanMoEV1ForCausalLM": ("hunyuan_v1", "HunYuanMoEV1ForCausalLM"),
    "HunYuanDenseV1ForCausalLM": ("hunyuan_v1", "HunYuanDenseV1ForCausalLM"),
    "HCXVisionForCausalLM": ("hyperclovax_vision", "HCXVisionForCausalLM"),
    "InternLMForCausalLM": ("llama", "LlamaForCausalLM"),
    "InternLM2ForCausalLM": ("internlm2", "InternLM2ForCausalLM"),
    "InternLM2VEForCausalLM": ("internlm2_ve", "InternLM2VEForCausalLM"),
    "InternLM3ForCausalLM": ("llama", "LlamaForCausalLM"),
    "IQuestCoderForCausalLM": ("llama", "LlamaForCausalLM"),
    "IQuestLoopCoderForCausalLM": ("iquest_loopcoder", "IQuestLoopCoderForCausalLM"),
    "JAISLMHeadModel": ("jais", "JAISLMHeadModel"),
    "Jais2ForCausalLM": ("jais2", "Jais2ForCausalLM"),
    "JambaForCausalLM": ("jamba", "JambaForCausalLM"),
    "KimiLinearForCausalLM": ("kimi_linear", "KimiLinearForCausalLM"),  # noqa: E501
    "Lfm2ForCausalLM": ("lfm2", "Lfm2ForCausalLM"),
    "Lfm2MoeForCausalLM": ("lfm2_moe", "Lfm2MoeForCausalLM"),
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
    "Llama4ForCausalLM": ("llama4", "Llama4ForCausalLM"),
    # For decapoda-research/llama-*
    "LLaMAForCausalLM": ("llama", "LlamaForCausalLM"),
    "LongcatFlashForCausalLM": ("longcat_flash", "LongcatFlashForCausalLM"),
    "MambaForCausalLM": ("mamba", "MambaForCausalLM"),
    "Mamba2ForCausalLM": ("mamba2", "Mamba2ForCausalLM"),
    "MiniCPMForCausalLM": ("minicpm", "MiniCPMForCausalLM"),
    "MiniCPM3ForCausalLM": ("minicpm3", "MiniCPM3ForCausalLM"),
    "MiniMaxForCausalLM": ("minimax_text_01", "MiniMaxText01ForCausalLM"),
    "MiniMaxText01ForCausalLM": ("minimax_text_01", "MiniMaxText01ForCausalLM"),
    "MiniMaxM1ForCausalLM": ("minimax_text_01", "MiniMaxText01ForCausalLM"),
    "MiniMaxM2ForCausalLM": ("minimax_m2", "MiniMaxM2ForCausalLM"),
    "MistralForCausalLM": ("llama", "LlamaForCausalLM"),
    "MistralLarge3ForCausalLM": ("mistral_large_3", "MistralLarge3ForCausalLM"),
    "MixtralForCausalLM": ("mixtral", "MixtralForCausalLM"),
    # transformers's mpt class has lower case
    "MptForCausalLM": ("mpt", "MPTForCausalLM"),
    "MPTForCausalLM": ("mpt", "MPTForCausalLM"),
    "MiMoForCausalLM": ("mimo", "MiMoForCausalLM"),
    "MiMoV2FlashForCausalLM": ("mimo_v2_flash", "MiMoV2FlashForCausalLM"),
    "NemotronForCausalLM": ("nemotron", "NemotronForCausalLM"),
    "NemotronHForCausalLM": ("nemotron_h", "NemotronHForCausalLM"),
    "OlmoForCausalLM": ("olmo", "OlmoForCausalLM"),
    "Olmo2ForCausalLM": ("olmo2", "Olmo2ForCausalLM"),
    "Olmo3ForCausalLM": ("olmo2", "Olmo2ForCausalLM"),
    "OlmoeForCausalLM": ("olmoe", "OlmoeForCausalLM"),
    "OPTForCausalLM": ("opt", "OPTForCausalLM"),
    "OrionForCausalLM": ("orion", "OrionForCausalLM"),
    "OuroForCausalLM": ("ouro", "OuroForCausalLM"),
    "PanguEmbeddedForCausalLM": ("openpangu", "PanguEmbeddedForCausalLM"),
    "PanguProMoEV2ForCausalLM": ("openpangu", "PanguProMoEV2ForCausalLM"),
    "PanguUltraMoEForCausalLM": ("openpangu", "PanguUltraMoEForCausalLM"),
    "PersimmonForCausalLM": ("persimmon", "PersimmonForCausalLM"),
    "PhiForCausalLM": ("phi", "PhiForCausalLM"),
    "Phi3ForCausalLM": ("phi3", "Phi3ForCausalLM"),
    "PhiMoEForCausalLM": ("phimoe", "PhiMoEForCausalLM"),
    "Plamo2ForCausalLM": ("plamo2", "Plamo2ForCausalLM"),
    "Plamo3ForCausalLM": ("plamo3", "Plamo3ForCausalLM"),
    "QWenLMHeadModel": ("qwen", "QWenLMHeadModel"),
    "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
    "Qwen2MoeForCausalLM": ("qwen2_moe", "Qwen2MoeForCausalLM"),
    "Qwen3ForCausalLM": ("qwen3", "Qwen3ForCausalLM"),
    "Qwen3MoeForCausalLM": ("qwen3_moe", "Qwen3MoeForCausalLM"),
    "RWForCausalLM": ("falcon", "FalconForCausalLM"),
    "SeedOssForCausalLM": ("seed_oss", "SeedOssForCausalLM"),
    "Step3TextForCausalLM": ("step3_text", "Step3TextForCausalLM"),
    "StableLMEpochForCausalLM": ("stablelm", "StablelmForCausalLM"),
    "StableLmForCausalLM": ("stablelm", "StablelmForCausalLM"),
    "Starcoder2ForCausalLM": ("starcoder2", "Starcoder2ForCausalLM"),
    "SolarForCausalLM": ("solar", "SolarForCausalLM"),
    "TeleChatForCausalLM": ("telechat2", "TeleChat2ForCausalLM"),
    "TeleChat2ForCausalLM": ("telechat2", "TeleChat2ForCausalLM"),
    "TeleFLMForCausalLM": ("teleflm", "TeleFLMForCausalLM"),
    "XverseForCausalLM": ("llama", "LlamaForCausalLM"),
    "Zamba2ForCausalLM": ("zamba2", "Zamba2ForCausalLM"),
}

_EMBEDDING_MODELS = {
    # [Text-only]
    "BertModel": ("bert", "BertEmbeddingModel"),
    "BertSpladeSparseEmbeddingModel": ("bert", "BertSpladeSparseEmbeddingModel"),
    "DeciLMForCausalLM": ("nemotron_nas", "DeciLMForCausalLM"),
    "Gemma2Model": ("gemma2", "Gemma2ForCausalLM"),
    "Gemma3TextModel": ("gemma3", "Gemma3Model"),
    "GlmForCausalLM": ("glm", "GlmForCausalLM"),
    "GPT2ForSequenceClassification": ("gpt2", "GPT2ForSequenceClassification"),
    "GritLM": ("gritlm", "GritLM"),
    "GteModel": ("bert_with_rope", "SnowflakeGteNewModel"),
    "GteNewModel": ("bert_with_rope", "GteNewModel"),
    "InternLM2ForRewardModel": ("internlm2", "InternLM2ForRewardModel"),
    "JambaForSequenceClassification": ("jamba", "JambaForSequenceClassification"),  # noqa: E501
    "LlamaBidirectionalModel": ("llama", "LlamaBidirectionalModel"),
    "LlamaModel": ("llama", "LlamaForCausalLM"),
    **{
        # Multiple models share the same architecture, so we include them all
        k: (mod, arch)
        for k, (mod, arch) in _TEXT_GENERATION_MODELS.items()
        if arch == "LlamaForCausalLM"
    },
    "MistralModel": ("llama", "LlamaForCausalLM"),
    "ModernBertModel": ("modernbert", "ModernBertModel"),
    "NomicBertModel": ("bert_with_rope", "NomicBertModel"),
    "Phi3ForCausalLM": ("phi3", "Phi3ForCausalLM"),
    "Qwen2Model": ("qwen2", "Qwen2ForCausalLM"),
    "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
    "Qwen2ForRewardModel": ("qwen2_rm", "Qwen2ForRewardModel"),
    "Qwen2ForProcessRewardModel": ("qwen2_rm", "Qwen2ForProcessRewardModel"),
    "RobertaForMaskedLM": ("roberta", "RobertaEmbeddingModel"),
    "RobertaModel": ("roberta", "RobertaEmbeddingModel"),
    "TeleChatForCausalLM": ("telechat2", "TeleChat2ForCausalLM"),
    "TeleChat2ForCausalLM": ("telechat2", "TeleChat2ForCausalLM"),
    "XLMRobertaModel": ("roberta", "RobertaEmbeddingModel"),
    # [Multimodal]
    "CLIPModel": ("clip", "CLIPEmbeddingModel"),
    "LlavaNextForConditionalGeneration": (
        "llava_next",
        "LlavaNextForConditionalGeneration",
    ),
    "Phi3VForCausalLM": ("phi3v", "Phi3VForCausalLM"),
    "Qwen2VLForConditionalGeneration": ("qwen2_vl", "Qwen2VLForConditionalGeneration"),  # noqa: E501
    "SiglipModel": ("siglip", "SiglipEmbeddingModel"),
    # Technically Terratorch models work on images, both in
    # input and output. I am adding it here because it piggy-backs on embedding
    # models for the time being.
    "PrithviGeoSpatialMAE": ("terratorch", "Terratorch"),
    "Terratorch": ("terratorch", "Terratorch"),
}

_CROSS_ENCODER_MODELS = {
    "BertForSequenceClassification": ("bert", "BertForSequenceClassification"),
    "BertForTokenClassification": ("bert", "BertForTokenClassification"),
    "GteNewForSequenceClassification": (
        "bert_with_rope",
        "GteNewForSequenceClassification",
    ),
    "JinaVLForRanking": ("jina_vl", "JinaVLForSequenceClassification"),
    "LlamaBidirectionalForSequenceClassification": (
        "llama",
        "LlamaBidirectionalForSequenceClassification",
    ),
    "ModernBertForSequenceClassification": (
        "modernbert",
        "ModernBertForSequenceClassification",
    ),
    "ModernBertForTokenClassification": (
        "modernbert",
        "ModernBertForTokenClassification",
    ),
    "RobertaForSequenceClassification": ("roberta", "RobertaForSequenceClassification"),
    "XLMRobertaForSequenceClassification": (
        "roberta",
        "RobertaForSequenceClassification",
    ),
}

_MULTIMODAL_MODELS = {
    # [Decoder-only]
    "AriaForConditionalGeneration": ("aria", "AriaForConditionalGeneration"),
    "AudioFlamingo3ForConditionalGeneration": (
        "audioflamingo3",
        "AudioFlamingo3ForConditionalGeneration",
    ),
    "AyaVisionForConditionalGeneration": (
        "aya_vision",
        "AyaVisionForConditionalGeneration",
    ),
    "BagelForConditionalGeneration": ("bagel", "BagelForConditionalGeneration"),
    "BeeForConditionalGeneration": ("bee", "BeeForConditionalGeneration"),
    "Blip2ForConditionalGeneration": ("blip2", "Blip2ForConditionalGeneration"),
    "ChameleonForConditionalGeneration": (
        "chameleon",
        "ChameleonForConditionalGeneration",
    ),
    "Cohere2VisionForConditionalGeneration": (
        "cohere2_vision",
        "Cohere2VisionForConditionalGeneration",
    ),
    "DeepseekVLV2ForCausalLM": ("deepseek_vl2", "DeepseekVLV2ForCausalLM"),
    "DeepseekOCRForCausalLM": ("deepseek_ocr", "DeepseekOCRForCausalLM"),
    "DotsOCRForCausalLM": ("dots_ocr", "DotsOCRForCausalLM"),
    "Ernie4_5_VLMoeForConditionalGeneration": (
        "ernie45_vl",
        "Ernie4_5_VLMoeForConditionalGeneration",
    ),
    "FuyuForCausalLM": ("fuyu", "FuyuForCausalLM"),
    "Gemma3ForConditionalGeneration": ("gemma3_mm", "Gemma3ForConditionalGeneration"),  # noqa: E501
    "Gemma3nForConditionalGeneration": (
        "gemma3n_mm",
        "Gemma3nForConditionalGeneration",
    ),
    "GlmAsrForConditionalGeneration": ("glmasr", "GlmAsrForConditionalGeneration"),
    "GLM4VForCausalLM": ("glm4v", "GLM4VForCausalLM"),
    "Glm4vForConditionalGeneration": ("glm4_1v", "Glm4vForConditionalGeneration"),  # noqa: E501
    "Glm4vMoeForConditionalGeneration": ("glm4_1v", "Glm4vMoeForConditionalGeneration"),  # noqa: E501
    "GraniteSpeechForConditionalGeneration": (
        "granite_speech",
        "GraniteSpeechForConditionalGeneration",
    ),
    "H2OVLChatModel": ("h2ovl", "H2OVLChatModel"),
    "HunYuanVLForConditionalGeneration": (
        "hunyuan_vision",
        "HunYuanVLForConditionalGeneration",
    ),
    "StepVLForConditionalGeneration": ("step_vl", "StepVLForConditionalGeneration"),
    "InternVLChatModel": ("internvl", "InternVLChatModel"),
    "NemotronH_Nano_VL_V2": ("nano_nemotron_vl", "NemotronH_Nano_VL_V2"),
    "OpenCUAForConditionalGeneration": (
        "opencua",
        "OpenCUAForConditionalGeneration",
    ),
    "InternS1ForConditionalGeneration": (
        "interns1",
        "InternS1ForConditionalGeneration",
    ),
    "InternVLForConditionalGeneration": (
        "interns1",
        "InternS1ForConditionalGeneration",
    ),
    "Idefics3ForConditionalGeneration": (
        "idefics3",
        "Idefics3ForConditionalGeneration",
    ),
    "IsaacForConditionalGeneration": ("isaac", "IsaacForConditionalGeneration"),
    "SmolVLMForConditionalGeneration": ("smolvlm", "SmolVLMForConditionalGeneration"),  # noqa: E501
    "KananaVForConditionalGeneration": ("kanana_v", "KananaVForConditionalGeneration"),
    "KeyeForConditionalGeneration": ("keye", "KeyeForConditionalGeneration"),
    "KeyeVL1_5ForConditionalGeneration": (
        "keye_vl1_5",
        "KeyeVL1_5ForConditionalGeneration",
    ),
    "RForConditionalGeneration": ("rvl", "RForConditionalGeneration"),
    "KimiVLForConditionalGeneration": ("kimi_vl", "KimiVLForConditionalGeneration"),  # noqa: E501
    "LightOnOCRForConditionalGeneration": (
        "lightonocr",
        "LightOnOCRForConditionalGeneration",
    ),
    "Lfm2VlForConditionalGeneration": ("lfm2_vl", "Lfm2VLForConditionalGeneration"),
    "Llama_Nemotron_Nano_VL": ("nemotron_vl", "LlamaNemotronVLChatModel"),
    "Llama4ForConditionalGeneration": ("mllama4", "Llama4ForConditionalGeneration"),  # noqa: E501
    "LlavaForConditionalGeneration": ("llava", "LlavaForConditionalGeneration"),
    "LlavaNextForConditionalGeneration": (
        "llava_next",
        "LlavaNextForConditionalGeneration",
    ),
    "LlavaNextVideoForConditionalGeneration": (
        "llava_next_video",
        "LlavaNextVideoForConditionalGeneration",
    ),
    "LlavaOnevisionForConditionalGeneration": (
        "llava_onevision",
        "LlavaOnevisionForConditionalGeneration",
    ),
    "MantisForConditionalGeneration": ("llava", "MantisForConditionalGeneration"),  # noqa: E501
    "MiDashengLMModel": ("midashenglm", "MiDashengLMModel"),
    "MiniMaxVL01ForConditionalGeneration": (
        "minimax_vl_01",
        "MiniMaxVL01ForConditionalGeneration",
    ),
    "MiniCPMO": ("minicpmo", "MiniCPMO"),
    "MiniCPMV": ("minicpmv", "MiniCPMV"),
    "Mistral3ForConditionalGeneration": (
        "mistral3",
        "Mistral3ForConditionalGeneration",
    ),
    "MolmoForCausalLM": ("molmo", "MolmoForCausalLM"),
    "Molmo2ForConditionalGeneration": ("molmo2", "Molmo2ForConditionalGeneration"),
    "NVLM_D": ("nvlm_d", "NVLM_D_Model"),
    "Ovis": ("ovis", "Ovis"),
    "Ovis2_5": ("ovis2_5", "Ovis2_5"),
    "PaddleOCRVLForConditionalGeneration": (
        "paddleocr_vl",
        "PaddleOCRVLForConditionalGeneration",
    ),
    "PaliGemmaForConditionalGeneration": (
        "paligemma",
        "PaliGemmaForConditionalGeneration",
    ),
    "Phi3VForCausalLM": ("phi3v", "Phi3VForCausalLM"),
    "Phi4MMForCausalLM": ("phi4mm", "Phi4MMForCausalLM"),
    "PixtralForConditionalGeneration": ("pixtral", "PixtralForConditionalGeneration"),  # noqa: E501
    "QwenVLForConditionalGeneration": ("qwen_vl", "QwenVLForConditionalGeneration"),  # noqa: E501
    "Qwen2VLForConditionalGeneration": ("qwen2_vl", "Qwen2VLForConditionalGeneration"),  # noqa: E501
    "Qwen2_5_VLForConditionalGeneration": (
        "qwen2_5_vl",
        "Qwen2_5_VLForConditionalGeneration",
    ),
    "Qwen2AudioForConditionalGeneration": (
        "qwen2_audio",
        "Qwen2AudioForConditionalGeneration",
    ),
    "Qwen2_5OmniModel": (
        "qwen2_5_omni_thinker",
        "Qwen2_5OmniThinkerForConditionalGeneration",
    ),
    "Qwen2_5OmniForConditionalGeneration": (
        "qwen2_5_omni_thinker",
        "Qwen2_5OmniThinkerForConditionalGeneration",
    ),
    "Qwen3OmniMoeForConditionalGeneration": (
        "qwen3_omni_moe_thinker",
        "Qwen3OmniMoeThinkerForConditionalGeneration",
    ),
    "Qwen3VLForConditionalGeneration": ("qwen3_vl", "Qwen3VLForConditionalGeneration"),  # noqa: E501
    "Qwen3VLMoeForConditionalGeneration": (
        "qwen3_vl_moe",
        "Qwen3VLMoeForConditionalGeneration",
    ),
    "SkyworkR1VChatModel": ("skyworkr1v", "SkyworkR1VChatModel"),
    "Step3VLForConditionalGeneration": ("step3_vl", "Step3VLForConditionalGeneration"),  # noqa: E501
    "TarsierForConditionalGeneration": ("tarsier", "TarsierForConditionalGeneration"),  # noqa: E501
    "Tarsier2ForConditionalGeneration": (
        "qwen2_vl",
        "Tarsier2ForConditionalGeneration",
    ),
    "UltravoxModel": ("ultravox", "UltravoxModel"),
    "VoxtralForConditionalGeneration": ("voxtral", "VoxtralForConditionalGeneration"),  # noqa: E501
    "VoxtralStreamingGeneration": ("voxtral_streaming", "VoxtralStreamingGeneration"),  # noqa: E501
    # [Encoder-decoder]
    "NemotronParseForConditionalGeneration": (
        "nemotron_parse",
        "NemotronParseForConditionalGeneration",
    ),
    "WhisperForConditionalGeneration": ("whisper", "WhisperForConditionalGeneration"),  # noqa: E501
}

_SPECULATIVE_DECODING_MODELS = {
    "MiMoMTPModel": ("mimo_mtp", "MiMoMTP"),
    "EagleLlamaForCausalLM": ("llama_eagle", "EagleLlamaForCausalLM"),
    "EagleLlama4ForCausalLM": ("llama4_eagle", "EagleLlama4ForCausalLM"),
    "EagleMiniCPMForCausalLM": ("minicpm_eagle", "EagleMiniCPMForCausalLM"),
    "Eagle3LlamaForCausalLM": ("llama_eagle3", "Eagle3LlamaForCausalLM"),
    "LlamaForCausalLMEagle3": ("llama_eagle3", "Eagle3LlamaForCausalLM"),
    "Eagle3Qwen2_5vlForCausalLM": ("llama_eagle3", "Eagle3LlamaForCausalLM"),
    "Eagle3Qwen3vlForCausalLM": ("llama_eagle3", "Eagle3LlamaForCausalLM"),
    "EagleMistralLarge3ForCausalLM": (
        "mistral_large_3_eagle",
        "EagleMistralLarge3ForCausalLM",
    ),
    "EagleDeepSeekMTPModel": ("deepseek_eagle", "EagleDeepseekV3ForCausalLM"),
    "DeepSeekMTPModel": ("deepseek_mtp", "DeepSeekMTP"),
    "ErnieMTPModel": ("ernie_mtp", "ErnieMTP"),
    "ExaoneMoeMTP": ("exaone_moe_mtp", "ExaoneMoeMTP"),
    "LongCatFlashMTPModel": ("longcat_flash_mtp", "LongCatFlashMTP"),
    "Glm4MoeMTPModel": ("glm4_moe_mtp", "Glm4MoeMTP"),
    "MedusaModel": ("medusa", "Medusa"),
    "OpenPanguMTPModel": ("openpangu_mtp", "OpenPanguMTP"),
    "Qwen3NextMTP": ("qwen3_next_mtp", "Qwen3NextMTP"),
    # Temporarily disabled.
    # # TODO(woosuk): Re-enable this once the MLP Speculator is supported in V1.
    # "MLPSpeculatorPreTrainedModel": ("mlp_speculator", "MLPSpeculator"),
}

_TRANSFORMERS_SUPPORTED_MODELS = {
    # Text generation models
    "SmolLM3ForCausalLM": ("transformers", "TransformersForCausalLM"),
    # Multimodal models
    "Emu3ForConditionalGeneration": (
        "transformers",
        "TransformersMultiModalForCausalLM",
    ),
}

_TRANSFORMERS_BACKEND_MODELS = {
    # Text generation models
    "TransformersForCausalLM": ("transformers", "TransformersForCausalLM"),
    "TransformersMoEForCausalLM": ("transformers", "TransformersMoEForCausalLM"),
    # Multimodal models
    "TransformersMultiModalForCausalLM": (
        "transformers",
        "TransformersMultiModalForCausalLM",
    ),
    "TransformersMultiModalMoEForCausalLM": (
        "transformers",
        "TransformersMultiModalMoEForCausalLM",
    ),
    # Embedding models
    "TransformersEmbeddingModel": ("transformers", "TransformersEmbeddingModel"),
    "TransformersMoEEmbeddingModel": ("transformers", "TransformersMoEEmbeddingModel"),
    "TransformersMultiModalEmbeddingModel": (
        "transformers",
        "TransformersMultiModalEmbeddingModel",
    ),
    # Sequence classification models
    "TransformersForSequenceClassification": (
        "transformers",
        "TransformersForSequenceClassification",
    ),
    "TransformersMoEForSequenceClassification": (
        "transformers",
        "TransformersMoEForSequenceClassification",
    ),
    "TransformersMultiModalForSequenceClassification": (
        "transformers",
        "TransformersMultiModalForSequenceClassification",
    ),
}

_VLLM_MODELS = {
    **_TEXT_GENERATION_MODELS,
    **_EMBEDDING_MODELS,
    **_CROSS_ENCODER_MODELS,
    **_MULTIMODAL_MODELS,
    **_SPECULATIVE_DECODING_MODELS,
    **_TRANSFORMERS_SUPPORTED_MODELS,
    **_TRANSFORMERS_BACKEND_MODELS,
}

# This variable is used as the args for subprocess.run(). We
# can modify  this variable to alter the args if needed. e.g.
# when we use par format to pack things together, sys.executable
# might not be the target we want to run.
_SUBPROCESS_COMMAND = [sys.executable, "-m", "vllm.model_executor.models.registry"]

_PREVIOUSLY_SUPPORTED_MODELS = {
    "MotifForCausalLM": "0.10.2",
    "Phi3SmallForCausalLM": "0.9.2",
    "Phi4FlashForCausalLM": "0.10.2",
    "Phi4MultimodalForCausalLM": "0.12.0",
    # encoder-decoder models except whisper
    # have been removed for V0 deprecation.
    "BartModel": "0.10.2",
    "BartForConditionalGeneration": "0.10.2",
    "DonutForConditionalGeneration": "0.10.2",
    "Florence2ForConditionalGeneration": "0.10.2",
    "MBartForConditionalGeneration": "0.10.2",
    "MllamaForConditionalGeneration": "0.10.2",
}


@dataclass(frozen=True)
class _ModelInfo:
    architecture: str
    is_text_generation_model: bool
    is_pooling_model: bool
    attn_type: AttnTypeStr
    default_seq_pooling_type: SequencePoolingType
    default_tok_pooling_type: TokenPoolingType
    supports_cross_encoding: bool
    supports_multimodal: bool
    supports_multimodal_raw_input_only: bool
    requires_raw_input_tokens: bool
    supports_multimodal_encoder_tp_data: bool
    supports_pp: bool
    has_inner_state: bool
    is_attention_free: bool
    is_hybrid: bool
    has_noops: bool
    supports_mamba_prefix_caching: bool
    supports_transcription: bool
    supports_transcription_only: bool

    @staticmethod
    def from_model_cls(model: type[nn.Module]) -> "_ModelInfo":
        return _ModelInfo(
            architecture=model.__name__,
            is_text_generation_model=is_text_generation_model(model),
            is_pooling_model=is_pooling_model(model),
            default_seq_pooling_type=get_default_seq_pooling_type(model),
            default_tok_pooling_type=get_default_tok_pooling_type(model),
            attn_type=get_attn_type(model),
            supports_cross_encoding=supports_cross_encoding(model),
            supports_multimodal=supports_multimodal(model),
            supports_multimodal_raw_input_only=supports_multimodal_raw_input_only(
                model
            ),
            requires_raw_input_tokens=requires_raw_input_tokens(model),
            supports_multimodal_encoder_tp_data=supports_multimodal_encoder_tp_data(
                model
            ),
            supports_pp=supports_pp(model),
            has_inner_state=has_inner_state(model),
            is_attention_free=is_attention_free(model),
            is_hybrid=is_hybrid(model),
            supports_mamba_prefix_caching=supports_mamba_prefix_caching(model),
            supports_transcription=supports_transcription(model),
            supports_transcription_only=(
                supports_transcription(model) and model.supports_transcription_only
            ),
            has_noops=has_noops(model),
        )


class _BaseRegisteredModel(ABC):
    @abstractmethod
    def inspect_model_cls(self) -> _ModelInfo:
        raise NotImplementedError

    @abstractmethod
    def load_model_cls(self) -> type[nn.Module]:
        raise NotImplementedError


@dataclass(frozen=True)
class _RegisteredModel(_BaseRegisteredModel):
    """
    Represents a model that has already been imported in the main process.
    """

    interfaces: _ModelInfo
    model_cls: type[nn.Module]

    @staticmethod
    def from_model_cls(model_cls: type[nn.Module]):
        return _RegisteredModel(
            interfaces=_ModelInfo.from_model_cls(model_cls),
            model_cls=model_cls,
        )

    def inspect_model_cls(self) -> _ModelInfo:
        return self.interfaces

    def load_model_cls(self) -> type[nn.Module]:
        return self.model_cls


@dataclass(frozen=True)
class _LazyRegisteredModel(_BaseRegisteredModel):
    """
    Represents a model that has not been imported in the main process.
    """

    module_name: str
    class_name: str

    @staticmethod
    def _get_cache_dir() -> Path:
        return Path(envs.VLLM_CACHE_ROOT) / "modelinfos"

    def _get_cache_filename(self) -> str:
        cls_name = f"{self.module_name}-{self.class_name}".replace(".", "-")
        return f"{cls_name}.json"

    def _load_modelinfo_from_cache(self, module_hash: str) -> _ModelInfo | None:
        try:
            try:
                modelinfo_path = self._get_cache_dir() / self._get_cache_filename()
                with open(modelinfo_path, encoding="utf-8") as file:
                    mi_dict = json.load(file)
            except FileNotFoundError:
                logger.debug(
                    "Cached model info file for class %s.%s not found",
                    self.module_name,
                    self.class_name,
                )
                return None

            if mi_dict["hash"] != module_hash:
                logger.debug(
                    "Cached model info file for class %s.%s is stale",
                    self.module_name,
                    self.class_name,
                )
                return None

            # file not changed, use cached _ModelInfo properties
            return _ModelInfo(**mi_dict["modelinfo"])
        except Exception:
            logger.debug(
                "Cached model info for class %s.%s error. ",
                self.module_name,
                self.class_name,
            )
            return None

    def _save_modelinfo_to_cache(self, mi: _ModelInfo, module_hash: str) -> None:
        """save dictionary json file to cache"""
        from vllm.model_executor.model_loader.weight_utils import atomic_writer

        try:
            modelinfo_dict = {
                "hash": module_hash,
                "modelinfo": asdict(mi),
            }
            cache_dir = self._get_cache_dir()
            cache_dir.mkdir(parents=True, exist_ok=True)
            modelinfo_path = cache_dir / self._get_cache_filename()
            with atomic_writer(modelinfo_path, encoding="utf-8") as f:
                json.dump(modelinfo_dict, f, indent=2)
        except Exception:
            logger.exception("Error saving model info cache.")

    @logtime(logger=logger, msg="Registry inspect model class")
    def inspect_model_cls(self) -> _ModelInfo:
        model_path = Path(__file__).parent / f"{self.module_name.split('.')[-1]}.py"
        module_hash = None

        if model_path.exists():
            with open(model_path, "rb") as f:
                module_hash = safe_hash(f.read(), usedforsecurity=False).hexdigest()

            mi = self._load_modelinfo_from_cache(module_hash)
            if mi is not None:
                logger.debug(
                    "Loaded model info for class %s.%s from cache",
                    self.module_name,
                    self.class_name,
                )
                return mi
            else:
                logger.debug(
                    "Cache model info for class %s.%s miss. Loading model instead.",
                    self.module_name,
                    self.class_name,
                )

        # Performed in another process to avoid initializing CUDA
        mi = _run_in_subprocess(
            lambda: _ModelInfo.from_model_cls(self.load_model_cls())
        )
        logger.debug(
            "Loaded model info for class %s.%s", self.module_name, self.class_name
        )

        # save cache file
        if module_hash is not None:
            self._save_modelinfo_to_cache(mi, module_hash)

        return mi

    def load_model_cls(self) -> type[nn.Module]:
        mod = importlib.import_module(self.module_name)
        return getattr(mod, self.class_name)


@lru_cache(maxsize=128)
def _try_load_model_cls(
    model_arch: str,
    model: _BaseRegisteredModel,
) -> type[nn.Module] | None:
    from vllm.platforms import current_platform

    current_platform.verify_model_arch(model_arch)
    try:
        return model.load_model_cls()
    except Exception:
        logger.exception("Error in loading model architecture '%s'", model_arch)
        return None


@lru_cache(maxsize=128)
def _try_inspect_model_cls(
    model_arch: str,
    model: _BaseRegisteredModel,
) -> _ModelInfo | None:
    try:
        return model.inspect_model_cls()
    except Exception:
        logger.exception("Error in inspecting model architecture '%s'", model_arch)
        return None


@dataclass
class _ModelRegistry:
    # Keyed by model_arch
    models: dict[str, _BaseRegisteredModel] = field(default_factory=dict)

    def get_supported_archs(self) -> Set[str]:
        return self.models.keys()

    def register_model(
        self,
        model_arch: str,
        model_cls: type[nn.Module] | str,
    ) -> None:
        """
        Register an external model to be used in vLLM.

        `model_cls` can be either:

        - A [`torch.nn.Module`][] class directly referencing the model.
        - A string in the format `<module>:<class>` which can be used to
          lazily import the model. This is useful to avoid initializing CUDA
          when importing the model and thus the related error
          `RuntimeError: Cannot re-initialize CUDA in forked subprocess`.
        """
        if not isinstance(model_arch, str):
            msg = f"`model_arch` should be a string, not a {type(model_arch)}"
            raise TypeError(msg)

        if model_arch in self.models:
            logger.warning(
                "Model architecture %s is already registered, and will be "
                "overwritten by the new model class %s.",
                model_arch,
                model_cls,
            )

        if isinstance(model_cls, str):
            split_str = model_cls.split(":")
            if len(split_str) != 2:
                msg = "Expected a string in the format `<module>:<class>`"
                raise ValueError(msg)

            model = _LazyRegisteredModel(*split_str)
        elif isinstance(model_cls, type) and issubclass(model_cls, nn.Module):
            model = _RegisteredModel.from_model_cls(model_cls)
        else:
            msg = (
                "`model_cls` should be a string or PyTorch model class, "
                f"not a {type(model_arch)}"
            )
            raise TypeError(msg)

        self.models[model_arch] = model

    def _raise_for_unsupported(self, architectures: list[str]):
        all_supported_archs = self.get_supported_archs()

        if any(arch in all_supported_archs for arch in architectures):
            raise ValueError(
                f"Model architectures {architectures} failed "
                "to be inspected. Please check the logs for more details."
            )

        for arch in architectures:
            if arch in _PREVIOUSLY_SUPPORTED_MODELS:
                previous_version = _PREVIOUSLY_SUPPORTED_MODELS[arch]

                raise ValueError(
                    f"Model architecture {arch} was supported in vLLM until "
                    f"v{previous_version}, and is not supported anymore. "
                    "Please use an older version of vLLM if you want to "
                    "use this model architecture."
                )

        raise ValueError(
            f"Model architectures {architectures} are not supported for now. "
            f"Supported architectures: {all_supported_archs}"
        )

    def _try_load_model_cls(self, model_arch: str) -> type[nn.Module] | None:
        if model_arch not in self.models:
            return None

        return _try_load_model_cls(model_arch, self.models[model_arch])

    def _try_inspect_model_cls(self, model_arch: str) -> _ModelInfo | None:
        if model_arch not in self.models:
            return None

        return _try_inspect_model_cls(model_arch, self.models[model_arch])

    def _try_resolve_transformers(
        self,
        architecture: str,
        model_config: ModelConfig,
    ) -> str | None:
        if architecture in _TRANSFORMERS_BACKEND_MODELS:
            return architecture

        auto_map: dict[str, str] = (
            getattr(model_config.hf_config, "auto_map", None) or dict()
        )

        # Make sure that config class is always initialized before model class,
        # otherwise the model class won't be able to access the config class,
        # the expected auto_map should have correct order like:
        # "auto_map": {
        #     "AutoConfig": "<your-repo-name>--<config-name>",
        #     "AutoModel": "<your-repo-name>--<config-name>",
        #     "AutoModelFor<Task>": "<your-repo-name>--<config-name>",
        # },
        for prefix in ("AutoConfig", "AutoModel"):
            for name, module in auto_map.items():
                if name.startswith(prefix):
                    try_get_class_from_dynamic_module(
                        module,
                        model_config.model,
                        revision=model_config.revision,
                        trust_remote_code=model_config.trust_remote_code,
                        warn_on_fail=False,
                    )

        model_module = getattr(transformers, architecture, None)

        if model_module is None:
            for name, module in auto_map.items():
                if name.startswith("AutoModel"):
                    model_module = try_get_class_from_dynamic_module(
                        module,
                        model_config.model,
                        revision=model_config.revision,
                        trust_remote_code=model_config.trust_remote_code,
                        warn_on_fail=True,
                    )
                    if model_module is not None:
                        break
            else:
                if model_config.model_impl != "transformers":
                    return None

                raise ValueError(
                    f"Cannot find model module. {architecture!r} is not a "
                    "registered model in the Transformers library (only "
                    "relevant if the model is meant to be in Transformers) "
                    "and 'AutoModel' is not present in the model config's "
                    "'auto_map' (relevant if the model is custom)."
                )

        if not model_module.is_backend_compatible():
            if model_config.model_impl != "transformers":
                return None

            raise ValueError(
                f"The Transformers implementation of {architecture!r} "
                "is not compatible with vLLM."
            )

        return model_config._get_transformers_backend_cls()

    def _normalize_arch(
        self,
        architecture: str,
        model_config: ModelConfig,
    ) -> str:
        if architecture in self.models:
            return architecture

        # This may be called in order to resolve runner_type and convert_type
        # in the first place, in which case we consider the default match
        match = try_match_architecture_defaults(
            architecture,
            runner_type=getattr(model_config, "runner_type", None),
            convert_type=getattr(model_config, "convert_type", None),
        )
        if match:
            suffix, _ = match

            # Get the name of the base model to convert
            for repl_suffix, _ in iter_architecture_defaults():
                base_arch = architecture.replace(suffix, repl_suffix)
                if base_arch in self.models:
                    return base_arch

        return architecture

    def inspect_model_cls(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> tuple[_ModelInfo, str]:
        if isinstance(architectures, str):
            architectures = [architectures]
        if not architectures:
            raise ValueError("No model architectures are specified")

        # Require transformers impl
        if model_config.model_impl == "transformers":
            arch = self._try_resolve_transformers(architectures[0], model_config)
            if arch is not None:
                model_info = self._try_inspect_model_cls(arch)
                if model_info is not None:
                    return (model_info, arch)
        elif model_config.model_impl == "terratorch":
            model_info = self._try_inspect_model_cls("Terratorch")
            return (model_info, "Terratorch")

        # Fallback to transformers impl (after resolving convert_type)
        if (
            all(arch not in self.models for arch in architectures)
            and model_config.model_impl == "auto"
            and getattr(model_config, "convert_type", "none") == "none"
        ):
            arch = self._try_resolve_transformers(architectures[0], model_config)
            if arch is not None:
                model_info = self._try_inspect_model_cls(arch)
                if model_info is not None:
                    return (model_info, arch)

        for arch in architectures:
            normalized_arch = self._normalize_arch(arch, model_config)
            model_info = self._try_inspect_model_cls(normalized_arch)
            if model_info is not None:
                return (model_info, arch)

        # Fallback to transformers impl (before resolving runner_type)
        if (
            all(arch not in self.models for arch in architectures)
            and model_config.model_impl == "auto"
        ):
            arch = self._try_resolve_transformers(architectures[0], model_config)
            if arch is not None:
                model_info = self._try_inspect_model_cls(arch)
                if model_info is not None:
                    return (model_info, arch)

        return self._raise_for_unsupported(architectures)

    def resolve_model_cls(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> tuple[type[nn.Module], str]:
        if isinstance(architectures, str):
            architectures = [architectures]
        if not architectures:
            raise ValueError("No model architectures are specified")

        # Require transformers impl
        if model_config.model_impl == "transformers":
            arch = self._try_resolve_transformers(architectures[0], model_config)
            if arch is not None:
                model_cls = self._try_load_model_cls(arch)
                if model_cls is not None:
                    return (model_cls, arch)
        elif model_config.model_impl == "terratorch":
            arch = "Terratorch"
            model_cls = self._try_load_model_cls(arch)
            if model_cls is not None:
                return (model_cls, arch)

        # Fallback to transformers impl (after resolving convert_type)
        if (
            all(arch not in self.models for arch in architectures)
            and model_config.model_impl == "auto"
            and getattr(model_config, "convert_type", "none") == "none"
        ):
            arch = self._try_resolve_transformers(architectures[0], model_config)
            if arch is not None:
                model_cls = self._try_load_model_cls(arch)
                if model_cls is not None:
                    return (model_cls, arch)

        for arch in architectures:
            normalized_arch = self._normalize_arch(arch, model_config)
            model_cls = self._try_load_model_cls(normalized_arch)
            if model_cls is not None:
                return (model_cls, arch)

        # Fallback to transformers impl (before resolving runner_type)
        if (
            all(arch not in self.models for arch in architectures)
            and model_config.model_impl == "auto"
        ):
            arch = self._try_resolve_transformers(architectures[0], model_config)
            if arch is not None:
                model_cls = self._try_load_model_cls(arch)
                if model_cls is not None:
                    return (model_cls, arch)

        return self._raise_for_unsupported(architectures)

    def is_text_generation_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> bool:
        model_cls, _ = self.inspect_model_cls(architectures, model_config)
        return model_cls.is_text_generation_model

    def is_pooling_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> bool:
        model_cls, _ = self.inspect_model_cls(architectures, model_config)
        return model_cls.is_pooling_model

    def is_cross_encoder_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> bool:
        model_cls, _ = self.inspect_model_cls(architectures, model_config)
        return model_cls.supports_cross_encoding

    def is_multimodal_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> bool:
        model_cls, _ = self.inspect_model_cls(architectures, model_config)
        return model_cls.supports_multimodal

    def is_multimodal_raw_input_only_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> bool:
        model_cls, _ = self.inspect_model_cls(architectures, model_config)
        return model_cls.supports_multimodal_raw_input_only

    def is_pp_supported_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> bool:
        model_cls, _ = self.inspect_model_cls(architectures, model_config)
        return model_cls.supports_pp

    def model_has_inner_state(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> bool:
        model_cls, _ = self.inspect_model_cls(architectures, model_config)
        return model_cls.has_inner_state

    def is_attention_free_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> bool:
        model_cls, _ = self.inspect_model_cls(architectures, model_config)
        return model_cls.is_attention_free

    def is_hybrid_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> bool:
        model_cls, _ = self.inspect_model_cls(architectures, model_config)
        return model_cls.is_hybrid

    def is_noops_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> bool:
        model_cls, _ = self.inspect_model_cls(architectures, model_config)
        return model_cls.has_noops

    def is_transcription_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> bool:
        model_cls, _ = self.inspect_model_cls(architectures, model_config)
        return model_cls.supports_transcription

    def is_transcription_only_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> bool:
        model_cls, _ = self.inspect_model_cls(architectures, model_config)
        return model_cls.supports_transcription_only


ModelRegistry = _ModelRegistry(
    {
        model_arch: _LazyRegisteredModel(
            module_name=f"vllm.model_executor.models.{mod_relname}",
            class_name=cls_name,
        )
        for model_arch, (mod_relname, cls_name) in _VLLM_MODELS.items()
    }
)

_T = TypeVar("_T")


def _run_in_subprocess(fn: Callable[[], _T]) -> _T:
    # NOTE: We use a temporary directory instead of a temporary file to avoid
    # issues like https://stackoverflow.com/questions/23212435/permission-denied-to-write-to-my-temporary-file
    with tempfile.TemporaryDirectory() as tempdir:
        output_filepath = os.path.join(tempdir, "registry_output.tmp")

        # `cloudpickle` allows pickling lambda functions directly
        import cloudpickle

        input_bytes = cloudpickle.dumps((fn, output_filepath))

        # cannot use `sys.executable __file__` here because the script
        # contains relative imports
        returned = subprocess.run(
            _SUBPROCESS_COMMAND, input=input_bytes, capture_output=True
        )

        # check if the subprocess is successful
        try:
            returned.check_returncode()
        except Exception as e:
            # wrap raised exception to provide more information
            raise RuntimeError(
                f"Error raised in subprocess:\n{returned.stderr.decode()}"
            ) from e

        with open(output_filepath, "rb") as f:
            return pickle.load(f)


def _run() -> None:
    # Setup plugins
    from vllm.plugins import load_general_plugins

    load_general_plugins()

    fn, output_file = pickle.loads(sys.stdin.buffer.read())

    result = fn()

    with open(output_file, "wb") as f:
        f.write(pickle.dumps(result))


if __name__ == "__main__":
    _run()
