# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checked metadata catalog for built-in models.

Unlisted implementations fall back to dynamic inspection.
"""

from typing import Any, TypeAlias

_Profile: TypeAlias = dict[str, Any]

_MODEL_INFO_DEFAULTS: _Profile = {
    "is_text_generation_model": False,
    "is_pooling_model": False,
    "attn_type": "decoder",
    "default_seq_pooling_type": "LAST",
    "default_tok_pooling_type": "ALL",
    "score_type": "bi-encoder",
    "supports_multimodal": False,
    "supports_multimodal_raw_input_only": False,
    "requires_raw_input_tokens": False,
    "supports_multimodal_encoder_tp_data": False,
    "supports_pp": False,
    "has_inner_state": False,
    "is_attention_free": False,
    "is_hybrid": False,
    "has_noops": False,
    "supports_mamba_prefix_caching": False,
    "supports_replayssm": False,
    "supports_transcription": False,
    "supports_transcription_only": False,
    "supported_video_pruning_methods": (),
    "supports_mm_device_do_normalize": False,
}


_TEXT_GENERATION_PP = dict(
    is_text_generation_model=True,
    supports_pp=True,
)

_MULTIMODAL_GENERATION_PP = dict(
    is_text_generation_model=True,
    supports_multimodal=True,
    supports_pp=True,
)

_TEXT_GENERATION = dict(
    is_text_generation_model=True,
)

_MULTIMODAL_ENCODER_TP_GENERATION_PP = dict(
    is_text_generation_model=True,
    supports_multimodal=True,
    supports_multimodal_encoder_tp_data=True,
    supports_pp=True,
)

_MULTIMODAL_GENERATION = dict(
    is_text_generation_model=True,
    supports_multimodal=True,
)

_HYBRID_GENERATION_PP = dict(
    is_text_generation_model=True,
    supports_pp=True,
    has_inner_state=True,
    is_hybrid=True,
)

_TRANSCRIPTION_GENERATION_PP = dict(
    is_text_generation_model=True,
    supports_multimodal=True,
    supports_pp=True,
    supports_transcription=True,
)

_DEFAULT_PROFILE: _Profile = {}

_ENCODER_POOLING = dict(
    is_pooling_model=True,
    attn_type="encoder_only",
)

_TRANSCRIPTION_ONLY_MULTIMODAL_GENERATION = dict(
    is_text_generation_model=True,
    supports_multimodal=True,
    supports_transcription=True,
    supports_transcription_only=True,
)

_CLS_DEFAULT = dict(
    default_seq_pooling_type="CLS",
)

_CLS_POOLING = dict(
    is_pooling_model=True,
    default_seq_pooling_type="CLS",
)

_CLS_CROSS_ENCODER = dict(
    is_pooling_model=True,
    default_seq_pooling_type="CLS",
    score_type="cross-encoder",
)

_HYBRID_VIDEO_PRUNING_MULTIMODAL_ENCODER_TP_GENERATION_PP = dict(
    is_text_generation_model=True,
    supports_multimodal=True,
    supports_multimodal_encoder_tp_data=True,
    supports_pp=True,
    is_hybrid=True,
    supported_video_pruning_methods=("evs", "vidcom2"),
)

_VIDEO_PRUNING_MULTIMODAL_ENCODER_TP_GENERATION_PP = dict(
    is_text_generation_model=True,
    supports_multimodal=True,
    supports_multimodal_encoder_tp_data=True,
    supports_pp=True,
    supported_video_pruning_methods=("evs", "vidcom2"),
)

_CLS_LATE_INTERACTION = dict(
    is_pooling_model=True,
    default_seq_pooling_type="CLS",
    score_type="late-interaction",
)

_MAMBA_HYBRID_GENERATION_PP = dict(
    is_text_generation_model=True,
    supports_pp=True,
    has_inner_state=True,
    is_hybrid=True,
    supports_mamba_prefix_caching=True,
)

_GENERATION_POOLING_PP = dict(
    is_text_generation_model=True,
    is_pooling_model=True,
    supports_pp=True,
)

_EVS_NORMALIZED_MULTIMODAL_ENCODER_TP_GENERATION_PP = dict(
    is_text_generation_model=True,
    supports_multimodal=True,
    supports_multimodal_encoder_tp_data=True,
    supports_pp=True,
    supported_video_pruning_methods=("evs",),
    supports_mm_device_do_normalize=True,
)

_CLS_CROSS_ENCODER_PP = dict(
    is_pooling_model=True,
    default_seq_pooling_type="CLS",
    score_type="cross-encoder",
    supports_pp=True,
)

_CLS_POOLING_PP = dict(
    is_pooling_model=True,
    default_seq_pooling_type="CLS",
    supports_pp=True,
)

_HYBRID_MULTIMODAL_ENCODER_TP_GENERATION_PP = dict(
    is_text_generation_model=True,
    supports_multimodal=True,
    supports_multimodal_encoder_tp_data=True,
    supports_pp=True,
    has_inner_state=True,
    is_hybrid=True,
)

_ModelInfoImplementation = tuple[str, str]
_ProfileGroup = tuple[_Profile, tuple[_ModelInfoImplementation, ...]]

_DECI_LM_PROFILE_GROUP: _ProfileGroup = (
    dict(
        is_text_generation_model=True,
        supports_pp=True,
        has_noops=True,
    ),
    (("nemotron_nas", "DeciLMForCausalLM"),),
)

_MODEL_INFO_PROFILE_GROUPS: tuple[_ProfileGroup, ...] = (
    (
        _ENCODER_POOLING,
        (
            ("bert", "BertForMaskedLM"),
            ("bert", "BertForTokenClassification"),
            ("modernbert", "ModernBertForTokenClassification"),
            ("openai_privacy_filter", "OpenAIPrivacyFilterForTokenClassification"),
            ("roberta", "RobertaForTokenClassification"),
        ),
    ),
    (
        dict(
            is_pooling_model=True,
            default_seq_pooling_type="CLS",
            score_type="late-interaction",
            has_inner_state=True,
            is_hybrid=True,
        ),
        (("colbert", "ColBERTLfm2Model"),),
    ),
    (
        dict(
            is_text_generation_model=True,
            is_pooling_model=True,
            default_seq_pooling_type="CLS",
            score_type="late-interaction",
            supports_multimodal=True,
            supports_multimodal_encoder_tp_data=True,
            supports_pp=True,
            is_hybrid=True,
            supported_video_pruning_methods=("evs", "vidcom2"),
        ),
        (("colqwen3_5", "ColQwen3_5Model"),),
    ),
    (
        dict(
            is_text_generation_model=True,
            is_pooling_model=True,
            default_seq_pooling_type="CLS",
            score_type="late-interaction",
            supports_multimodal=True,
            supports_multimodal_encoder_tp_data=True,
            supports_pp=True,
            supported_video_pruning_methods=("evs", "vidcom2"),
        ),
        (("colqwen3", "ColQwen3Model"),),
    ),
    (
        dict(
            is_text_generation_model=True,
            is_pooling_model=True,
            default_seq_pooling_type="CLS",
            score_type="late-interaction",
            supports_multimodal=True,
            supports_pp=True,
        ),
        (("colpali", "ColPaliModel"),),
    ),
    (
        dict(
            is_pooling_model=True,
            default_seq_pooling_type="CLS",
            score_type="cross-encoder",
            supports_multimodal=True,
            supports_pp=True,
        ),
        (("transformers", "TransformersMultiModalForSequenceClassification"),),
    ),
    (
        _CLS_CROSS_ENCODER_PP,
        (
            ("transformers", "TransformersForSequenceClassification"),
            ("transformers", "TransformersMoEForSequenceClassification"),
        ),
    ),
    (
        _CLS_CROSS_ENCODER,
        (
            ("bert", "BertForSequenceClassification"),
            ("bert_with_rope", "GteNewForSequenceClassification"),
            ("modernbert", "ModernBertForSequenceClassification"),
            ("roberta", "RobertaForSequenceClassification"),
        ),
    ),
    (
        dict(
            is_pooling_model=True,
            default_seq_pooling_type="CLS",
            score_type="late-interaction",
            supports_multimodal=True,
        ),
        (("colmodernvbert", "ColModernVBertForRetrieval"),),
    ),
    (
        _CLS_LATE_INTERACTION,
        (
            ("colbert", "ColBERTJinaRobertaModel"),
            ("colbert", "ColBERTModel"),
            ("colbert", "ColBERTModernBertModel"),
        ),
    ),
    (
        dict(
            is_pooling_model=True,
            default_seq_pooling_type="CLS",
            supports_multimodal=True,
            supports_pp=True,
        ),
        (("transformers", "TransformersMultiModalEmbeddingModel"),),
    ),
    (
        dict(
            is_pooling_model=True,
            default_seq_pooling_type="CLS",
            supports_multimodal=True,
        ),
        (("siglip", "SiglipEmbeddingModel"),),
    ),
    (
        _CLS_POOLING_PP,
        (
            ("transformers", "TransformersEmbeddingModel"),
            ("transformers", "TransformersMoEEmbeddingModel"),
        ),
    ),
    (
        _CLS_POOLING,
        (
            ("bert", "BertEmbeddingModel"),
            ("bert", "BertSpladeSparseEmbeddingModel"),
            ("roberta", "BgeM3EmbeddingModel"),
            ("roberta", "RobertaEmbeddingModel"),
        ),
    ),
    (
        _CLS_DEFAULT,
        (
            ("bert_with_rope", "GteNewModel"),
            ("bert_with_rope", "NomicBertModel"),
            ("bert_with_rope", "SnowflakeGteNewModel"),
            ("modernbert", "ModernBertModel"),
        ),
    ),
    (
        dict(
            is_text_generation_model=True,
            is_pooling_model=True,
            default_seq_pooling_type="MEAN",
            supports_pp=True,
        ),
        (("gritlm", "GritLM"),),
    ),
    (
        dict(
            is_pooling_model=True,
            default_tok_pooling_type="STEP",
            supports_pp=True,
        ),
        (("qwen2_rm", "Qwen2ForProcessRewardModel"),),
    ),
    (
        dict(
            is_text_generation_model=True,
            supports_pp=True,
            has_inner_state=True,
            is_attention_free=True,
            supports_mamba_prefix_caching=True,
        ),
        (("mamba", "MambaForCausalLM"),),
    ),
    (
        dict(
            is_text_generation_model=True,
            has_inner_state=True,
            is_attention_free=True,
            supports_mamba_prefix_caching=True,
        ),
        (("mamba2", "Mamba2ForCausalLM"),),
    ),
    (
        dict(
            is_text_generation_model=True,
            is_pooling_model=True,
            supports_pp=True,
            has_inner_state=True,
            is_hybrid=True,
            supports_mamba_prefix_caching=True,
        ),
        (("jamba", "JambaForSequenceClassification"),),
    ),
    (
        dict(
            is_text_generation_model=True,
            supports_multimodal=True,
            has_inner_state=True,
            is_hybrid=True,
            supported_video_pruning_methods=("evs",),
        ),
        (("nano_nemotron_vl", "NemotronH_Nano_VL_V2"),),
    ),
    (
        dict(
            is_text_generation_model=True,
            supports_pp=True,
            has_inner_state=True,
            is_hybrid=True,
            supports_mamba_prefix_caching=True,
            supports_replayssm=True,
        ),
        (("nemotron_h", "NemotronHForCausalLM"),),
    ),
    (
        _MAMBA_HYBRID_GENERATION_PP,
        (
            ("falcon_h1", "FalconH1ForCausalLM"),
            ("granitemoehybrid", "GraniteMoeHybridForCausalLM"),
            ("jamba", "JambaForCausalLM"),
        ),
    ),
    (
        dict(
            is_text_generation_model=True,
            has_inner_state=True,
            is_hybrid=True,
            supports_mamba_prefix_caching=True,
        ),
        (("zamba2", "Zamba2ForCausalLM"),),
    ),
    (
        _HYBRID_MULTIMODAL_ENCODER_TP_GENERATION_PP,
        (("minicpmv4_6", "MiniCPMV4_6ForConditionalGeneration"),),
    ),
    (
        _HYBRID_GENERATION_PP,
        (
            ("bailing_moe_linear", "BailingMoeV25ForCausalLM"),
            ("bailing_moe_v3", "BailingMoeV3ForCausalLM"),
            ("lfm2", "Lfm2ForCausalLM"),
            ("lfm2_moe", "Lfm2MoeForCausalLM"),
            ("olmo_hybrid", "OlmoHybridForCausalLM"),
            ("qwen3_5", "Qwen3_5ForCausalLM"),
            ("qwen3_5", "Qwen3_5MoeForCausalLM"),
            ("qwen3_next", "Qwen3NextForCausalLM"),
        ),
    ),
    _DECI_LM_PROFILE_GROUP,
    (
        _HYBRID_VIDEO_PRUNING_MULTIMODAL_ENCODER_TP_GENERATION_PP,
        (
            ("interns2_mobius", "InternS2MobiusForConditionalGeneration"),
            ("interns2_preview", "InternS2PreviewForConditionalGeneration"),
            ("qwen3_5", "Qwen3_5ForConditionalGeneration"),
            ("qwen3_5", "Qwen3_5MoeForConditionalGeneration"),
        ),
    ),
    (
        dict(
            is_text_generation_model=True,
            supports_multimodal=True,
            supports_pp=True,
            is_hybrid=True,
        ),
        (("lfm2_vl", "Lfm2VLForConditionalGeneration"),),
    ),
    (
        dict(
            is_text_generation_model=True,
            is_pooling_model=True,
            score_type="cross-encoder",
            supports_multimodal=True,
            supports_multimodal_encoder_tp_data=True,
            supports_pp=True,
            supports_mm_device_do_normalize=True,
        ),
        (("jina_vl", "JinaVLForSequenceClassification"),),
    ),
    (
        dict(
            is_text_generation_model=True,
            is_pooling_model=True,
            score_type="cross-encoder",
            supports_multimodal=True,
            supports_pp=True,
        ),
        (("nemotron_vl", "LlamaNemotronVLForSequenceClassification"),),
    ),
    (
        dict(
            is_text_generation_model=True,
            is_pooling_model=True,
            score_type="cross-encoder",
            supports_pp=True,
        ),
        (("llama", "LlamaBidirectionalForSequenceClassification"),),
    ),
    (
        dict(
            is_text_generation_model=True,
            is_pooling_model=True,
            supports_multimodal=True,
            supports_pp=True,
            supports_transcription=True,
        ),
        (("qwen3_asr_forced_aligner", "Qwen3ASRForcedAlignerForTokenClassification"),),
    ),
    (
        dict(
            is_text_generation_model=True,
            is_pooling_model=True,
            supports_multimodal=True,
            supports_pp=True,
        ),
        (("nemotron_vl", "LlamaNemotronVLForEmbedding"),),
    ),
    (
        _GENERATION_POOLING_PP,
        (
            ("internlm2", "InternLM2ForRewardModel"),
            ("jina", "JinaEmbeddingsV5Model"),
            ("llama", "LlamaBidirectionalModel"),
        ),
    ),
    (
        dict(
            is_pooling_model=True,
            score_type="cross-encoder",
        ),
        (("gpt2", "GPT2ForSequenceClassification"),),
    ),
    (
        dict(
            is_pooling_model=True,
            score_type="late-interaction",
        ),
        (("jina", "JinaForRanking"),),
    ),
    (
        dict(
            is_pooling_model=True,
            supports_multimodal=True,
        ),
        (("clip", "CLIPEmbeddingModel"),),
    ),
    (
        dict(
            is_pooling_model=True,
            supports_pp=True,
        ),
        (("qwen2_rm", "Qwen2ForRewardModel"),),
    ),
    (
        dict(
            is_text_generation_model=True,
            supports_multimodal=True,
            requires_raw_input_tokens=True,
            supports_pp=True,
            supports_transcription=True,
        ),
        (("voxtral_realtime", "VoxtralRealtimeGeneration"),),
    ),
    (
        dict(
            is_text_generation_model=True,
            supports_multimodal=True,
            requires_raw_input_tokens=True,
            supports_pp=True,
        ),
        (("cheers", "CheersForConditionalGeneration"),),
    ),
    (
        _VIDEO_PRUNING_MULTIMODAL_ENCODER_TP_GENERATION_PP,
        (
            ("cosmos3", "Cosmos3ForConditionalGeneration"),
            ("interns1_pro", "InternS1ProForConditionalGeneration"),
            ("qwen3_vl", "Qwen3VLForConditionalGeneration"),
            ("qwen3_vl_moe", "Qwen3VLMoeForConditionalGeneration"),
        ),
    ),
    (
        _EVS_NORMALIZED_MULTIMODAL_ENCODER_TP_GENERATION_PP,
        (
            ("exaone4_5", "Exaone4_5_ForConditionalGeneration"),
            ("opencua", "OpenCUAForConditionalGeneration"),
            ("qwen2_5_vl", "Qwen2_5_VLForConditionalGeneration"),
        ),
    ),
    (
        dict(
            is_text_generation_model=True,
            supports_multimodal=True,
            supports_multimodal_encoder_tp_data=True,
            supports_pp=True,
            supports_mm_device_do_normalize=True,
        ),
        (("qwen2_vl", "Qwen2VLForConditionalGeneration"),),
    ),
    (
        _MULTIMODAL_ENCODER_TP_GENERATION_PP,
        (
            ("cosmos3_edge", "Cosmos3EdgeForConditionalGeneration"),
            ("dots_ocr", "DotsOCRForCausalLM"),
            ("eagle2_5_vl", "Eagle2_5_VLForConditionalGeneration"),
            ("glm4_1v", "Glm4vForConditionalGeneration"),
            ("glm4_1v", "Glm4vMoeForConditionalGeneration"),
            ("glm_ocr", "GlmOcrForConditionalGeneration"),
            ("h2ovl", "H2OVLChatModel"),
            ("internvl", "InternVLChatModel"),
            ("isaac", "IsaacForConditionalGeneration"),
            ("kimi_k25", "KimiK25ForConditionalGeneration"),
            ("kimi_vl", "KimiVLForConditionalGeneration"),
            ("minicpmo", "MiniCPMO"),
            ("minicpmv", "MiniCPMV"),
            ("mllama4", "Llama4ForConditionalGeneration"),
            ("nvlm_d", "NVLM_D_Model"),
            ("qianfan_ocr", "QianfanOCRForConditionalGeneration"),
            ("step3_vl", "Step3VLForConditionalGeneration"),
            ("step3p7", "Step3p7ForConditionalGeneration"),
            ("step_vl", "StepVLForConditionalGeneration"),
            ("vllm.models.minimax_m3", "MiniMaxM3SparseForConditionalGeneration"),
        ),
    ),
    (
        dict(
            is_text_generation_model=True,
            supports_multimodal=True,
            supports_pp=True,
            supports_transcription=True,
            supports_transcription_only=True,
        ),
        (("moss_transcribe_diarize", "MossTranscribeDiarizeForConditionalGeneration"),),
    ),
    (
        _TRANSCRIPTION_GENERATION_PP,
        (
            ("glmasr", "GlmAsrForConditionalGeneration"),
            ("granite_speech", "GraniteSpeechForConditionalGeneration"),
            ("granite_speech_plus", "GraniteSpeechPlusForConditionalGeneration"),
            ("kimi_audio", "KimiAudioForConditionalGeneration"),
            ("qwen3_asr", "Qwen3ASRForConditionalGeneration"),
            ("qwen3_asr_realtime", "Qwen3ASRRealtimeGeneration"),
            ("qwen3_omni_moe_thinker", "Qwen3OmniMoeThinkerForConditionalGeneration"),
            ("voxtral", "VoxtralForConditionalGeneration"),
        ),
    ),
    (
        _MULTIMODAL_GENERATION_PP,
        (
            ("audioflamingo3", "AudioFlamingo3ForConditionalGeneration"),
            ("bagel", "BagelForConditionalGeneration"),
            ("bee", "BeeForConditionalGeneration"),
            ("blip2", "Blip2ForConditionalGeneration"),
            ("chameleon", "ChameleonForConditionalGeneration"),
            ("cohere2_vision", "Cohere2VisionForConditionalGeneration"),
            ("deepseek_ocr", "DeepseekOCRForCausalLM"),
            ("deepseek_ocr2", "DeepseekOCR2ForCausalLM"),
            ("deepseek_vl2", "DeepseekVLV2ForCausalLM"),
            ("ernie45_vl", "Ernie4_5_VLMoeForConditionalGeneration"),
            ("funaudiochat", "FunAudioChatForConditionalGeneration"),
            ("gemma3_mm", "Gemma3ForConditionalGeneration"),
            ("gemma4_mm", "Gemma4ForConditionalGeneration"),
            ("gemma4_unified", "Gemma4UnifiedForConditionalGeneration"),
            ("glm4v", "GLM4VForCausalLM"),
            ("granite4_vision", "Granite4VisionForConditionalGeneration"),
            ("hyperclovax_vision_v2", "HCXVisionV2ForCausalLM"),
            ("interns1", "InternS1ForConditionalGeneration"),
            ("keye", "KeyeForConditionalGeneration"),
            ("keye_vl1_5", "KeyeVL1_5ForConditionalGeneration"),
            ("lightonocr", "LightOnOCRForConditionalGeneration"),
            ("llava", "LlavaForConditionalGeneration"),
            ("llava_next", "LlavaNextForConditionalGeneration"),
            ("llava_next_video", "LlavaNextVideoForConditionalGeneration"),
            ("llava_onevision", "LlavaOnevisionForConditionalGeneration"),
            ("llava_onevision2", "LlavaOnevision2ForConditionalGeneration"),
            ("midashenglm", "MiDashengLMModel"),
            ("mimo_v2_omni", "MiMoV2OmniForCausalLM"),
            ("mistral3", "Mistral3ForConditionalGeneration"),
            ("molmo", "MolmoForCausalLM"),
            ("molmo2", "Molmo2ForConditionalGeneration"),
            ("moondream3", "Moondream3ForCausalLM"),
            ("moss_audio", "MossAudioModel"),
            ("muse_glimmer", "MuseGlimmerForCausalLM"),
            ("nemotron_vl", "LlamaNemotronVLChatModel"),
            ("openpangu_vl", "OpenPanguVLForConditionalGeneration"),
            ("openvla", "OpenVLAForActionPrediction"),
            ("ovis", "Ovis"),
            ("ovis2_5", "Ovis2_5"),
            ("paligemma", "PaliGemmaForConditionalGeneration"),
            ("phi3v", "Phi3VForCausalLM"),
            ("phi4siglip", "Phi4ForCausalLMV"),
            ("pixtral", "PixtralForConditionalGeneration"),
            ("qwen2_5_omni_thinker", "Qwen2_5OmniThinkerForConditionalGeneration"),
            ("qwen2_audio", "Qwen2AudioForConditionalGeneration"),
            ("rvl", "RForConditionalGeneration"),
            ("skyworkr1v", "SkyworkR1VChatModel"),
            ("transformers", "TransformersMultiModalForCausalLM"),
            ("transformers", "TransformersMultiModalMoEForCausalLM"),
            ("ultravox", "UltravoxModel"),
            ("unlimited_ocr", "UnlimitedOCRForCausalLM"),
            ("vllm.models.inkling", "InklingForConditionalGeneration"),
        ),
    ),
    (
        _TRANSCRIPTION_ONLY_MULTIMODAL_GENERATION,
        (
            ("cohere_asr", "CohereAsrForConditionalGeneration"),
            ("fireredasr2", "FireRedASR2ForConditionalGeneration"),
            ("fireredlid", "FireRedLIDForConditionalGeneration"),
            ("funasr", "FunASRForConditionalGeneration"),
            ("whisper", "WhisperForConditionalGeneration"),
        ),
    ),
    (
        dict(
            is_text_generation_model=True,
            supports_multimodal=True,
            supports_transcription=True,
        ),
        (("gemma3n_mm", "Gemma3nForConditionalGeneration"),),
    ),
    (
        _MULTIMODAL_GENERATION,
        (
            ("aria", "AriaForConditionalGeneration"),
            ("diffusion_gemma", "DiffusionGemmaForConditionalGeneration"),
            ("exaone4_5_mtp", "Exaone4_5_MTP"),
            ("idefics3", "Idefics3ForConditionalGeneration"),
            ("interns2_mobius", "InternS2MobiusMTP"),
            ("mimo_v2_mtp", "MiMoV2OmniMTP"),
            ("nemotron_parse", "NemotronParseForConditionalGeneration"),
            ("paddleocr_vl", "PaddleOCRVLForConditionalGeneration"),
            ("phi4mm", "Phi4MMForCausalLM"),
            ("qwen3_5_mtp", "Qwen3_5MTP"),
            ("qwen3_5_mtp", "Qwen3_5MoeMTP"),
            ("smolvlm", "SmolVLMForConditionalGeneration"),
        ),
    ),
    (
        _TEXT_GENERATION_PP,
        (
            ("AXK1", "AXK1ForCausalLM"),
            ("afmoe", "AfmoeForCausalLM"),
            ("apertus", "ApertusForCausalLM"),
            ("arcee", "ArceeForCausalLM"),
            ("arctic", "ArcticForCausalLM"),
            ("bailing_moe", "BailingMoeForCausalLM"),
            ("bailing_moe", "BailingMoeV2ForCausalLM"),
            ("bailing_moe_v3_mtp", "BailingMoeV3MTPModel"),
            ("bloom", "BloomForCausalLM"),
            ("chatglm", "ChatGLMForCausalLM"),
            ("cohere2_moe", "Cohere2MoeForCausalLM"),
            ("commandr", "CohereForCausalLM"),
            ("dbrx", "DbrxForCausalLM"),
            ("deepseek_v2", "DeepseekForCausalLM"),
            ("deepseek_v2", "DeepseekV2ForCausalLM"),
            ("deepseek_v2", "DeepseekV3ForCausalLM"),
            ("ernie45", "Ernie4_5ForCausalLM"),
            ("ernie45_moe", "Ernie4_5_MoeForCausalLM"),
            ("exaone", "ExaoneForCausalLM"),
            ("exaone4", "Exaone4ForCausalLM"),
            ("exaone_moe", "ExaoneMoeForCausalLM"),
            ("fairseq2_llama", "Fairseq2LlamaForCausalLM"),
            ("falcon", "FalconForCausalLM"),
            ("flex_olmo", "FlexOlmoForCausalLM"),
            ("gemma", "GemmaForCausalLM"),
            ("gemma2", "Gemma2ForCausalLM"),
            ("gemma3", "Gemma3ForCausalLM"),
            ("gemma4", "Gemma4ForCausalLM"),
            ("glm", "GlmForCausalLM"),
            ("glm4", "Glm4ForCausalLM"),
            ("glm4_moe", "Glm4MoeForCausalLM"),
            ("glm4_moe_lite", "Glm4MoeLiteForCausalLM"),
            ("glm4_moe_lite_mtp", "Glm4MoeLiteMTP"),
            ("glm_ocr_mtp", "GlmOcrMTP"),
            ("gpt2", "GPT2LMHeadModel"),
            ("gpt_j", "GPTJForCausalLM"),
            ("gpt_neox", "GPTNeoXForCausalLM"),
            ("gpt_oss", "GptOssForCausalLM"),
            ("granite", "GraniteForCausalLM"),
            ("granitemoe", "GraniteMoeForCausalLM"),
            ("granitemoeshared", "GraniteMoeSharedForCausalLM"),
            ("hy_v3", "HYV3ForCausalLM"),
            ("hyperclovax", "HyperCLOVAXForCausalLM"),
            ("internlm2", "InternLM2ForCausalLM"),
            ("jais2", "Jais2ForCausalLM"),
            ("laguna", "LagunaForCausalLM"),
            ("llama", "LlamaForCausalLM"),
            ("llama4", "Llama4ForCausalLM"),
            ("longcat_flash", "LongcatFlashForCausalLM"),
            ("longcat_flash_ngram", "LongcatFlashNgramForCausalLM"),
            ("mellum", "MellumForCausalLM"),
            ("mimo", "MiMoForCausalLM"),
            ("mimo_v2", "MiMoV2FlashForCausalLM"),
            ("mimo_v2", "MiMoV2ForCausalLM"),
            ("minicpm", "MiniCPMForCausalLM"),
            ("minicpm3", "MiniCPM3ForCausalLM"),
            ("minimax_m2", "MiniMaxM2ForCausalLM"),
            ("mistral", "MistralForCausalLM"),
            ("mistral_large_3", "MistralLarge3ForCausalLM"),
            ("mixtral", "MixtralForCausalLM"),
            ("mpt", "MPTForCausalLM"),
            ("nemotron", "NemotronForCausalLM"),
            ("olmo3", "Olmo3ForCausalLM"),
            ("olmoe", "OlmoeForCausalLM"),
            ("openpangu", "PanguEmbeddedForCausalLM"),
            ("openpangu", "PanguProMoEV2ForCausalLM"),
            ("openpangu", "PanguUltraMoEForCausalLM"),
            ("opt", "OPTForCausalLM"),
            ("orion", "OrionForCausalLM"),
            ("param2moe", "Param2MoEForCausalLM"),
            ("phi", "PhiForCausalLM"),
            ("phi3", "Phi3ForCausalLM"),
            ("phimoe", "PhiMoEForCausalLM"),
            ("plamo3", "Plamo3ForCausalLM"),
            ("qwen2", "Qwen2ForCausalLM"),
            ("qwen2_moe", "Qwen2MoeForCausalLM"),
            ("qwen3", "Qwen3ForCausalLM"),
            ("qwen3_moe", "Qwen3MoeForCausalLM"),
            ("rnj1", "Rnj1ForCausalLM"),
            ("sarvam", "SarvamMLAForCausalLM"),
            ("sarvam", "SarvamMoEForCausalLM"),
            ("seed_oss", "SeedOssForCausalLM"),
            ("solar", "SolarForCausalLM"),
            ("stablelm", "StablelmForCausalLM"),
            ("step1", "Step1ForCausalLM"),
            ("step3_text", "Step3TextForCausalLM"),
            ("step3p5", "Step3p5ForCausalLM"),
            ("telechat2", "TeleChat2ForCausalLM"),
            ("teleflm", "TeleFLMForCausalLM"),
            ("transformers", "TransformersForCausalLM"),
            ("transformers", "TransformersMoEForCausalLM"),
            ("vllm.models.deepseek_v4", "DeepseekV4ForCausalLM"),
            ("vllm.models.inkling", "InklingForCausalLM"),
            ("vllm.models.minimax_m3", "MiniMaxM3SparseForCausalLM"),
        ),
    ),
    (
        _TEXT_GENERATION,
        (
            ("bailing_moe_mtp", "BailingMoeV25MTPModel"),
            ("cohere_eagle", "EagleCohereForCausalLM"),
            ("deepseek_eagle", "EagleDeepseekV3ForCausalLM"),
            ("deepseek_eagle3", "Eagle3DeepseekV2ForCausalLM"),
            ("deepseek_mtp", "DeepSeekMTP"),
            ("ernie_mtp", "ErnieMTP"),
            ("gemma3n", "Gemma3nForCausalLM"),
            ("gemma4_dspark", "Gemma4DSparkForCausalLM"),
            ("gemma4_mtp", "Gemma4MTP"),
            ("glm4_moe_mtp", "Glm4MoeMTP"),
            ("hrm_text", "HrmTextForCausalLM"),
            ("iquest_loopcoder", "IQuestLoopCoderForCausalLM"),
            ("laguna_dflash", "DFlashLagunaForCausalLM"),
            ("llama4_eagle", "EagleLlama4ForCausalLM"),
            ("llama_eagle", "EagleLlamaForCausalLM"),
            ("llama_eagle3", "Eagle3LlamaForCausalLM"),
            ("mimo_mtp", "MiMoMTP"),
            ("mimo_v2_mtp", "MiMoV2MTP"),
            ("minicpm_eagle", "EagleMiniCPMForCausalLM"),
            ("mistral_eagle", "EagleMistralForCausalLM"),
            ("mistral_large_3_eagle", "EagleMistralLarge3ForCausalLM"),
            ("openpangu_mtp", "OpenPanguMTP"),
            ("qwen3_dflash", "DFlashQwen3ForCausalLM"),
            ("qwen3_dspark", "Qwen3DSparkForCausalLM"),
            ("qwen3_eagle3", "Eagle3Qwen3ForCausalLM"),
            ("qwen3_next_mtp", "Qwen3NextMTP"),
            ("step3p5_mtp", "Step3p5MTP"),
            ("vllm.models.deepseek_v4", "DSparkDeepseekV4ForCausalLM"),
            ("vllm.models.deepseek_v4", "DeepSeekV4MTP"),
            ("vllm.models.inkling", "InklingMTP"),
            ("vllm.models.kimi_k3", "KimiK3MTP"),
            ("vllm.models.kimi_k3.nvidia.dspark_mla", "K3DSparkForCausalLM"),
            ("vllm.models.minimax_m3", "MiniMaxM3MTP"),
        ),
    ),
    (
        dict(
            supports_pp=True,
        ),
        (("nemotron_h_mtp", "NemotronHMTP"),),
    ),
    (
        _DEFAULT_PROFILE,
        (
            ("exaone_moe_mtp", "ExaoneMoeMTP"),
            ("extract_hidden_states", "ExtractHiddenStatesModel"),
            ("gemma3", "Gemma3Model"),
            ("hy_v3_mtp", "HYV3MTP"),
            ("longcat_flash_mtp", "LongCatFlashMTP"),
            ("medusa", "Medusa"),
            ("voyage", "VoyageQwen3BidirectionalEmbedModel"),
        ),
    ),
)


def _build_builtin_model_info_profiles() -> dict[_ModelInfoImplementation, _Profile]:
    profiles: dict[_ModelInfoImplementation, _Profile] = {}
    for profile, implementations in _MODEL_INFO_PROFILE_GROUPS:
        for implementation in implementations:
            if implementation in profiles:
                raise RuntimeError(
                    f"duplicate model info profile for {implementation!r}"
                )
            profiles[implementation] = profile
    return profiles


_BUILTIN_MODEL_INFO_PROFILES = _build_builtin_model_info_profiles()


def _get_builtin_model_info(module_name: str, class_name: str) -> _Profile | None:
    module_name = module_name.removeprefix("vllm.model_executor.models.")
    profile = _BUILTIN_MODEL_INFO_PROFILES.get((module_name, class_name))
    return None if profile is None else _MODEL_INFO_DEFAULTS | profile
