# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import cast

import numpy as np
import pytest

from vllm.config import ModelConfig
from vllm.config.speech_to_text import SpeechToTextParams
from vllm.inputs import TokensPrompt
from vllm.model_executor.models.lfm3_audio import (
    Lfm3AudioForConditionalGeneration,
    Lfm3AudioProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.transformers_utils.configs.lfm3_audio import Lfm3AudioConfig
from vllm.transformers_utils.processors.lfm3_audio import Lfm3AudioFeatureExtractor

LAYER_TYPES = ["conv", "conv", "full_attention"] * 5 + ["full_attention"] * 2

# The hub checkpoints (training export) and the transformers-native export
TRAINING_EXPORT = {
    "lfm": {"model_type": "lfm3", "dtype": "float32"},
    "encoder": {
        "dim_in": 128,
        "dim_conv": 256,
        "dim_out": 768,
        "sliding_window": 32,
        "lfm": {
            "model_type": "lfm2",
            "vocab_size": 2,
            "hidden_size": 768,
            "intermediate_size": 2560,
            "num_hidden_layers": 17,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "layer_types": LAYER_TYPES,
            "rope_parameters": {"rope_type": "default", "rope_theta": 1e6},
        },
    },
    "depthformer": {"model_type": "lfm3"},
    "codebooks": 12,
}
TRANSFORMERS_NATIVE = {
    "text_config": {"model_type": "lfm3"},
    "audio_encoder_config": {
        "num_mel_bins": 128,
        "conv_channels": 256,
        "text_config": {
            "model_type": "lfm3",
            "vocab_size": 2,
            "hidden_size": 768,
            "intermediate_size": 1792,
            "num_hidden_layers": 17,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 96,
            "sliding_window": 32,
            "use_attention_gate": False,
            "use_sandwich_norm": False,
            "layer_types": [
                "sliding_attention" if t == "full_attention" else t for t in LAYER_TYPES
            ],
            "rope_parameters": {
                "sliding_attention": {"rope_type": "default", "rope_theta": 1e6}
            },
        },
    },
    "audio_token_id": 125031,
}


@pytest.mark.parametrize("layout", [TRAINING_EXPORT, TRANSFORMERS_NATIVE])
def test_config_layouts(layout):
    config = Lfm3AudioConfig(**layout)
    encoder = config.audio_encoder_config
    assert config.audio_token_id == 125031
    assert config.text_config.model_type == "lfm3"
    assert (encoder.num_mel_bins, encoder.text_config.hidden_size) == (128, 768)
    assert encoder.sliding_window == 32
    backbone = encoder.text_config
    assert backbone.model_type == "lfm2"
    assert backbone.layer_types == LAYER_TYPES
    assert backbone.rope_parameters["rope_theta"] == 1e6
    assert backbone.block_auto_adjust_ff_dim == (layout is TRAINING_EXPORT)
    restored = Lfm3AudioConfig(**config.to_dict())
    assert restored.audio_encoder_config.to_dict() == encoder.to_dict()


@pytest.mark.parametrize(
    ("num_samples", "num_frames", "num_tokens"),
    [(1, 1, 1), (400, 1, 1), (401, 2, 1), (16000, 99, 13), (16160, 100, 13)],
)
def test_feature_extractor_lengths(num_samples, num_frames, num_tokens):
    extractor = Lfm3AudioFeatureExtractor()
    features = extractor(np.zeros(num_samples, dtype=np.float32))
    assert features["input_features"].shape == (1, 128, num_frames)
    assert extractor.num_frames(num_samples) == num_frames
    assert extractor.num_encoder_tokens(num_frames) == num_tokens


MODEL = "LiquidAI/LFM3-Audio-2B"


@pytest.fixture(scope="module")
def model_config():
    return ModelConfig(MODEL, dtype="bfloat16", mm_processor_cache_gb=0)


def _stt_params(model_config, num_samples, language=None, task_type="transcribe"):
    model = Lfm3AudioForConditionalGeneration
    return SpeechToTextParams(
        audio=np.zeros(num_samples, dtype=np.float32),
        stt_config=model.get_speech_to_text_config(model_config, task_type),
        model_config=model_config,
        language=language,
        task_type=task_type,
    )


@pytest.mark.parametrize(
    ("language", "system_prompt"),
    [
        (None, "Perform ASR."),
        ("fr", "Perform ASR in French."),
        # Not in the model card's list, but a known language: named, not rejected
        ("hi", "Perform ASR in Hindi."),
    ],
)
def test_transcription_prompt(model_config, language, system_prompt):
    params = _stt_params(model_config, 16000, language)
    prompt = cast(
        TokensPrompt, Lfm3AudioForConditionalGeneration.get_generation_prompt(params)
    )

    text = cached_tokenizer_from_config(model_config).decode(prompt["prompt_token_ids"])
    assert system_prompt in text
    assert text.count("<|audio|>") == 1
    assert text.index(system_prompt) < text.index("<|audio|>")
    mm_data = prompt.get("multi_modal_data")
    assert mm_data is not None
    assert mm_data["audio"] is params.audio


def test_transcription_rejects_translation(model_config):
    params = _stt_params(model_config, 16000, task_type="translate")
    with pytest.raises(ValueError, match="only supports transcription"):
        Lfm3AudioForConditionalGeneration.get_generation_prompt(params)


def test_speech_to_text_config(model_config):
    config = Lfm3AudioForConditionalGeneration.get_speech_to_text_config(
        model_config, "transcribe"
    )
    assert config.sample_rate == 16000
    assert config.max_audio_clip_s == 30


@pytest.mark.parametrize("num_samples", [16000, 16160, 480000])
def test_transcription_audio_tokens_match_processor(model_config, num_samples):
    """The duration estimate must equal the placeholders the processor inserts."""
    model = Lfm3AudioForConditionalGeneration
    params = _stt_params(model_config, num_samples)
    prompt = cast(TokensPrompt, model.get_generation_prompt(params))
    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    processed = processor(
        prompt["prompt_token_ids"],
        mm_items=processor.info.parse_mm_data(prompt.get("multi_modal_data") or {}),
        hf_processor_mm_kwargs={},
    )

    audio_token_id = model_config.hf_config.audio_token_id
    inserted = processed["prompt_token_ids"].count(audio_token_id)
    estimated = model.get_num_audio_tokens(
        num_samples / params.stt_config.sample_rate, params.stt_config, model_config
    )
    assert inserted == estimated
    extractor = cast(Lfm3AudioProcessingInfo, processor.info).get_feature_extractor()
    assert estimated == extractor.num_encoder_tokens(extractor.num_frames(num_samples))
