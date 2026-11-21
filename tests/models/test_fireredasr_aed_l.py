# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.models.fireredasr_aed_l import _encoder_token_count
from vllm.transformers_utils.configs.fireredasr import FireRedASRConfig
from vllm.transformers_utils.processors import FireRedASRProcessor


@pytest.mark.parametrize(
    ("frames", "expected"),
    [
        (147, 37),
        (152, 38),
        (300, 75),
        (600, 150),
    ],
)
def test_encoder_token_count(frames: int, expected: int):
    """The encoder prompt length must match Conv2dSubsampling's output."""
    assert _encoder_token_count(frames) == expected


def test_encoder_token_count_is_at_least_one():
    assert _encoder_token_count(1) == 1


def test_firered_asr_config_defaults():
    config = FireRedASRConfig()

    assert config.model_type == "whisper"
    assert config.vocab_size == 7832
    assert config.d_model == 1280
    assert config.encoder_layers == 16
    assert config.decoder_layers == 16
    assert config.encoder_attention_heads == 20
    assert config.decoder_attention_heads == 20
    assert config.sos_id == 3
    assert config.eos_id == 4
    assert config.pad_id == 2
    assert config.is_encoder_decoder


def test_firered_asr_processor_is_registered():
    assert FireRedASRProcessor.feature_extractor_class == "WhisperFeatureExtractor"
