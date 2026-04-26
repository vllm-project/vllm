# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.transformers_utils.configs.mistral import _remap_mistral_audio_args


def _make_audio_config(**overrides):
    config = {
        "dim": 4096,
        "n_layers": 32,
        "n_heads": 32,
        "head_dim": 128,
        "hidden_dim": 14336,
        "vocab_size": 32000,
        "multimodal": {
            "whisper_model_args": {
                "encoder_args": {
                    "dim": 1024,
                    "n_layers": 24,
                    "hidden_dim": 4096,
                    "n_heads": 16,
                    "head_dim": 64,
                    "vocab_size": 200,
                    "audio_encoding_args": {
                        "num_mel_bins": 128,
                        "window_size": 400,
                        "sampling_rate": 16000,
                        "hop_length": 160,
                    },
                },
                "downsample_args": {
                    "downsample_factor": 2,
                },
            },
        },
    }
    config.update(overrides)
    return config


def test_audio_config_without_max_position_embeddings():
    config = _make_audio_config()
    assert "max_position_embeddings" not in config
    result = _remap_mistral_audio_args(config)
    expected = 1 * 128_000  # block_pool_size(1) * default(128_000)
    assert result["audio_config"].max_position_embeddings == expected


def test_audio_config_with_max_position_embeddings():
    config = _make_audio_config(max_position_embeddings=32768)
    result = _remap_mistral_audio_args(config)
    expected = 1 * 32768  # block_pool_size(1) * provided value
    assert result["audio_config"].max_position_embeddings == expected


def test_audio_config_causal_without_max_position_embeddings():
    config = _make_audio_config()
    config["multimodal"]["whisper_model_args"]["encoder_args"]["causal"] = True
    result = _remap_mistral_audio_args(config)
    expected = 2 * 128_000  # block_pool_size(downsample_factor=2) * default
    assert result["audio_config"].max_position_embeddings == expected
