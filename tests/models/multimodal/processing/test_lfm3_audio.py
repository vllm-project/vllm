# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest

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
