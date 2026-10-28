# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch
from transformers import (
    ParakeetEncoder,
    ParakeetEncoderConfig,
    ParakeetFeatureExtractor,
)
from transformers.models.canary.modeling_canary import (
    CanaryPositionalEmbedding as HFCanaryPositionalEmbedding,
)

from vllm.model_executor.models.canary import (
    CanaryForConditionalGeneration,
    CanaryPositionalEmbedding,
    _subsampled_length,
)

pytestmark = pytest.mark.core_model


def test_positional_embedding_matches_transformers():
    max_positions, hidden_size = 64, 32
    ours = CanaryPositionalEmbedding(max_positions, hidden_size)
    ref = HFCanaryPositionalEmbedding(max_positions, hidden_size)
    ref.positional_embedding.copy_(ref.compute_default_singular_positional_embedding())

    positions = torch.tensor([0, 1, 7, 63])
    torch.testing.assert_close(ours(positions), ref(positions))


@pytest.mark.parametrize("num_samples", [1600, 16000, 16000 * 7 + 123, 16000 * 40])
def test_num_audio_tokens_matches_encoder(num_samples):
    """The placeholder count must equal the encoder's output length."""
    # Subsampling settings of nvidia/canary-1b-v2; the rest is shrunk.
    encoder_config = ParakeetEncoderConfig(
        hidden_size=32,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=1,
        num_mel_bins=128,
        subsampling_conv_channels=8,
        subsampling_factor=8,
    )
    encoder = ParakeetEncoder(encoder_config).eval()
    features = ParakeetFeatureExtractor(feature_size=128)(
        np.random.default_rng(0).standard_normal(num_samples).astype(np.float32),
        sampling_rate=16000,
        return_tensors="pt",
    )
    with torch.no_grad():
        out = encoder(**features)

    expected = int(out.attention_mask.sum())
    num_frames = int(features["attention_mask"].sum())
    assert _subsampled_length(num_frames, encoder_config) == expected


def test_validate_language():
    assert CanaryForConditionalGeneration.validate_language(None) == "en"
    assert CanaryForConditionalGeneration.validate_language("de") == "de"
    with pytest.raises(ValueError, match="Unsupported language"):
        CanaryForConditionalGeneration.validate_language("ja")
