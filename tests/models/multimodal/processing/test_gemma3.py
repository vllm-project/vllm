# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.models.gemma3n_audio_utils import (
    adjust_audio_features_to_expected_length,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from ....conftest import ImageTestAssets
from ...utils import build_model_context

# Gemma3 (image) model
GEMMA3_MODEL_ID = "google/gemma-3-4b-it"

# Gemma3n (multimodal with audio) model
GEMMA3N_MODEL_ID = "google/gemma-3n-E2B-it"

# Expected audio tokens for Gemma3n (audio_soft_tokens_per_image)
GEMMA3N_EXPECTED_AUDIO_TOKENS = 188


class TestGemma3nAudioTensorLogic:
    """CPU-based tests for Gemma3n audio feature tensor manipulation.

    These tests validate the padding/truncation logic in
    adjust_audio_features_to_expected_length() which fixes the
    integer overflow in _process_audio_input when audio_seq_len > 188.
    """

    def test_padding_when_audio_short(self):
        """Test that short audio is padded to expected length."""
        batch_size, seq_len, embed_dim = 1, 100, 256
        expected_tokens = GEMMA3N_EXPECTED_AUDIO_TOKENS

        audio_features = torch.randn(batch_size, seq_len, embed_dim)
        padding_embs = torch.zeros(1, 1, embed_dim)

        result, tokens_truncated = adjust_audio_features_to_expected_length(
            audio_features, expected_tokens, padding_embs
        )

        assert result.shape == (batch_size, expected_tokens, embed_dim)
        assert tokens_truncated == 0
        # First 100 tokens should be original, rest should be padding (zeros)
        assert torch.allclose(result[:, :seq_len, :], audio_features)
        assert torch.allclose(
            result[:, seq_len:, :],
            torch.zeros(batch_size, expected_tokens - seq_len, embed_dim),
        )

    def test_truncation_when_audio_long(self):
        """Test that long audio is truncated to expected length.

        This is the key test for the overflow fix. Previously, when
        audio_seq_len > expected_tokens, the code would compute a negative
        padding value causing: RuntimeError: numel: integer multiplication overflow
        """
        batch_size, seq_len, embed_dim = 1, 192, 256  # 192 > 188
        expected_tokens = GEMMA3N_EXPECTED_AUDIO_TOKENS

        audio_features = torch.randn(batch_size, seq_len, embed_dim)
        padding_embs = torch.zeros(1, 1, embed_dim)

        result, tokens_truncated = adjust_audio_features_to_expected_length(
            audio_features, expected_tokens, padding_embs
        )

        assert result.shape == (batch_size, expected_tokens, embed_dim)
        assert tokens_truncated == seq_len - expected_tokens  # 192 - 188 = 4
        # Result should be first 188 tokens of original
        assert torch.allclose(result, audio_features[:, :expected_tokens, :])

    def test_no_change_when_exact_length(self):
        """Test that exact-length audio passes through unchanged."""
        batch_size, embed_dim = 1, 256
        expected_tokens = GEMMA3N_EXPECTED_AUDIO_TOKENS

        audio_features = torch.randn(batch_size, expected_tokens, embed_dim)
        padding_embs = torch.zeros(1, 1, embed_dim)

        result, tokens_truncated = adjust_audio_features_to_expected_length(
            audio_features, expected_tokens, padding_embs
        )

        assert result.shape == audio_features.shape
        assert tokens_truncated == 0
        assert torch.allclose(result, audio_features)

    def test_original_bug_would_fail(self):
        """Verify the original buggy implementation would cause overflow.

        The original code always tried to pad, which fails when
        audio_seq_len > expected_tokens because expand() gets negative size.
        """
        batch_size, seq_len, embed_dim = 1, 192, 256
        expected_tokens = GEMMA3N_EXPECTED_AUDIO_TOKENS

        padding_embs = torch.zeros(1, 1, embed_dim)

        # Original buggy logic (always pads, never truncates)
        extra_padding_tokens = expected_tokens - seq_len  # = -4 (negative!)

        with pytest.raises(RuntimeError):
            # This should fail with negative size error
            padding_embs.expand(batch_size, extra_padding_tokens, embed_dim)

    @pytest.mark.parametrize(
        "seq_len",
        [50, 100, 150, 187, 188, 189, 192, 200, 300],
    )
    def test_various_audio_lengths(self, seq_len: int):
        """Test padding/truncation with various audio lengths."""
        batch_size, embed_dim = 1, 256
        expected_tokens = GEMMA3N_EXPECTED_AUDIO_TOKENS

        audio_features = torch.randn(batch_size, seq_len, embed_dim)
        padding_embs = torch.zeros(1, 1, embed_dim)

        # Should not raise any errors
        result, tokens_truncated = adjust_audio_features_to_expected_length(
            audio_features, expected_tokens, padding_embs
        )

        # Output should always be expected_tokens length
        assert result.shape == (batch_size, expected_tokens, embed_dim)

        # Verify truncation count is correct
        if seq_len > expected_tokens:
            assert tokens_truncated == seq_len - expected_tokens
        else:
            assert tokens_truncated == 0

    def test_batch_processing(self):
        """Test that batch processing works correctly."""
        batch_size, seq_len, embed_dim = 4, 192, 256
        expected_tokens = GEMMA3N_EXPECTED_AUDIO_TOKENS

        audio_features = torch.randn(batch_size, seq_len, embed_dim)
        padding_embs = torch.zeros(1, 1, embed_dim)

        result, tokens_truncated = adjust_audio_features_to_expected_length(
            audio_features, expected_tokens, padding_embs
        )

        assert result.shape == (batch_size, expected_tokens, embed_dim)
        assert tokens_truncated == seq_len - expected_tokens


@pytest.mark.parametrize("model_id", [GEMMA3_MODEL_ID])
def test_get_image_size_with_most_features(
    image_assets: ImageTestAssets, model_id: str
):
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs={"do_pan_and_scan": True},
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    hf_processor_mm_kwargs: dict[str, object] = {}
    hf_processor = processor.info.get_hf_processor(**hf_processor_mm_kwargs)

    max_image_size = processor.info.get_image_size_with_most_features()
    max_tokens = processor.info.get_num_image_tokens(
        image_width=max_image_size.width,
        image_height=max_image_size.height,
        processor=hf_processor,
    )

    prompt = "<start_of_image>"
    image_seq_length = hf_processor.image_seq_length

    for asset in image_assets:
        mm_data = {"image": [asset.pil_image]}
        processed_inputs = processor.apply(prompt, mm_data, hf_processor_mm_kwargs)
        mm_kwargs_data = processed_inputs["mm_kwargs"].get_data()
        num_patches_tensor = mm_kwargs_data["num_patches"]
        tokens = int(num_patches_tensor.item()) * image_seq_length
        assert tokens <= max_tokens
