# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for Gemma3n audio processing, specifically the fix for integer overflow
when audio_seq_len > expected_tokens (audio_soft_tokens_per_image).

This addresses the RuntimeError: numel: integer multiplication overflow
that occurred in _process_audio_input when the audio encoder produced more
tokens than expected (e.g., 192 vs 188).

NOTE: These tests are standalone and do not import the full vLLM model
infrastructure to avoid CUDA dependency issues in test environments.
Instead, they test the core audio processing logic directly.
"""

import logging

import pytest
import torch


def process_audio_features_fixed(
    audio_features: torch.Tensor,
    expected_tokens: int,
    audio_padding_embs: torch.Tensor,
    logger: logging.Logger | None = None,
) -> torch.Tensor:
    """
    Standalone implementation of the fixed audio processing logic.

    This mirrors the logic in Gemma3nForConditionalGeneration._process_audio_input
    to validate the fix without requiring the full model infrastructure.

    Args:
        audio_features: Tensor of shape (batch_size, audio_seq_len, embed_dim)
        expected_tokens: Expected number of tokens (audio_soft_tokens_per_image)
        audio_padding_embs: Padding embeddings of shape (1, 1, embed_dim)
        logger: Optional logger for warning messages

    Returns:
        Tensor of shape (batch_size, expected_tokens, embed_dim)
    """
    audio_batch_size, audio_seq_len, audio_embed_dim = audio_features.shape

    if audio_seq_len < expected_tokens:
        extra_padding_tokens = expected_tokens - audio_seq_len
        extra_padding_features = audio_padding_embs.expand(
            audio_batch_size, extra_padding_tokens, audio_embed_dim
        )
        audio_features = torch.cat((audio_features, extra_padding_features), dim=1)
    elif audio_seq_len > expected_tokens:
        if logger:
            logger.warning(
                "Gemma3n audio encoder produced %d tokens, but expected %d. "
                "Truncating to match placeholder count.",
                audio_seq_len,
                expected_tokens,
            )
        audio_features = audio_features[:, :expected_tokens, :]
    # else: audio_seq_len == expected_tokens, no modification needed

    return audio_features


def process_audio_features_original_buggy(
    audio_features: torch.Tensor,
    expected_tokens: int,
    audio_padding_embs: torch.Tensor,
) -> torch.Tensor:
    """
    Original buggy implementation that causes integer overflow.

    This demonstrates the bug where negative extra_padding_tokens
    causes tensor.expand() to overflow.
    """
    audio_batch_size, audio_seq_len, audio_embed_dim = audio_features.shape

    # BUG: This can be negative when audio_seq_len > expected_tokens
    extra_padding_tokens = expected_tokens - audio_seq_len

    # BUG: expand() with negative dimension causes integer overflow
    extra_padding_features = audio_padding_embs.expand(
        audio_batch_size, extra_padding_tokens, audio_embed_dim
    )
    audio_features = torch.cat((audio_features, extra_padding_features), dim=1)
    return audio_features


class TestGemma3nAudioProcessing:
    """Test cases for Gemma3n audio processing overflow fix"""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger to capture warning messages"""
        logger = logging.getLogger("test_gemma3n_audio")
        logger.setLevel(logging.WARNING)
        return logger

    def test_audio_padding_when_short(self):
        """Test that audio features are padded when shorter than expected"""
        audio_seq_len = 150
        expected_tokens = 188
        batch_size = 2
        embed_dim = 2048

        # Create test audio features (shorter than expected)
        audio_features = torch.randn(batch_size, audio_seq_len, embed_dim)
        audio_padding_embs = torch.randn(1, 1, embed_dim)

        # Apply the fixed processing logic
        result = process_audio_features_fixed(
            audio_features, expected_tokens, audio_padding_embs
        )

        # Verify the output shape is padded to expected length
        assert result.shape == (batch_size, expected_tokens, embed_dim)

    def test_audio_truncation_when_long(self, mock_logger):
        """Test that audio features are truncated when longer than expected.

        This is the key test for the overflow fix. Previously, when
        audio_seq_len > expected_tokens, the code would try to expand
        a tensor with negative dimensions causing integer overflow.
        """
        audio_seq_len = 200
        expected_tokens = 188
        batch_size = 2
        embed_dim = 2048

        audio_features = torch.randn(batch_size, audio_seq_len, embed_dim)
        audio_padding_embs = torch.randn(1, 1, embed_dim)

        # Apply the fixed processing logic (should not raise overflow error)
        result = process_audio_features_fixed(
            audio_features, expected_tokens, audio_padding_embs, mock_logger
        )

        # Verify the output is truncated to expected length
        assert result.shape == (batch_size, expected_tokens, embed_dim)
        # Verify the content is the first expected_tokens from the input
        torch.testing.assert_close(result, audio_features[:, :expected_tokens, :])

    def test_audio_exact_match(self, mock_logger):
        """Test that no modification occurs when audio length matches exactly"""
        audio_seq_len = 188
        expected_tokens = 188
        batch_size = 2
        embed_dim = 2048

        audio_features = torch.randn(batch_size, audio_seq_len, embed_dim)
        audio_padding_embs = torch.randn(1, 1, embed_dim)

        # Apply the fixed processing logic
        result = process_audio_features_fixed(
            audio_features, expected_tokens, audio_padding_embs, mock_logger
        )

        # Verify no modification occurred
        assert result.shape == (batch_size, expected_tokens, embed_dim)
        torch.testing.assert_close(result, audio_features)

    def test_overflow_scenario_192_vs_188(self):
        """Test the specific overflow scenario from the bug report:
        - audio_soft_tokens_per_image = 188
        - actual audio tokens = 192 (due to BOA/EOA tokens)

        This test validates that the fix prevents the integer overflow
        that was occurring in production.
        """
        # Exact scenario from the bug report
        audio_seq_len = 192  # As seen in the error logs
        expected_tokens = 188  # From audio_soft_tokens_per_image
        batch_size = 1
        embed_dim = 2048

        audio_features = torch.randn(batch_size, audio_seq_len, embed_dim)
        audio_padding_embs = torch.randn(1, 1, embed_dim)

        # The fixed version should NOT raise any error
        try:
            result = process_audio_features_fixed(
                audio_features, expected_tokens, audio_padding_embs
            )
        except RuntimeError as e:
            if "integer multiplication overflow" in str(e):
                pytest.fail(
                    f"Integer overflow still occurring! The fix did not work: {e}"
                )
            raise

        # Verify result is correctly truncated
        assert result.shape == (batch_size, expected_tokens, embed_dim)

    def test_original_buggy_implementation_fails(self):
        """Verify that the original buggy implementation does cause overflow.

        This test documents the original bug behavior by showing that
        the unfixed code raises a RuntimeError when audio_seq_len > expected_tokens.
        """
        audio_seq_len = 192
        expected_tokens = 188
        batch_size = 1
        embed_dim = 2048

        audio_features = torch.randn(batch_size, audio_seq_len, embed_dim)
        audio_padding_embs = torch.randn(1, 1, embed_dim)

        # The original buggy implementation should raise RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            process_audio_features_original_buggy(
                audio_features, expected_tokens, audio_padding_embs
            )

        # Verify it's the expected overflow error (or related negative size error)
        error_msg = str(exc_info.value).lower()
        assert any(msg in error_msg for msg in ["overflow", "negative", "invalid"]), (
            f"Unexpected error message: {exc_info.value}"
        )

    def test_large_overflow_scenario(self):
        """Test with a significantly larger token count to ensure robustness"""
        audio_seq_len = 500
        expected_tokens = 188
        batch_size = 1
        embed_dim = 2048

        audio_features = torch.randn(batch_size, audio_seq_len, embed_dim)
        audio_padding_embs = torch.randn(1, 1, embed_dim)

        # Apply the fixed processing logic
        result = process_audio_features_fixed(
            audio_features, expected_tokens, audio_padding_embs
        )

        # Verify result is truncated to expected length
        assert result.shape == (batch_size, expected_tokens, embed_dim)

    def test_batch_processing(self):
        """Test that batch processing works correctly with mixed scenarios"""
        expected_tokens = 188
        batch_size = 4
        embed_dim = 2048

        # Test with different audio lengths (all shorter)
        audio_features = torch.randn(batch_size, 150, embed_dim)
        audio_padding_embs = torch.randn(1, 1, embed_dim)

        result = process_audio_features_fixed(
            audio_features, expected_tokens, audio_padding_embs
        )

        assert result.shape == (batch_size, expected_tokens, embed_dim)
        # First 150 tokens should match original
        torch.testing.assert_close(result[:, :150, :], audio_features)

    def test_device_compatibility(self):
        """Test that the fix works on both CPU and (if available) CUDA"""
        audio_seq_len = 192
        expected_tokens = 188
        batch_size = 1
        embed_dim = 2048

        # Test on CPU
        audio_features = torch.randn(batch_size, audio_seq_len, embed_dim)
        audio_padding_embs = torch.randn(1, 1, embed_dim)

        result_cpu = process_audio_features_fixed(
            audio_features, expected_tokens, audio_padding_embs
        )
        assert result_cpu.shape == (batch_size, expected_tokens, embed_dim)

        # Test on CUDA if available
        if torch.cuda.is_available():
            audio_features_cuda = audio_features.cuda()
            audio_padding_embs_cuda = audio_padding_embs.cuda()

            result_cuda = process_audio_features_fixed(
                audio_features_cuda, expected_tokens, audio_padding_embs_cuda
            )
            assert result_cuda.shape == (batch_size, expected_tokens, embed_dim)
            assert result_cuda.device.type == "cuda"
