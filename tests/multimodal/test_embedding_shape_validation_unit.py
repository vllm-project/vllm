# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for embedding shape validation.

Simple, fast unit tests that can run without server fixtures.
Run with: pytest tests/multimodal/test_embedding_shape_validation_unit.py -v
"""

import pytest
import torch

from vllm.multimodal.parse import (
    AudioEmbeddingItems,
    ImageEmbeddingItems,
)


class TestImageEmbedBasicValidation:
    """Test basic ndim validation in image embeddings via ImageEmbeddingItems."""

    def test_valid_2d_tensor_accepted(self):
        """Baseline: 2D tensors should be accepted."""
        valid_tensor = torch.randn(10, 768, dtype=torch.float32)

        # Should not raise - 2D is valid
        items = ImageEmbeddingItems(valid_tensor)
        assert items.get_count() == 10

    def test_valid_3d_tensor_accepted(self):
        """Baseline: 3D tensors should be accepted."""
        valid_tensor = torch.randn(2, 10, 768, dtype=torch.float32)

        # Should not raise - 3D is valid
        items = ImageEmbeddingItems(valid_tensor)
        assert items.get_count() == 2

    def test_valid_list_of_2d_tensors_accepted(self):
        """Baseline: List of 2D tensors should be accepted."""
        tensors = [
            torch.randn(10, 768, dtype=torch.float32),
            torch.randn(15, 768, dtype=torch.float32),
        ]

        # Should not raise
        items = ImageEmbeddingItems(tensors)
        assert items.get_count() == 2

    def test_1d_tensor_rejected(self):
        """Security: 1D tensors should be rejected (invalid ndim)."""
        invalid_tensor = torch.randn(768, dtype=torch.float32)  # 1D

        with pytest.raises(ValueError) as exc_info:
            ImageEmbeddingItems(invalid_tensor)

        assert "must be 2D" in str(exc_info.value) or "3D" in str(exc_info.value)

    def test_4d_tensor_rejected(self):
        """Security: 4D tensors should be rejected (invalid ndim)."""
        invalid_tensor = torch.randn(1, 2, 10, 768, dtype=torch.float32)  # 4D

        with pytest.raises(ValueError) as exc_info:
            ImageEmbeddingItems(invalid_tensor)

        assert "must be 2D" in str(exc_info.value) or "3D" in str(exc_info.value)

    def test_hidden_size_validation_correct_size(self):
        """Embeddings with correct hidden size should be accepted."""
        expected_hidden_size = 768
        valid_tensor = torch.randn(10, expected_hidden_size, dtype=torch.float32)

        # Should not raise
        items = ImageEmbeddingItems(
            valid_tensor, expected_hidden_size=expected_hidden_size
        )
        assert items.get_count() == 10

    def test_hidden_size_validation_wrong_size_rejected(self):
        """Embeddings with wrong hidden size should be rejected."""
        expected_hidden_size = 768
        wrong_hidden_size = 4096
        invalid_tensor = torch.randn(10, wrong_hidden_size, dtype=torch.float32)

        with pytest.raises(ValueError) as exc_info:
            ImageEmbeddingItems(
                invalid_tensor, expected_hidden_size=expected_hidden_size
            )

        error_msg = str(exc_info.value)
        assert "hidden dimension mismatch" in error_msg.lower()
        assert str(wrong_hidden_size) in error_msg
        assert str(expected_hidden_size) in error_msg


class TestAudioEmbedBasicValidation:
    """Test basic ndim validation in audio embeddings via AudioEmbeddingItems."""

    def test_valid_2d_tensor_accepted(self):
        """Baseline: 2D tensors should be accepted."""
        valid_tensor = torch.randn(10, 768, dtype=torch.float32)

        # Should not raise - 2D is valid
        items = AudioEmbeddingItems(valid_tensor)
        assert items.get_count() == 10

    def test_valid_3d_tensor_accepted(self):
        """Baseline: 3D tensors should be accepted."""
        valid_tensor = torch.randn(2, 10, 768, dtype=torch.float32)

        # Should not raise - 3D is valid
        items = AudioEmbeddingItems(valid_tensor)
        assert items.get_count() == 2

    def test_valid_list_of_2d_tensors_accepted(self):
        """Baseline: List of 2D tensors should be accepted."""
        tensors = [
            torch.randn(10, 768, dtype=torch.float32),
            torch.randn(15, 768, dtype=torch.float32),
        ]

        # Should not raise
        items = AudioEmbeddingItems(tensors)
        assert items.get_count() == 2

    def test_1d_tensor_rejected(self):
        """Security: 1D tensors should be rejected (invalid ndim)."""
        invalid_tensor = torch.randn(768, dtype=torch.float32)  # 1D

        with pytest.raises(ValueError) as exc_info:
            AudioEmbeddingItems(invalid_tensor)

        assert "must be 2D" in str(exc_info.value) or "3D" in str(exc_info.value)

    def test_scalar_rejected(self):
        """Security: Scalar tensors should be rejected."""
        invalid_tensor = torch.tensor(1.0)  # 0D (scalar)

        with pytest.raises(ValueError):
            AudioEmbeddingItems(invalid_tensor)

    def test_hidden_size_validation_correct_size(self):
        """Embeddings with correct hidden size should be accepted."""
        expected_hidden_size = 768
        valid_tensor = torch.randn(10, expected_hidden_size, dtype=torch.float32)

        # Should not raise
        items = AudioEmbeddingItems(
            valid_tensor, expected_hidden_size=expected_hidden_size
        )
        assert items.get_count() == 10

    def test_hidden_size_validation_wrong_size_rejected(self):
        """Embeddings with wrong hidden size should be rejected."""
        expected_hidden_size = 768
        wrong_hidden_size = 4096
        invalid_tensor = torch.randn(10, wrong_hidden_size, dtype=torch.float32)

        with pytest.raises(ValueError) as exc_info:
            AudioEmbeddingItems(
                invalid_tensor, expected_hidden_size=expected_hidden_size
            )

        error_msg = str(exc_info.value)
        assert "hidden dimension mismatch" in error_msg.lower()
        assert str(wrong_hidden_size) in error_msg
        assert str(expected_hidden_size) in error_msg


class TestShapeValidationDoSPrevention:
    """
    Tests for DoS prevention through shape validation.

    Verifies that embeddings with incorrect shapes are rejected early,
    preventing crashes during model inference.
    """

    def test_prevent_crash_from_wrong_shape_image_embeds(self):
        """
        Prevent crash scenario: wrong hidden size in image embeddings.

        Without validation, this would pass initial checks but crash later
        during model forward pass when dimensions don't match.
        """
        expected_hidden_size = 768  # Typical model hidden size
        wrong_hidden_size = 4096  # Wrong size (e.g., Llama-sized)

        wrong_embedding = torch.randn(100, wrong_hidden_size, dtype=torch.float32)

        # Should be rejected at instantiation time, not during inference
        with pytest.raises(ValueError) as exc_info:
            ImageEmbeddingItems(
                wrong_embedding, expected_hidden_size=expected_hidden_size
            )

        error_msg = str(exc_info.value)
        assert "hidden dimension mismatch" in error_msg.lower()
        assert str(expected_hidden_size) in error_msg  # Expected
        assert str(wrong_hidden_size) in error_msg  # Received

    def test_prevent_crash_from_wrong_shape_audio_embeds(self):
        """
        Prevent crash scenario: wrong hidden size in audio embeddings.
        """
        expected_hidden_size = 768
        wrong_hidden_size = 4096

        wrong_embedding = torch.randn(100, wrong_hidden_size, dtype=torch.float32)

        with pytest.raises(ValueError) as exc_info:
            AudioEmbeddingItems(
                wrong_embedding, expected_hidden_size=expected_hidden_size
            )

        error_msg = str(exc_info.value)
        assert "hidden dimension mismatch" in error_msg.lower()

    def test_extremely_large_hidden_size_rejected(self):
        """Security: Prevent DoS from extremely large embeddings."""
        expected_hidden_size = 768
        huge_hidden_size = 100000  # Large but not extreme to avoid test OOM

        invalid_tensor = torch.randn(10, huge_hidden_size, dtype=torch.float32)

        with pytest.raises(ValueError) as exc_info:
            ImageEmbeddingItems(
                invalid_tensor, expected_hidden_size=expected_hidden_size
            )

        assert "hidden dimension mismatch" in str(exc_info.value).lower()

    def test_batch_with_mixed_hidden_sizes_rejected(self):
        """All embeddings in a list must have the same hidden size."""
        expected_hidden_size = 768

        # One correct, one wrong
        batch = [
            torch.randn(10, expected_hidden_size, dtype=torch.float32),
            torch.randn(10, expected_hidden_size + 100, dtype=torch.float32),  # Wrong!
        ]

        # Should fail on the second one
        with pytest.raises(ValueError) as exc_info:
            ImageEmbeddingItems(batch, expected_hidden_size=expected_hidden_size)

        assert "hidden dimension mismatch" in str(exc_info.value).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
