# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Embedding shape validation in multimodal APIs.

Tests verify that embeddings with correct ndim but incorrect hidden_size
are rejected before they can cause crashes during model inference.

Validation is performed by the parser (MultiModalDataParser) and EmbeddingItems
classes, not by CompletionRenderer or MediaIO classes.
"""

import pytest
import torch

from vllm.multimodal.parse import (
    AudioEmbeddingItems,
    ImageEmbeddingItems,
    MultiModalDataParser,
    VideoEmbeddingItems,
)


class TestMultiModalParserShapeValidation:
    """Test hidden_size validation in MultiModalDataParser."""

    def test_image_embeddings_correct_hidden_size_accepted(self):
        """Baseline: Image embeddings with correct hidden_size should work."""
        expected_hidden_size = 768
        parser = MultiModalDataParser(expected_hidden_size=expected_hidden_size)

        valid_embeds = torch.randn(2, 100, expected_hidden_size)

        result = parser.parse_mm_data({"image": valid_embeds})

        assert "image" in result
        assert isinstance(result["image"], ImageEmbeddingItems)
        assert result["image"].get_count() == 2

    def test_image_embeddings_wrong_hidden_size_rejected(self):
        """Security: Image embeddings with wrong hidden_size should be rejected."""
        expected_hidden_size = 768
        wrong_hidden_size = 4096
        parser = MultiModalDataParser(expected_hidden_size=expected_hidden_size)

        invalid_embeds = torch.randn(2, 100, wrong_hidden_size)

        with pytest.raises(ValueError) as exc_info:
            parser.parse_mm_data({"image": invalid_embeds})

        error_msg = str(exc_info.value).lower()
        assert "image" in error_msg
        assert "hidden dimension mismatch" in error_msg

    def test_audio_embeddings_wrong_hidden_size_rejected(self):
        """Security: Audio embeddings with wrong hidden_size should be rejected."""
        expected_hidden_size = 768
        wrong_hidden_size = 2048
        parser = MultiModalDataParser(expected_hidden_size=expected_hidden_size)

        invalid_embeds = torch.randn(2, 100, wrong_hidden_size)

        with pytest.raises(ValueError) as exc_info:
            parser.parse_mm_data({"audio": invalid_embeds})

        error_msg = str(exc_info.value).lower()
        assert "audio" in error_msg
        assert "hidden dimension mismatch" in error_msg

    def test_video_embeddings_wrong_hidden_size_rejected(self):
        """Security: Video embeddings with wrong hidden_size should be rejected."""
        expected_hidden_size = 768
        wrong_hidden_size = 512
        parser = MultiModalDataParser(expected_hidden_size=expected_hidden_size)

        invalid_embeds = torch.randn(2, 100, wrong_hidden_size)

        with pytest.raises(ValueError) as exc_info:
            parser.parse_mm_data({"video": invalid_embeds})

        error_msg = str(exc_info.value).lower()
        assert "video" in error_msg
        assert "hidden dimension mismatch" in error_msg

    def test_list_of_embeddings_validates_each(self):
        """Security: Each embedding in list should be validated."""
        expected_hidden_size = 768
        wrong_hidden_size = 1024
        parser = MultiModalDataParser(expected_hidden_size=expected_hidden_size)

        # List with second tensor having wrong hidden_size
        invalid_embeds = [
            torch.randn(100, expected_hidden_size),
            torch.randn(100, wrong_hidden_size),
        ]

        with pytest.raises(ValueError) as exc_info:
            parser.parse_mm_data({"image": invalid_embeds})

        # Should identify which embedding failed
        assert "[1]" in str(exc_info.value)

    def test_validation_disabled_allows_any_size(self):
        """When validation disabled (legacy), any hidden_size allowed."""
        parser = MultiModalDataParser(expected_hidden_size=None)

        any_hidden_size = 12345
        embeds = torch.randn(2, 100, any_hidden_size)

        # Should not raise
        result = parser.parse_mm_data({"image": embeds})
        assert "image" in result
        assert isinstance(result["image"], ImageEmbeddingItems)


class TestEmbeddingItemsDirectValidation:
    """Direct tests for EmbeddingItems hidden_size validation."""

    def test_image_embedding_items_validates_batched_tensor(self):
        """Test validation for batched (3D) image embeddings."""
        expected = 768
        wrong = 1024

        # Valid
        valid = torch.randn(2, 100, expected)
        items = ImageEmbeddingItems(valid, expected_hidden_size=expected)
        assert items.get_count() == 2

        # Invalid
        invalid = torch.randn(2, 100, wrong)
        with pytest.raises(ValueError) as exc_info:
            ImageEmbeddingItems(invalid, expected_hidden_size=expected)

        assert str(wrong) in str(exc_info.value)
        assert str(expected) in str(exc_info.value)

    def test_image_embedding_items_validates_list_of_tensors(self):
        """Test validation for list of 2D image embeddings."""
        expected = 768
        wrong = 512

        # Valid list
        valid_list = [torch.randn(100, expected), torch.randn(50, expected)]
        items = ImageEmbeddingItems(valid_list, expected_hidden_size=expected)
        assert items.get_count() == 2

        # Invalid list
        invalid_list = [torch.randn(100, expected), torch.randn(50, wrong)]
        with pytest.raises(ValueError) as exc_info:
            ImageEmbeddingItems(invalid_list, expected_hidden_size=expected)

        assert "[1]" in str(exc_info.value)

    def test_audio_embedding_items_validates(self):
        """Test validation for audio embeddings."""
        expected = 768
        wrong = 256

        invalid = torch.randn(2, 100, wrong)
        with pytest.raises(ValueError) as exc_info:
            AudioEmbeddingItems(invalid, expected_hidden_size=expected)

        assert "audio" in str(exc_info.value).lower()

    def test_video_embedding_items_validates(self):
        """Test validation for video embeddings."""
        expected = 768
        wrong = 384

        invalid = torch.randn(2, 100, wrong)
        with pytest.raises(ValueError) as exc_info:
            VideoEmbeddingItems(invalid, expected_hidden_size=expected)

        assert "video" in str(exc_info.value).lower()


class TestShapeValidationIntegration:
    """Integration tests verifying attack scenarios are blocked."""

    def test_attack_scenario_multimodal_image(self):
        """
        Simulate attack through Chat API with image embeddings.

        Verifies validation occurs in multimodal parser path.
        """
        expected_hidden_size = 768
        wrong_hidden_size = 4096
        parser = MultiModalDataParser(expected_hidden_size=expected_hidden_size)

        attack_tensor = torch.randn(1, 100, wrong_hidden_size)

        with pytest.raises(ValueError):
            parser.parse_mm_data({"image": attack_tensor})

    def test_attack_scenario_multimodal_audio(self):
        """
        Simulate attack through Chat API with audio embeddings.

        Verifies validation occurs in multimodal parser path.
        """
        expected_hidden_size = 768
        wrong_hidden_size = 2048
        parser = MultiModalDataParser(expected_hidden_size=expected_hidden_size)

        attack_tensor = torch.randn(1, 100, wrong_hidden_size)

        with pytest.raises(ValueError):
            parser.parse_mm_data({"audio": attack_tensor})

    def test_attack_scenario_multimodal_video(self):
        """
        Simulate attack through Chat API with video embeddings.

        Verifies validation occurs in multimodal parser path.
        """
        expected_hidden_size = 768
        wrong_hidden_size = 1024
        parser = MultiModalDataParser(expected_hidden_size=expected_hidden_size)

        attack_tensor = torch.randn(1, 100, wrong_hidden_size)

        with pytest.raises(ValueError):
            parser.parse_mm_data({"video": attack_tensor})
