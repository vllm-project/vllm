# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Embedding shape validation in multimodal APIs.

Tests verify that embeddings with correct ndim but incorrect hidden_size
are rejected before they can cause crashes during model inference.
"""

import base64
import io

import pytest
import torch

from vllm.entrypoints.renderer import CompletionRenderer
from vllm.multimodal.audio import AudioEmbeddingMediaIO
from vllm.multimodal.image import ImageEmbeddingMediaIO
from vllm.multimodal.parse import (
    AudioEmbeddingItems,
    ImageEmbeddingItems,
    MultiModalDataParser,
    VideoEmbeddingItems,
)


def _encode_tensor(tensor: torch.Tensor) -> bytes:
    """Helper to encode a tensor as base64 bytes."""
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.read())


def _create_valid_embedding(hidden_size: int = 768) -> torch.Tensor:
    """Create a valid 2D embedding tensor."""
    return torch.randn(10, hidden_size, dtype=torch.float32)


def _create_valid_batched_embedding(hidden_size: int = 768) -> torch.Tensor:
    """Create a valid 3D batched embedding tensor."""
    return torch.randn(2, 10, hidden_size, dtype=torch.float32)


def _create_wrong_hidden_size_embedding(
    expected: int = 768, wrong: int = 4096
) -> torch.Tensor:
    """Create an embedding with wrong hidden_size (attack payload)."""
    return torch.randn(10, wrong, dtype=torch.float32)


class TestPromptEmbedsShapeValidation:
    """Test hidden_size validation in prompt embeddings (Completions API)."""

    def test_valid_hidden_size_accepted(self, model_config):
        """Baseline: Embeddings with correct hidden_size should work."""
        renderer = CompletionRenderer(model_config)

        hidden_size = model_config.hf_text_config.hidden_size
        valid_tensor = _create_valid_embedding(hidden_size)
        encoded = _encode_tensor(valid_tensor)

        # Should not raise any exception
        result = renderer.load_prompt_embeds(encoded)
        assert len(result) == 1
        assert result[0]["prompt_embeds"].shape == valid_tensor.shape

    def test_wrong_hidden_size_rejected(self, model_config):
        """Security: Embeddings with wrong hidden_size should be rejected."""
        renderer = CompletionRenderer(model_config)

        expected_hidden_size = model_config.hf_text_config.hidden_size
        wrong_hidden_size = 4096  # Different from expected
        invalid_tensor = torch.randn(10, wrong_hidden_size, dtype=torch.float32)
        encoded = _encode_tensor(invalid_tensor)

        with pytest.raises(ValueError) as exc_info:
            renderer.load_prompt_embeds(encoded)

        error_msg = str(exc_info.value)
        assert "hidden" in error_msg.lower() and "mismatch" in error_msg.lower()
        assert str(wrong_hidden_size) in error_msg
        assert str(expected_hidden_size) in error_msg

    def test_extremely_large_hidden_size_rejected(self, model_config):
        """Security: Embeddings with huge hidden_size should be rejected."""
        renderer = CompletionRenderer(model_config)

        huge_hidden_size = 1000000
        invalid_tensor = torch.randn(10, huge_hidden_size, dtype=torch.float32)
        encoded = _encode_tensor(invalid_tensor)

        with pytest.raises(ValueError) as exc_info:
            renderer.load_prompt_embeds(encoded)

        assert "mismatch" in str(exc_info.value).lower()

    def test_tiny_hidden_size_rejected(self, model_config):
        """Security: Embeddings with undersized hidden_size should be rejected."""
        renderer = CompletionRenderer(model_config)

        tiny_hidden_size = 8
        invalid_tensor = torch.randn(10, tiny_hidden_size, dtype=torch.float32)
        encoded = _encode_tensor(invalid_tensor)

        with pytest.raises(ValueError):
            renderer.load_prompt_embeds(encoded)

    def test_batch_with_wrong_hidden_size_rejected(self, model_config):
        """Security: Batch with wrong hidden_size tensor should be rejected."""
        renderer = CompletionRenderer(model_config)

        expected_hidden_size = model_config.hf_text_config.hidden_size
        wrong_hidden_size = expected_hidden_size + 100

        batch = [
            _encode_tensor(torch.randn(10, expected_hidden_size, dtype=torch.float32)),
            _encode_tensor(torch.randn(10, wrong_hidden_size, dtype=torch.float32)),
        ]

        with pytest.raises(ValueError):
            renderer.load_prompt_embeds(batch)


class TestImageEmbedsShapeValidation:
    """Test ndim validation in image embeddings (Chat API)."""

    def test_valid_2d_tensor_accepted(self):
        """Baseline: 2D tensors should be accepted."""
        io_handler = ImageEmbeddingMediaIO()

        valid_tensor = torch.randn(10, 768, dtype=torch.float32)
        encoded = _encode_tensor(valid_tensor)

        result = io_handler.load_base64("", encoded.decode("utf-8"))
        assert result.shape == valid_tensor.shape

    def test_valid_3d_tensor_accepted(self):
        """Baseline: 3D tensors should be accepted."""
        io_handler = ImageEmbeddingMediaIO()

        valid_tensor = torch.randn(2, 10, 768, dtype=torch.float32)
        encoded = _encode_tensor(valid_tensor)

        result = io_handler.load_base64("", encoded.decode("utf-8"))
        assert result.shape == valid_tensor.shape

    def test_1d_tensor_rejected(self):
        """Security: 1D tensors should be rejected."""
        io_handler = ImageEmbeddingMediaIO()

        invalid_tensor = torch.randn(768, dtype=torch.float32)
        encoded = _encode_tensor(invalid_tensor)

        with pytest.raises(ValueError) as exc_info:
            io_handler.load_base64("", encoded.decode("utf-8"))

        assert "2D" in str(exc_info.value) or "3D" in str(exc_info.value)

    def test_4d_tensor_rejected(self):
        """Security: 4D tensors should be rejected."""
        io_handler = ImageEmbeddingMediaIO()

        invalid_tensor = torch.randn(1, 2, 10, 768, dtype=torch.float32)
        encoded = _encode_tensor(invalid_tensor)

        with pytest.raises(ValueError) as exc_info:
            io_handler.load_base64("", encoded.decode("utf-8"))

        assert "2D" in str(exc_info.value) or "3D" in str(exc_info.value)


class TestAudioEmbedsShapeValidation:
    """Test ndim validation in audio embeddings (Chat API)."""

    def test_valid_2d_tensor_accepted(self):
        """Baseline: 2D tensors should be accepted."""
        io_handler = AudioEmbeddingMediaIO()

        valid_tensor = torch.randn(10, 768, dtype=torch.float32)
        encoded = _encode_tensor(valid_tensor)

        result = io_handler.load_base64("", encoded.decode("utf-8"))
        assert result.shape == valid_tensor.shape

    def test_valid_3d_tensor_accepted(self):
        """Baseline: 3D tensors should be accepted."""
        io_handler = AudioEmbeddingMediaIO()

        valid_tensor = torch.randn(2, 10, 768, dtype=torch.float32)
        encoded = _encode_tensor(valid_tensor)

        result = io_handler.load_base64("", encoded.decode("utf-8"))
        assert result.shape == valid_tensor.shape

    def test_1d_tensor_rejected(self):
        """Security: 1D tensors should be rejected."""
        io_handler = AudioEmbeddingMediaIO()

        invalid_tensor = torch.randn(768, dtype=torch.float32)
        encoded = _encode_tensor(invalid_tensor)

        with pytest.raises(ValueError) as exc_info:
            io_handler.load_base64("", encoded.decode("utf-8"))

        assert "2D" in str(exc_info.value) or "3D" in str(exc_info.value)

    def test_scalar_rejected(self):
        """Security: Scalar tensors should be rejected."""
        io_handler = AudioEmbeddingMediaIO()

        invalid_tensor = torch.tensor(1.0)
        encoded = _encode_tensor(invalid_tensor)

        with pytest.raises(ValueError):
            io_handler.load_base64("", encoded.decode("utf-8"))


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

    def test_attack_scenario_completions_api(self, model_config):
        """
        Simulate attack through Completions API with wrong hidden_size.

        Attack scenario:
        1. Attacker crafts embedding with wrong hidden_size
        2. Encodes it as base64
        3. Sends to /v1/completions with prompt_embeds parameter
        4. Server should reject before inference crashes
        """
        renderer = CompletionRenderer(model_config)

        expected_hidden_size = model_config.hf_text_config.hidden_size
        wrong_hidden_size = expected_hidden_size * 2

        # Attacker payload
        attack_tensor = torch.randn(100, wrong_hidden_size, dtype=torch.float32)
        attack_payload = _encode_tensor(attack_tensor)

        # Should be rejected
        with pytest.raises(ValueError) as exc_info:
            renderer.load_prompt_embeds(attack_payload)

        error_msg = str(exc_info.value)
        assert str(expected_hidden_size) in error_msg
        assert str(wrong_hidden_size) in error_msg

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

    def test_multiple_valid_embeddings_accepted(self, model_config):
        """
        Regression test: Multiple valid embeddings should still work.
        """
        renderer = CompletionRenderer(model_config)

        hidden_size = model_config.hf_text_config.hidden_size
        valid_tensors = [
            _encode_tensor(torch.randn(10, hidden_size, dtype=torch.float32)),
            _encode_tensor(torch.randn(20, hidden_size, dtype=torch.float32)),
            _encode_tensor(torch.randn(15, hidden_size, dtype=torch.float32)),
        ]

        result = renderer.load_prompt_embeds(valid_tensors)
        assert len(result) == 3


# Pytest fixtures
@pytest.fixture
def model_config():
    """ModelConfig for testing with prompt embeds enabled."""
    from vllm.config import ModelConfig

    return ModelConfig(
        model="facebook/opt-125m",
        tokenizer="facebook/opt-125m",
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="float32",
        seed=0,
        enable_prompt_embeds=True,
    )

