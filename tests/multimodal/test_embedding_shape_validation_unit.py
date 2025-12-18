# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for embedding shape validation.

Simple, fast unit tests that can run without server fixtures.
Run with: pytest tests/multimodal/test_embedding_shape_validation_unit.py -v
"""

import base64
import io

import pytest
import torch

from vllm.config import ModelConfig
from vllm.entrypoints.renderer import CompletionRenderer
from vllm.multimodal.audio import AudioEmbeddingMediaIO
from vllm.multimodal.image import ImageEmbeddingMediaIO


def _encode_tensor(tensor: torch.Tensor) -> bytes:
    """Helper to encode a tensor as base64 bytes."""
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.read())


@pytest.fixture(scope="module")
def opt_125m_model_config():
    """Shared ModelConfig for OPT-125M with prompt embeds enabled.

    Used by TestPromptEmbedShapeValidation and TestShapeValidationDoSPrevention.
    """
    return ModelConfig(
        model="facebook/opt-125m",
        tokenizer="facebook/opt-125m",
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="float16",
        seed=0,
        enable_prompt_embeds=True,
    )


class TestPromptEmbedShapeValidation:
    """Test shape validation in prompt embeddings to prevent crashes."""

    def test_correct_hidden_size_accepted(self, opt_125m_model_config):
        """Baseline: Embeddings with correct hidden size should work."""
        renderer = CompletionRenderer(opt_125m_model_config)

        # OPT-125M has hidden_size=768
        correct_hidden_size = opt_125m_model_config.hf_text_config.hidden_size
        valid_tensor = torch.randn(10, correct_hidden_size, dtype=torch.float32)
        encoded = _encode_tensor(valid_tensor)

        # Should not raise any exception
        result = renderer.load_prompt_embeds(encoded)
        assert len(result) == 1
        assert result[0]["prompt_embeds"].shape == valid_tensor.shape

    def test_wrong_hidden_size_rejected(self, opt_125m_model_config):
        """Security: Embeddings with wrong hidden size should be rejected."""
        renderer = CompletionRenderer(opt_125m_model_config)

        # OPT-125M expects 768, send wrong size
        correct_hidden_size = opt_125m_model_config.hf_text_config.hidden_size
        wrong_hidden_size = 4096  # Wrong!
        assert wrong_hidden_size != correct_hidden_size

        invalid_tensor = torch.randn(10, wrong_hidden_size, dtype=torch.float32)
        encoded = _encode_tensor(invalid_tensor)

        # Should raise ValueError about shape mismatch
        with pytest.raises(ValueError) as exc_info:
            renderer.load_prompt_embeds(encoded)

        error_msg = str(exc_info.value)
        assert "hidden dimension mismatch" in error_msg.lower()
        assert str(wrong_hidden_size) in error_msg
        assert str(correct_hidden_size) in error_msg

    def test_extremely_large_hidden_size_rejected(self, opt_125m_model_config):
        """Security: Prevent DoS from extremely large tensors."""
        renderer = CompletionRenderer(opt_125m_model_config)

        # Try to send a huge tensor that would consume massive memory
        huge_hidden_size = 1000000  # 1 million dimensions
        invalid_tensor = torch.randn(10, huge_hidden_size, dtype=torch.float32)
        encoded = _encode_tensor(invalid_tensor)

        with pytest.raises(ValueError) as exc_info:
            renderer.load_prompt_embeds(encoded)

        assert "hidden dimension mismatch" in str(exc_info.value).lower()

    def test_tiny_hidden_size_rejected(self, opt_125m_model_config):
        """Security: Embeddings that are too small should be rejected."""
        renderer = CompletionRenderer(opt_125m_model_config)

        # Try to send undersized tensor
        tiny_hidden_size = 10  # Way too small
        invalid_tensor = torch.randn(10, tiny_hidden_size, dtype=torch.float32)
        encoded = _encode_tensor(invalid_tensor)

        with pytest.raises(ValueError):
            renderer.load_prompt_embeds(encoded)


class TestImageEmbedBasicValidation:
    """Test basic ndim validation in image embeddings."""

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
        """Security: 1D tensors should be rejected (invalid ndim)."""
        io_handler = ImageEmbeddingMediaIO()

        invalid_tensor = torch.randn(768, dtype=torch.float32)  # 1D
        encoded = _encode_tensor(invalid_tensor)

        with pytest.raises(ValueError) as exc_info:
            io_handler.load_base64("", encoded.decode("utf-8"))

        assert "must be 2D" in str(exc_info.value) or "3D" in str(exc_info.value)

    def test_4d_tensor_rejected(self):
        """Security: 4D tensors should be rejected (invalid ndim)."""
        io_handler = ImageEmbeddingMediaIO()

        invalid_tensor = torch.randn(1, 2, 10, 768, dtype=torch.float32)  # 4D
        encoded = _encode_tensor(invalid_tensor)

        with pytest.raises(ValueError) as exc_info:
            io_handler.load_base64("", encoded.decode("utf-8"))

        assert "must be 2D" in str(exc_info.value) or "3D" in str(exc_info.value)


class TestAudioEmbedBasicValidation:
    """Test basic ndim validation in audio embeddings."""

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
        """Security: 1D tensors should be rejected (invalid ndim)."""
        io_handler = AudioEmbeddingMediaIO()

        invalid_tensor = torch.randn(768, dtype=torch.float32)  # 1D
        encoded = _encode_tensor(invalid_tensor)

        with pytest.raises(ValueError) as exc_info:
            io_handler.load_base64("", encoded.decode("utf-8"))

        assert "must be 2D" in str(exc_info.value) or "3D" in str(exc_info.value)

    def test_scalar_rejected(self):
        """Security: Scalar tensors should be rejected."""
        io_handler = AudioEmbeddingMediaIO()

        invalid_tensor = torch.tensor(1.0)  # 0D (scalar)
        encoded = _encode_tensor(invalid_tensor)

        with pytest.raises(ValueError):
            io_handler.load_base64("", encoded.decode("utf-8"))


class TestShapeValidationDoSPrevention:
    """
    Tests for DoS prevention through shape validation.

    Verifies that embeddings with incorrect shapes are rejected early,
    preventing crashes during model inference.
    """

    def test_prevent_crash_from_wrong_shape_prompt_embeds(self, opt_125m_model_config):
        """
        Prevent crash scenario: wrong hidden size in prompt embeddings.

        Without validation, this would pass initial checks but crash later
        during matrix multiplication in the model forward pass.
        """
        renderer = CompletionRenderer(opt_125m_model_config)

        # Send embedding with Llama-sized hidden_size to OPT model
        wrong_embedding = torch.randn(100, 4096, dtype=torch.float32)  # Llama size
        encoded = _encode_tensor(wrong_embedding)

        # Should be rejected at loading time, not during inference
        with pytest.raises(ValueError) as exc_info:
            renderer.load_prompt_embeds(encoded)

        error_msg = str(exc_info.value)
        assert "hidden dimension mismatch" in error_msg.lower()
        assert "768" in error_msg  # Expected
        assert "4096" in error_msg  # Received

    def test_batch_with_mixed_shapes_rejected(self, opt_125m_model_config):
        """All embeddings in a batch must have the same hidden size."""
        renderer = CompletionRenderer(opt_125m_model_config)

        correct_size = opt_125m_model_config.hf_text_config.hidden_size

        # One correct, one wrong
        batch = [
            _encode_tensor(torch.randn(10, correct_size, dtype=torch.float32)),
            _encode_tensor(
                torch.randn(10, correct_size + 100, dtype=torch.float32)
            ),  # Wrong!
        ]

        # Should fail on the second one
        with pytest.raises(ValueError):
            renderer.load_prompt_embeds(batch)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
