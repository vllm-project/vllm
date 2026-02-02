# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests verify that malicious sparse tensors are rejected before they can trigger
out-of-bounds memory writes during to_dense() operations.
"""

import base64
import io

import pytest
import torch

from vllm.multimodal.media import AudioEmbeddingMediaIO, ImageEmbeddingMediaIO
from vllm.renderers.embed_utils import safe_load_prompt_embeds


@pytest.fixture
def model_config():
    """Mock ModelConfig for testing."""
    from vllm.config import ModelConfig

    return ModelConfig(
        model="facebook/opt-125m",
        tokenizer="facebook/opt-125m",
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="float32",
        seed=0,
        enable_prompt_embeds=True,  # Required for prompt embeds tests
    )


def _encode_tensor(tensor: torch.Tensor) -> bytes:
    """Helper to encode a tensor as base64 bytes."""
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.read())


def _create_malicious_sparse_tensor() -> torch.Tensor:
    """
    Create a malicious sparse COO tensor with out-of-bounds indices.

    This tensor has indices that point beyond the declared shape, which would
    cause an out-of-bounds write when converted to dense format without
    validation.
    """
    # Create a 3x3 sparse tensor but with indices pointing to (10, 10)
    indices = torch.tensor([[10], [10]])  # Out of bounds for 3x3 shape
    values = torch.tensor([1.0])
    shape = (3, 3)

    # Create sparse tensor (this will be invalid)
    sparse_tensor = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)
    return sparse_tensor


def _create_valid_sparse_tensor() -> torch.Tensor:
    """Create a valid sparse COO tensor for baseline testing."""
    indices = torch.tensor([[0, 1, 2], [0, 1, 2]])
    values = torch.tensor([1.0, 2.0, 3.0])
    shape = (3, 3)

    sparse_tensor = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)
    return sparse_tensor


def _create_valid_dense_tensor() -> torch.Tensor:
    """Create a valid dense tensor for baseline testing."""
    return torch.randn(10, 768, dtype=torch.float32)  # (seq_len, hidden_size)


class TestPromptEmbedsValidation:
    """Test sparse tensor validation in prompt embeddings (Completions API)."""

    def test_valid_dense_tensor_accepted(self, model_config):
        """Baseline: Valid dense tensors should work normally."""
        valid_tensor = _create_valid_dense_tensor()
        encoded = _encode_tensor(valid_tensor)

        # Should not raise any exception
        result = safe_load_prompt_embeds(model_config, encoded)
        assert result.shape == valid_tensor.shape

    def test_valid_sparse_tensor_accepted(self):
        """Baseline: Valid sparse tensors should load successfully."""
        io_handler = ImageEmbeddingMediaIO()

        valid_sparse = _create_valid_sparse_tensor()
        encoded = _encode_tensor(valid_sparse)

        # Should not raise any exception (sparse tensors remain sparse)
        result = io_handler.load_base64("", encoded.decode("utf-8"))
        assert result.shape == valid_sparse.shape

    def test_malicious_sparse_tensor_rejected(self, model_config):
        """Security: Malicious sparse tensors should be rejected."""
        malicious_tensor = _create_malicious_sparse_tensor()
        encoded = _encode_tensor(malicious_tensor)

        # Should raise RuntimeError due to invalid sparse tensor
        with pytest.raises((RuntimeError, ValueError)) as exc_info:
            safe_load_prompt_embeds(model_config, encoded)

        # Error should indicate sparse tensor validation failure
        error_msg = str(exc_info.value).lower()
        assert "sparse" in error_msg or "index" in error_msg or "bounds" in error_msg

    def test_extremely_large_indices_rejected(self, model_config):
        """Security: Sparse tensors with extremely large indices should be rejected."""
        # Create tensor with indices far beyond reasonable bounds
        indices = torch.tensor([[999999], [999999]])
        values = torch.tensor([1.0])
        shape = (10, 10)

        malicious_tensor = torch.sparse_coo_tensor(
            indices, values, shape, dtype=torch.float32
        )
        encoded = _encode_tensor(malicious_tensor)

        with pytest.raises((RuntimeError, ValueError)):
            safe_load_prompt_embeds(model_config, encoded)

    def test_negative_indices_rejected(self, model_config):
        """Security: Sparse tensors with negative indices should be rejected."""
        # Create tensor with negative indices
        indices = torch.tensor([[-1], [-1]])
        values = torch.tensor([1.0])
        shape = (10, 10)

        malicious_tensor = torch.sparse_coo_tensor(
            indices, values, shape, dtype=torch.float32
        )
        encoded = _encode_tensor(malicious_tensor)

        with pytest.raises((RuntimeError, ValueError)):
            safe_load_prompt_embeds(model_config, encoded)


class TestImageEmbedsValidation:
    """Test sparse tensor validation in image embeddings (Chat API)."""

    def test_valid_dense_tensor_accepted(self):
        """Baseline: Valid dense tensors should work normally."""
        io_handler = ImageEmbeddingMediaIO()

        valid_tensor = _create_valid_dense_tensor()
        encoded = _encode_tensor(valid_tensor)

        # Should not raise any exception
        result = io_handler.load_base64("", encoded.decode("utf-8"))
        assert result.shape == valid_tensor.shape

    def test_valid_sparse_tensor_accepted(self):
        """Baseline: Valid sparse tensors should load successfully."""
        io_handler = AudioEmbeddingMediaIO()

        valid_sparse = _create_valid_sparse_tensor()
        encoded = _encode_tensor(valid_sparse)

        # Should not raise any exception (sparse tensors remain sparse)
        result = io_handler.load_base64("", encoded.decode("utf-8"))
        assert result.shape == valid_sparse.shape

    def test_malicious_sparse_tensor_rejected(self):
        """Security: Malicious sparse tensors should be rejected."""
        io_handler = ImageEmbeddingMediaIO()

        malicious_tensor = _create_malicious_sparse_tensor()
        encoded = _encode_tensor(malicious_tensor)

        # Should raise RuntimeError due to invalid sparse tensor
        with pytest.raises((RuntimeError, ValueError)) as exc_info:
            io_handler.load_base64("", encoded.decode("utf-8"))

        error_msg = str(exc_info.value).lower()
        assert "sparse" in error_msg or "index" in error_msg or "bounds" in error_msg

    def test_load_bytes_validates(self):
        """Security: Validation should also work for load_bytes method."""
        io_handler = ImageEmbeddingMediaIO()

        malicious_tensor = _create_malicious_sparse_tensor()
        buffer = io.BytesIO()
        torch.save(malicious_tensor, buffer)
        buffer.seek(0)

        with pytest.raises((RuntimeError, ValueError)):
            io_handler.load_bytes(buffer.read())


class TestAudioEmbedsValidation:
    """Test sparse tensor validation in audio embeddings (Chat API)."""

    def test_valid_dense_tensor_accepted(self):
        """Baseline: Valid dense tensors should work normally."""
        io_handler = AudioEmbeddingMediaIO()

        valid_tensor = _create_valid_dense_tensor()
        encoded = _encode_tensor(valid_tensor)

        # Should not raise any exception
        result = io_handler.load_base64("", encoded.decode("utf-8"))
        assert result.shape == valid_tensor.shape

    def test_valid_sparse_tensor_accepted(self):
        """Baseline: Valid sparse tensors should be converted successfully."""
        io_handler = AudioEmbeddingMediaIO()

        valid_sparse = _create_valid_sparse_tensor()
        encoded = _encode_tensor(valid_sparse)

        # Should not raise any exception
        result = io_handler.load_base64("", encoded.decode("utf-8"))
        assert result.is_sparse is False

    def test_malicious_sparse_tensor_rejected(self):
        """Security: Malicious sparse tensors should be rejected."""
        io_handler = AudioEmbeddingMediaIO()

        malicious_tensor = _create_malicious_sparse_tensor()
        encoded = _encode_tensor(malicious_tensor)

        # Should raise RuntimeError due to invalid sparse tensor
        with pytest.raises((RuntimeError, ValueError)) as exc_info:
            io_handler.load_base64("", encoded.decode("utf-8"))

        error_msg = str(exc_info.value).lower()
        assert "sparse" in error_msg or "index" in error_msg or "bounds" in error_msg

    def test_load_bytes_validates(self):
        """Security: Validation should also work for load_bytes method."""
        io_handler = AudioEmbeddingMediaIO()

        malicious_tensor = _create_malicious_sparse_tensor()
        buffer = io.BytesIO()
        torch.save(malicious_tensor, buffer)
        buffer.seek(0)

        with pytest.raises((RuntimeError, ValueError)):
            io_handler.load_bytes(buffer.read())


class TestSparseTensorValidationIntegration:
    """
    These tests verify the complete attack chain is blocked at all entry points.
    """

    def test_attack_scenario_completions_api(self, model_config):
        """
        Simulate a complete attack through the Completions API.

        Attack scenario:
        1. Attacker crafts malicious sparse tensor
        2. Encodes it as base64
        3. Sends to /v1/completions with prompt_embeds parameter
        4. Server should reject before memory corruption occurs
        """
        # Step 1-2: Attacker creates malicious payload
        attack_payload = _encode_tensor(_create_malicious_sparse_tensor())

        # Step 3-4: Server processes and should reject
        with pytest.raises((RuntimeError, ValueError)):
            safe_load_prompt_embeds(model_config, attack_payload)

    def test_attack_scenario_chat_api_image(self):
        """
        Simulate attack through Chat API with image_embeds.

        Verifies the image embeddings path is protected.
        """
        io_handler = ImageEmbeddingMediaIO()
        attack_payload = _encode_tensor(_create_malicious_sparse_tensor())

        with pytest.raises((RuntimeError, ValueError)):
            io_handler.load_base64("", attack_payload.decode("utf-8"))

    def test_attack_scenario_chat_api_audio(self):
        """
        Simulate attack through Chat API with audio_embeds.

        Verifies the audio embeddings path is protected.
        """
        io_handler = AudioEmbeddingMediaIO()
        attack_payload = _encode_tensor(_create_malicious_sparse_tensor())

        with pytest.raises((RuntimeError, ValueError)):
            io_handler.load_base64("", attack_payload.decode("utf-8"))
