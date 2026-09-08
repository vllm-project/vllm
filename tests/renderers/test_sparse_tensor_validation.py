# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests verify that malicious sparse tensors are rejected before they can trigger
out-of-bounds memory writes, or an unbounded allocation, during to_dense()
operations.

A sparse tensor carries its own declared shape, so the two failure modes are
independent: invalid *indices* are what the invariant checks catch, while a
valid tensor with an enormous declared *shape* passes those checks and is
bounded by `VLLM_MAX_EMBED_DECODE_BYTES` instead.
"""

import io

import numpy as np
import pybase64 as base64
import pytest
import torch

import vllm.envs as envs
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.media import AudioEmbeddingMediaIO, ImageEmbeddingMediaIO
from vllm.multimodal.media.video import VideoEmbeddingMediaIO
from vllm.renderers.embed_utils import safe_load_prompt_embeds
from vllm.utils.sparse_utils import safe_to_dense


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

    # Create sparse tensor (this will be invalid). Pass `check_invariants=False`
    # explicitly so this fixture is robust to process-wide invariant-check state
    # left enabled by other tests (the global flag isn't thread-local, and
    # concurrent users of the `check_sparse_tensor_invariants` context manager
    # can leak the "enabled" state across tests).
    sparse_tensor = torch.sparse_coo_tensor(
        indices, values, shape, dtype=torch.float32, check_invariants=False
    )
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
            indices, values, shape, dtype=torch.float32, check_invariants=False
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
            indices, values, shape, dtype=torch.float32, check_invariants=False
        )
        encoded = _encode_tensor(malicious_tensor)

        with pytest.raises((RuntimeError, ValueError)):
            safe_load_prompt_embeds(model_config, encoded)

    def test_hidden_size_mismatch_rejected(self, model_config):
        """Tensors whose trailing dim doesn't match the model's hidden_size
        must be rejected at parse time."""
        # opt-125m has hidden_size=768, passing 512 triggers the check.
        wrong_hidden = torch.randn(10, 512, dtype=torch.float32)
        encoded = _encode_tensor(wrong_hidden)

        with pytest.raises(VLLMValidationError, match="hidden_size"):
            safe_load_prompt_embeds(model_config, encoded)

    def test_float_dtype_mismatch_cast_to_model_dtype(self, model_config):
        """Tensors whose dtype doesn't match the model's dtype but are still
        floating-point are cast, since API clients generally can't know the
        server's `--dtype` setting ahead of time."""
        # Fixture pins model dtype to float32, upload a bfloat16 tensor.
        mismatched_float = torch.randn(10, 768, dtype=torch.bfloat16)
        encoded = _encode_tensor(mismatched_float)

        result = safe_load_prompt_embeds(model_config, encoded)

        assert result.dtype == torch.float32
        assert result.shape == mismatched_float.shape

    def test_non_float_dtype_rejected(self, model_config):
        """Non-floating-point dtypes cannot be safely cast for embeddings
        (e.g. integer tensors almost certainly indicate caller confusion),
        so they are rejected at parse time."""
        non_float = torch.randint(0, 100, (10, 768), dtype=torch.int32)
        encoded = _encode_tensor(non_float)

        with pytest.raises(VLLMValidationError, match="floating-point"):
            safe_load_prompt_embeds(model_config, encoded)

    def test_non_2d_tensor_rejected(self, model_config):
        """Tensors that aren't 2D (even after squeezing a leading dim)
        must be rejected with a clear error."""
        # A 1D tensor cannot be interpreted as (num_tokens, hidden_size).
        bad = torch.randn(768, dtype=torch.float32)
        encoded = _encode_tensor(bad)

        with pytest.raises(VLLMValidationError, match="2D tensor"):
            safe_load_prompt_embeds(model_config, encoded)

    def test_non_tensor_payload_rejected(self, model_config):
        """Deserializing to a non-Tensor object must raise a clear error
        instead of propagating an AssertionError."""
        # `torch.save` will serialize a plain dict; `weights_only=True` allows
        # loading built-in containers, so this exercises the isinstance check.
        buffer = io.BytesIO()
        torch.save({"not": "a tensor"}, buffer)
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read())

        with pytest.raises(VLLMValidationError, match="torch.Tensor"):
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

    def test_valid_numpy_tensor_accepted(self):
        """numpy .npy format should load and return correct tensor."""
        io_handler = ImageEmbeddingMediaIO()

        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        buf = io.BytesIO()
        np.save(buf, arr)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")

        result = io_handler.load_base64("", encoded)
        assert isinstance(result, torch.Tensor)
        assert result.shape == torch.Size([2, 3])
        assert result.dtype == torch.float32
        assert torch.allclose(result, torch.from_numpy(arr))

    def test_numpy_int32_tensor_accepted(self):
        """numpy int32 arrays should round-trip correctly."""

        io_handler = ImageEmbeddingMediaIO()

        arr = np.arange(280, dtype=np.int32)
        buf = io.BytesIO()
        np.save(buf, arr)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")

        result = io_handler.load_base64("", encoded)
        assert result.dtype == torch.int32
        assert result.shape == torch.Size([280])
        assert (result == torch.from_numpy(arr)).all()

    def test_load_file_numpy_tensor_accepted(self, tmp_path):
        """numpy .npy files should load correctly via load_file."""

        io_handler = ImageEmbeddingMediaIO()

        arr = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
        npy_path = tmp_path / "image_embeds.npy"
        np.save(npy_path, arr)

        result = io_handler.load_file(npy_path)
        assert isinstance(result, torch.Tensor)
        assert result.shape == torch.Size([2, 2])
        assert result.dtype == torch.float32
        assert torch.allclose(result, torch.from_numpy(arr))


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


# The real attack payload: 4 TiB of float32 once densified, from a single
# stored element. Only ever measured, never decoded.
BOMB_SHAPE = (2**20, 2**20)

# What the size-limit tests below actually build. Each of them also lowers
# `VLLM_MAX_EMBED_DECODE_BYTES` to `TEST_LIMIT_BYTES`, so the guard still has
# something to reject while nothing here can allocate more than 4 MiB even if
# the guard regressed. Keeping the 4 TiB shape in these tests would make the
# outcome depend on the runner: with `vm.overcommit_memory=0` the allocator
# refuses it outright, but with `vm.overcommit_memory=1` the malloc succeeds
# and the zero-fill is what fails, which would OOM the test process.
OVERSIZED_SHAPE = (1024, 1024)  # 4 MiB dense
TEST_LIMIT_BYTES = 4096

EMBEDDING_MEDIA_IO = [
    (ImageEmbeddingMediaIO, "image_embeds"),
    (AudioEmbeddingMediaIO, "audio_embeds"),
    (VideoEmbeddingMediaIO, "video_embeds"),
]


def _create_oversized_sparse_tensor(shape: tuple[int, int]) -> torch.Tensor:
    """A *valid* sparse tensor with one stored element and a declared `shape`.

    Indices are in bounds, so `check_sparse_tensor_invariants()` accepts it.
    That is the point: the invariant checks are not what stops this one.
    """
    indices = torch.tensor([[0], [0]])
    values = torch.tensor([1.0])
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)


@pytest.fixture
def small_decode_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_MAX_EMBED_DECODE_BYTES", str(TEST_LIMIT_BYTES))


class TestEmbeddingDecodeSizeLimit:
    """Bound the dense allocation a payload can declare.

    `weights_only=True` stops code execution and the invariant checks stop
    out-of-bounds indices, but neither bounds the declared shape.
    """

    def test_oversized_sparse_tensor_rejected(self, small_decode_limit):
        with pytest.raises(VLLMValidationError) as exc_info:
            safe_to_dense(
                _create_oversized_sparse_tensor(OVERSIZED_SHAPE),
                parameter="prompt_embeds",
            )

        assert "VLLM_MAX_EMBED_DECODE_BYTES" in str(exc_info.value)
        assert "prompt_embeds" in str(exc_info.value)

    def test_default_limit_would_reject_the_real_payload(self):
        """The shipped default has to bound the actual attack, not just the
        shrunken shape used above. Checked arithmetically so no test ever asks
        the allocator for 4 TiB."""
        dense_bytes = BOMB_SHAPE[0] * BOMB_SHAPE[1] * 4
        assert dense_bytes > envs.VLLM_MAX_EMBED_DECODE_BYTES > 0

    def test_attack_payload_is_tiny(self):
        """The payload really is small - this is amplification, not an upload."""
        assert len(_encode_tensor(_create_oversized_sparse_tensor(BOMB_SHAPE))) < 4096

    def test_valid_tensors_still_densify(self):
        assert safe_to_dense(
            _create_valid_dense_tensor(), parameter="prompt_embeds"
        ).shape == (10, 768)

        dense = safe_to_dense(_create_valid_sparse_tensor(), parameter="image_embeds")
        assert dense.shape == (3, 3)
        assert not dense.is_sparse

    def test_limit_is_configurable(self, monkeypatch: pytest.MonkeyPatch):
        # 32 x 32 float32 == 4096 bytes, exactly at the limit.
        monkeypatch.setenv("VLLM_MAX_EMBED_DECODE_BYTES", str(TEST_LIMIT_BYTES))

        at_limit = _create_oversized_sparse_tensor((32, 32))
        assert safe_to_dense(at_limit, parameter="image_embeds").shape == (32, 32)

        over_limit = _create_oversized_sparse_tensor((33, 32))
        with pytest.raises(VLLMValidationError):
            safe_to_dense(over_limit, parameter="image_embeds")

    def test_limit_can_be_disabled(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("VLLM_MAX_EMBED_DECODE_BYTES", "0")
        tensor = _create_oversized_sparse_tensor(OVERSIZED_SHAPE)

        assert safe_to_dense(tensor, parameter="image_embeds").shape == OVERSIZED_SHAPE

    @pytest.mark.parametrize("io_cls,parameter", EMBEDDING_MEDIA_IO)
    def test_media_io_rejects_oversized_base64(
        self, io_cls, parameter, small_decode_limit
    ):
        encoded = _encode_tensor(_create_oversized_sparse_tensor(OVERSIZED_SHAPE))

        with pytest.raises(VLLMValidationError) as exc_info:
            io_cls().load_base64("", encoded.decode("utf-8"))

        assert parameter in str(exc_info.value)

    @pytest.mark.parametrize("io_cls,parameter", EMBEDDING_MEDIA_IO)
    def test_media_io_rejects_oversized_file(
        self, io_cls, parameter, tmp_path, small_decode_limit
    ):
        path = tmp_path / "embeds.pt"
        torch.save(_create_oversized_sparse_tensor(OVERSIZED_SHAPE), path)

        with pytest.raises(VLLMValidationError):
            io_cls().load_file(path)

    @pytest.mark.parametrize("io_cls,parameter", EMBEDDING_MEDIA_IO)
    def test_media_io_rejects_non_tensor_payload(self, io_cls, parameter):
        """These three had no `isinstance` check, so a non-tensor payload
        surfaced as an `AttributeError` (500) rather than a 4xx."""
        buffer = io.BytesIO()
        torch.save({"not": "a tensor"}, buffer)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        with pytest.raises(VLLMValidationError) as exc_info:
            io_cls().load_base64("", encoded)

        assert "torch.Tensor" in str(exc_info.value)

    @pytest.mark.parametrize("io_cls,parameter", EMBEDDING_MEDIA_IO)
    def test_media_io_still_accepts_valid_embeddings(self, io_cls, parameter):
        tensor = torch.randn(4, 16, dtype=torch.float32)
        encoded = _encode_tensor(tensor)

        assert torch.equal(io_cls().load_base64("", encoded.decode("utf-8")), tensor)
