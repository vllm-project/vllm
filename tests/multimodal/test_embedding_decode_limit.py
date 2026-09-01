# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests that client-supplied embedding payloads cannot amplify a tiny request
into an arbitrarily large dense allocation.

`torch.load(..., weights_only=True)` stops arbitrary code execution, and the
sparse invariant checks stop out-of-bounds indices, but neither bounds the
*declared shape* of a sparse tensor: a ~1.5 KB payload can declare a
(2**20, 2**20) float32 shape whose dense form is 4 TiB.

Run with: pytest tests/multimodal/test_embedding_decode_limit.py -v
"""

import io

import pybase64
import pytest
import torch

import vllm.envs as envs
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.media import AudioEmbeddingMediaIO, ImageEmbeddingMediaIO
from vllm.multimodal.media.video import VideoEmbeddingMediaIO
from vllm.utils.sparse_utils import safe_to_dense

# The real attack payload: 4 TiB of float32 once densified, from a handful of
# stored elements. Only ever measured, never decoded -- see
# `test_bomb_payload_is_tiny`.
BOMB_SHAPE = (2**20, 2**20)

# What the rejection tests actually build. Every such test also lowers
# `VLLM_MAX_EMBED_DECODE_BYTES` to `TEST_LIMIT_BYTES`, so the guard still has
# something to reject while nothing here can allocate more than 4 MiB even if
# the guard regressed. Keeping the 4 TiB shape in these tests would make the
# outcome depend on the runner: with `vm.overcommit_memory=0` the allocator
# refuses it (measured: `RuntimeError: DefaultCPUAllocator: can't allocate
# memory: you tried to allocate 4398046511104 bytes`, peak RSS 498 -> 504 MiB),
# but with `vm.overcommit_memory=1` the `malloc` succeeds and the zero-fill is
# what fails, which would OOM the test process instead of raising.
OVERSIZED_SHAPE = (1024, 1024)  # 4 MiB dense
TEST_LIMIT_BYTES = 4096


def _sparse_tensor(shape: tuple[int, int]) -> torch.Tensor:
    """A *valid* sparse tensor with one stored element and a declared `shape`.

    Indices are in bounds, so `check_sparse_tensor_invariants()` accepts it.
    That is the point: the invariant checks are not what stops this.
    """
    indices = torch.tensor([[0], [0]])
    values = torch.tensor([1.0])
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)


@pytest.fixture
def small_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_MAX_EMBED_DECODE_BYTES", str(TEST_LIMIT_BYTES))


def _encode_base64(tensor: torch.Tensor) -> str:
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return pybase64.b64encode(buffer.getvalue()).decode("utf-8")


class TestSafeToDense:
    def test_oversized_sparse_tensor_rejected(self, small_limit):
        with pytest.raises(VLLMValidationError) as exc_info:
            safe_to_dense(_sparse_tensor(OVERSIZED_SHAPE), parameter="prompt_embeds")

        assert "VLLM_MAX_EMBED_DECODE_BYTES" in str(exc_info.value)
        assert "prompt_embeds" in str(exc_info.value)

    def test_default_limit_would_reject_the_real_payload(self):
        """The shipped default has to bound the actual attack, not just the
        shrunken shape the other tests use. Checked arithmetically so no test
        ever asks the allocator for 4 TiB."""
        dense_bytes = BOMB_SHAPE[0] * BOMB_SHAPE[1] * 4
        assert dense_bytes > envs.VLLM_MAX_EMBED_DECODE_BYTES > 0

    def test_bomb_payload_is_tiny(self):
        """The payload really is small — this is amplification, not a big upload."""
        assert len(_encode_base64(_sparse_tensor(BOMB_SHAPE))) < 4096

    def test_dense_tensor_accepted(self):
        tensor = torch.randn(10, 768, dtype=torch.float32)
        assert safe_to_dense(tensor, parameter="prompt_embeds").shape == (10, 768)

    def test_small_sparse_tensor_accepted(self):
        indices = torch.tensor([[0, 1, 2], [0, 1, 2]])
        values = torch.tensor([1.0, 2.0, 3.0])
        tensor = torch.sparse_coo_tensor(indices, values, (3, 3))

        dense = safe_to_dense(tensor, parameter="image_embeds")
        assert dense.shape == (3, 3)
        assert not dense.is_sparse

    def test_non_tensor_payload_rejected(self):
        with pytest.raises(VLLMValidationError) as exc_info:
            safe_to_dense({"not": "a tensor"}, parameter="prompt_embeds")

        assert "torch.Tensor" in str(exc_info.value)

    def test_limit_is_configurable(self, monkeypatch: pytest.MonkeyPatch):
        # 32 x 32 float32 == 4096 bytes, exactly at the limit.
        monkeypatch.setenv("VLLM_MAX_EMBED_DECODE_BYTES", "4096")
        indices = torch.tensor([[0], [0]])
        values = torch.tensor([1.0])

        at_limit = torch.sparse_coo_tensor(indices, values, (32, 32))
        assert safe_to_dense(at_limit, parameter="image_embeds").shape == (32, 32)

        over_limit = torch.sparse_coo_tensor(indices, values, (33, 32))
        with pytest.raises(VLLMValidationError):
            safe_to_dense(over_limit, parameter="image_embeds")

    def test_limit_can_be_disabled(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("VLLM_MAX_EMBED_DECODE_BYTES", "0")
        indices = torch.tensor([[0], [0]])
        values = torch.tensor([1.0])
        tensor = torch.sparse_coo_tensor(indices, values, (1000, 1000))

        assert safe_to_dense(tensor, parameter="image_embeds").shape == (1000, 1000)


@pytest.mark.parametrize(
    "io_cls,parameter",
    [
        (ImageEmbeddingMediaIO, "image_embeds"),
        (AudioEmbeddingMediaIO, "audio_embeds"),
        (VideoEmbeddingMediaIO, "video_embeds"),
    ],
)
class TestEmbeddingMediaIO:
    """These are the objects reached from `/v1/chat/completions` content parts."""

    def test_oversized_sparse_tensor_rejected(self, io_cls, parameter, small_limit):
        encoded = _encode_base64(_sparse_tensor(OVERSIZED_SHAPE))

        with pytest.raises(VLLMValidationError) as exc_info:
            io_cls().load_base64("", encoded)

        assert parameter in str(exc_info.value)

    def test_valid_embedding_accepted(self, io_cls, parameter):
        tensor = torch.randn(4, 16, dtype=torch.float32)
        encoded = _encode_base64(tensor)

        loaded = io_cls().load_base64("", encoded)
        assert torch.equal(loaded, tensor)

    def test_oversized_file_rejected(self, io_cls, parameter, tmp_path, small_limit):
        path = tmp_path / "embeds.pt"
        torch.save(_sparse_tensor(OVERSIZED_SHAPE), path)

        with pytest.raises(VLLMValidationError):
            io_cls().load_file(path)
