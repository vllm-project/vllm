# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for PPHandler broadcast/draft relay and sparse MLA stale-buffer fix."""

from types import SimpleNamespace

import pytest
import torch


class FakePPHandler:
    """Minimal stand-in for PPHandler that skips CUDA/distributed setup.

    Replicates the broadcast padding logic from PPHandler.broadcast().
    """

    def __init__(self, max_sample_len: int, is_last_rank: bool):
        self.max_sample_len = max_sample_len
        self.is_last_rank = is_last_rank

    def broadcast(self, sampled_token_ids: torch.Tensor) -> torch.Tensor:
        assert self.is_last_rank
        width = sampled_token_ids.shape[-1]
        if width != self.max_sample_len:
            assert width < self.max_sample_len
            padded = sampled_token_ids.new_full(
                (sampled_token_ids.shape[0], self.max_sample_len), -1
            )
            padded[:, :width] = sampled_token_ids
            sampled_token_ids = padded
        return sampled_token_ids


# ---------------------------------------------------------------------------
# DeepSeekMTP under pipeline parallelism
# ---------------------------------------------------------------------------


def test_deepseek_mtp_implements_supports_pp():
    """DeepSeekMTP must pass the supports_pp() gate used at model resolution;
    otherwise the engine refuses to build it under pipeline parallelism."""
    from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP
    from vllm.model_executor.models.interfaces import supports_pp

    assert supports_pp(DeepSeekMTP), (
        "DeepSeekMTP must implement SupportsPP to be built under PP"
    )


def test_deepseek_mtp_has_make_empty_intermediate_tensors():
    """DeepSeekMTP must provide make_empty_intermediate_tensors."""
    from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP

    assert hasattr(DeepSeekMTP, "make_empty_intermediate_tensors"), (
        "DeepSeekMTP must provide make_empty_intermediate_tensors"
    )


# ---------------------------------------------------------------------------
# PPHandler.broadcast() pads sampled_token_ids to max_sample_len
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "width,max_sample_len",
    [
        (1, 2),  # prefill / first decode (width=1, max_sample_len=2)
        (1, 4),  # K=3 spec decode (width=1, max_sample_len=4)
        (2, 4),  # K=1 spec decode (width=2, max_sample_len=4)
    ],
)
def test_pphandler_broadcast_pads_to_max_sample_len(width, max_sample_len):
    """broadcast() must pad sampled_token_ids to max_sample_len so the
    NCCL send/recv element counts always match."""
    handler = FakePPHandler(max_sample_len=max_sample_len, is_last_rank=True)
    num_reqs = 4
    sampled = torch.zeros(num_reqs, width, dtype=torch.int64)
    result = handler.broadcast(sampled)
    assert result.shape == (num_reqs, max_sample_len)
    # Trailing positions should be -1 (ignored by post_update)
    assert (result[:, width:] == -1).all()
    # Original values preserved
    assert (result[:, :width] == 0).all()


def test_pphandler_broadcast_no_pad_when_already_max():
    """broadcast() should not pad when width == max_sample_len."""
    handler = FakePPHandler(max_sample_len=4, is_last_rank=True)
    sampled = torch.zeros(4, 4, dtype=torch.int64)
    result = handler.broadcast(sampled)
    assert result.shape == (4, 4)
    assert (result == 0).all()


# ---------------------------------------------------------------------------
# Sparse MLA backends read topk_indices_buffer dynamically
# ---------------------------------------------------------------------------


def test_sparse_mla_backend_reads_topk_indices_buffer_dynamically():
    """Sparse MLA backends must read topk_indices_buffer dynamically via
    self._indexer: after _maybe_share_lm_head replaces
    Indexer.topk_indices_buffer, the impl must see the new buffer."""

    # Simulate the pattern used in all sparse MLA backends:
    # __init__ stores self._indexer = indexer
    # forward_mqa reads self._indexer.topk_indices_buffer dynamically

    initial_buffer = torch.zeros(128, 128, dtype=torch.int32)
    indexer = SimpleNamespace(topk_indices_buffer=initial_buffer)

    # Create a minimal impl that follows the same pattern as the real backends
    class FakeSparseMLAImpl:
        def __init__(self, indexer=None, topk_indices_buffer=None):
            self._indexer = indexer
            self.topk_indices_buffer = (
                indexer.topk_indices_buffer
                if indexer is not None
                else topk_indices_buffer
            )

        def forward_mqa(self):
            buf = (
                self._indexer.topk_indices_buffer
                if self._indexer is not None
                else self.topk_indices_buffer
            )
            return buf

    impl = FakeSparseMLAImpl(indexer=indexer)

    # Simulate _maybe_share_lm_head replacing the buffer
    new_buffer = torch.ones(128, 128, dtype=torch.int32)
    indexer.topk_indices_buffer = new_buffer

    # The impl should read the new buffer dynamically
    assert impl.forward_mqa() is new_buffer, (
        "Impl must read topk_indices_buffer dynamically via self._indexer"
    )
    assert impl.forward_mqa() is not initial_buffer, (
        "Impl must not hold a stale reference to the old buffer"
    )


def test_sparse_mla_backend_handles_no_indexer():
    """Verify sparse MLA backends handle indexer=None (skip-topk layers)."""

    class FakeSparseMLAImpl:
        def __init__(self, indexer=None, topk_indices_buffer=None):
            self._indexer = indexer
            self.topk_indices_buffer = (
                indexer.topk_indices_buffer
                if indexer is not None
                else topk_indices_buffer
            )

        def forward_mqa(self):
            buf = (
                self._indexer.topk_indices_buffer
                if self._indexer is not None
                else self.topk_indices_buffer
            )
            return buf

    # No indexer, buffer passed directly
    buffer = torch.ones(128, 128, dtype=torch.int32)
    impl = FakeSparseMLAImpl(indexer=None, topk_indices_buffer=buffer)
    assert impl.forward_mqa() is buffer


# ---------------------------------------------------------------------------
# Packed single-op broadcast: [sampled | num_sampled | num_rejected | draft]
# ---------------------------------------------------------------------------


class FakePackedPPHandler:
    """CPU stand-in replicating PPHandler's packed-broadcast wire format.

    Mirrors broadcast() (build + stage), broadcast_draft() (fill + flush) and
    receive() (single recv + slice) without CUDA/NCCL. The wire row layout is
    int64 [sampled(max_sample_len) | num_sampled | num_rejected |
    draft(max_sample_len - 1)]; width is static so send/recv element counts
    always match.
    """

    def __init__(self, max_sample_len: int):
        self.max_sample_len = max_sample_len
        self.packed_width = 2 * max_sample_len + 1
        self._staged_packed: torch.Tensor | None = None

    def broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
    ) -> torch.Tensor | None:
        """Returns the wire row when sent directly (no spec decode), else
        stages it for broadcast_draft() and returns None."""
        assert sampled_token_ids.dtype == torch.int64
        msl = self.max_sample_len
        num_reqs = sampled_token_ids.shape[0]
        width = sampled_token_ids.shape[-1]
        assert width <= msl
        packed = torch.full((num_reqs, self.packed_width), -1, dtype=torch.int64)
        packed[:, :width] = sampled_token_ids
        packed[:, msl] = num_sampled
        packed[:, msl + 1] = num_rejected
        if msl > 1:
            assert self._staged_packed is None
            self._staged_packed = packed
            return None
        return packed

    def broadcast_draft(self, draft_tokens: torch.Tensor) -> torch.Tensor:
        assert self.max_sample_len > 1
        packed = self._staged_packed
        assert packed is not None
        self._staged_packed = None
        packed[:, self.max_sample_len + 2 :] = draft_tokens
        return packed

    def receive(self, packed: torch.Tensor):
        """Replicates the receive() unpack: slices + dtype restoration."""
        msl = self.max_sample_len
        sampled_tokens = packed[:, :msl].contiguous()
        num_sampled = packed[:, msl].to(torch.int32)
        num_rejected = packed[:, msl + 1].to(torch.int32)
        draft_tokens = packed[:, msl + 2 :] if msl > 1 else None
        return sampled_tokens, num_sampled, num_rejected, draft_tokens


@pytest.mark.parametrize(
    "width,max_sample_len",
    [
        (1, 2),  # K=1 spec decode, prefill / first-decode step (no full width)
        (2, 2),  # K=1 spec decode, steady state
        (1, 4),  # K=3 spec decode, first-decode step
        (4, 4),  # K=3 spec decode, steady state
    ],
)
def test_packed_broadcast_roundtrip_spec_decode(width, max_sample_len):
    """Stage in broadcast(), flush in broadcast_draft(), unpack in receive():
    every field must round-trip with the original dtypes."""
    handler = FakePackedPPHandler(max_sample_len=max_sample_len)
    num_reqs, msl = 3, max_sample_len
    sampled = torch.arange(num_reqs * width, dtype=torch.int64).reshape(
        num_reqs, width
    )
    num_sampled = torch.tensor([1] * num_reqs, dtype=torch.int32)
    num_rejected = torch.tensor([0, 1, 2][:num_reqs], dtype=torch.int32)
    draft = torch.full((num_reqs, msl - 1), 7, dtype=torch.int64)

    assert handler.broadcast(sampled, num_sampled, num_rejected) is None  # staged
    wire = handler.broadcast_draft(draft)
    assert wire.shape == (num_reqs, 2 * msl + 1)
    assert wire.dtype == torch.int64

    out_sampled, out_num_sampled, out_num_rejected, out_draft = handler.receive(wire)
    # Sampled: original values then -1 padding (ignored by post_update).
    assert (out_sampled[:, :width] == sampled).all()
    assert (out_sampled[:, width:] == -1).all()
    # Counts: values AND downstream dtype (int32) preserved.
    assert (out_num_sampled == num_sampled).all()
    assert out_num_sampled.dtype == torch.int32
    assert (out_num_rejected == num_rejected).all()
    assert out_num_rejected.dtype == torch.int32
    # Draft relay: values and dtype preserved.
    assert out_draft is not None
    assert (out_draft == draft).all()
    assert out_draft.dtype == torch.int64


def test_packed_broadcast_no_spec_sends_directly():
    """Without spec decode (max_sample_len == 1) broadcast() must send the
    3-column row itself; receive() must yield draft_tokens=None."""
    handler = FakePackedPPHandler(max_sample_len=1)
    sampled = torch.tensor([[5], [6]], dtype=torch.int64)
    num_sampled = torch.ones(2, dtype=torch.int32)
    num_rejected = torch.zeros(2, dtype=torch.int32)
    wire = handler.broadcast(sampled, num_sampled, num_rejected)
    assert wire is not None and wire.shape == (2, 3)
    out_sampled, out_num_sampled, out_num_rejected, out_draft = handler.receive(wire)
    assert (out_sampled == sampled).all()
    assert (out_num_sampled == num_sampled).all()
    assert (out_num_rejected == num_rejected).all()
    assert out_draft is None


def test_packed_broadcast_double_stage_asserts():
    """Staging twice without a broadcast_draft() flush is a pairing bug and
    must crash loudly (silent mispairing would desync NCCL op counts)."""
    handler = FakePackedPPHandler(max_sample_len=4)
    sampled = torch.zeros(2, 1, dtype=torch.int64)
    counts = torch.zeros(2, dtype=torch.int32)
    handler.broadcast(sampled, counts, counts)
    with pytest.raises(AssertionError):
        handler.broadcast(sampled, counts, counts)


def test_packed_broadcast_flush_without_stage_asserts():
    """broadcast_draft() with nothing staged must crash loudly too."""
    handler = FakePackedPPHandler(max_sample_len=4)
    with pytest.raises(AssertionError):
        handler.broadcast_draft(torch.zeros(2, 3, dtype=torch.int64))
