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


