# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chunk selection in the rejection sampler's verification loop.

Adaptive verification compacts the logits on device but leaves
``cu_num_logits_np`` describing the pre-trim layout, so the CPU prefix sums must
not be used to slice the compacted tensor.
"""

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
    RejectionSampler,
    _iter_request_chunks,
)


class _FakeInputBatch:
    def __init__(self, cu_num_logits_np: np.ndarray, num_logits: int):
        self.num_reqs = cu_num_logits_np.size - 1
        # Pre-trim layout, as prepare_inputs leaves it.
        self.cu_num_logits_np = cu_num_logits_np
        # Compacted device layout, as reallocate_drafts produces it.
        step = num_logits // self.num_reqs
        device_cu = np.arange(self.num_reqs + 1, dtype=np.int32) * step
        device_cu[-1] = num_logits
        self.cu_num_logits = torch.from_numpy(device_cu)
        self.idx_mapping = torch.arange(self.num_reqs)
        self.idx_mapping_np = np.arange(self.num_reqs)
        self.expanded_idx_mapping = torch.zeros(num_logits, dtype=torch.int32)
        self.expanded_local_pos = torch.zeros(num_logits, dtype=torch.int32)


def _run(sampler, logits, input_batch, max_chunk_logits):
    return RejectionSampler._verify_in_chunks(
        sampler,
        logits,
        input_batch,
        None,
        torch.zeros(logits.shape[0], dtype=torch.int64),
        torch.zeros(logits.shape[0], dtype=torch.int32),
        max_chunk_logits,
        -1,  # NO_LOGPROBS
    )


@pytest.fixture
def sampler():
    """A RejectionSampler stub that records the ranges _verify is asked for."""
    obj = object.__new__(RejectionSampler)
    obj.sampler = type("S", (), {"logprobs_mode": "raw_logprobs"})()
    obj.calls = []

    def fake_verify(logits, draft_logits, draft_sampled, pos, cu_num_logits, *args):
        obj.calls.append((logits.shape[0], cu_num_logits.tolist()))
        n = cu_num_logits.shape[0] - 1
        return (
            None,
            torch.zeros(n, dtype=torch.int64),
            torch.zeros(n, dtype=torch.int32),
        )

    def fake_logprobs(*args, **kwargs):
        return None

    obj._verify = fake_verify
    obj._get_logprobs_tensors = fake_logprobs
    return obj


def test_compacted_logits_do_not_use_stale_cpu_offsets(sampler):
    """pre_trim_total > chunk_limit >= compacted_total must stay one chunk.

    Chunking off the stale CPU prefix sums would slice the compacted logits with
    pre-trim offsets, silently reading the wrong rows for every request past the
    first chunk boundary.
    """
    num_reqs, chunk_limit = 8, 40
    # Scheduler handed out 8 logits per request; the budget trimmed it to 4.
    pre_trim = np.arange(num_reqs + 1, dtype=np.int32) * 8  # total 64 > 40
    compacted_total = 32  # <= 40
    assert pre_trim[-1] > chunk_limit >= compacted_total

    input_batch = _FakeInputBatch(pre_trim, compacted_total)
    logits = torch.zeros(compacted_total, 16)
    _run(sampler, logits, input_batch, chunk_limit)

    assert len(sampler.calls) == 1, "compacted batch must verify in a single chunk"
    num_rows, cu_num_logits = sampler.calls[0]
    assert num_rows == compacted_total
    assert cu_num_logits == input_batch.cu_num_logits.tolist(), (
        "must verify against the device offsets, not the pre-trim CPU ones"
    )


def test_oversized_batch_still_chunks(sampler):
    """Without compaction, a batch past the limit is still split per request."""
    num_reqs, chunk_limit = 8, 24
    cu = np.arange(num_reqs + 1, dtype=np.int32) * 8  # total 64 > 24
    input_batch = _FakeInputBatch(cu, int(cu[-1]))
    logits = torch.zeros(int(cu[-1]), 16)
    _run(sampler, logits, input_batch, chunk_limit)

    assert len(sampler.calls) > 1
    assert sum(rows for rows, _ in sampler.calls) == int(cu[-1])
    assert all(rows <= chunk_limit for rows, _ in sampler.calls)


def test_iter_request_chunks_never_splits_a_request():
    cu = np.array([0, 3, 6, 9, 12], dtype=np.int32)
    chunks = list(_iter_request_chunks(cu, 7))
    assert chunks[0][0] == 0 and chunks[-1][1] == 4
    for start, end in chunks:
        assert end > start
