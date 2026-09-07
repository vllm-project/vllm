# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for InputBatch.logits_cover_all_tokens.

The property decides whether the model runner and the sampler may slice the
forward-pass output instead of gathering it with logits_indices. Getting it
wrong silently feeds the wrong hidden states to compute_logits, so the tests
below check it against logits_indices built by the real Triton kernel.
"""

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    combine_sampled_and_draft_tokens,
)

# The CPU tests below run anywhere; the kernel-agreement tests need a GPU.
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for the Triton kernel"
)

# --------------------------------------------------------------------------
# The predicate itself. Pure CPU: it only reads two integers off the batch.
# --------------------------------------------------------------------------

_predicate = InputBatch.logits_cover_all_tokens.fget


class _Batch:
    """Minimal stand-in exposing the two attributes the property reads."""

    def __init__(self, num_logits: int, num_tokens: int):
        self.logits_indices = torch.empty(num_logits, dtype=torch.int64)
        self.num_tokens = num_tokens


@pytest.mark.parametrize(
    ("num_logits", "num_tokens", "expected"),
    [
        # Pure decode: one row per request, one logit per request.
        pytest.param(8, 8, True, id="decode"),
        # Speculative verification, 4 requests x (1 bonus + 3 draft).
        pytest.param(16, 16, True, id="spec-verify-uniform"),
        # Ragged verification: draft counts differ, counts still match.
        pytest.param(11, 11, True, id="spec-verify-ragged"),
        # A prefilling request contributes many query rows but one logit.
        pytest.param(4, 260, False, id="chunked-prefill"),
        pytest.param(1, 1, True, id="single-request"),
    ],
)
def test_predicate(num_logits: int, num_tokens: int, expected: bool):
    assert _predicate(_Batch(num_logits, num_tokens)) is expected


# --------------------------------------------------------------------------
# Agreement with the real kernel. Requires CUDA + triton.
# --------------------------------------------------------------------------

DEVICE = torch.device("cuda")
MAX_SPEC_STEPS = 4


def _build_logits_indices(num_scheduled: list[int], num_draft: list[int]):
    """Drive the real kernel the way prepare_inputs does.

    Returns (logits_indices, num_tokens).
    """
    num_reqs = len(num_scheduled)
    # prepare_inputs asserts this; the property's reasoning depends on it.
    assert all(s >= d + 1 for s, d in zip(num_scheduled, num_draft))

    query_start_loc_np = np.zeros(num_reqs + 1, dtype=np.int32)
    np.cumsum(num_scheduled, out=query_start_loc_np[1:])
    num_tokens = int(query_start_loc_np[-1])

    num_logits_np = np.asarray(num_draft, dtype=np.int32) + 1
    cu_num_logits_np = np.zeros(num_reqs + 1, dtype=np.int32)
    np.cumsum(num_logits_np, out=cu_num_logits_np[1:])
    total_num_logits = int(cu_num_logits_np[-1])

    # seq_len == prefill_len keeps every request past its prefill, so the kernel
    # takes the draft-token branch rather than the prefill one.
    seq_lens = torch.full((num_reqs,), 64, dtype=torch.int32, device=DEVICE)

    logits_indices = combine_sampled_and_draft_tokens(
        input_ids=torch.zeros(num_tokens, dtype=torch.int32, device=DEVICE),
        idx_mapping=torch.arange(num_reqs, dtype=torch.int64, device=DEVICE),
        last_sampled_tokens=torch.zeros(
            num_reqs, MAX_SPEC_STEPS + 1, dtype=torch.int32, device=DEVICE
        ),
        query_start_loc=torch.from_numpy(query_start_loc_np).to(DEVICE),
        seq_lens=seq_lens,
        prefill_len=seq_lens.clone(),
        draft_tokens=torch.zeros(
            num_reqs, MAX_SPEC_STEPS, dtype=torch.int32, device=DEVICE
        ),
        cu_num_logits=torch.from_numpy(cu_num_logits_np).to(DEVICE),
        num_logits=total_num_logits,
    )
    return logits_indices, num_tokens


@requires_cuda
@pytest.mark.parametrize(
    ("num_scheduled", "num_draft"),
    [
        pytest.param([1, 1, 1, 1], [0, 0, 0, 0], id="decode"),
        pytest.param([4, 4, 4], [3, 3, 3], id="spec-verify-uniform"),
        pytest.param([4, 2, 5], [3, 1, 4], id="spec-verify-ragged"),
        pytest.param([1], [0], id="single-request"),
        # Mismatched counts: request 1 brings 6 query rows but only 2 logits.
        pytest.param([4, 6, 4], [3, 1, 3], id="not-covered"),
    ],
)
def test_predicate_agrees_with_kernel(num_scheduled: list[int], num_draft: list[int]):
    """The predicate must be true exactly when slicing is safe."""
    logits_indices, num_tokens = _build_logits_indices(num_scheduled, num_draft)
    batch = _Batch(logits_indices.shape[0], num_tokens)

    slicing_is_valid = torch.equal(
        logits_indices, torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)
    )
    assert _predicate(batch) is slicing_is_valid


@requires_cuda
def test_slice_matches_gather_on_hidden_states():
    """End result: the slice must select the same rows the gather would."""
    logits_indices, num_tokens = _build_logits_indices([4, 2, 5], [3, 1, 4])
    hidden_states = torch.randn(num_tokens, 128, device=DEVICE)

    assert _predicate(_Batch(logits_indices.shape[0], num_tokens))
    assert torch.equal(hidden_states[:num_tokens], hidden_states[logits_indices])
