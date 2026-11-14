# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_K31_BEAT_EAGERPREP's identity logits slice (``logits_indices_is_identity``,
``select_logits_rows``): for spec-decode batches the sampler's hidden rows, and
so the logits, are the gather's bit for bit with the flag on or off; the slice
is taken exactly when logits_indices is the identity, every other batch keeps
the gather."""

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.input_batch import (
    combine_sampled_and_draft_tokens,
    logits_indices_is_identity,
    select_logits_rows,
)

MAX_NUM_REQS = 8
HIDDEN = 64
VOCAB = 96

# (num_scheduled_tokens, num_draft_tokens, num_bonus, num_reqs_padded,
#  extra hidden rows (token padding), identity)
CASES = {
    "c1_verify": ([8], [7], 1, None, 0, True),
    "verify_uniform": ([8, 8, 8], [7, 7, 7], 1, None, 0, True),
    "verify_ragged": ([8, 3, 1, 6], [7, 2, 0, 5], 1, None, 0, True),
    "no_drafts": ([1, 1], [0, 0], 1, None, 0, True),
    "req_and_token_padding": ([8, 8], [7, 7], 1, 4, 16, True),
    "no_bonus": ([7, 2], [7, 2], 0, 4, 5, True),
    "chunked_prefill": ([8, 100, 8], [7, 0, 7], 1, 4, 8, False),
    "prefill_first": ([50, 8], [0, 7], 1, None, 0, False),
    "prefill_last": ([8, 33], [7, 0], 1, None, 0, False),
    "fewer_drafts": ([8], [3], 1, None, 0, False),
    "no_bonus_short": ([8, 3], [7, 2], 0, None, 0, False),
}


def _host_batch(num_scheduled, num_drafts, num_bonus, num_reqs_padded):
    """prepare_inputs' host cumsums for a batch with drafts: cu_num_logits
    [num_reqs + 1], query_start_loc padded with num_tokens then cut to
    num_reqs_padded + 1."""
    num_scheduled = np.array(num_scheduled, dtype=np.int32)
    num_logits = np.array(num_drafts, dtype=np.int32) + num_bonus
    assert (num_scheduled >= num_logits).all()
    num_reqs = len(num_scheduled)
    cu_num_logits = np.empty(num_reqs + 1, dtype=np.int32)
    cu_num_logits[0] = 0
    np.cumsum(num_logits, out=cu_num_logits[1:])
    query_start_loc = np.empty(MAX_NUM_REQS + 1, dtype=np.int32)
    query_start_loc[0] = 0
    np.cumsum(num_scheduled, out=query_start_loc[1 : num_reqs + 1])
    query_start_loc[num_reqs + 1 :] = int(num_scheduled.sum())
    num_reqs_padded = num_reqs_padded or num_reqs
    return cu_num_logits, query_start_loc[: num_reqs_padded + 1], num_reqs


def _ref_logits_indices(cu_num_logits, query_start_loc, num_reqs):
    """combine_sampled_and_draft_tokens' logits_indices: request i's rows
    [query_end - num_logits, query_end) at cu_num_logits[i]."""
    out = np.empty(int(cu_num_logits[num_reqs]), dtype=np.int64)
    for i in range(num_reqs):
        start, end = int(cu_num_logits[i]), int(cu_num_logits[i + 1])
        query_end = int(query_start_loc[i + 1])
        out[start:end] = np.arange(query_end - (end - start), query_end)
    return torch.from_numpy(out)


@pytest.mark.parametrize("case", list(CASES))
def test_identity_only_when_indices_are_identity(case):
    num_scheduled, num_drafts, num_bonus, padded, _, identity = CASES[case]
    cu, qsl, num_reqs = _host_batch(num_scheduled, num_drafts, num_bonus, padded)
    indices = _ref_logits_indices(cu, qsl, num_reqs)
    is_arange = torch.equal(indices, torch.arange(indices.shape[0]))
    assert is_arange is identity
    assert logits_indices_is_identity(cu, qsl, num_reqs) is identity


@pytest.mark.parametrize("case", list(CASES))
def test_logits_match_gather_flag_on_and_off(case):
    num_scheduled, num_drafts, num_bonus, padded, extra, _ = CASES[case]
    cu, qsl, num_reqs = _host_batch(num_scheduled, num_drafts, num_bonus, padded)
    indices = _ref_logits_indices(cu, qsl, num_reqs)
    num_tokens = int(sum(num_scheduled))
    gen = torch.Generator().manual_seed(len(case))
    hidden = torch.randn(num_tokens + extra, HIDDEN, generator=gen)
    hidden[num_tokens:] = float("nan")  # padded token rows: never selected
    lm_head = torch.randn(HIDDEN, VOCAB, generator=gen)

    off = select_logits_rows(hidden, indices, False)  # flag off: never set
    on = select_logits_rows(
        hidden, indices, logits_indices_is_identity(cu, qsl, num_reqs)
    )
    assert torch.equal(off, hidden[indices])
    assert torch.equal(on, off)
    assert torch.equal(on @ lm_head, off @ lm_head)
    if logits_indices_is_identity(cu, qsl, num_reqs):
        assert on.data_ptr() == hidden.data_ptr()  # a view, no gather


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (Triton)")
@pytest.mark.parametrize("case", list(CASES))
def test_ref_indices_match_combine_kernel(case):
    """The CPU reference above is the kernel's logits_indices."""
    num_scheduled, num_drafts, num_bonus, padded, _, _ = CASES[case]
    cu, qsl, num_reqs = _host_batch(num_scheduled, num_drafts, num_bonus, padded)
    device = torch.device("cuda")
    num_tokens = int(sum(num_scheduled))
    # the placement only reads cu_num_logits; draft rows wide enough for
    # every request's logits
    width = int(np.diff(cu).max())
    indices = combine_sampled_and_draft_tokens(
        torch.zeros(num_tokens, dtype=torch.int32, device=device),
        torch.arange(num_reqs, dtype=torch.int64, device=device),
        torch.zeros(num_reqs, dtype=torch.int32, device=device),
        torch.from_numpy(qsl).to(device),
        torch.full((len(qsl) - 1,), 1000, dtype=torch.int32, device=device),
        torch.full((num_reqs,), 10, dtype=torch.int32, device=device),
        torch.zeros(num_reqs, width, dtype=torch.int32, device=device),
        torch.from_numpy(cu).to(device),
        int(cu[num_reqs]),
        num_bonus,
    )
    assert torch.equal(indices.cpu(), _ref_logits_indices(cu, qsl, num_reqs))
