# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DiffusionSampler logprobs assembly (issue #45689)."""

import numpy as np
import torch

from vllm.model_executor.models.diffusion_gemma import DiffusionSampler
from vllm.v1.outputs import LogprobsTensors


def _make_stash(n_rows: int, n_lp: int, fill: int) -> LogprobsTensors:
    return LogprobsTensors(
        logprob_token_ids=torch.full((n_rows, n_lp), fill, dtype=torch.int32),
        logprobs=torch.full((n_rows, n_lp), float(fill)),
        selected_token_ranks=torch.full((n_rows,), fill, dtype=torch.int32),
    )


def test_only_committing_request_pops_its_logprobs():
    """Regression for #45689.

    In a mixed batch where request A (slot 0) commits while request B (slot 1)
    merely converged this step, only A's stash may be popped. B's stash must
    survive until B's own commit; popping it early makes the committed
    response return fewer logprob rows than tokens (IndexError downstream).
    """
    slots_np = np.array([0, 1], dtype=np.int64)
    is_decode_np = np.array([True, True])
    pending = {0: _make_stash(3, 2, 0), 1: _make_stash(4, 2, 1)}

    out = DiffusionSampler._assemble_committed_logprobs(
        num_reqs=2,
        slots_np=slots_np,
        is_decode_np=is_decode_np,
        committing_slots={0},  # only slot 0 commits this step
        pending_logprobs=pending,
    )

    assert out is not None
    assert out.logprobs.shape[0] == 3  # only A emitted
    assert 0 not in pending  # A's stash consumed
    assert 1 in pending and pending[1].logprobs.shape[0] == 4  # B untouched


def test_both_committing_emit_in_request_order():
    slots_np = np.array([0, 1], dtype=np.int64)
    is_decode_np = np.array([True, True])
    pending = {0: _make_stash(3, 2, 0), 1: _make_stash(4, 2, 1)}

    out = DiffusionSampler._assemble_committed_logprobs(
        num_reqs=2,
        slots_np=slots_np,
        is_decode_np=is_decode_np,
        committing_slots={0, 1},
        pending_logprobs=pending,
    )

    assert out is not None
    assert out.logprobs.shape[0] == 7  # 3 + 4
    # One cumulative offset per request, in request order.
    assert out.cu_num_generated_tokens == [0, 3]
    assert pending == {}


def test_no_committing_request_returns_none():
    slots_np = np.array([0, 1], dtype=np.int64)
    is_decode_np = np.array([True, True])
    pending = {0: _make_stash(3, 2, 0)}

    out = DiffusionSampler._assemble_committed_logprobs(
        num_reqs=2,
        slots_np=slots_np,
        is_decode_np=is_decode_np,
        committing_slots=set(),  # nobody commits this step
        pending_logprobs=pending,
    )

    assert out is None
    assert 0 in pending  # stash preserved


def test_non_decode_request_is_skipped():
    """A committing slot that is not a decode request must not be popped."""
    slots_np = np.array([0, 1], dtype=np.int64)
    is_decode_np = np.array([False, True])  # req 0 is a prefill
    pending = {0: _make_stash(3, 2, 0), 1: _make_stash(4, 2, 1)}

    out = DiffusionSampler._assemble_committed_logprobs(
        num_reqs=2,
        slots_np=slots_np,
        is_decode_np=is_decode_np,
        committing_slots={0, 1},
        pending_logprobs=pending,
    )

    assert out is not None
    assert out.logprobs.shape[0] == 4  # only the decode request (slot 1)
    assert 0 in pending  # prefill slot's stash untouched
    assert 1 not in pending
