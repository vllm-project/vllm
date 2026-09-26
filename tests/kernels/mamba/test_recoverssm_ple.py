# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PLE short-conv RecoverSSM against the sequential reference.

Reference: the stock decode path fed the accepted tokens one at a time.
RecoverSSM: one spec-mode conv over the whole window from the checkpoint
(offset 0), then ``_PleConvCommit`` compacts the accepted suffix to the front.
The conv history is pure data movement, so the states must be bit-identical,
and the outputs of the accepted tokens must match as well.
"""

import pytest
import torch

from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.models.qwen4_exp.nvidia.ops.ple import ple_conv
from vllm.v1.attention.backends.ple_recoverssm import _PleConvCommit

C, KSIZE, DIL, SQ, NB = 512, 4, 3, 4, 8
STATE_LEN = (KSIZE - 1) * DIL
WIDTH = STATE_LEN + SQ - 1
DEV = "cuda"


def _state():
    if is_conv_state_dim_first():
        raw = torch.zeros(NB, C, WIDTH, dtype=torch.bfloat16, device=DEV)
        return raw, raw
    raw = torch.zeros(NB, WIDTH, C, dtype=torch.bfloat16, device=DEV)
    return raw, raw.transpose(-1, -2)


def _i32(xs):
    return torch.tensor(xs, dtype=torch.int32, device=DEV)


def _setup(seed, blocks):
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = torch.randn(C, KSIZE, generator=g, device=DEV).to(torch.bfloat16)
    x = torch.randn(len(blocks) * SQ, C, generator=g, device=DEV).to(torch.bfloat16)
    outer = torch.randn_like(x)
    raw, view = _state()
    for b in blocks:
        view[b, :, :STATE_LEN] = torch.randn(C, STATE_LEN, generator=g, device=DEV).to(
            torch.bfloat16
        )
    return w, x, outer, raw, view


def _sequential(w, x, outer, view, block, first, count):
    """Stock decode, one token at a time, on a copy of the checkpoint block."""
    ref = view.clone()
    res = torch.zeros(count, C, dtype=torch.bfloat16, device=DEV)
    for i in range(count):
        t = first + i
        ple_conv(
            inputs=x[t : t + 1],
            residual=res[i : i + 1],
            conv_state=ref,
            conv_weights=w,
            state_indices=_i32([block]),
            outer_residual=outer[t : t + 1],
            mode="decode",
            dilation=DIL,
        )
    return ref[block, :, :STATE_LEN].clone(), res


def _spec(w, x, outer, view, blocks):
    res = torch.zeros(len(blocks) * SQ, C, dtype=torch.bfloat16, device=DEV)
    ple_conv(
        inputs=x,
        residual=res,
        conv_state=view,
        conv_weights=w,
        state_indices=_i32(blocks),
        outer_residual=outer,
        mode="spec",
        dilation=DIL,
        query_start_loc=_i32(list(range(0, len(blocks) * SQ + 1, SQ))),
        num_accepted_tokens=_i32([1] * len(blocks)),
        spec_query_len=SQ,
    )
    return res


@pytest.mark.parametrize("accepted", [1, 2, SQ])
@pytest.mark.parametrize("seed", [0, 1])
def test_ple_commit_matches_sequential(accepted, seed):
    block = 3
    w, x, outer, raw, view = _setup(seed, [block])
    want_state, want_out = _sequential(w, x, outer, view, block, 0, accepted)
    history = torch.cat([view[block, :, :STATE_LEN], x[:accepted].T], dim=1)

    got_out = _spec(w, x, outer, view, [block])
    ctx = _PleConvCommit([raw], spec_query_len=SQ, max_num_reqs=NB)
    ctx.commit(_i32([accepted]), _i32([block]), _i32([0, SQ]))

    got_state = view[block, :, :STATE_LEN]
    assert torch.equal(got_state, want_state)
    assert torch.equal(got_state, history[:, -STATE_LEN:])
    assert torch.equal(got_out[:accepted], want_out)


@pytest.mark.parametrize(
    ("num_computed", "accepted", "crosses"),
    [(6, 3, True), (6, 2, True), (1, 2, False), (7, SQ, True)],
)
def test_ple_commit_align_boundary(num_computed, accepted, crosses):
    block_size, src, nxt = 8, 2, 5
    w, x, outer, raw, view = _setup(7, [src])
    boundary_len = block_size - num_computed
    want_final, _ = _sequential(w, x, outer, view, src, 0, accepted)
    want_boundary, _ = _sequential(w, x, outer, view, src, 0, min(boundary_len, SQ))

    _spec(w, x, outer, view, [src])
    ctx = _PleConvCommit([raw], spec_query_len=SQ, max_num_reqs=NB)
    ctx.commit(
        _i32([accepted]),
        _i32([src]),
        _i32([0, SQ]),
        block_table=_i32([[src, nxt]]),
        num_computed_tokens=_i32([num_computed]),
        mamba_block_size=block_size,
    )

    final_block = nxt if num_computed + accepted >= block_size else src
    assert torch.equal(view[final_block, :, :STATE_LEN], want_final)
    if crosses:
        assert torch.equal(view[src, :, :STATE_LEN], want_boundary)


def test_ple_commit_non_contiguous_request_rows():
    blocks, accepted = [1, 6], [3, 1]
    w, x, outer, raw, view = _setup(11, blocks)
    want = [
        _sequential(w, x, outer, view, b, i * SQ, a)[0]
        for i, (b, a) in enumerate(zip(blocks, accepted))
    ]

    _spec(w, x, outer, view, blocks)
    # Spec rows 0 and 1 belong to requests 1 and 3 of a four-request batch.
    counts = _i32([9, accepted[0], 9, accepted[1]])
    ctx = _PleConvCommit([raw], spec_query_len=SQ, max_num_reqs=NB)
    ctx.commit(
        counts,
        _i32(blocks),
        _i32([0, SQ, 2 * SQ]),
        request_indices=_i32([1, 3]),
    )

    for b, ref in zip(blocks, want):
        assert torch.equal(view[b, :, :STATE_LEN], ref)
