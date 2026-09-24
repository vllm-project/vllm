# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.mamba._mamba_ssm_ssd import generate_continuous_batched_examples
from vllm.model_executor.layers.mamba.ops.ssd_combined import (
    mamba_chunk_scan_combined_varlen,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mamba2_attn import compute_varlen_chunk_metadata

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="Mamba2 SSD Triton kernels require a CUDA-alike or XPU device.",
)


@pytest.mark.parametrize("itype", [torch.float32])
@pytest.mark.parametrize("n_heads", [4, 8])
@pytest.mark.parametrize("d_head", [5, 16, 32])
@pytest.mark.parametrize(
    "seq_len_chunk_size_cases",
    [
        # small-ish chunk_size (8)
        (64, 8, 2, [(64, 32), (64, 32)]),
        (64, 8, 2, [(8, 8), (8, 8), (8, 8)]),  # chunk size boundary
        (
            64,
            8,
            2,
            [(4, 4), (4, 4), (4, 4), (4, 4)],
        ),  # chunk_size larger than cont batches
        (64, 8, 5, [(64, 32, 16, 8, 8)]),
        # large-ish chunk_size (256)
        (64, 256, 1, [(5,), (1,), (1,), (1,)]),  # irregular sizes with small sequences
        (
            64,
            256,
            2,
            [(5, 30), (1, 2), (1, 2), (1, 2)],
        ),  # irregular sizes with small sequences
        # we also need to test some large seqlen
        # to catch errors with init states decay
        (768, 128, 2, [(138, 225), (138, 225)]),
    ],
)
def test_mamba_chunk_scan_cont_batch(d_head, n_heads, seq_len_chunk_size_cases, itype):
    # this test with multiple examples in a continuous batch
    # (i.e. chunked prefill)

    seqlen, chunk_size, num_examples, cases = seq_len_chunk_size_cases

    # This test can have larger error for longer sequences
    if seqlen > 256:
        atol, rtol = 1e-2, 5e-3
    else:
        atol, rtol = 5e-3, 5e-3

    # hold state during the cutting process so we know if an
    # example has been exhausted and needs to cycle
    last_taken: dict = {}  # map: eg -> pointer to last taken sample
    exhausted: dict = {}  # map: eg -> boolean indicating example is exhausted

    states = None
    for Y_min, cu_seqlens, _token_seq_idx, (
        A,
        dt,
        X,
        B,
        C,
    ) in generate_continuous_batched_examples(
        cases, num_examples, seqlen, last_taken, exhausted, n_heads, d_head, itype
    ):
        cu_chunk_seqlens, last_chunk_indices, seq_idx_chunks = (
            compute_varlen_chunk_metadata(cu_seqlens, chunk_size)
        )

        Y = torch.empty_like(X)
        new_states = mamba_chunk_scan_combined_varlen(
            X,
            dt,
            A,
            B,
            C,
            chunk_size,
            cu_seqlens=cu_seqlens.to(torch.int32),
            cu_chunk_seqlens=cu_chunk_seqlens,
            last_chunk_indices=last_chunk_indices,
            seq_idx=seq_idx_chunks,
            out=Y,
            D=None,
            initial_states=states,
        )

        # just test the last in sequence
        for i in range(num_examples):
            # just test one dim and dstate
            Y_eg = Y[cu_seqlens[i] : cu_seqlens[i + 1], 0, 0]
            Y_min_eg = Y_min[i][:, 0, 0]
            torch.testing.assert_close(Y_eg, Y_min_eg, atol=atol, rtol=rtol)

        # update states
        states = new_states
        for i, clear in exhausted.items():
            if clear:
                states[i].fill_(0.0)
                exhausted[i] = False


@pytest.mark.parametrize("chunk_size", [8, 256])
@pytest.mark.parametrize(
    "seqlens",
    [(16, 20), (270, 88, 212, 203)],
)
def test_mamba_chunk_scan_cont_batch_prefill_chunking(chunk_size, seqlens):
    # This test verifies the correctness of the chunked prefill implementation
    # in the mamba2 ssd kernels, by comparing concatenation (in the sequence
    # dimension) of chunked results with the full sequence result.
    # It is different from test_mamba_chunk_scan_cont_batch by:
    # 1. Not using the naive torch implementation (ssd_minimal_discrete) to get
    #    reference outputs. Instead, it compares chunked kernel outputs to full
    #    sequence kernel outputs. This is the most straightforward way to
    #    assert chunked prefill correctness.
    # 2. It focuses on cases where sequences change in the middle of mamba
    #    chunks, and not necessarily on chunk boundaries.

    max_seqlen = max(seqlens)
    # This test can have larger error for longer sequences
    if max_seqlen > 256:
        atol, rtol = 1e-2, 5e-3
    else:
        atol, rtol = 5e-3, 5e-3

    num_sequences = len(seqlens)
    n_heads = 16
    d_head = 64
    itype = torch.float32

    # hold state during the cutting process so we know if an
    # example has been exhausted and needs to cycle
    last_taken: dict = {}  # map: eg -> pointer to last taken sample
    exhausted: dict = {}  # map: eg -> boolean indicating example is exhausted
    _, cu_seqlens, seq_idx, (A, dt, X, B, C) = next(
        generate_continuous_batched_examples(
            [seqlens],
            num_sequences,
            max_seqlen,
            last_taken,
            exhausted,
            n_heads,
            d_head,
            itype,
            return_naive_ref=False,
        )
    )
    seqlens = torch.tensor(seqlens, dtype=torch.int32, device=X.device)
    device = X.device

    ## full seqlen computation
    cu_chunk_seqlens, last_chunk_indices, seq_idx_chunks = (
        compute_varlen_chunk_metadata(cu_seqlens, chunk_size)
    )
    Y_ref = torch.empty_like(X)
    state_ref = mamba_chunk_scan_combined_varlen(
        X,
        dt,
        A,
        B,
        C,
        chunk_size,
        cu_seqlens=cu_seqlens.to(torch.int32),
        cu_chunk_seqlens=cu_chunk_seqlens,
        last_chunk_indices=last_chunk_indices,
        seq_idx=seq_idx_chunks,
        out=Y_ref,
        D=None,
        initial_states=None,
    )

    ## chunked seqlen computation
    # first chunk
    chunked_seqlens = seqlens // 2
    chunked_cu_seqlens = torch.cat(
        [torch.tensor([0], device=device), torch.cumsum(chunked_seqlens, dim=0)], dim=0
    )
    chunked_input_seq_len = chunked_cu_seqlens[-1]
    X_chunked = torch.zeros_like(X)[:chunked_input_seq_len, ...]
    dt_chunked = torch.zeros_like(dt)[:chunked_input_seq_len, ...]
    B_chunked = torch.zeros_like(B)[:chunked_input_seq_len, ...]
    C_chunked = torch.zeros_like(C)[:chunked_input_seq_len, ...]
    for i in range(num_sequences):
        chunk_f = lambda x, i: x[
            cu_seqlens[i] : cu_seqlens[i] + chunked_seqlens[i], ...
        ]

        X_chunked[chunked_cu_seqlens[i] : chunked_cu_seqlens[i + 1], ...] = chunk_f(
            X, i
        )
        dt_chunked[chunked_cu_seqlens[i] : chunked_cu_seqlens[i + 1], ...] = chunk_f(
            dt, i
        )
        B_chunked[chunked_cu_seqlens[i] : chunked_cu_seqlens[i + 1], ...] = chunk_f(
            B, i
        )
        C_chunked[chunked_cu_seqlens[i] : chunked_cu_seqlens[i + 1], ...] = chunk_f(
            C, i
        )

    cu_chunk_seqlens, last_chunk_indices, seq_idx_chunks = (
        compute_varlen_chunk_metadata(chunked_cu_seqlens, chunk_size)
    )
    Y_partial = torch.empty_like(X_chunked)
    partial_state = mamba_chunk_scan_combined_varlen(
        X_chunked,
        dt_chunked,
        A,
        B_chunked,
        C_chunked,
        chunk_size,
        cu_seqlens=chunked_cu_seqlens.to(torch.int32),
        cu_chunk_seqlens=cu_chunk_seqlens,
        last_chunk_indices=last_chunk_indices,
        seq_idx=seq_idx_chunks,
        out=Y_partial,
        D=None,
        initial_states=None,
    )

    # remaining chunk
    remaining_chunked_seqlens = seqlens - chunked_seqlens
    remaining_chunked_cu_seqlens = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(remaining_chunked_seqlens, dim=0),
        ],
        dim=0,
    )
    remaining_chunked_input_seq_len = remaining_chunked_cu_seqlens[-1]
    remaining_X_chunked = torch.zeros_like(X)[:remaining_chunked_input_seq_len, ...]
    remaining_dt_chunked = torch.zeros_like(dt)[:remaining_chunked_input_seq_len, ...]
    remaining_B_chunked = torch.zeros_like(B)[:remaining_chunked_input_seq_len, ...]
    remaining_C_chunked = torch.zeros_like(C)[:remaining_chunked_input_seq_len, ...]
    for i in range(num_sequences):
        remaining_chunk_f = lambda x, i: x[
            cu_seqlens[i] + chunked_seqlens[i] : cu_seqlens[i + 1], ...
        ]

        remaining_X_chunked[
            remaining_chunked_cu_seqlens[i] : remaining_chunked_cu_seqlens[i + 1], ...
        ] = remaining_chunk_f(X, i)
        remaining_dt_chunked[
            remaining_chunked_cu_seqlens[i] : remaining_chunked_cu_seqlens[i + 1], ...
        ] = remaining_chunk_f(dt, i)
        remaining_B_chunked[
            remaining_chunked_cu_seqlens[i] : remaining_chunked_cu_seqlens[i + 1], ...
        ] = remaining_chunk_f(B, i)
        remaining_C_chunked[
            remaining_chunked_cu_seqlens[i] : remaining_chunked_cu_seqlens[i + 1], ...
        ] = remaining_chunk_f(C, i)

    # assert input chunking is correct
    concat_chunk_f = lambda pt1, pt2, i: torch.cat(
        [
            pt1[chunked_cu_seqlens[i] : chunked_cu_seqlens[i + 1], ...],
            pt2[
                remaining_chunked_cu_seqlens[i] : remaining_chunked_cu_seqlens[i + 1],
                ...,
            ],
        ],
        dim=0,
    )
    concat_batch_f = lambda pt1, pt2: torch.cat(
        [concat_chunk_f(pt1, pt2, i) for i in range(num_sequences)], dim=0
    )

    assert concat_batch_f(X_chunked, remaining_X_chunked).equal(X)
    assert concat_batch_f(dt_chunked, remaining_dt_chunked).equal(dt)
    assert concat_batch_f(B_chunked, remaining_B_chunked).equal(B)
    assert concat_batch_f(C_chunked, remaining_C_chunked).equal(C)

    cu_chunk_seqlens, last_chunk_indices, seq_idx_chunks = (
        compute_varlen_chunk_metadata(remaining_chunked_cu_seqlens, chunk_size)
    )

    Y_chunked = torch.empty_like(remaining_X_chunked)
    state_chunked = mamba_chunk_scan_combined_varlen(
        remaining_X_chunked,
        remaining_dt_chunked,
        A,
        remaining_B_chunked,
        remaining_C_chunked,
        chunk_size,
        cu_seqlens=remaining_chunked_cu_seqlens.to(torch.int32),
        cu_chunk_seqlens=cu_chunk_seqlens,
        last_chunk_indices=last_chunk_indices,
        seq_idx=seq_idx_chunks,
        out=Y_chunked,
        D=None,
        initial_states=partial_state,
    )
    Y = concat_batch_f(Y_partial, Y_chunked)

    # kernel chunked is same as kernel overall
    for i in range(num_sequences):
        Y_seq = Y[cu_seqlens[i] : cu_seqlens[i + 1], ...]
        Y_ref_seq = Y_ref[cu_seqlens[i] : cu_seqlens[i + 1], ...]
        torch.testing.assert_close(
            Y_seq[: chunked_seqlens[i], ...],
            Y_ref_seq[: chunked_seqlens[i], ...],
            atol=atol,
            rtol=rtol,
            msg=lambda x, i=i: f"seq{i} output part1 " + x,
        )
        torch.testing.assert_close(
            Y_seq[chunked_seqlens[i] :, ...],
            Y_ref_seq[chunked_seqlens[i] :, ...],
            atol=atol,
            rtol=rtol,
            msg=lambda x, i=i: f"seq{i} output part2 " + x,
        )

        state_seq = state_chunked[i]
        state_seq_ref = state_ref[i]
        torch.testing.assert_close(
            state_seq,
            state_seq_ref,
            atol=atol,
            rtol=rtol,
            msg=lambda x, i=i: f"seq{i} state " + x,
        )
