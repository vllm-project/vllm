# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from vllm.model_executor.layers.mamba.ops.ssd_chunk_scan import (
    _chunk_scan_fwd,
    _chunk_scan_fwd_kernel,
)
from vllm.model_executor.layers.mamba.ops.ssd_combined import (
    mamba_chunk_scan_combined_varlen,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.mamba2_attn import compute_varlen_chunk_metadata

# All kernels exercised here are pure Triton, so they run on any backend
# that the vLLM platform layer treats as a CUDA-alike device or as XPU.
DEVICE = current_platform.device_type

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="Mamba2 SSD Triton kernels require a CUDA-alike or XPU device.",
)

# Added by the IBM Team, 2024

# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/modules/ssd_minimal.py


# this is the segsum implementation taken from above
def segsum(x):
    """Calculates segment sum."""
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd_minimal_discrete(X, A, B, C, block_len, initial_states=None):
    """Arguments:
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)

    Return:
        Y: (batch, length, n_heads, d_head)

    """
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0

    # Rearrange into blocks/chunks
    X, A, B, C = (
        rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)
    )

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at
    #    chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms
    # (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state


def generate_random_inputs(batch_size, seqlen, n_heads, d_head, itype, device=DEVICE):
    set_random_seed(0)
    A = -torch.exp(torch.rand(n_heads, dtype=itype, device=device))
    dt = F.softplus(
        torch.randn(batch_size, seqlen, n_heads, dtype=itype, device=device) - 4
    )
    X = torch.randn((batch_size, seqlen, n_heads, d_head), dtype=itype, device=device)
    B = torch.randn((batch_size, seqlen, n_heads, d_head), dtype=itype, device=device)
    C = torch.randn((batch_size, seqlen, n_heads, d_head), dtype=itype, device=device)

    return A, dt, X, B, C


def generate_continuous_batched_examples(
    example_lens_by_batch,
    num_examples,
    full_length,
    last_taken,
    exhausted,
    n_heads,
    d_head,
    itype,
    device=DEVICE,
    return_naive_ref=True,
):
    # this function generates a random examples of certain length
    # and then cut according to "example_lens_by_batch" and feed
    # them in continuous batches to the kernels.
    # If if return_naive_ref=True, the naive torch implementation
    # ssd_minimal_discrete will be used to compute and return
    # reference output.

    # generate the full-length example
    A, dt, X, B, C = generate_random_inputs(
        num_examples, full_length, n_heads, d_head, itype
    )

    if return_naive_ref:
        Y_min, final_state_min = ssd_minimal_discrete(
            X * dt.unsqueeze(-1), A * dt, B, C, block_len=full_length // 4
        )

    # internal function that outputs a cont batch of examples
    # given a tuple of lengths for each example in the batch
    # e.g., example_lens=(8, 4) means take 8 samples from first eg,
    #       4 examples from second eg, etc
    def get_continuous_batch(example_lens: tuple[int, ...]):
        indices = []
        for i, x in enumerate(example_lens):
            c = last_taken.get(i, 0)
            indices.append((c, c + x))
            last_taken[i] = (c + x) % full_length
            exhausted[i] = last_taken[i] == 0

        return (
            torch.concat([x[i, s:e] for i, (s, e) in enumerate(indices)]).unsqueeze(0)
            for x in (dt, X, B, C)
        )

    # internal function that maps "n" to the appropriate right boundary
    # value when forming continuous batches from examples of length given
    # by "full_length".
    # - e.g., when n > full_length, returns n % full_length
    #         when n == full_length, returns full_length
    def end_boundary(n: int):
        return n - ((n - 1) // full_length) * full_length

    IND_E = None
    for spec in example_lens_by_batch:
        # get the (maybe partial) example seen in this cont batch
        dt2, X2, B2, C2 = get_continuous_batch(spec)

        # get the metadata
        cu_seqlens = torch.tensor((0,) + spec, device=device).cumsum(dim=0)
        seq_idx = torch.zeros(
            cu_seqlens[-1], dtype=torch.int32, device=cu_seqlens.device
        )
        for i, (srt, end) in enumerate(
            zip(
                cu_seqlens,
                cu_seqlens[1:],
            )
        ):
            seq_idx[srt:end] = i

        # for cont batch
        if IND_E is None:
            IND_S = [0 for _ in range(len(spec))]
        else:
            IND_S = [x % full_length for x in IND_E]
        IND_E = [end_boundary(x + y) for x, y in zip(IND_S, spec)]

        # varlen has implicit batch=1
        dt2 = dt2.squeeze(0)
        X2 = X2.squeeze(0)
        B2 = B2.squeeze(0)
        C2 = C2.squeeze(0)
        yield (
            [Y_min[s, IND_S[s] : IND_E[s]] for s in range(num_examples)]
            if return_naive_ref
            else None,
            cu_seqlens,
            seq_idx,
            (A, dt2, X2, B2, C2),
        )


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("n_heads", [4, 16, 32])
@pytest.mark.parametrize("d_head", [5, 8, 32, 128])
@pytest.mark.parametrize("seq_len_chunk_size", [(112, 16), (128, 32)])
def test_mamba_chunk_scan_single_example(d_head, n_heads, seq_len_chunk_size, itype):
    # this tests the kernels on a single example (bs=1)

    # TODO: the bfloat16 case requires higher thresholds. To be investigated

    if itype == torch.bfloat16:
        atol, rtol = 5e-2, 5e-2
    else:
        atol, rtol = 8e-3, 5e-3

    # set seed
    batch_size = 1  # batch_size
    # ssd_minimal_discrete requires chunk_size divide seqlen
    # - this is only required for generating the reference seqs,
    #   it is not an operational limitation.
    seqlen, chunk_size = seq_len_chunk_size

    A, dt, X, B, C = generate_random_inputs(batch_size, seqlen, n_heads, d_head, itype)

    Y_min, final_state_min = ssd_minimal_discrete(
        X * dt.unsqueeze(-1), A * dt, B, C, chunk_size
    )

    cu_seqlens = torch.tensor((0, seqlen), device=DEVICE).cumsum(dim=0)
    cu_chunk_seqlens, last_chunk_indices, seq_idx_chunks = (
        compute_varlen_chunk_metadata(cu_seqlens, chunk_size)
    )
    # varlen has implicit batch=1
    X = X.squeeze(0)
    dt = dt.squeeze(0)
    A = A.squeeze(0)
    B = B.squeeze(0)
    C = C.squeeze(0)
    Y = torch.empty_like(X)
    final_state = mamba_chunk_scan_combined_varlen(
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
    )

    # just test the last in sequence
    torch.testing.assert_close(Y[-1], Y_min[0, -1], atol=atol, rtol=rtol)

    # just test the last head
    # NOTE, in the kernel we always cast states to fp32
    torch.testing.assert_close(
        final_state[:, -1].to(torch.float32),
        final_state_min[:, -1].to(torch.float32),
        atol=atol,
        rtol=rtol,
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


def test_chunk_scan_fwd_every_config_matches(monkeypatch):
    """The autotuner picks a config by speed only, so every config in the list
    must give the same finite output. On sm90, configs that pipeline a wgmma
    with `cb` in registers wrote NaN into whole rows (#61603)."""
    from triton.runtime.errors import OutOfResources

    set_random_seed(0)
    nheads, headdim, ngroups, dstate, chunk_size = 64, 64, 8, 128, 128
    cu_chunk_seqlens = torch.tensor([0, 128, 181], dtype=torch.int32, device=DEVICE)
    seqlen, nchunks = 181, 2
    x = torch.randn(seqlen, nheads, headdim, dtype=torch.bfloat16, device=DEVICE)
    C = torch.randn(seqlen, ngroups, dstate, dtype=torch.bfloat16, device=DEVICE)
    cb = torch.randn(nchunks, ngroups, chunk_size, chunk_size, device=DEVICE)
    dt = torch.rand(nheads, nchunks, chunk_size, device=DEVICE) * 0.1
    dA_cumsum = -torch.rand(nheads, nchunks, chunk_size, device=DEVICE).cumsum(-1)
    states = torch.randn(nchunks, nheads, headdim, dstate, device=DEVICE) * 0.1
    seq_idx = torch.zeros(nchunks, dtype=torch.int32, device=DEVICE)
    D = torch.randn(nheads, headdim, device=DEVICE) * 0.1

    ref = None
    for config in list(_chunk_scan_fwd_kernel.configs):
        monkeypatch.setattr(_chunk_scan_fwd_kernel, "configs", [config])
        for _ in range(10):
            out = torch.empty_like(x)
            try:
                _chunk_scan_fwd(
                    cb, x, dt, dA_cumsum, C, states, cu_chunk_seqlens, out, seq_idx, D
                )
            except OutOfResources:
                # The autotuner skips configs that do not fit this device.
                break
            assert torch.isfinite(out).all(), f"non-finite output with {config}"
            if ref is None:
                ref = out
            torch.testing.assert_close(
                out,
                ref,
                atol=2e-2,
                rtol=2e-2,
                msg=lambda m, config=config: f"{config}: {m}",
            )


@pytest.mark.parametrize(
    "shape",
    [
        (64, 64, 128, 1, 128),
        (64, 64, 128, 8, 128),
        (16, 32, 64, 4, 64),
        (32, 64, 256, 4, 128),
        (16, 128, 128, 2, 256),
    ],
    ids=["granite", "grouped", "small-chunks", "large-state", "large-chunks"],
)
@pytest.mark.parametrize("state_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("fp32_inputs", [False, True])
@pytest.mark.parametrize(
    "lengths,initial_slots,final_slots,valid,checkpoints,write_final",
    [
        ([32], [1], [1], [True], False, True),
        ([257], [2], [3], [True], False, True),
        ([33], [-1], [1], [False], False, True),
        ([32, 257], [1, 1], [1, 3], [True, True], False, True),
        ([257, 33], [1, 2], [2, 1], [True, True], False, True),
        ([1, 257, 33], [1, -1, 1], [1, 2, 3], [True, False, True], False, True),
        ([257], [1], [1], [True], True, False),
        ([257, 33], [1, 1], [2, 3], [True, True], True, False),
        ([257], [1], [1], [True], False, False),
    ],
    ids=[
        "in-place-first-call",
        "multi-chunk",
        "masked-invalid-slot",
        "shared-source-overwritten",
        "crossed-slots",
        "mixed-validity",
        "single-checkpoint-fallback",
        "shared-checkpoint-fallback",
        "returned-final-state",
    ],
)
def test_indexed_initial_state_handoff(
    monkeypatch,
    shape,
    state_dtype,
    fp32_inputs,
    lengths,
    initial_slots,
    final_slots,
    valid,
    checkpoints,
    write_final,
):
    _check_indexed_initial_state_handoff(
        monkeypatch,
        shape,
        state_dtype,
        fp32_inputs,
        lengths,
        initial_slots,
        final_slots,
        valid,
        checkpoints,
        write_final,
    )


def _check_indexed_initial_state_handoff(
    monkeypatch,
    shape,
    state_dtype,
    fp32_inputs,
    lengths,
    initial_slots,
    final_slots,
    valid,
    checkpoints,
    write_final,
    *,
    guard_autotune=False,
    force_snapshot=False,
    guard_gather_warmup=False,
    initial_columns=2,
):
    """Cache inputs must behave like a snapshot, including the first aliasing call.

    Checkpoint callers must still see inclusive states, and shared/crossed input
    slots must survive other requests' final writes. Padding is a write sentinel.
    """
    from vllm.model_executor.layers.mamba.ops import (
        ssd_combined,
        ssd_state_passing,
        triton_helpers,
    )
    from vllm.triton_utils import triton

    set_random_seed(19)
    monkeypatch.setattr(triton_helpers, "_autotuned_configs", {})
    for kernel, config in [
        (
            _chunk_scan_fwd_kernel,
            triton.Config(
                {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
                num_stages=1,
            ),
        ),
        (
            ssd_state_passing._state_passing_fwd_kernel,
            triton.Config({"BLOCK_SIZE": 512}),
        ),
    ]:
        monkeypatch.setattr(kernel, "configs", [config])
        monkeypatch.setattr(kernel, "cache", {})
    heads, dim, dstate, groups, chunk = shape
    length = sum(lengths)
    storage = torch.randn(
        5, heads * dim * dstate + 128, device=DEVICE, dtype=state_dtype
    )
    reference_storage = storage.clone()
    cache = storage[:, : heads * dim * dstate].view(5, heads, dim, dstate)
    reference_cache = reference_storage[:, : heads * dim * dstate].view_as(cache)
    indices = torch.tensor(initial_slots, device=DEVICE, dtype=torch.int32)
    indices = indices.repeat_interleave(2)[::2]
    mask = torch.tensor(valid, device=DEVICE).repeat_interleave(2)[::2]
    if guard_gather_warmup:
        # Match metadata's decode-row slice followed by prefill's [:, 0].
        table = torch.zeros(
            len(lengths) + 1, initial_columns, device=DEVICE, dtype=torch.int32
        )
        table[1:, 0] = indices
        indices = table[1:, 0]
        mask = torch.tensor(valid, device=DEVICE)
    destinations = torch.tensor(final_slots, device=DEVICE, dtype=torch.int32)
    destinations = destinations.repeat_interleave(2)[::2]
    input_dtype = (
        torch.float32
        if fp32_inputs
        else (torch.float16 if state_dtype == torch.float16 else torch.bfloat16)
    )
    x = torch.randn(length, heads, dim, device=DEVICE, dtype=input_dtype) * 0.1
    dt = torch.randn(length, heads, device=DEVICE) - 3
    A = -torch.rand(heads, device=DEVICE)
    B = torch.randn(length, groups, dstate, device=DEVICE, dtype=x.dtype) * 0.1
    C = torch.randn_like(B) * 0.1
    cu = torch.tensor(
        [0] + list(torch.tensor(lengths).cumsum(0).tolist()),
        device=DEVICE,
        dtype=torch.int32,
    )
    cu_chunk, last, seq = compute_varlen_chunk_metadata(cu, chunk)
    common = dict(
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        chunk_size=chunk,
        cu_seqlens=cu,
        cu_chunk_seqlens=cu_chunk,
        last_chunk_indices=last,
        seq_idx=seq,
        dt_softplus=True,
        state_dtype=state_dtype,
        return_intermediate_states=checkpoints,
        final_state_indices=destinations if write_final else None,
    )
    output, reference_output = torch.empty_like(x), torch.empty_like(x)
    if fp32_inputs and state_dtype == torch.float32:
        with pytest.raises(AssertionError, match="must have the same dtype"):
            mamba_chunk_scan_combined_varlen(
                **dict(common, state_dtype=torch.bfloat16),
                out=output,
                initial_state_cache=cache,
                initial_state_indices=indices,
                has_initial_state=mask,
                final_states_out=cache if write_final else None,
            )
        torch.testing.assert_close(storage, reference_storage, rtol=0, atol=0)
    snapshot_calls = []
    with monkeypatch.context() as first_call:
        if guard_gather_warmup:
            from collections import defaultdict

            from vllm.model_executor.layers.mamba.ops.gather_initial_states import (
                _gather_initial_states_kernel,
            )
            from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry

            kernel = _gather_initial_states_kernel.kernel
            # Earlier tests or the disk cache must not hide a missing key.
            first_call.setattr(
                kernel,
                "device_caches",
                defaultdict(kernel.device_caches.default_factory),
            )
            registry = JitWarmupRegistry(None)
            with registry.activate():
                for _ in range(2):
                    _gather_initial_states_kernel.register_warmup(
                        row_size=heads * dim * dstate, state_dtype=state_dtype
                    )
            assert len(registry) == 1
            raw_run = kernel.run

            def require_compile_only(*args, **kwargs):
                assert kwargs.get("warmup"), "Warmup must not launch the gather"
                return raw_run(*args, **kwargs)

            def forbid_allocation(*args, **kwargs):
                pytest.fail("Gather warmup must not allocate real tensors")

            with monkeypatch.context() as startup:
                startup.setattr(kernel, "run", require_compile_only)
                startup.setattr(torch, "empty", forbid_allocation)
                startup.setattr(torch, "empty_strided", forbid_allocation)
                registry.warmup()
            previous_hook = triton.knobs.runtime.jit_post_compile_hook

            def forbid_gather_compile(**kwargs):
                if getattr(kwargs.get("fn"), "name", "") == kernel.__name__:
                    pytest.fail("Strided snapshot compiled after startup warmup")
                if previous_hook is not None:
                    return previous_hook(**kwargs)

            first_call.setattr(
                triton.knobs.runtime, "jit_post_compile_hook", forbid_gather_compile
            )
        if force_snapshot:
            first_call.setattr(current_platform, "is_cuda", lambda: False)
            raw_passing = ssd_combined._state_passing_fwd

            def require_snapshot(*args, **kwargs):
                assert kwargs.get("initial_state_indices") is None
                snapshot_calls.append(True)
                return raw_passing(*args, **kwargs)

            first_call.setattr(ssd_combined, "_state_passing_fwd", require_snapshot)
        if guard_autotune:
            kernel = ssd_state_passing._state_passing_fwd_kernel
            first_call.setattr(
                kernel,
                "configs",
                [triton.Config({"BLOCK_SIZE": size}) for size in (256, 512, 1024)],
            )

            def forbid_autotune(*args, **kwargs):
                pytest.fail("Indexed aliasing must bypass the autotuner")

            first_call.setattr(kernel, "run", forbid_autotune)
            first_call.setattr(ssd_state_passing, "launch_autotuned", forbid_autotune)
        actual_states = mamba_chunk_scan_combined_varlen(
            **common,
            out=output,
            initial_state_cache=cache,
            initial_state_indices=indices,
            has_initial_state=mask,
            final_states_out=cache if write_final else None,
        )
    if force_snapshot:
        assert snapshot_calls == [True]
    dense = torch.where(mask[:, None, None, None], reference_cache[indices], 0)
    expected_states = mamba_chunk_scan_combined_varlen(
        **common,
        out=reference_output,
        initial_states=dense,
        final_states_out=reference_cache if write_final else None,
    )
    torch.testing.assert_close(output, reference_output, rtol=0, atol=0)
    torch.testing.assert_close(storage, reference_storage, rtol=0, atol=0)
    if write_final:
        assert actual_states is expected_states is None
    else:
        torch.testing.assert_close(actual_states, expected_states, rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Indexed CUDA path")
def test_indexed_aliasing_bypasses_autotuner(monkeypatch):
    """A first in-place multi-chunk call must not run destructive autotuning."""
    _check_indexed_initial_state_handoff(
        monkeypatch,
        (32, 64, 256, 4, 128),
        torch.float16,
        False,
        [257],
        [1],
        [1],
        [True],
        False,
        True,
        guard_autotune=True,
    )


def test_indexed_cache_uses_snapshot_off_cuda(monkeypatch):
    """Unvalidated platforms retain snapshot semantics even for one request."""
    _check_indexed_initial_state_handoff(
        monkeypatch,
        (16, 32, 64, 4, 64),
        torch.bfloat16,
        False,
        [257],
        [1],
        [1],
        [True],
        False,
        True,
        force_snapshot=True,
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Mamba2 CUDA warmup")
@pytest.mark.parametrize("state_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("columns", [1, 2, 3, 16, 17, 32])
def test_strided_snapshot_does_not_compile_after_warmup(
    monkeypatch, state_dtype, columns
):
    """A real B>1 fallback must reuse compile-only startup's gather keys."""
    _check_indexed_initial_state_handoff(
        monkeypatch,
        (16, 32, 64, 4, 64),
        state_dtype,
        False,
        [32, 33],
        [1, 1],
        [1, 2],
        [True, True],
        False,
        True,
        guard_gather_warmup=True,
        initial_columns=columns,
    )
