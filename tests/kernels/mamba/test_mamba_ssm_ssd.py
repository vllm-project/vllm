# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from tests.kernels.mamba.utils import selective_state_update_ref
from vllm.config.mamba import MambaConfig
from vllm.model_executor.layers.mamba.checkpoint import (
    MambaPrefillCheckpointExporter,
    MambaPrefillCheckpointMetadata,
)
from vllm.model_executor.layers.mamba.ops.ssd_chunk_scan import (
    _chunk_scan_fwd,
    _chunk_scan_fwd_kernel,
)
from vllm.model_executor.layers.mamba.ops.ssd_combined import (
    mamba_chunk_scan_combined_varlen,
)
from vllm.model_executor.layers.mamba.ops.ssd_dispatch import (
    get_mamba_ssd_backend,
    initialize_mamba_ssd_backend,
)
from vllm.model_executor.warmup.mamba_ssd_warmup import mamba_ssd_warmup
from vllm.platforms import current_platform
from vllm.utils.flashinfer import is_flashinfer_ssd_supported
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.mamba2_attn import (
    Mamba2AttentionMetadata,
    build_ssd_prefill_metadata,
    compute_varlen_chunk_metadata,
)
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec

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
    ("provider", "checkpoints", "state_dtype"),
    [
        ("triton", True, torch.float32),
        ("flashinfer", False, torch.float16),
        ("flashinfer", True, torch.float32),
    ],
    ids=["triton_checkpoint", "flashinfer", "flashinfer_checkpoint"],
)
def test_mamba_ssd_dispatch_preserves_outputs_cache_and_checkpoint_resume(
    monkeypatch, provider, checkpoints, state_dtype
):
    """Exercise mixer dispatch with real SSD kernels and an FP64 recurrence."""
    import vllm.model_executor.layers.mamba.mamba_mixer2 as mod

    if provider == "flashinfer" and not is_flashinfer_ssd_supported():
        pytest.skip("FlashInfer SSD is unavailable on this platform")

    set_random_seed(0)
    seqlens = (224, 129, 3) if checkpoints else (127, 126, 3)
    heads, groups, dim, dstate, chunk_size = 8, 2, 64, 128, 256
    tokens, width = sum(seqlens), heads * dim + 2 * groups * dstate
    x = torch.randn(tokens, heads, dim, device=DEVICE, dtype=torch.bfloat16)
    dt = torch.randn(tokens, heads, device=DEVICE, dtype=torch.bfloat16) - 4
    B = torch.randn(tokens, groups, dstate, device=DEVICE, dtype=torch.bfloat16)
    C = torch.randn_like(B)
    A = -torch.exp(torch.rand(heads, device=DEVICE))
    D = torch.randn(heads, device=DEVICE, dtype=torch.bfloat16)
    bias = torch.randn_like(D)
    raw_qkv = torch.cat((x.flatten(1), B.flatten(1), C.flatten(1)), dim=-1)
    projected = torch.cat((raw_qkv, dt), dim=-1)
    payload = heads * dim * dstate
    storage = torch.full((7, payload + 13), 17, device=DEVICE, dtype=state_dtype)
    cache = storage[:, :payload].view(7, heads, dim, dstate)
    cache[2].normal_()
    before = storage.clone()
    indices = torch.tensor([5, 2, 4], dtype=torch.int32, device=DEVICE)
    has_initial = torch.tensor([False, True, False], device=DEVICE)
    initial = torch.where(has_initial[:, None, None, None], cache[indices], 0)
    conv_state = torch.full((7, width, 5), 17, device=DEVICE, dtype=torch.bfloat16)

    def metadata(lengths, slots, has_states, computed, checkpoint=None):
        starts = torch.tensor((0, *lengths), dtype=torch.int32).cumsum(0).int()
        return build_ssd_prefill_metadata(
            Mamba2AttentionMetadata(
                num_prefills=len(lengths),
                num_prefill_tokens=sum(lengths),
                num_decodes=0,
                num_decode_tokens=0,
                num_reqs=len(lengths),
                has_initial_states_p=has_states,
                query_start_loc_p=starts.to(DEVICE),
                state_indices_tensor_p=slots,
                state_indices_tensor_d=None,
                query_start_loc_d=None,
                num_accepted_tokens=None,
                seq_lens=torch.tensor(lengths, dtype=torch.int32, device=DEVICE),
                prep_initial_states=any(computed),
                chunk_size=chunk_size,
                checkpoint_meta=checkpoint,
            ),
            torch.tensor(computed),
            starts,
            (heads, dim, dstate),
            state_dtype,
        )

    def reference(x, dt, B, C, initial, lengths):
        # FP64 avoids TF32 in the independent output contraction.
        state = initial.double().clone()
        x, dt, B, C = (value.double() for value in (x, dt, B, C))
        dt = dt.unsqueeze(-1).expand(-1, -1, dim)
        A_ref = A.double()[:, None, None].expand(-1, dim, dstate)
        D_ref = D.double()[:, None].expand(-1, dim)
        bias_ref = bias.double()[:, None].expand(-1, dim)
        output, checkpoint_state, start = torch.empty_like(x), None, 0
        for row, length in enumerate(lengths):
            for token in range(start, start + length):
                output[token] = selective_state_update_ref(
                    state[row : row + 1],
                    x[token : token + 1],
                    dt[token : token + 1],
                    A_ref,
                    B[token : token + 1],
                    C[token : token + 1],
                    D=D_ref,
                    dt_bias=bias_ref,
                    dt_softplus=True,
                )[0]
                if row == 0 and token == 95:
                    checkpoint_state = state[row].clone()
            start += length
        return output, state, checkpoint_state

    checkpoint = (
        MambaPrefillCheckpointMetadata(
            torch.tensor([96, 0, 0], dtype=torch.int32, device=DEVICE),
            torch.tensor([1, 0, 0], dtype=torch.int32, device=DEVICE),
            offsets=[96, 0, 0],
        )
        if checkpoints
        else None
    )
    mixer = mod.MambaMixer2.__new__(mod.MambaMixer2)
    torch.nn.Module.__init__(mixer)
    mixer.prefix = "mixer"
    mixer.tped_intermediate_size, mixer.tped_conv_size, mixer.tped_dt_size = (
        0,
        width,
        heads,
    )
    mixer.num_heads, mixer.head_dim, mixer.n_groups, mixer.tp_size = (
        heads,
        dim,
        groups,
        1,
    )
    mixer.ssm_state_size, mixer.num_spec, mixer.use_replayssm = dstate, 0, False
    mixer.model_config = SimpleNamespace(
        dtype=x.dtype, get_mamba_chunk_size=lambda: chunk_size
    )
    mixer.cache_config = SimpleNamespace(
        mamba_block_size=256,
        mamba_cache_mode="align" if checkpoints else "none",
        mamba_cache_dtype="auto",
        mamba_ssm_cache_dtype=str(state_dtype).removeprefix("torch."),
    )
    mixer.conv_weights, mixer.conv1d = torch.empty(0), SimpleNamespace(bias=None)
    mixer.activation, mixer.conv_kernel_size = "silu", 4
    mixer._checkpoint_exporter = MambaPrefillCheckpointExporter(state_len=3)
    mixer.A, mixer.D, mixer.dt_bias = A, D, bias
    vllm_config = SimpleNamespace(
        mamba_config=MambaConfig(prefill_backend=provider),
        model_config=mixer.model_config,
        cache_config=mixer.cache_config,
        compilation_config=SimpleNamespace(static_forward_context={"mixer": mixer}),
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.ops.ssd_dispatch._mamba_ssd_backend", None
    )
    mixer._ssd_kernels_warmed_up = False
    kv_cache_config = KVCacheConfig(
        num_blocks=7,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["mixer"],
                kv_cache_spec=MambaSpec(
                    block_size=256,
                    shapes=((heads, dim, dstate),),
                    dtypes=(state_dtype,),
                ),
            )
        ],
    )
    initialize_mamba_ssd_backend(vllm_config.mamba_config, kv_cache_config)
    backend = get_mamba_ssd_backend()
    mamba_ssd_warmup(
        SimpleNamespace(vllm_config=vllm_config, kv_cache_config=kv_cache_config)
    )
    attn_metadata = metadata(seqlens, indices, has_initial, [0, 5, 0], checkpoint)
    mixer.kv_cache = (conv_state, cache)
    mixer.split_hidden_states_B_C_fn = lambda values: values.split(
        (heads * dim, groups * dstate, groups * dstate), dim=-1
    )
    assert backend.name == provider
    monkeypatch.setattr(mod, "is_conv_state_dim_first", lambda: True)
    monkeypatch.setattr(
        mod, "causal_conv1d_fn", lambda values, *args, **kwargs: values.clone()
    )

    def forward_context():
        return SimpleNamespace(
            attn_metadata={mixer.prefix: attn_metadata},
        )

    monkeypatch.setattr(mod, "get_forward_context", forward_context)
    expected_out, expected_states, expected_checkpoint = reference(
        x, dt, B, C, initial, seqlens
    )
    output = torch.empty(tokens, heads * dim, device=DEVICE, dtype=x.dtype)
    mixer.conv_ssm_forward(projected, output)
    torch.testing.assert_close(
        output.view_as(x).double(), expected_out, atol=0.1, rtol=0.06
    )
    torch.testing.assert_close(
        cache[indices].double(), expected_states, atol=0.05, rtol=0.05
    )
    untouched = [0, 3, 6] + ([] if checkpoints else [1])
    torch.testing.assert_close(storage[untouched], before[untouched], atol=0, rtol=0)
    torch.testing.assert_close(
        storage[:, payload:], before[:, payload:], atol=0, rtol=0
    )
    if checkpoints:
        torch.testing.assert_close(
            cache[1].double(), expected_checkpoint, atol=0.05, rtol=0.05
        )
        torch.testing.assert_close(
            conv_state[1, :, :3], raw_qkv[93:96].T, atol=0, rtol=0
        )
        assert (conv_state[1, :, 3:] == 17).all()
        expected_suffix, expected_final, _ = reference(
            x[96:224], dt[96:224], B[96:224], C[96:224], cache[1:2], (128,)
        )
        attn_metadata = metadata(
            (128,), indices.new_tensor([1]), has_initial.new_tensor([True]), [96]
        )
        suffix = torch.empty(128, heads * dim, device=DEVICE, dtype=x.dtype)
        mixer.conv_ssm_forward(projected[96:224], suffix)
        torch.testing.assert_close(
            suffix.view(128, heads, dim).double(), expected_suffix, atol=0.1, rtol=0.06
        )
        torch.testing.assert_close(
            cache[1:2].double(), expected_final, atol=0.05, rtol=0.05
        )

    # A batch shorter than one 128-token chunk with no cached state, written
    # into occupied cache slots.
    cold_lengths = (90, 5)
    cold_tokens = sum(cold_lengths)
    cold_slots = indices.new_tensor([3, 6])
    attn_metadata = metadata(cold_lengths, cold_slots, has_initial.new_zeros(2), [0, 0])
    expected_cold, expected_cold_states, _ = reference(
        x[:cold_tokens],
        dt[:cold_tokens],
        B[:cold_tokens],
        C[:cold_tokens],
        torch.zeros_like(cache[cold_slots]),
        cold_lengths,
    )
    before_cold = storage.clone()
    cold_output = torch.empty(cold_tokens, heads * dim, device=DEVICE, dtype=x.dtype)
    mixer.conv_ssm_forward(projected[:cold_tokens], cold_output)
    torch.testing.assert_close(
        cold_output.view(-1, heads, dim).double(), expected_cold, atol=0.1, rtol=0.06
    )
    torch.testing.assert_close(
        cache[cold_slots].double(), expected_cold_states, atol=0.05, rtol=0.05
    )
    torch.testing.assert_close(
        storage[[0, 1, 2, 4, 5]], before_cold[[0, 1, 2, 4, 5]], atol=0, rtol=0
    )
    torch.testing.assert_close(
        storage[:, payload:], before_cold[:, payload:], atol=0, rtol=0
    )
