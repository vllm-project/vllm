# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from vllm.model_executor.layers.mamba.ops import (
    ssd_bmm,
    ssd_chunk_scan,
    ssd_chunk_state,
    ssd_state_passing,
)
from vllm.model_executor.layers.mamba.ops.ssd_combined import (
    mamba_chunk_scan_combined_varlen,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl
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


# TD path of the SSD scan/state kernels: run TD and the pointer path at pinned
# tiles and compare them; a launch spy asserts which path ran.

requires_td = pytest.mark.skipif(
    not hasattr(tl, "make_tensor_descriptor")
    or not (
        current_platform.is_xpu()
        or (current_platform.is_cuda() and current_platform.has_device_capability(90))
    ),
    reason="SSD TD path is tested on XPU and on CUDA sm90+ (TMA) only; it needs "
    "tl.make_tensor_descriptor",
)


def _pin_ssd_configs(
    monkeypatch, scan_cfg=(64, 64, 32, 2, 4), state_cfg=(64, 64, 32, 4, 2)
):
    """Pin each autotuned SSD kernel to one config: (M, N, K, num_stages, num_warps)."""

    def tile(m, n, k):
        return {"BLOCK_SIZE_M": m, "BLOCK_SIZE_N": n, "BLOCK_SIZE_K": k}

    pins = [
        (ssd_chunk_scan._chunk_scan_fwd_kernel, tile(*scan_cfg[:3]), scan_cfg[3:]),
        (ssd_chunk_state._chunk_state_fwd_kernel, tile(*state_cfg[:3]), state_cfg[3:]),
        (ssd_bmm._bmm_chunk_fwd_kernel, tile(64, 64, 32), (4, 2)),
        (ssd_chunk_state._chunk_cumsum_fwd_kernel, {"BLOCK_SIZE_H": 8}, None),
        (ssd_state_passing._state_passing_fwd_kernel, {"BLOCK_SIZE": 256}, None),
    ]
    for kernel, kwargs, stages_warps in pins:
        (config,) = [
            c
            for c in kernel.configs
            if c.kwargs == kwargs
            and stages_warps in (None, (c.num_stages, c.num_warps))
        ]
        monkeypatch.setattr(kernel, "configs", [config])


@pytest.fixture
def pin_ssd_configs(monkeypatch):
    _pin_ssd_configs(monkeypatch)


def _spy_use_td(monkeypatch) -> dict[str, list[bool]]:
    """Record the USE_TD constexpr of every launch of both kernels."""
    calls: dict[str, list[bool]] = {"scan": [], "state": []}
    for name, kernel in (
        ("scan", ssd_chunk_scan._chunk_scan_fwd_kernel),
        ("state", ssd_chunk_state._chunk_state_fwd_kernel),
    ):

        def spy(*args, _run=kernel.run, _calls=calls[name], **kwargs):
            _calls.append(kwargs["USE_TD"])
            return _run(*args, **kwargs)

        monkeypatch.setattr(kernel, "run", spy)
    return calls


def _misaligned(t: torch.Tensor) -> torch.Tensor:
    """Copy of ``t`` with data_ptr one element off 16 B alignment."""
    out = torch.empty(t.numel() + 1, dtype=t.dtype, device=t.device)[1:]
    return out.view(t.shape).copy_(t)


def _make_td_inputs(
    nheads, headdim, dstate, ngroups, seqlens, dtype, *, z, hdim_D, state_dtype
):
    """Inputs as in the mixer2 prefill call: x, B, C are views of one buffer."""
    set_random_seed(0)
    T = sum(seqlens)
    xBC = torch.randn(
        T, nheads * headdim + 2 * ngroups * dstate, device=DEVICE, dtype=dtype
    )
    x, B, C = torch.split(
        xBC, [nheads * headdim, ngroups * dstate, ngroups * dstate], dim=-1
    )
    D_shape = (nheads, headdim) if hdim_D else (nheads,)
    return dict(
        x=x.view(T, nheads, headdim),
        B=B.view(T, ngroups, dstate),
        C=C.view(T, ngroups, dstate),
        dt=torch.randn(T, nheads, device=DEVICE, dtype=dtype) * 0.5,
        A=-torch.exp(torch.rand(nheads, device=DEVICE, dtype=torch.float32)),
        D=torch.rand(D_shape, device=DEVICE, dtype=torch.float32),
        z=torch.randn(T, nheads, headdim, device=DEVICE, dtype=dtype) if z else None,
        dt_bias=torch.rand(nheads, device=DEVICE, dtype=torch.float32),
        state_dtype=state_dtype,
    )


def _run_ssd(inputs, seqlens, chunk_size, initial_states, misaligned_out=False):
    cu_seqlens = torch.tensor((0, *seqlens), device=DEVICE).cumsum(0).to(torch.int32)
    cu_chunk_seqlens, last_chunk_indices, seq_idx = compute_varlen_chunk_metadata(
        cu_seqlens, chunk_size
    )
    out = torch.empty_like(inputs["x"])
    if misaligned_out:
        out = _misaligned(out)
    final_states = mamba_chunk_scan_combined_varlen(
        **inputs,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        cu_chunk_seqlens=cu_chunk_seqlens,
        last_chunk_indices=last_chunk_indices,
        seq_idx=seq_idx,
        out=out,
        initial_states=initial_states,
        dt_softplus=True,
    )
    return out, final_states


def _run_both_paths(monkeypatch, *run_args, **run_kwargs):
    results, calls = {}, {}
    for use_td in (False, True):
        monkeypatch.setenv("VLLM_TRITON_USE_TD", "1" if use_td else "0")
        with monkeypatch.context() as m:
            calls[use_td] = _spy_use_td(m)
            results[use_td] = _run_ssd(*run_args, **run_kwargs)
    assert calls[False]["scan"] and not any(calls[False]["scan"])
    assert calls[False]["state"] and not any(calls[False]["state"])
    return results, calls


def _assert_same_result(results):
    # Same tiles on both paths: allow rounding-level differences only.
    for got, want in zip(results[True], results[False]):
        torch.testing.assert_close(got, want)


# Ragged: a 1-token sequence, partial and exact chunks.
TD_SEQLENS = (300, 1, 127, 256, 17)


@requires_td
@pytest.mark.parametrize(
    "nheads, headdim, dstate, ngroups, chunk_size",
    [
        # headdim not a multiple of BLOCK_SIZE_N
        (8, 80, 128, 8, 256),
        # 8 heads per B/C group
        (8, 64, 128, 1, 128),
        # dstate > 128: K-looped prev_states branch
        (4, 64, 256, 2, 128),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "variant",
    [
        # fresh prefill, per-head D, no gate
        "fresh",
        # chunked prefill / prefix hit, fp32 states (NemotronH)
        "cont_fp32_state",
        # states in the input dtype, gate z, per-headdim D
        "cont_state_in_dtype_z_hdim_D",
    ],
)
def test_ssd_td_matches_pointer(
    nheads,
    headdim,
    dstate,
    ngroups,
    chunk_size,
    dtype,
    variant,
    pin_ssd_configs,
    monkeypatch,
):
    """TD matches the pointer path at the same config."""
    full = variant == "cont_state_in_dtype_z_hdim_D"
    state_dtype = dtype if full else torch.float32
    inputs = _make_td_inputs(
        nheads,
        headdim,
        dstate,
        ngroups,
        TD_SEQLENS,
        dtype,
        z=full,
        hdim_D=full,
        state_dtype=state_dtype,
    )
    initial_states = None
    if variant != "fresh":
        initial_states = 0.1 * torch.randn(
            len(TD_SEQLENS), nheads, headdim, dstate, device=DEVICE
        ).to(state_dtype)

    results, calls = _run_both_paths(
        monkeypatch, inputs, TD_SEQLENS, chunk_size, initial_states
    )

    assert all(calls[True]["scan"]) and all(calls[True]["state"]), calls[True]
    _assert_same_result(results)


@requires_td
@pytest.mark.parametrize(
    "dstate, scan_cfg, state_cfg",
    [
        # (M, N, K, num_stages, num_warps); first case: tiles picked in E2E runs
        (128, (32, 64, 32, 1, 4), (32, 64, 32, 5, 2)),
        # M=128: taller than the ragged chunks and headdim
        (128, (128, 32, 32, 4, 4), (128, 32, 32, 4, 4)),
        # N=256: wider than headdim
        (256, (64, 256, 32, 4, 4), (64, 256, 32, 4, 4)),
        # K=64: K-looped prev_states step
        (256, (128, 64, 64, 4, 4), (64, 128, 64, 2, 4)),
    ],
)
def test_ssd_td_matches_pointer_across_tiles(dstate, scan_cfg, state_cfg, monkeypatch):
    """Same check at other tile shapes."""
    _pin_ssd_configs(monkeypatch, scan_cfg, state_cfg)
    nheads, headdim, chunk_size = 4, 80, 128
    inputs = _make_td_inputs(
        nheads,
        headdim,
        dstate,
        2,
        TD_SEQLENS,
        torch.bfloat16,
        z=True,
        hdim_D=True,
        state_dtype=torch.bfloat16,
    )
    initial_states = 0.1 * torch.randn(
        len(TD_SEQLENS),
        nheads,
        headdim,
        dstate,
        device=DEVICE,
        dtype=torch.bfloat16,
    )

    results, calls = _run_both_paths(
        monkeypatch, inputs, TD_SEQLENS, chunk_size, initial_states
    )

    assert all(calls[True]["scan"]) and all(calls[True]["state"]), calls[True]
    _assert_same_result(results)


@requires_td
@pytest.mark.parametrize(
    "case, scan_td, state_td",
    [
        # 16 B aligned but rows under the 64 B 2D block minimum
        ("headdim_16", False, False),
        ("chunk_size_8", False, True),
        # one operand off 16 B alignment
        ("x", False, False),
        ("B", True, False),
        ("C", False, True),
        ("z", False, True),
        ("initial_states", False, True),
        ("out", False, True),
    ],
)
def test_ssd_td_falls_back_per_operand(
    case, scan_td, state_td, pin_ssd_configs, monkeypatch
):
    """A kernel with an operand a descriptor cannot cover falls back to the
    pointer path; the result still matches."""
    seqlens = (300, 17, 500)
    headdim = 16 if case == "headdim_16" else 64
    chunk_size = 8 if case == "chunk_size_8" else 128
    inputs = _make_td_inputs(
        4,
        headdim,
        128,
        1,
        seqlens,
        torch.bfloat16,
        z=True,
        hdim_D=False,
        state_dtype=torch.bfloat16,
    )
    initial_states = 0.1 * torch.randn(
        len(seqlens), 4, headdim, 128, device=DEVICE, dtype=torch.bfloat16
    )
    if case == "initial_states":
        initial_states = _misaligned(initial_states)
    elif case in inputs:
        inputs[case] = _misaligned(inputs[case])

    results, calls = _run_both_paths(
        monkeypatch,
        inputs,
        seqlens,
        chunk_size,
        initial_states,
        misaligned_out=case == "out",
    )

    assert calls[True]["scan"] and all(v is scan_td for v in calls[True]["scan"])
    assert calls[True]["state"] and all(v is state_td for v in calls[True]["state"])
    _assert_same_result(results)


@requires_td
@pytest.mark.parametrize(
    "d_head, uses_td",
    [
        (128, True),
        # rows under the 64 B minimum
        (8, False),
    ],
)
@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
def test_ssd_td_single_example_vs_reference(
    d_head, uses_td, itype, pin_ssd_configs, monkeypatch
):
    """TD and the pointer fallback both match the torch reference."""
    monkeypatch.setenv("VLLM_TRITON_USE_TD", "1")
    calls = _spy_use_td(monkeypatch)

    test_mamba_chunk_scan_single_example(d_head, 16, (128, 32), itype)

    assert calls["scan"] and all(v is uses_td for v in calls["scan"])
    assert calls["state"] and all(v is uses_td for v in calls["state"])
