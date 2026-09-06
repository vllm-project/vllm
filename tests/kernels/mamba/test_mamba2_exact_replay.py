# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba2 exact-replay mode reproduces single-shot prefill bit for bit.

The schedule below mixes, in the same batches, sequences that are at different
chunk offsets: a first prefill call that splits prompts at unaligned points, a
second call that resumes them, then token-by-token decode until each sequence
reaches its length (so the batch composition changes as sequences finish and
chunk boundaries are crossed at different steps). Every produced output row
must equal the row of a single-shot prefill of the whole sequence.
"""

import pytest
import torch

from vllm.model_executor.layers.mamba.exact_replay import (
    ExactReplayBuffers,
    exact_replay_ssd,
)
from vllm.model_executor.layers.mamba.ops.ssd_combined import (
    mamba_chunk_scan_combined_varlen,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mamba2_attn import build_exact_replay_metadata

DEVICE = current_platform.device_type

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Mamba2 SSD Triton kernels require a CUDA-alike device.",
)


def _single_shot(x, dt, A, B, C, D, dt_bias, chunk_size):
    """Reference: the whole sequence in one chunked-scan call from position 0."""
    seqlen = x.shape[0]
    cu_chunk = list(range(0, seqlen, chunk_size)) + [seqlen]
    n_chunks = len(cu_chunk) - 1
    i32 = lambda v: torch.tensor(v, dtype=torch.int32, device=x.device)  # noqa: E731
    out = torch.empty_like(x)
    mamba_chunk_scan_combined_varlen(
        x,
        dt,
        A,
        B,
        C,
        chunk_size=chunk_size,
        cu_seqlens=i32([0, seqlen]),
        cu_chunk_seqlens=i32(cu_chunk),
        last_chunk_indices=i32([n_chunks - 1]),
        seq_idx=i32([0] * n_chunks),
        out=out,
        D=D,
        z=None,
        dt_bias=dt_bias,
        initial_states=None,
        dt_softplus=True,
        dt_limit=(0.0, float("inf")),
        state_dtype=torch.float32,
    )
    return out


def _carve_states(num_slots, shapes, dtypes, device, pad_bytes):
    """Carve state tensors out of one paged buffer the way MambaBase does.

    Every slot is one page; the per-state views are therefore strided in the
    slot dimension, which is the layout the production code sees.
    """
    sizes = [
        int(torch.empty(shape, dtype=dtype).numel() * dtype.itemsize)
        for shape, dtype in zip(shapes, dtypes)
    ]
    page_bytes = sum(sizes) + pad_bytes
    pages = torch.zeros(num_slots, page_bytes, dtype=torch.uint8, device=device)
    states, offset = [], 0
    for shape, dtype, nbytes in zip(shapes, dtypes, sizes):
        state = pages[:, offset : offset + nbytes].view(dtype)
        states.append(state.view(-1, *shape))
        offset += nbytes
    return states


@pytest.mark.parametrize("chunk_size", [64])
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("layout", ["contiguous", "paged"])
def test_exact_replay_matches_single_shot_prefill(chunk_size, seed, layout):
    torch.manual_seed(seed)
    device = torch.device(DEVICE)
    nheads, head_dim, ngroups, dstate = 8, 64, 1, 64
    dtype = torch.bfloat16

    # Realistic Mamba2 initialisation: slow heads keep a large part of their
    # state across a chunk, which is what makes boundary handling observable.
    A = -(torch.rand(nheads, device=device) * 15 + 1)
    dt_target = torch.exp(
        torch.rand(nheads, device=device)
        * (torch.log(torch.tensor(0.1)) - torch.log(torch.tensor(1e-3)))
        + torch.log(torch.tensor(1e-3))
    )
    dt_bias = dt_target + torch.log(-torch.expm1(-dt_target))
    D = torch.ones(nheads, device=device)

    # total length, prompt length and the split point of the first prefill call
    total_lens = [150, 210, 64, 130]
    prompt_lens = [100, 133, 20, 64]
    first_split = [37, 100, 20, 30]
    n = len(total_lens)

    xs = [torch.randn(L, nheads, head_dim, device=device).to(dtype) for L in total_lens]
    dts = [(0.5 * torch.randn(L, nheads, device=device)).to(dtype) for L in total_lens]
    Bs = [torch.randn(L, ngroups, dstate, device=device).to(dtype) for L in total_lens]
    Cs = [torch.randn(L, ngroups, dstate, device=device).to(dtype) for L in total_lens]
    refs = [
        _single_shot(xs[i], dts[i], A, Bs[i], Cs[i], D, dt_bias, chunk_size)
        for i in range(n)
    ]

    # engine-side state: slot 0 is the null block, sequences use slots 1..n
    num_slots = n + 1
    shapes = [
        (nheads, head_dim, dstate),
        (chunk_size, nheads, head_dim),
        (chunk_size, nheads),
        (chunk_size, ngroups, dstate),
        (chunk_size, ngroups, dstate),
    ]
    dtypes = [torch.float32, dtype, dtype, dtype, dtype]
    if layout == "contiguous":
        states = [
            torch.zeros(num_slots, *shape, dtype=dt_, device=device)
            for shape, dt_ in zip(shapes, dtypes)
        ]
    else:
        states = _carve_states(num_slots, shapes, dtypes, device, pad_bytes=4096)
    ssm_state = states[0]
    buffers = ExactReplayBuffers(*states[1:])
    all_slots = torch.arange(1, n + 1, dtype=torch.int32, device=device)
    outs = [torch.empty_like(x) for x in xs]
    computed = [0] * n

    def run_step(active, lens):
        x = torch.cat(
            [
                xs[i][computed[i] : computed[i] + length]
                for i, length in zip(active, lens)
            ]
        )
        dt = torch.cat(
            [
                dts[i][computed[i] : computed[i] + length]
                for i, length in zip(active, lens)
            ]
        )
        B = torch.cat(
            [
                Bs[i][computed[i] : computed[i] + length]
                for i, length in zip(active, lens)
            ]
        )
        C = torch.cat(
            [
                Cs[i][computed[i] : computed[i] + length]
                for i, length in zip(active, lens)
            ]
        )
        slots = all_slots[torch.tensor(active, device=device)]
        meta = build_exact_replay_metadata(
            [computed[i] for i in active], lens, slots, chunk_size, device
        )
        out = torch.empty_like(x)
        exact_replay_ssd(
            x,
            dt,
            B,
            C,
            A=A,
            D=D,
            dt_bias=dt_bias,
            out=out,
            ssm_state=ssm_state,
            slots=slots,
            meta=meta,
            chunk_size=chunk_size,
            buffers=buffers,
        )
        off = 0
        for i, length in zip(active, lens):
            outs[i][computed[i] : computed[i] + length] = out[off : off + length]
            off += length
            computed[i] += length

    # 1) first prefill call: every prompt split at an unaligned point (or whole)
    run_step(list(range(n)), first_split)
    # 2) resume the remaining prompt tokens of the split prompts
    rest = [i for i in range(n) if computed[i] < prompt_lens[i]]
    run_step(rest, [prompt_lens[i] - computed[i] for i in rest])
    # 3) decode: one token per active sequence per step, batch shrinks over time
    while any(computed[i] < total_lens[i] for i in range(n)):
        active = [i for i in range(n) if computed[i] < total_lens[i]]
        run_step(active, [1] * len(active))

    for i in range(n):
        assert torch.equal(outs[i].view(torch.int16), refs[i].view(torch.int16)), (
            f"sequence {i}: exact replay diverged from single-shot prefill"
        )
