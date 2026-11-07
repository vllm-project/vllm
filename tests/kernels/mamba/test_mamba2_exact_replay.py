# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba2 exact-replay mode reproduces single-shot prefill bit for bit.

Two layers with different state slots share one metadata object, as the KV
cache groups of a hybrid model do. The schedule mixes, in the same batches,
sequences that are at different chunk offsets: a first prefill call that splits
prompts at unaligned points, a second call that resumes them, then
token-by-token decode until each sequence reaches its length, so the batch
composition changes as sequences finish and chunk boundaries are crossed at
different steps. Every produced output row must equal the row of a single-shot
prefill of the whole sequence, and every layer must keep its partial chunk in
its own slots.
"""

import pytest
import torch

from vllm.model_executor.layers.mamba.exact_replay import (
    ExactReplayBuffers,
    build_exact_replay_metadata,
    exact_replay_ssd,
)
from vllm.model_executor.layers.mamba.ops.ssd_combined import (
    mamba_chunk_scan_combined_varlen,
)
from vllm.platforms import current_platform

DEVICE = current_platform.device_type

requires_cuda = pytest.mark.skipif(
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


def _carve_states(num_slots, shapes, dtypes, device, pad_bytes=4096):
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


@requires_cuda
def test_exact_replay_matches_single_shot_prefill():
    torch.manual_seed(0)
    device = torch.device(DEVICE)
    chunk_size = 64
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
    num_layers = 2

    def make_layer():
        xs = [
            torch.randn(L, nheads, head_dim, device=device).to(dtype)
            for L in total_lens
        ]
        dts = [
            (0.5 * torch.randn(L, nheads, device=device)).to(dtype) for L in total_lens
        ]
        Bs = [
            torch.randn(L, ngroups, dstate, device=device).to(dtype) for L in total_lens
        ]
        Cs = [
            torch.randn(L, ngroups, dstate, device=device).to(dtype) for L in total_lens
        ]
        refs = [
            _single_shot(xs[i], dts[i], A, Bs[i], Cs[i], D, dt_bias, chunk_size)
            for i in range(n)
        ]
        return xs, dts, Bs, Cs, refs, [torch.empty_like(x) for x in xs]

    layers = [make_layer() for _ in range(num_layers)]

    # Engine-side state carved out of pages: slot 0 is the null block, layer l
    # owns slots l*n+1 .. l*n+n.
    states = _carve_states(
        num_layers * n + 1,
        [
            (nheads, head_dim, dstate),
            (chunk_size, nheads, head_dim),
            (chunk_size, nheads),
            (chunk_size, ngroups, dstate),
        ],
        [torch.float32, dtype, dtype, dtype],
        device,
    )
    ssm_state, buffers = states[0], ExactReplayBuffers(*states[1:])
    slots = [
        torch.arange(li * n + 1, li * n + n + 1, dtype=torch.int32, device=device)
        for li in range(num_layers)
    ]
    computed = [0] * n

    def run_step(active, lens):
        # one metadata object for every layer, as the model runner does
        meta = build_exact_replay_metadata(
            [computed[i] for i in active], lens, chunk_size, device
        )
        rows = torch.tensor(active, device=device)
        for li, (xs, dts, Bs, Cs, _refs, outs) in enumerate(layers):
            gather = (
                lambda seqs: torch.cat(  # noqa: E731
                    [
                        seqs[i][computed[i] : computed[i] + k]
                        for i, k in zip(active, lens)
                    ]
                )
            )
            x, dt, B, C = gather(xs), gather(dts), gather(Bs), gather(Cs)
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
                slots=slots[li][rows],
                meta=meta,
                chunk_size=chunk_size,
                buffers=buffers,
            )
            off = 0
            for i, k in zip(active, lens):
                outs[i][computed[i] : computed[i] + k] = out[off : off + k]
                off += k
        for i, k in zip(active, lens):
            computed[i] += k

    # 1) first prefill call: every prompt split at an unaligned point (or whole)
    run_step(list(range(n)), first_split)
    # 2) resume the remaining prompt tokens of the split prompts
    rest = [i for i in range(n) if computed[i] < prompt_lens[i]]
    run_step(rest, [prompt_lens[i] - computed[i] for i in rest])
    # after the prompts, every layer's trailing partial chunk sits in its own slot
    for li, (xs, _dts, _Bs, _Cs, _refs, _outs) in enumerate(layers):
        for i, prompt in enumerate(prompt_lens):
            tail = prompt % chunk_size
            stored = buffers.x[int(slots[li][i]), :tail]
            assert torch.equal(stored, xs[i][prompt - tail : prompt]), (
                f"layer {li}, sequence {i}: partial chunk stored in another slot"
            )
    # 3) decode: one token per active sequence per step, batch shrinks over time
    while any(computed[i] < total_lens[i] for i in range(n)):
        active = [i for i in range(n) if computed[i] < total_lens[i]]
        run_step(active, [1] * len(active))

    for li, (_xs, _dts, _Bs, _Cs, refs, outs) in enumerate(layers):
        for i in range(n):
            assert torch.equal(outs[i].view(torch.int16), refs[i].view(torch.int16)), (
                f"layer {li}, sequence {i}: exact replay diverged from single-shot"
            )


META_CHUNK = 256


def _metadata(num_computed, query_lens):
    return build_exact_replay_metadata(
        num_computed, query_lens, META_CHUNK, torch.device("cpu")
    )


def test_metadata_for_decode_rows():
    # three decode rows: 300 computed (44 tokens into a chunk), exactly at a
    # boundary, and one token short of completing a chunk
    m = _metadata([300, 512, 767], [1, 1, 1])
    assert m.num_aug_tokens == 44 + 1 + 1 + 255 + 1
    assert m.cu_seqlens.tolist() == [0, 45, 46, 302]
    # every augmented sequence is a single chunk
    assert m.cu_chunk_seqlens.tolist() == [0, 45, 46, 302]
    assert m.last_chunk_indices.tolist() == [0, 1, 2]
    assert m.seq_idx.tolist() == [0, 1, 2]
    assert m.has_boundary_state.tolist() == [True, True, True]
    # only the third row fills its chunk this step
    assert m.boundary_rows.tolist() == [2]
    assert m.boundary_chunk_idx.tolist() == [2]
    # buffered tokens: 44 of row 0, none of row 1, 255 of row 2
    assert m.buffered_seq.tolist() == [0] * 44 + [2] * 255
    assert m.buffered_pos.tolist() == list(range(44)) + list(range(255))
    assert m.buffered_dst.tolist() == list(range(44)) + list(range(46, 301))
    assert m.step_dst.tolist() == [44, 45, 301]
    # rows 0 and 1 append their token to the buffer; row 2 completed a chunk
    assert m.store_src.tolist() == [44, 45]
    assert m.store_seq.tolist() == [0, 1]
    assert m.store_pos.tolist() == [44, 0]


def test_metadata_for_prefill_rows():
    # fresh 600-token prefill; resume at 300 with 213 tokens; resume at 100
    # with 413 tokens
    m = _metadata([0, 300, 100], [600, 213, 413])
    assert m.num_aug_tokens == 600 + (44 + 213) + (100 + 413)
    assert m.cu_seqlens.tolist() == [0, 600, 857, 1370]
    assert m.cu_chunk_seqlens.tolist() == [0, 256, 512, 600, 856, 857, 1113, 1369, 1370]
    assert m.last_chunk_indices.tolist() == [2, 4, 7]
    assert m.seq_idx.tolist() == [0, 0, 0, 1, 1, 2, 2, 2]
    # row 1 resumes at 300 -> boundary 256 holds a state; the others start
    # from zero (row 2's boundary is 0)
    assert m.has_boundary_state.tolist() == [False, True, False]
    # last completed chunks: row 0 -> [256,512) (chunk 1); row 1 -> the chunk
    # ending at 512 (chunk 3); row 2 -> the chunk ending at 512 (chunk 6)
    assert m.boundary_rows.tolist() == [0, 1, 2]
    assert m.boundary_chunk_idx.tolist() == [1, 3, 6]
    # buffered tokens re-fed for rows 1 and 2
    assert m.buffered_seq.tolist() == [1] * 44 + [2] * 100
    assert m.buffered_dst.tolist() == list(range(600, 644)) + list(range(857, 957))
    # trailing partial chunks stored back: 88 tokens of row 0, 1 of row 1,
    # 1 of row 2, all at buffer positions starting from 0
    assert m.store_src.tolist() == list(range(512, 600)) + [856] + [1369]
    assert m.store_seq.tolist() == [0] * 88 + [1] + [2]
    assert m.store_pos.tolist() == list(range(88)) + [0] + [0]
