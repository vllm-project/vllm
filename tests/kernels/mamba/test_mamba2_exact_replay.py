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
            [computed[i] for i in active], lens, chunk_size, device
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


def test_shared_metadata_writes_each_groups_own_slots():
    """One metadata object must serve layers with different state slots.

    In a hybrid model the model runner builds the Mamba metadata once and hands
    the other KV cache groups a copy with only the block table swapped, so two
    layers see the same ``ExactReplayMetadata`` but different ``slots``. Each
    layer must read and store its partial-chunk inputs at its own slots.
    """
    torch.manual_seed(0)
    device = torch.device(DEVICE)
    nheads, head_dim, ngroups, dstate, chunk_size = 4, 32, 1, 32, 64
    dtype = torch.bfloat16
    A = -(torch.rand(nheads, device=device) * 15 + 1)
    dt_bias = torch.zeros(nheads, device=device)
    D = torch.ones(nheads, device=device)
    total, prompt = 100, 70  # prompt ends 6 tokens into the second chunk

    def make_layer():
        x = torch.randn(total, nheads, head_dim, device=device).to(dtype)
        dt = (0.5 * torch.randn(total, nheads, device=device)).to(dtype)
        B = torch.randn(total, ngroups, dstate, device=device).to(dtype)
        C = torch.randn(total, ngroups, dstate, device=device).to(dtype)
        ref = _single_shot(x, dt, A, B, C, D, dt_bias, chunk_size)
        return x, dt, B, C, ref

    layers = [make_layer(), make_layer()]
    # both layers' states live in the same paged tensors (KV cache groups of a
    # hybrid model alias one allocation); layer 0 owns slot 1, layer 1 slot 2
    num_slots = 3
    shapes = [
        (nheads, head_dim, dstate),
        (chunk_size, nheads, head_dim),
        (chunk_size, nheads),
        (chunk_size, ngroups, dstate),
        (chunk_size, ngroups, dstate),
    ]
    dtypes = [torch.float32, dtype, dtype, dtype, dtype]
    states = _carve_states(num_slots, shapes, dtypes, device, pad_bytes=512)
    ssm_state, buffers = states[0], ExactReplayBuffers(*states[1:])
    slots = [
        torch.tensor([1], dtype=torch.int32, device=device),
        torch.tensor([2], dtype=torch.int32, device=device),
    ]
    outs = [torch.empty_like(layer[0]) for layer in layers]

    def run(step_start, step_len):
        # one metadata object for both layers, as the model runner does
        meta = build_exact_replay_metadata([step_start], [step_len], chunk_size, device)
        for li, (x, dt, B, C, _ref) in enumerate(layers):
            sl = slice(step_start, step_start + step_len)
            exact_replay_ssd(
                x[sl],
                dt[sl],
                B[sl],
                C[sl],
                A=A,
                D=D,
                dt_bias=dt_bias,
                out=outs[li][sl],
                ssm_state=ssm_state,
                slots=slots[li],
                meta=meta,
                chunk_size=chunk_size,
                buffers=buffers,
            )

    run(0, prompt)
    # after the prefill every layer's trailing partial chunk sits in its own slot
    for li, (x, _dt, _B, _C, _ref) in enumerate(layers):
        stored = buffers.x[li + 1, : prompt % chunk_size]
        assert torch.equal(
            stored.view(torch.int16), x[chunk_size:prompt].view(torch.int16)
        ), f"layer {li} stored its partial chunk in another layer's slot"
    for pos in range(prompt, total):
        run(pos, 1)
    for li, (_x, _dt, _B, _C, ref) in enumerate(layers):
        assert torch.equal(outs[li].view(torch.int16), ref.view(torch.int16)), (
            f"layer {li}: exact replay diverged from single-shot prefill"
        )
