# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-K sparse decode over bf16 and FP8 NoPE caches.

The split kernel must agree with an fp32 reference over the same dequantized
rows, and with the unsplit ragged kernel it replaces at decode, for every split
count, including splits left empty by short or fully-invalid requests.
"""

from __future__ import annotations

import pytest
import torch

from vllm.platforms import current_platform

LATENT = 512
SCALE = LATENT**-0.5
KV_SCALE = 0.05
SLOTS = 8192

gpu = pytest.mark.skipif(
    not (current_platform.is_rocm() and torch.cuda.is_available()),
    reason="needs a ROCm GPU",
)


def _case(kv_dtype, heads, lengths, seed=0, invalid_every=0):
    g = torch.Generator().manual_seed(seed)
    latent = torch.randn(SLOTS, LATENT, generator=g).to(torch.bfloat16)
    if kv_dtype == "fp8":
        cache = (latent.float() / KV_SCALE).to(current_platform.fp8_dtype()).cuda()
        rows = cache.cpu().to(torch.bfloat16).float() * KV_SCALE
        kv_scale = KV_SCALE
    else:
        cache = latent.cuda()
        rows = latent.float()
        kv_scale = 1.0
    q = torch.randn(len(lengths), heads, LATENT, generator=g).to(torch.bfloat16)
    idx = [
        torch.randint(0, SLOTS, (n,), generator=g, dtype=torch.int32) for n in lengths
    ]
    if invalid_every:
        for t in idx:
            t[::invalid_every] = -1
    indptr = torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist())
    return (
        q.cuda(),
        cache,
        rows,
        kv_scale,
        torch.cat(idx).cuda(),
        indptr.to(torch.int32).cuda(),
    )


def _reference(q, rows, indices, indptr, sink):
    q = q.float().cpu()
    out = torch.zeros_like(q)
    for i in range(q.shape[0]):
        idx = indices[indptr[i] : indptr[i + 1]].cpu().long()
        idx = idx[idx >= 0]
        scores = q[i] @ rows[idx].T * SCALE
        if sink is not None:
            scores = torch.cat([scores, sink.cpu()[:, None]], dim=1)
        if scores.shape[1] == 0:
            continue
        p = torch.softmax(scores, dim=1)[:, : idx.numel()]
        out[i] = p @ rows[idx]
    return out


CASES = {
    "uniform_topk": [2048] * 4,
    "ragged": [2048, 1500, 513, 37],
    "empty_request": [2048, 0, 700],
}


@gpu
@pytest.mark.parametrize("kv_dtype", ["fp8", "bf16"])
@pytest.mark.parametrize("heads", [16, 8])
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("num_splits", [1, 2, 7, 32])
@pytest.mark.parametrize("with_sink", [False, True])
def test_split_matches_reference_and_ragged(
    kv_dtype, heads, case, num_splits, with_sink
):
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _rocm_sparse_attn_prefill_ragged_triton,
        _rocm_sparse_attn_split_decode_triton,
    )

    q, cache, rows, kv_scale, indices, indptr = _case(
        kv_dtype, heads, CASES[case], invalid_every=11
    )
    sink = (
        torch.linspace(-2.0, 2.0, heads, device="cuda", dtype=torch.float32)
        if with_sink
        else None
    )
    split = _rocm_sparse_attn_split_decode_triton(
        q, cache, indices, indptr, SCALE, sink, LATENT, 0, num_splits, kv_scale
    )
    ragged = _rocm_sparse_attn_prefill_ragged_triton(
        q, cache, indices, indptr, SCALE, sink, LATENT, 0, kv_scale=kv_scale
    )
    ref = _reference(q, rows, indices, indptr, sink)

    torch.testing.assert_close(split.float().cpu(), ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(split.float(), ragged.float(), atol=2e-2, rtol=2e-2)


@gpu
def test_public_op_writes_a_narrower_output_in_place():
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        rocm_sparse_attn_split_decode,
    )

    q, cache, rows, kv_scale, indices, indptr = _case("fp8", 16, CASES["uniform_topk"])
    output = torch.empty(q.shape, dtype=torch.float32, device="cuda")
    rocm_sparse_attn_split_decode(
        q=q,
        kv=cache.view(SLOTS, 1, -1),
        scale=SCALE,
        head_dim=LATENT,
        nope_head_dim=LATENT,
        rope_head_dim=0,
        attn_sink=None,
        output=output,
        ragged_indices=indices,
        ragged_indptr=indptr,
        num_splits=16,
        kv_scale=kv_scale,
    )
    ref = _reference(q, rows, indices, indptr, None)
    torch.testing.assert_close(output.cpu(), ref, atol=2e-2, rtol=2e-2)


def test_num_splits_floor_and_tile_clamp(monkeypatch):
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    monkeypatch.setattr(mod, "_decode_num_splits", lambda *args: 32)
    monkeypatch.setattr(mod, "_decode_gfx950_num_splits", lambda *args: 32)
    floor = mod._SPARSE_DECODE_SPLIT_MIN_LEN
    block_k = mod._SPARSE_DECODE_SPLIT_BLOCK_K
    assert mod.rocm_sparse_decode_num_splits(1, 16, floor - 1) == 1
    assert mod.rocm_sparse_decode_num_splits(1, 16, floor) == floor // block_k
    assert mod.rocm_sparse_decode_num_splits(1, 16, 2048) == 32
