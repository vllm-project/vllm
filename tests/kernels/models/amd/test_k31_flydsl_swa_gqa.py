# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The FlyDSL sliding-window GQA attention of Kimi-K3.1's release DSpark draft
(kimi_k3/amd/ops/flydsl_swa_gqa, backend KIMI_K3_FLYDSL_SWA_GQA) against an
fp32 torch reference, at the draft's TP8 shapes: 12 query / 2 KV heads, head
dim 64, a 2048-token causal window.

The cache is a block-outermost (BLHNC) multi-layer cache, as the DSA indexer
pins the worker to, read through one layer's strided view, with block sizes
that are not powers of 2 and shuffled block tables. Cases: contexts around the
window (2041 to 2049) and long ones (9000, 130000), block-boundary offsets,
1 and 2 requests, 1 to 16 query rows, bf16 and fp8 e4m3 (per-tensor scales),
and a CUDA graph captured with seq_lens 1 and an empty (padded) request, then
replayed on real data. The backend's impl is checked against TRITON_ATTN's on
the same metadata, and for its Triton fallback past 16 rows.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="the FlyDSL kernel is for ROCm gfx950"
)

D, HQ, HKV, W = 64, 12, 2, 2048
SCALE = 1.0 / math.sqrt(D)
FP8 = torch.float8_e4m3fn
K_SCALE, V_SCALE = 0.37, 1.9


@pytest.fixture(scope="module", autouse=True)
def _gfx950():
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        pytest.skip("gfx950 only")
    from vllm.models.kimi_k3.amd.ops.flydsl_swa_gqa import flydsl_swa_gqa_reserve

    flydsl_swa_gqa_reserve("cuda", HKV, W)


def _cache(num_blocks, bs, dtype):
    # BLHNC: physical [B, L, H, N, C]; layer 1's logical view [B, H, N, C].
    phys = torch.randn(num_blocks, 3, HKV, bs, 2 * D, device="cuda")
    phys = (phys * 0.5).to(FP8) if dtype == FP8 else phys.to(dtype)
    return phys[:, 1]


def _problem(ctxs, rows, bs, num_blocks, seed=0):
    g = torch.Generator().manual_seed(seed)
    seq_lens = [c + rows for c in ctxs]
    max_blocks = max(-(-s // bs) for s in seq_lens)
    bt = torch.zeros(len(ctxs), max_blocks + 3, dtype=torch.int32)
    for r, s in enumerate(seq_lens):
        nb = -(-s // bs)
        if nb < num_blocks:
            bt[r, :nb] = torch.randperm(num_blocks - 1, generator=g)[:nb] + 1
        else:
            bt[r, :nb] = torch.randint(1, num_blocks, (nb,), generator=g)
    qsl = torch.arange(len(ctxs) + 1, dtype=torch.int32) * rows
    qkv = torch.randn(len(ctxs) * rows, (HQ + 2 * HKV) * D, device="cuda")
    q = qkv.to(torch.bfloat16)[:, : HQ * D].view(-1, HQ, D)  # token-strided
    return (
        q,
        bt.cuda(),
        torch.tensor(seq_lens, dtype=torch.int32, device="cuda"),
        qsl.cuda(),
    )


def _reference(q, kv, bt, sl, qsl, k_scale=1.0, v_scale=1.0):
    out = torch.zeros(q.shape, dtype=torch.float32, device="cuda")
    bs = kv.shape[2]
    for r in range(sl.shape[0]):
        s, a, b = int(sl[r]), int(qsl[r]), int(qsl[r + 1])
        if a == b:
            continue
        toks = torch.arange(max(0, s - (b - a) - (W - 1)), s, device="cuda")
        kvr = kv[bt[r, toks // bs].long(), :, toks % bs].float()
        k = (kvr[..., :D] * k_scale).repeat_interleave(HQ // HKV, dim=1)
        v = (kvr[..., D:] * v_scale).repeat_interleave(HQ // HKV, dim=1)
        scores = torch.einsum("qhd,nhd->hqn", q[a:b].float(), k) * SCALE
        pos = s - (b - a) + torch.arange(b - a, device="cuda")
        mask = (toks[None] <= pos[:, None]) & (toks[None] > pos[:, None] - W)
        p = torch.softmax(scores.masked_fill(~mask[None], -math.inf), dim=-1)
        out[a:b] = torch.einsum("hqn,nhd->qhd", p, v)
    return out


def _scales(dtype):
    if dtype != FP8:
        return None, None, 1.0, 1.0
    ks = torch.tensor([K_SCALE], device="cuda")
    vs = torch.tensor([V_SCALE], device="cuda")
    return ks, vs, K_SCALE, V_SCALE


CASES = [([c], 7) for c in (1, 100, 2041, 2042, 2048, 2049, 9000, 130000)]
CASES += [([1024 + off], 7) for off in (-1, 0, 1, 15, 16, 17)]
CASES += [([2048, 9000], 7), ([1000, 130000], 8), ([5000, 33], 5), ([3000, 7], 6)]
CASES += [([4099], 16), ([17, 9000], 14), ([1500, 2500], 10), ([123456], 1)]


@pytest.mark.parametrize("bs", [400, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, FP8], ids=["bf16", "fp8"])
@pytest.mark.parametrize("ctxs,rows", CASES)
def test_kernel_matches_reference(bs, dtype, ctxs, rows):
    from vllm.models.kimi_k3.amd.ops.flydsl_swa_gqa import flydsl_swa_gqa_fwd

    nblk = min(max(-(-(c + rows) // bs) for c in ctxs) * len(ctxs) + 8, 4096)
    kv = _cache(nblk, bs, dtype)
    q, bt, sl, qsl = _problem(ctxs, rows, bs, nblk, seed=rows)
    ks, vs, kf, vf = _scales(dtype)
    out = torch.zeros_like(q)
    flydsl_swa_gqa_fwd(q, kv, bt, sl, qsl, rows, out, SCALE, W, ks, vs)
    ref = _reference(q, kv, bt, sl, qsl, kf, vf)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=0)


def test_cuda_graph_padded_requests():
    """Captured as the builder captures (seq_lens 1) with an empty request in
    the middle; replays on real lengths."""
    from vllm.models.kimi_k3.amd.ops.flydsl_swa_gqa import flydsl_swa_gqa_fwd

    bs = 448
    kv = _cache(512, bs, torch.bfloat16)
    q, bt, sl, _ = _problem([3000, 50, 9000], 7, bs, 512)
    qsl = torch.tensor([0, 7, 7, 14], dtype=torch.int32, device="cuda")
    run_sl = sl.clone().fill_(1)
    out = torch.zeros_like(q)
    flydsl_swa_gqa_fwd(q, kv, bt, run_sl, qsl, 7, out, SCALE, W)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flydsl_swa_gqa_fwd(q, kv, bt, run_sl, qsl, 7, out, SCALE, W)
    run_sl.copy_(sl)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    ref = _reference(q, kv, bt, run_sl, qsl)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=0)


def _metadata(q, bt, sl, qsl, rows):
    from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata

    T = q.shape[0]
    return TritonAttentionMetadata(
        num_actual_tokens=T,
        max_query_len=rows,
        query_start_loc=qsl,
        max_seq_len=int(sl.max()),
        seq_lens=sl,
        block_table=bt,
        slot_mapping=torch.zeros(T, dtype=torch.int64, device="cuda"),
        seq_threshold_3D=64,
        num_par_softmax_segments=16,
        softmax_segm_output=torch.empty(T, HQ, 16, D, device="cuda"),
        softmax_segm_max=torch.empty(T, HQ, 16, device="cuda"),
        softmax_segm_expsum=torch.empty(T, HQ, 16, device="cuda"),
        causal=True,
    )


@pytest.mark.parametrize("cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("rows", [7, 17])
def test_impl_matches_triton_attn(cache_dtype, rows, monkeypatch):
    from vllm.models.kimi_k3.amd import flydsl_swa_gqa_attn
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

    args = dict(
        num_heads=HQ,
        head_size=D,
        scale=SCALE,
        num_kv_heads=HKV,
        alibi_slopes=None,
        sliding_window=W,
        kv_cache_dtype=cache_dtype,
    )
    fly = flydsl_swa_gqa_attn.KimiK3FlyDSLSwaGqaImpl(**args)
    fly.process_weights_after_loading(torch.bfloat16)
    assert fly._off_reason is None
    tri = TritonAttentionImpl(**args)
    calls = []
    real_fwd = flydsl_swa_gqa_attn.flydsl_swa_gqa_fwd
    monkeypatch.setattr(
        flydsl_swa_gqa_attn,
        "flydsl_swa_gqa_fwd",
        lambda *a: calls.append(1) or real_fwd(*a),
    )

    dtype = FP8 if cache_dtype == "fp8" else torch.bfloat16
    bs = 448
    kv = _cache(160, bs, dtype)
    q, bt, sl, qsl = _problem([2500, 30000], rows, bs, 160)
    ks, vs, _, _ = _scales(dtype)
    one = torch.ones(1, device="cuda")
    layer = SimpleNamespace(
        _q_scale=one, _k_scale=ks if ks is not None else one,
        _v_scale=vs if vs is not None else one,
    )
    md = _metadata(q, bt, sl, qsl, rows)
    out_f = torch.zeros_like(q)
    out_t = torch.zeros_like(q)
    fly.forward(layer, q, None, None, kv, md, out_f)
    tri.forward(layer, q, None, None, kv, md, out_t)
    assert len(calls) == (1 if rows <= 16 else 0)
    torch.testing.assert_close(out_f.float(), out_t.float(), atol=2e-2, rtol=0)
