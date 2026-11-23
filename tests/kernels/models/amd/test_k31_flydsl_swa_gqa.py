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
1 to 48 requests (both tiles-per-workgroup configs), 1 to 16 query rows, bf16
and fp8 e4m3 (per-tensor scales, [1] or 0-dim as vLLM registers them), and
CUDA graphs captured with seq_lens 1 and an empty (padded) request, or right
after the warmup, then replayed on real data. The backend's impl is checked
against TRITON_ATTN's on the same metadata and bf16 query (the Attention layer
must not quantize q for it on an fp8 cache), and for its Triton fallback past
16 rows.
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


def _num_blocks(ctxs, rows, bs):
    return min(max(-(-(c + rows) // bs) for c in ctxs) * len(ctxs) + 8, 4096)


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


def _scales(dtype, dim0=False):
    """(k, v scale tensors or None, k, v floats); dim0: 0-dim tensors, as the
    Attention layer registers _k_scale / _v_scale."""
    if dtype != FP8:
        return None, None, 1.0, 1.0
    shape = () if dim0 else (1,)
    ks = torch.full(shape, K_SCALE, device="cuda")
    vs = torch.full(shape, V_SCALE, device="cuda")
    return ks, vs, K_SCALE, V_SCALE


CASES = [([c], 7) for c in (1, 100, 2041, 2042, 2048, 2049, 9000, 130000)]
CASES += [([1024 + off], 7) for off in (-1, 0, 1, 15, 16, 17)]
CASES += [([2048, 9000], 7), ([1000, 130000], 8), ([5000, 33], 5), ([3000, 7], 6)]
CASES += [([4099], 16), ([17, 9000], 14), ([1500, 2500], 10), ([123456], 1)]
# Many requests: 3-5 rows take 8 key tiles a workgroup from TPC8_MIN_REQS on.
CASES += [([2049 + 997 * i for i in range(8)], 4), ([50000 + i for i in range(20)], 4)]
CASES += [([7, 2048, 3001, 20000, 64], 3), ([1 + 4093 * i for i in range(48)], 5)]
CASES += [([9000] * 4, 4), ([3000 + i for i in range(8)], 2)]
CASES += [([3000 + 31 * i for i in range(8)], 7)]


@pytest.mark.parametrize("bs", [400, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, FP8], ids=["bf16", "fp8"])
@pytest.mark.parametrize("ctxs,rows", CASES)
def test_kernel_matches_reference(bs, dtype, ctxs, rows):
    from vllm.models.kimi_k3.amd.ops.flydsl_swa_gqa import flydsl_swa_gqa_fwd

    nblk = _num_blocks(ctxs, rows, bs)
    kv = _cache(nblk, bs, dtype)
    q, bt, sl, qsl = _problem(ctxs, rows, bs, nblk, seed=rows)
    ks, vs, kf, vf = _scales(dtype)
    out = torch.zeros_like(q)
    flydsl_swa_gqa_fwd(q, kv, bt, sl, qsl, rows, out, SCALE, W, ks, vs)
    ref = _reference(q, kv, bt, sl, qsl, kf, vf)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=0)


@pytest.mark.parametrize("ctxs,rows", [([2500, 30000], 7), ([9000] * 8, 4)])
def test_kernel_0dim_scales(ctxs, rows):
    """0-dim k / v scales (vLLM's) give the [1]-scale result bit for bit."""
    from vllm.models.kimi_k3.amd.ops.flydsl_swa_gqa import flydsl_swa_gqa_fwd

    bs = 64
    nblk = _num_blocks(ctxs, rows, bs)
    kv = _cache(nblk, bs, FP8)
    q, bt, sl, qsl = _problem(ctxs, rows, bs, nblk)
    outs = []
    for dim0 in (False, True):
        ks, vs, kf, vf = _scales(FP8, dim0)
        assert ks.dim() == (0 if dim0 else 1)
        outs.append(torch.zeros_like(q))
        flydsl_swa_gqa_fwd(q, kv, bt, sl, qsl, rows, outs[-1], SCALE, W, ks, vs)
    torch.testing.assert_close(outs[1], outs[0], atol=0, rtol=0)
    ref = _reference(q, kv, bt, sl, qsl, kf, vf)
    torch.testing.assert_close(outs[1].float(), ref, atol=2e-2, rtol=0)


def test_config_tiles_per_workgroup():
    from vllm.models.kimi_k3.amd.ops.flydsl_swa_gqa import TPC8_MIN_REQS, _config

    # The served c16-c48 shapes (4 rows, 8 / 13 / 20 / 48 requests) take 8.
    for reqs in (TPC8_MIN_REQS, 8, 13, 20, 48, 256):
        for rows in (3, 4, 5):
            assert _config(rows, reqs) == (2, 8)
    for rows in (3, 4, 5):
        assert _config(rows, TPC8_MIN_REQS - 1) == (2, 4)
    for reqs in (1, 8, 48):
        assert _config(2, reqs) == (1, 4)
        assert _config(7, reqs) == (3, 4)
        assert _config(16, reqs) == (6, 2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, FP8], ids=["bf16", "fp8"])
def test_tiles_per_workgroup_8_matches_4(dtype, monkeypatch):
    """At the c32 shape (20 requests x 4 rows) both configs agree with the
    reference, and with each other to bf16 rounding (split order only)."""
    import vllm.models.kimi_k3.amd.ops.flydsl_swa_gqa as ops

    bs, rows = 64, 4
    ctxs = [40000 + 1999 * i for i in range(20)]
    nblk = _num_blocks(ctxs, rows, bs)
    kv = _cache(nblk, bs, dtype)
    q, bt, sl, qsl = _problem(ctxs, rows, bs, nblk)
    ks, vs, kf, vf = _scales(dtype, dim0=True)
    ref = _reference(q, kv, bt, sl, qsl, kf, vf)
    stock = ops._config
    outs = {}
    for tpc in (4, 8):
        monkeypatch.setattr(ops, "_config", lambda r, n, t=tpc: (stock(r, n)[0], t))
        outs[tpc] = torch.zeros_like(q)
        ops.flydsl_swa_gqa_fwd(q, kv, bt, sl, qsl, rows, outs[tpc], SCALE, W, ks, vs)
    errs = {t: (o.float() - ref).abs().max().item() for t, o in outs.items()}
    assert max(errs.values()) < 2e-2, errs
    torch.testing.assert_close(outs[8], outs[4], atol=1e-3, rtol=0)


@pytest.mark.parametrize(
    "dtype,bs,ctxs,qlens",
    [
        (torch.bfloat16, 448, [3000, 50, 9000], [7, 0, 7]),
        # 8 key tiles a workgroup: new requests, and padded ones (no rows, no
        # keys, null blocks) mid-batch and at the end.
        (FP8, 64, [0, 5000, 0, 3, 2045, 60000, 0, 0], [4, 1, 0, 3, 4, 2, 0, 0]),
    ],
    ids=["bf16", "fp8"],
)
def test_cuda_graph_padded_requests(dtype, bs, ctxs, qlens):
    """Captured as the builder captures (seq_lens 1) with empty requests;
    replays on real lengths. Rows past the last request stay untouched."""
    from vllm.models.kimi_k3.amd.ops.flydsl_swa_gqa import flydsl_swa_gqa_fwd

    rows = max(qlens)
    nblk = _num_blocks(ctxs, rows, bs)
    kv = _cache(nblk, bs, dtype)
    q, bt, _, _ = _problem(ctxs, rows, bs, nblk)
    sl = torch.tensor(
        [c + n for c, n in zip(ctxs, qlens)], dtype=torch.int32, device="cuda"
    )
    bt[sl == 0] = 0
    qsl = torch.tensor([0, *qlens], device="cuda").cumsum(0).to(torch.int32)
    ks, vs, kf, vf = _scales(dtype, dim0=True)
    run_sl = sl.clone().fill_(1)
    out = torch.zeros_like(q)
    flydsl_swa_gqa_fwd(q, kv, bt, run_sl, qsl, rows, out, SCALE, W, ks, vs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flydsl_swa_gqa_fwd(q, kv, bt, run_sl, qsl, rows, out, SCALE, W, ks, vs)
    run_sl.copy_(sl)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    ref = _reference(q, kv, bt, sl, qsl, kf, vf)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=0)


@pytest.mark.parametrize("num_reqs", [1, 8, 48])
@pytest.mark.parametrize("dtype", [torch.bfloat16, FP8], ids=["bf16", "fp8"])
def test_cuda_graph_after_warmup(dtype, num_reqs, monkeypatch):
    """The warmup alone compiles every config a capture can meet (both tile
    configs of 4 rows): nothing is refused or compiled inside the capture."""
    import vllm.models.kimi_k3.amd.ops.flydsl_swa_gqa as ops

    monkeypatch.setattr(ops, "_COMPILED", set())
    ops.flydsl_swa_gqa_warmup("cuda", HKV, W, dtype)
    bs, rows = 64, 4
    ctxs = [20000 + 777 * i for i in range(num_reqs)]
    nblk = _num_blocks(ctxs, rows, bs)
    kv = _cache(nblk, bs, dtype)
    q, bt, sl, qsl = _problem(ctxs, rows, bs, nblk)
    ks, vs, kf, vf = _scales(dtype, dim0=True)
    run_sl = sl.clone().fill_(1)
    out = torch.zeros_like(q)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        reason = ops.flydsl_swa_gqa_call_reason(
            num_reqs=num_reqs, max_query_len=rows, window=W, kv_cache=kv
        )
        if reason is None:
            ops.flydsl_swa_gqa_fwd(q, kv, bt, run_sl, qsl, rows, out, SCALE, W, ks, vs)
    assert reason is None, reason
    run_sl.copy_(sl)
    graph.replay()
    torch.accelerator.synchronize()
    ref = _reference(q, kv, bt, sl, qsl, kf, vf)
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


def _impl_args(cache_dtype):
    return dict(
        num_heads=HQ,
        head_size=D,
        scale=SCALE,
        num_kv_heads=HKV,
        alibi_slopes=None,
        sliding_window=W,
        kv_cache_dtype=cache_dtype,
    )


@pytest.mark.parametrize("cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize(
    "ctxs,rows",
    [
        ([2500, 30000], 7),
        ([2500, 30000], 17),
        ([30000 + 3001 * i for i in range(20)], 4),
    ],
)
def test_impl_matches_triton_attn(cache_dtype, ctxs, rows, monkeypatch):
    """Scales are 0-dim and q is bf16, as the Attention layer hands them over."""
    from vllm.models.kimi_k3.amd import flydsl_swa_gqa_attn
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

    fly = flydsl_swa_gqa_attn.KimiK3FlyDSLSwaGqaImpl(**_impl_args(cache_dtype))
    fly.process_weights_after_loading(torch.bfloat16)
    assert fly._off_reason is None
    assert not fly.supports_quant_query_input
    tri = TritonAttentionImpl(**_impl_args(cache_dtype))
    calls = []
    real_fwd = flydsl_swa_gqa_attn.flydsl_swa_gqa_fwd
    monkeypatch.setattr(
        flydsl_swa_gqa_attn,
        "flydsl_swa_gqa_fwd",
        lambda *a: calls.append(1) or real_fwd(*a),
    )

    dtype = FP8 if cache_dtype == "fp8" else torch.bfloat16
    bs = 448
    nblk = max(_num_blocks(ctxs, rows, bs), 160)
    kv = _cache(nblk, bs, dtype)
    q, bt, sl, qsl = _problem(ctxs, rows, bs, nblk)
    ks, vs, _, _ = _scales(dtype, dim0=True)
    one = torch.ones((), device="cuda")
    layer = SimpleNamespace(
        _q_scale=one,
        _k_scale=ks if ks is not None else one,
        _v_scale=vs if vs is not None else one,
    )
    md = _metadata(q, bt, sl, qsl, rows)
    out_f = torch.zeros_like(q)
    out_t = torch.zeros_like(q)
    fly.forward(layer, q, None, None, kv, md, out_f)
    tri.forward(layer, q, None, None, kv, md, out_t)
    assert len(calls) == (1 if rows <= 16 else 0)
    assert fly._off_reason is None, fly._off_reason
    torch.testing.assert_close(out_f.float(), out_t.float(), atol=2e-2, rtol=0)


@pytest.mark.parametrize("cache_dtype", ["auto", "fp8"])
def test_attention_layer_keeps_bf16_query(cache_dtype):
    """On an fp8 cache the Attention layer quantizes q for TRITON_ATTN (gfx950)
    but not for KIMI_K3_FLYDSL_SWA_GQA, whose kernel takes bf16 q only."""
    from vllm.config import CacheConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.attention import Attention
    from vllm.models.kimi_k3.amd.flydsl_swa_gqa_attn import (
        KimiK3FlyDSLSwaGqaBackend,
    )
    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

    layers = {}
    with set_current_vllm_config(VllmConfig()), torch.device("cuda"):
        for name, backend in (
            ("fly", KimiK3FlyDSLSwaGqaBackend),
            ("tri", TritonAttentionBackend),
        ):
            layers[name] = Attention(
                num_heads=HQ,
                head_size=D,
                scale=SCALE,
                num_kv_heads=HKV,
                cache_config=CacheConfig(cache_dtype=cache_dtype),
                prefix=f"draft.{name}.{cache_dtype}.attn",
                per_layer_sliding_window=W,
                attn_backend=backend,
            )
    assert layers["fly"].query_quant is None
    if cache_dtype == "fp8":
        assert layers["tri"].query_quant is not None
