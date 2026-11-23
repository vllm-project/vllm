# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""amd_kernels on the Kimi-K3.1 DSA decode path, single GPU: the integrated ops
against the stock op sequence on real shapes (the DCP / multi-stream wiring is
tested by tests/kimi_dsams/run_dsa_akr.py on 8 GPUs)."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm amd_kernels wiring"
)

HEADS, TOPK, EPS = 32, 2048, 1e-5
ALL = ("PROLOGUE", "SCORE_TOPK", "GATHER_MERGE", "QGATHER", "SPARSE_MLA", "COMBINE")


@pytest.fixture(autouse=True)
def _akr(monkeypatch):
    pytest.importorskip("amd_kernels")
    from vllm.models.kimi_k3.amd import akr_config

    if not akr_config._available():
        pytest.skip("amd_kernels / gfx950 not available")
    for n in ALL:
        monkeypatch.setenv(f"VLLM_KIMI_AKR_{n}", "1")


def _cache(pages: int):
    storage = torch.full((pages, 46592), 173, dtype=torch.uint8, device="cuda")
    return storage, storage[:, 41984:46336].view(pages, 64, 68)


def _knorm():
    from vllm.models.kimi_k3.amd.dsa import KimiK3DSAIndexerKNorm

    norm = KimiK3DSAIndexerKNorm(128, EPS).cuda()
    with torch.no_grad():
        norm.weight.copy_(torch.randn(128, device="cuda") * 0.5 + 1)
        norm.bias.copy_(torch.randn(128, device="cuda") * 0.1)
    return norm


def _selected_scores(out, scores):
    v = torch.where(out >= 0, scores, torch.full_like(scores, float("-inf")))
    return v.sort(dim=1).values


@pytest.mark.parametrize("tokens", [4, 16, 64])
@pytest.mark.parametrize("width", [4096, 131072])
@pytest.mark.parametrize("prepped", [False, True])
def test_fused_score_topk_matches_stock(monkeypatch, tokens, width, prepped):
    """VLLM_KIMI_AKR_SCORE_TOPK: score + select in one launch picks the stock
    pair's keys (same scores of the selected keys, ties at the k-th key aside).
    ``prepped``: the caller's prep rows exceed the decode rows (graph padding)."""
    from vllm.config import VllmConfig
    from vllm.forward_context import set_forward_context
    from vllm.models.kimi_k3.amd import dsa  # noqa: F401
    from vllm.models.kimi_k3.amd.ops.fp4_indexer import (
        quantize_queries_and_weights,
        store_keys,
        store_keys_and_reset,
    )
    from vllm.v1.attention.backends.mla.indexer import (
        DeepSeekV32IndexerDecodeMetadata,
        DeepseekV32IndexerMetadata,
    )

    torch.manual_seed(tokens + width)
    padded = tokens + 8
    pages = min(width, 8192) // 64
    reqs = tokens // 4
    config = VllmConfig()
    knorm = _knorm()
    q = torch.randn((padded, HEADS, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((padded, 128), device="cuda", dtype=torch.bfloat16)
    weights = torch.rand((padded, HEADS), device="cuda", dtype=torch.bfloat16)
    keys = min(width, 8192)
    history = torch.randn((reqs * keys, 128), device="cuda", dtype=torch.bfloat16)
    lengths = torch.randint(keys // 2, keys, (reqs,), device="cuda")
    table = torch.zeros((reqs, width // 64), device="cuda", dtype=torch.int32)
    table[:, :pages] = torch.arange(
        reqs * pages, device="cuda", dtype=torch.int32
    ).view(reqs, pages)
    rows = torch.arange(tokens, device="cuda")
    req = rows // 4
    ends = (lengths[req] - 3 + rows % 4).to(torch.int32)
    slots = (req * keys + ends - 1).to(torch.int64)
    starts = torch.arange(0, tokens + 1, 4, device="cuda", dtype=torch.int32)
    batch = DeepseekV32IndexerMetadata(
        seq_lens=lengths.to(torch.int32),
        max_seq_len=keys,
        slot_mapping=slots,
        num_decodes=reqs,
        num_decode_tokens=tokens,
        num_prefills=0,
        num_prefill_tokens=0,
        decode=DeepSeekV32IndexerDecodeMetadata(
            block_table=table[req],
            seq_lens=ends,
            decode_lens=torch.full((reqs,), 4, device="cuda"),
            requires_padding=False,
            schedule_metadata=torch.empty(0, device="cuda", dtype=torch.int32),
            query_start_loc=starts,
            write_max_decode_len=4,
        ),
        prefill=None,
        block_table=table,
    )
    scale = HEADS**-0.5 * 128**-0.5
    initial, cache = _cache(reqs * pages)
    store_keys(knorm(history), cache, torch.arange(reqs * keys, device="cuda"))

    def run(akr: bool):
        for n in ("SCORE_TOPK",):
            monkeypatch.setenv(f"VLLM_KIMI_AKR_{n}", "1" if akr else "0")
        storage = initial.clone()
        c = storage[:, 41984:46336].view(-1, 64, 68)
        out = torch.full((padded, TOPK), 9, device="cuda", dtype=torch.int32)
        sc = torch.zeros(out.shape, device="cuda")
        prep = {}
        with set_forward_context({"indexer": batch}, config):
            if prepped:
                store_keys_and_reset(knorm(k)[:tokens], c, slots, out[:padded])
                payload, q_scales, q_weights, _ = quantize_queries_and_weights(
                    q, weights, scale, starts, 0, padded
                )
                prep = dict(q_payload=payload, q_scales=q_scales, q_weights=q_weights)
            torch.ops.vllm.kimi_fp4_indexer(
                q,
                knorm(k) if not prepped else k,
                weights,
                c,
                out,
                "indexer",
                width,
                weight_scale=scale,
                scores=sc,
                **prep,
            )
        torch.accelerator.synchronize()
        return storage, out, sc

    s0, o0, c0 = run(False)
    s1, o1, c1 = run(True)
    torch.testing.assert_close(s1, s0, rtol=0, atol=0)
    assert bool((o1[tokens:] == -1).all())
    assert bool(((o0 >= 0).sum(1) == (o1 >= 0).sum(1)).all())
    torch.testing.assert_close(
        _selected_scores(o1, c1), _selected_scores(o0, c0), rtol=0, atol=0
    )


@pytest.mark.parametrize("tokens", [1, 12, 65, 192])
def test_prologue_matches_stock_ops(default_vllm_config, tokens):
    """K_A: q_a / kv_a norms, fp8_ds_mla KV insert, indexer key store, head weights."""
    from types import SimpleNamespace

    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.models.kimi_k3.amd import akr_dsa as A
    from vllm.models.kimi_k3.amd.ops.fp4_indexer import store_keys
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseImpl,  # noqa: F401
    )

    torch.manual_seed(tokens)
    dev = "cuda"
    qn, kn = (
        RMSNorm(1536, eps=EPS).to(dev).bfloat16(),
        RMSNorm(512, eps=EPS).to(dev).bfloat16(),
    )
    with torch.no_grad():
        qn.weight.normal_(1, 0.1)
        kn.weight.normal_(1, 0.1)
    knorm = _knorm()
    knorm.weight_bf16, knorm.bias_bf16 = knorm.weight.bfloat16(), knorm.bias.bfloat16()
    w = SimpleNamespace(
        q_a_layernorm=qn,
        kv_a_layernorm=kn,
        q_lora_rank=1536,
        kv_lora_rank=512,
        indexer=SimpleNamespace(
            k_index_norm=knorm,
            index_num_heads=32,
            indexer_weight_scale=32**-0.5,
            index_softmax_scale=128**-0.5,
        ),
    )
    qkv = (torch.randn((tokens, 2272), device=dev) * 2).bfloat16()
    pages = 8
    kv_cache = torch.zeros((pages, 64, 656), dtype=torch.uint8, device=dev)
    ref_cache = kv_cache.clone()
    _, idx_cache = _cache(pages)
    ref_idx = idx_cache.clone()
    slots = torch.randperm(pages * 64, device=dev)[:tokens].to(torch.int64)
    slots[::4] = -1
    islots = slots.clone()
    islots[1::5] = -1
    q_c, kv_c, idx_w = A.prologue_a(w, qkv, kv_cache, slots, idx_cache, islots)

    def rms(x, norm):
        # the AITER rms_norm contract: fp32 statistics, one bf16 rounding
        xf = x.float()
        inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
        return (xf * inv * norm.weight.float()).bfloat16()

    q_ref = rms(qkv[:, :1536], qn)
    kv_ref = rms(qkv[:, 1536:2048], kn)
    from vllm import _custom_ops as ops

    ops.concat_and_cache_mla(
        kv_ref,
        qkv[:, 2048:2112],
        ref_cache,
        slots,
        kv_cache_dtype="fp8_ds_mla",
        scale=torch.ones(1, device=dev),
    )
    ok = islots >= 0
    store_keys(knorm(qkv[:, 2112:2240])[ok], ref_idx, islots[ok])
    torch.accelerator.synchronize()
    assert float((q_c == q_ref).float().mean()) > 0.999
    assert float((kv_c == kv_ref).float().mean()) > 0.999
    assert float((kv_cache == ref_cache).float().mean()) > 0.999
    assert float((idx_cache == ref_idx).float().mean()) > 0.999
    ref_w = qkv[:, 2240:].float() * (32**-0.5 * 128**-0.5)
    torch.testing.assert_close(idx_w, ref_w, rtol=0, atol=0)


def test_sparse_mla_gate_prefers_flydsl_where_it_wins(monkeypatch):
    """The hook leaves the measured FlyDSL-faster row bands to the stock kernel, only
    when the FlyDSL 96-head variant the decode call needs is built."""
    from vllm._aiter_ops import rocm_aiter_ops as ops
    from vllm.models.kimi_k3.amd import akr_dsa as A

    monkeypatch.setattr(ops, "_FLYDSL_SPARSE_MLA", True)
    monkeypatch.setattr(ops, "_AITER_ENABLED", True)
    monkeypatch.setattr(ops, "_FLYDSL_SPARSE_MLA_FAILED", set())
    monkeypatch.setattr(ops, "_FLYDSL_SPARSE_MLA_COMPILED", {(96, False, True)})
    stock = {r for lo, hi in A.FLYDSL_SPARSE_ROWS for r in range(lo, hi + 1)}
    assert {43, 85, 129, 175, 215} <= stock and not {42, 86, 128, 216, 240} & stock
    for rows in range(1, A.MAX_ROWS_SPARSE + 1):
        q = torch.empty((rows, 96, 576), device="meta")
        assert A._flydsl_sparse_wins(q) == (rows in stock), rows
    q = torch.empty((64, 96, 576), device="meta")
    monkeypatch.setattr(ops, "_FLYDSL_SPARSE_MLA_COMPILED", {(96, True, True)})
    assert not A._flydsl_sparse_wins(q)
    monkeypatch.setattr(ops, "_FLYDSL_SPARSE_MLA_COMPILED", {(96, False, True)})
    monkeypatch.setattr(ops, "_FLYDSL_SPARSE_MLA_FAILED", {96})
    assert not A._flydsl_sparse_wins(q)
    monkeypatch.setattr(ops, "_FLYDSL_SPARSE_MLA_FAILED", set())
    monkeypatch.setattr(ops, "_FLYDSL_SPARSE_MLA", False)
    assert not A._flydsl_sparse_wins(q)


@pytest.mark.parametrize("rows", [1, 12, 43, 100, 192, 240, 256])
def test_sparse_mla_decode_matches_triton(monkeypatch, rows):
    """The DCP sparse MLA decode kernel against Triton sparse_mla_fwd (out and lse)."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.kimi_k3.amd import akr_dsa as A

    monkeypatch.setattr(rocm_aiter_ops, "_FLYDSL_SPARSE_MLA", False)  # no gate
    torch.manual_seed(rows)
    dev = "cuda"
    pages = 64
    # valid fp8_ds_mla records, written by the stock cache op
    from vllm import _custom_ops as ops

    cache = torch.zeros((pages, 64, 656), dtype=torch.uint8, device=dev)
    ops.concat_and_cache_mla(
        torch.randn((pages * 64, 512), device=dev).bfloat16(),
        torch.randn((pages * 64, 64), device=dev).bfloat16(),
        cache,
        torch.arange(pages * 64, device=dev, dtype=torch.int64),
        kv_cache_dtype="fp8_ds_mla",
        scale=torch.ones(1, device=dev),
    )
    rows_view = cache.view(-1, 656)
    q = (torch.randn((rows, 96, 576), device=dev) * 0.7).bfloat16()
    counts = torch.randint(0, 700, (rows,), device=dev).to(torch.int32)
    counts[::7] = 0
    counts[-1] = 300
    indptr = torch.zeros(rows + 1, dtype=torch.int32, device=dev)
    indptr[1:] = counts.cumsum(0)
    packed = torch.randint(0, pages * 64, (rows * 2048,), device=dev).to(torch.int32)
    scale = 0.07
    res = A.sparse_mla_decode(q, rows_view, indptr, packed, scale, torch.bfloat16)
    assert res is not None
    out, lse = res
    ref = torch.empty((rows, 96, 512), dtype=torch.bfloat16, device=dev)
    ref_lse = rocm_aiter_ops.triton_sparse_mla_fwd(
        q,
        rows_view.view(torch.uint8).unsqueeze(1),
        ref,
        scale,
        indptr,
        packed,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        has_invalid=False,
        return_lse=True,
    )
    assert ref_lse is not None
    torch.accelerator.synchronize()
    live = counts > 0
    err = (out[live].float() - ref[live].float()).norm() / ref[live].float().norm()
    assert float(err) < 6e-3
    assert float((lse[live] - ref_lse[live]).abs().max()) < 2e-2


def _ragged_metadata(rows_per_req, keys_per_req, pad_reqs, width):
    """Decode indexer metadata of requests with `rows_per_req` rows each and
    `keys_per_req` keys (rank-local), plus `pad_reqs` padded requests (no rows)."""
    from vllm.v1.attention.backends.mla.indexer import (
        DeepSeekV32IndexerDecodeMetadata,
        DeepseekV32IndexerMetadata,
    )

    dev = "cuda"
    rows_per_req = list(rows_per_req) + [0] * pad_reqs
    keys_per_req = list(keys_per_req) + [0] * pad_reqs
    nreq, tokens = len(rows_per_req), sum(rows_per_req)
    pages = width // 64
    req = torch.tensor(
        [r for r, n in enumerate(rows_per_req) for _ in range(n)], device=dev
    )
    within = torch.tensor(
        [j for n in rows_per_req for j in range(n)], device=dev, dtype=torch.int64
    )
    keys = torch.tensor(keys_per_req, device=dev, dtype=torch.int64)
    new = torch.tensor(rows_per_req, device=dev, dtype=torch.int64)
    raw_ends = keys[req] - new[req] + within + 1
    ends = raw_ends.clamp(min=0).to(torch.int32)
    table = torch.zeros((nreq, pages), device=dev, dtype=torch.int32)
    used = 0
    for r, n in enumerate(keys_per_req):
        nb = -(-n // 64)
        table[r, :nb] = torch.arange(used, used + nb, device=dev, dtype=torch.int32)
        used += nb
    starts = torch.zeros(nreq + 1, device=dev, dtype=torch.int32)
    starts[1:] = torch.tensor(rows_per_req, device=dev).cumsum(0)
    last = (ends - 1).clamp(min=0)
    slots = table[req, last // 64].to(torch.int64) * 64 + last % 64
    # Rows beyond a short request's keys store nothing: several rows writing one
    # slot leave whichever key lands last, a different one in each run.
    slots = torch.where(raw_ends > 0, slots, -1)
    stored = slots[slots >= 0]
    assert stored.unique().numel() == stored.numel()
    meta = DeepseekV32IndexerMetadata(
        seq_lens=keys.to(torch.int32),
        max_seq_len=max(keys_per_req),
        slot_mapping=slots,
        num_decodes=nreq - pad_reqs,
        num_decode_tokens=tokens,
        num_prefills=0,
        num_prefill_tokens=0,
        decode=DeepSeekV32IndexerDecodeMetadata(
            block_table=table[req],
            seq_lens=ends,
            decode_lens=torch.tensor(rows_per_req, device=dev),
            requires_padding=False,
            schedule_metadata=torch.empty(0, device=dev, dtype=torch.int32),
            query_start_loc=starts,
            write_max_decode_len=max(rows_per_req),
        ),
        prefill=None,
        block_table=table,
    )
    return meta, used, tokens


def test_fused_score_topk_across_ragged_batches(monkeypatch):
    """The fused kernel's per-launch state (ready counters of every (request, row
    tile) group, also of padded requests and short requests' empty tiles) must not
    leak into the next launch: ragged / padded batches back to back give the stock
    pair's selections."""
    from vllm.config import VllmConfig
    from vllm.forward_context import set_forward_context
    from vllm.models.kimi_k3.amd import dsa  # noqa: F401
    from vllm.models.kimi_k3.amd.ops.fp4_indexer import decode_tile_size, store_keys

    torch.manual_seed(0)
    width, scale = 131072, HEADS**-0.5 * 128**-0.5
    # DCP8: rank-local width of the model length, every batch's rows one tile (the
    # fused kernel serves single-tile calls only)
    max_model_len = 8 * width
    knorm = _knorm()
    config = VllmConfig()
    total_pages = 3000
    _, cache = _cache(total_pages)
    store_keys(
        knorm(torch.randn((total_pages * 64, 128), device="cuda").bfloat16()),
        cache,
        torch.arange(total_pages * 64, device="cuda"),
    )
    batches = [
        ([1, 2, 5, 3, 4, 1], [3000, 3000, 4000, 3500, 5000, 2100], 0),
        ([4] * 8, [30, 60, 90, 20, 100, 1, 2, 70], 0),
        ([4] * 23, [int(c) for c in torch.randint(2000, 9000, (23,))], 2),
        ([4] * 25, [int(c) for c in torch.randint(2000, 9000, (25,))], 0),
        ([4] * 3, [5000, 6000, 7000], 5),
        ([4] * 13, [int(c) for c in torch.randint(2000, 9000, (13,))], 0),
    ]
    for rows_per_req, keys_per_req, pad in batches:
        meta, used, tokens = _ragged_metadata(rows_per_req, keys_per_req, pad, width)
        assert used <= total_pages
        assert tokens <= decode_tile_size(max_model_len, width, TOPK, HEADS)
        q = torch.randn((tokens, HEADS, 128), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((tokens, 128), device="cuda", dtype=torch.bfloat16)
        w = torch.rand((tokens, HEADS), device="cuda", dtype=torch.bfloat16)
        results = []
        for akr in (False, True):
            monkeypatch.setenv("VLLM_KIMI_AKR_SCORE_TOPK", "1" if akr else "0")
            out = torch.full((tokens, TOPK), 9, device="cuda", dtype=torch.int32)
            sc = torch.zeros(out.shape, device="cuda")
            with set_forward_context({"indexer": meta}, config):
                torch.ops.vllm.kimi_fp4_indexer(
                    q, knorm(k), w, cache, out, "indexer", max_model_len,
                    weight_scale=scale, scores=sc,
                )  # fmt: skip
            torch.accelerator.synchronize()
            results.append((out, sc))
        (o0, c0), (o1, c1) = results
        assert bool(((o0 >= 0).sum(1) == (o1 >= 0).sum(1)).all()), rows_per_req
        torch.testing.assert_close(
            _selected_scores(o1, c1), _selected_scores(o0, c0), rtol=0, atol=0
        )
