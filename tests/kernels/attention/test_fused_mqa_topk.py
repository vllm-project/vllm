# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-launch fused MQA-logits + top-k prefill vs the dense indexer path."""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.kernels.attention.dsa.fused_mqa_topk import (
    FUSED_MQA_TOPK_MAX_ROWS,
    fused_mqa_topk_available,
    fused_mqa_topk_prefill,
)
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder

requires_fused = pytest.mark.skipif(
    not torch.cuda.is_available() or not fused_mqa_topk_available(),
    reason="Requires the SM100 fused MQA+top-k extension",
)

TOPKS = (512, 1024, 2048)
TOPK = 2048
LOGITS_BUDGET = 512 * 1024**2


def _dense(q, k, scales, weights, starts, ends, topk=TOPK):
    """Upstream dense path: budgeted DeepGEMM logits + top_k_per_row_prefill."""
    from vllm.model_executor.layers.sparse_attn_indexer import as_fp32_logits
    from vllm.utils.deep_gemm import (
        fp8_fp4_mqa_logits,
        mqa_logits_mx_inputs,
        mqa_logits_use_mx_scales,
    )

    k_scale = scales
    if mqa_logits_use_mx_scales():
        # Power-of-two scales as four equal UE8M0 bytes, like the K cache.
        exp = torch.log2(scales).round().to(torch.int64) + 127
        k_scale = (exp * 0x01010101).to(torch.int32)
    rows = q.shape[0]
    out = torch.full((rows, topk), -1, device=q.device, dtype=torch.int32)
    lens = ends - starts
    max_len = max(1, int(lens.max()))
    step = max(1, LOGITS_BUDGET // (4 * max_len))
    for i in range(0, rows, step):
        j = min(rows, i + step)
        q_scale, w = mqa_logits_mx_inputs(q[i:j], None, weights[i:j])
        logits = as_fp32_logits(
            fp8_fp4_mqa_logits(
                (q[i:j], q_scale), (k, k_scale), w, starts[i:j], ends[i:j], max_len
            ),
            lens[i:j],
        )
        # Compressed logits: row r holds keys [starts[r], ends[r]) from column 0.
        ops.top_k_per_row_prefill(
            logits,
            torch.zeros_like(lens[i:j]),
            lens[i:j],
            out[i:j],
            j - i,
            logits.stride(0),
            logits.stride(1),
            topk,
        )
    return out


def _logits(q, k, scales, weights, starts, ends):
    """fp32 reference scores: sum_h w * relu(q . k) * scale, -inf off-range."""
    kf = k.float()
    out = torch.zeros(q.shape[0], k.shape[0], device=q.device)
    for h in range(q.shape[1]):
        out += weights[:, h, None] * torch.relu(q[:, h].float() @ kf.T)
    out *= scales[None]
    n = torch.arange(k.shape[0], device=q.device)
    off = (n[None] < starts[:, None]) | (n[None] >= ends[:, None])
    return out.masked_fill(off, -torch.inf)


def _check(rel, q, k, scales, weights, starts, ends, scores=True):
    """Indices relative to the row start: count, bounds, uniqueness, scores.

    ``scores`` compares the selected scores with the exact fp32 top-k; the
    SM100 dense path rounds its logits to bf16, so it is checked without.
    """
    topk = rel.shape[1]
    visible = (ends - starts).clamp(min=0)
    valid = rel >= 0
    assert (valid.sum(1) == visible.clamp(max=topk)).all(), "valid count"
    assert (rel[~valid] == -1).all(), "padding must be -1"
    idx = rel.long() + starts[:, None].long()
    assert ((~valid) | ((idx >= starts[:, None]) & (idx < ends[:, None]))).all()
    srt = idx.masked_fill(~valid, -1).sort().values
    assert ((srt[:, 1:] < 0) | (srt.diff(dim=1) > 0)).all(), "duplicates"
    if not scores:
        return
    # Check scores in row blocks so the reference logits stay small.
    step = max(1, LOGITS_BUDGET // (4 * k.shape[0]))
    for i in range(0, q.shape[0], step):
        j = min(q.shape[0], i + step)
        logits = _logits(q[i:j], k, scales, weights[i:j], starts[i:j], ends[i:j])
        ref = logits.topk(topk, dim=1).values
        chosen = logits.gather(1, idx[i:j].clamp(min=0)).masked_fill(
            ~valid[i:j], -torch.inf
        )
        torch.testing.assert_close(
            chosen.sort(descending=True).values, ref, rtol=2e-4, atol=1e-4
        )


def _inputs(rows, keys, seed=913):
    torch.manual_seed(seed)
    q = torch.randn(rows, 32, 128, device="cuda").to(torch.float8_e4m3fn)
    k = torch.randn(keys, 128, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.ones(keys, device="cuda")
    weights = torch.randn(rows, 32, device="cuda") / 32
    return q, k, scales, weights


@requires_fused
@pytest.mark.parametrize("topk", TOPKS)
@pytest.mark.parametrize(
    "keys,rows,case",
    [
        (4096, 64, "causal"),
        (16384, 256, "causal"),
        (16384, 16, "early"),
        (32768, 16, "bounds"),
        (65536, 16, "ties"),
        (131072, 16, "hot"),
        (262145, 9, "padding"),
        (1048576, 16, "causal"),
    ],
)
def test_single_request_matches_dense(keys, rows, case, topk):
    q, k, scales, weights = _inputs(rows, keys)
    starts = torch.zeros(rows, device="cuda", dtype=torch.int32)
    ends = torch.arange(keys - rows + 1, keys + 1, device="cuda", dtype=torch.int32)
    if case == "early":  # rows around each top-k's visible-key boundary
        ends = torch.tensor(
            [
                0,
                1,
                511,
                512,
                513,
                1023,
                1024,
                1025,
                2047,
                2048,
                2049,
                4096,
                8192,
                12000,
                16000,
                keys,
            ],
            device="cuda",
            dtype=torch.int32,
        )
    elif case == "bounds":
        starts.fill_(333)
        ends[::2] = keys // 2
    elif case == "ties":
        weights.zero_()
    elif case == "hot":
        weights = weights.abs()
        scales[:12288] = 4
    out = torch.empty(rows, topk, device="cuda", dtype=torch.int32)
    assert fused_mqa_topk_prefill(q, k, scales, weights, starts, ends, out)
    _check(out, q, k, scales, weights, starts, ends)
    # Same convention and valid counts as the dense path (key sets may differ
    # only among equal scores at the top-k boundary).
    dense = _dense(q, k, scales, weights, starts, ends, topk)
    _check(dense, q, k, scales, weights, starts, ends, scores=False)
    assert ((out >= 0).sum(1) == (dense >= 0).sum(1)).all()


@requires_fused
@pytest.mark.parametrize("topk", TOPKS)
@pytest.mark.parametrize(
    "lens", [[3000, 5000, 1000], [40000, 1500, 70000, 9], [600000, 200000]]
)
def test_multi_request_chunk_matches_dense(lens, topk):
    """Packed requests: rows of request r see [cu[r], cu[r] + causal_end)."""
    q_lens = [min(n, 1024) for n in lens]
    rows, keys = sum(q_lens), sum(lens)
    q, k, scales, weights = _inputs(rows, keys, seed=917)
    starts, ends, base = [], [], 0
    for n, m in zip(lens, q_lens):
        starts += [base] * m
        ends += list(range(base + n - m + 1, base + n + 1))
        base += n
    starts = torch.tensor(starts, device="cuda", dtype=torch.int32)
    ends = torch.tensor(ends, device="cuda", dtype=torch.int32)
    out = torch.empty(rows, topk, device="cuda", dtype=torch.int32)
    assert fused_mqa_topk_prefill(q, k, scales, weights, starts, ends, out)
    _check(out, q, k, scales, weights, starts, ends)
    dense = _dense(q, k, scales, weights, starts, ends, topk)
    _check(dense, q, k, scales, weights, starts, ends, scores=False)
    assert ((out >= 0).sum(1) == (dense >= 0).sum(1)).all()
    # Both paths agree on (nearly) every selected key. The dense path's bf16
    # logits blur the k-th boundary, the more so for small k at long contexts
    # (~98.6% overlap for top-512 over 600K keys; the fused keys are exact).
    overlap = sum(
        len(set(a[a >= 0].tolist()) & set(b[b >= 0].tolist()))
        for a, b in zip(out.cpu(), dense.cpu())
    )
    assert overlap >= 0.98 * int((dense >= 0).sum())


@requires_fused
@pytest.mark.parametrize("rows", [32768, 32770])
def test_large_chunk_single_call(rows):
    """Prefill chunks above 16384 rows (e.g. --max-num-batched-tokens 32768)
    run in one launch; the per-row workspace grows to fit."""
    from vllm.model_executor.kernels.attention.dsa.fused_mqa_topk import (
        get_fused_mqa_topk_workspace,
    )

    keys = 65536
    q, k, scales, weights = _inputs(rows, keys, seed=921)
    starts = torch.zeros(rows, device="cuda", dtype=torch.int32)
    ends = torch.arange(keys - rows + 1, keys + 1, device="cuda", dtype=torch.int32)
    out = torch.empty(rows, TOPK, device="cuda", dtype=torch.int32)
    assert fused_mqa_topk_prefill(q, k, scales, weights, starts, ends, out)
    assert get_fused_mqa_topk_workspace(q.device, rows).counts.numel() >= rows
    _check(out, q, k, scales, weights, starts, ends)


@requires_fused
def test_rows_start_at_zero_skips_conversion():
    rows, keys = 32, 8192
    q, k, scales, weights = _inputs(rows, keys)
    starts = torch.zeros(rows, device="cuda", dtype=torch.int32)
    ends = torch.arange(keys - rows + 1, keys + 1, device="cuda", dtype=torch.int32)
    a = torch.empty(rows, TOPK, device="cuda", dtype=torch.int32)
    b = torch.empty_like(a)
    assert fused_mqa_topk_prefill(q, k, scales, weights, starts, ends, a)
    assert fused_mqa_topk_prefill(
        q, k, scales, weights, starts, ends, b, rows_start_at_zero=True
    )
    torch.testing.assert_close(a.sort(1).values, b.sort(1).values)


@requires_fused
def test_cuda_graph_replay():
    rows, keys = 64, 65536
    q, k, scales, weights = _inputs(rows, keys)
    starts = torch.zeros(rows, device="cuda", dtype=torch.int32)
    ends = torch.arange(keys - rows + 1, keys + 1, device="cuda", dtype=torch.int32)
    out = torch.empty(rows, TOPK, device="cuda", dtype=torch.int32)
    fn = lambda: fused_mqa_topk_prefill(  # noqa: E731
        q, k, scales, weights, starts, ends, out, rows_start_at_zero=True
    )
    fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    out.fill_(-7)
    for _ in range(3):
        graph.replay()
    torch.accelerator.synchronize()
    _check(out, q, k, scales, weights, starts, ends)


@requires_fused
def test_unsupported_inputs_fall_back():
    q, k, scales, weights = _inputs(16, 4096)
    starts = torch.zeros(16, device="cuda", dtype=torch.int32)
    ends = torch.full((16,), 4096, device="cuda", dtype=torch.int32)
    out = torch.empty(16, TOPK, device="cuda", dtype=torch.int32)
    assert not fused_mqa_topk_prefill(
        q[:, :16], k, scales, weights[:, :16], starts, ends, out
    )
    assert not fused_mqa_topk_prefill(q, k, scales, weights, starts, ends, out[:, :768])
    q17 = torch.zeros(16385, 32, 128, device="cuda").to(torch.float8_e4m3fn)
    assert not fused_mqa_topk_prefill(q17, k, scales, weights, starts, ends, out)


def _split(seq_lens, q_lens, fused, offset=0):
    return DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks(
        torch.tensor(seq_lens),
        torch.tensor(q_lens),
        1048576,
        LOGITS_BUDGET,
        request_offset=offset,
        use_fused_mqa_topk=fused,
    )


@pytest.mark.parametrize("q_rows", [16384, 32768])
@pytest.mark.parametrize("fused", [False, True])
def test_planner_one_fused_call_per_chunk(fused, q_rows):
    # One fused call per FUSED_MQA_TOPK_MAX_ROWS rows (None: per chunk); dense
    # splits by the logits budget.
    chunks = _split([524288], [q_rows], fused)
    cap = FUSED_MQA_TOPK_MAX_ROWS or q_rows
    assert len(chunks) == (-(-q_rows // cap) if fused else q_rows // 256)
    assert sum(q.stop - q.start for _, q in chunks) == q_rows


def test_planner_fused_any_length_and_short_queries():
    # No length threshold and no minimum query count on the fused path.
    assert _split([524288], [512], True) == [(slice(0, 1), slice(0, 512))]
    assert _split([4096], [4096], True) == [(slice(0, 1), slice(0, 4096))]
    # A 40000-row step: split at FUSED_MQA_TOPK_MAX_ROWS (None: one call).
    cap = FUSED_MQA_TOPK_MAX_ROWS or 40000
    assert _split([65536], [40000], True) == [
        (slice(0, 1), slice(lo, min(lo + cap, 40000))) for lo in range(0, 40000, cap)
    ]


def test_planner_packs_requests_in_one_fused_call():
    # Requests share one fused call up to the key limit; dense needs a split.
    assert _split([4096] * 4, [4096] * 4, True) == [(slice(0, 4), slice(0, 16384))]
    assert len(_split([4096] * 4, [4096] * 4, False)) > 1
    chunks = _split([524288, 262144], [16384, 9], True, offset=2)
    if FUSED_MQA_TOPK_MAX_ROWS is None or FUSED_MQA_TOPK_MAX_ROWS >= 16393:
        assert chunks == [(slice(2, 4), slice(0, 16393))]
    else:  # the second request would push the call past the row cap
        assert chunks == [(slice(2, 3), slice(0, 16384)), (slice(3, 4), slice(0, 9))]


def test_planner_fused_never_packs_past_key_limit():
    # Two 600K-key continuation steps of 50 rows: dense packs them into one
    # logits-budget chunk; the fused planner keeps one fused chunk per request so
    # no chunk reaches the dense path (whose logits buffer is not reserved).
    args = (
        torch.tensor([600_000, 600_000]),
        torch.tensor([50, 50]),
        4_000_000,
        LOGITS_BUDGET,
    )
    dense = DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks(*args)
    fused = DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks(
        *args, use_fused_mqa_topk=True
    )
    assert dense == [(slice(0, 2), slice(0, 100))]
    assert fused == [(slice(0, 1), slice(0, 50)), (slice(1, 2), slice(0, 50))]


def test_planner_keeps_budget_beyond_fused_key_limit():
    seq_lens, q_lens = [2_000_000], [256]
    fused = DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks(
        torch.tensor(seq_lens),
        torch.tensor(q_lens),
        4_000_000,
        LOGITS_BUDGET,
        use_fused_mqa_topk=True,
    )
    dense = DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks(
        torch.tensor(seq_lens),
        torch.tensor(q_lens),
        4_000_000,
        LOGITS_BUDGET,
    )
    assert fused == dense and len(dense) == 4


KPOOL = 4


def _kpool_inputs(pool_lens=(20000, 70000), q_rows=96, tail=2, seed=919):
    """Two packed GLM-5.3-Flash-style requests: K holds 4-token pools; the
    last ``q_rows`` tokens of each request score the pools they can see."""
    q, k, _, weights = _inputs(q_rows * len(pool_lens), sum(pool_lens), seed)
    starts, ends, positions, base = [], [], [], 0
    for n in pool_lens:
        tokens = KPOOL * n + tail
        pos = torch.arange(tokens - q_rows, tokens)
        positions.append(pos)
        starts += [base] * q_rows
        ends += (base + (pos + 1) // KPOOL).tolist()
        base += n
    i32 = dict(device="cuda", dtype=torch.int32)
    # Unit power-of-two scales as four UE8M0 bytes, like the SM100 K cache.
    scales = torch.full((k.shape[0], 4), 127, device="cuda", dtype=torch.uint8)
    return (
        q,
        k,
        scales,
        weights,
        torch.tensor(starts, **i32),
        torch.tensor(ends, **i32),
        torch.cat(positions).cuda(),
    )


def _run_kpool_indexer(monkeypatch, inputs, fused, fused_fn=None, budget_mb=512):
    """Run the GLM-5.3-Flash kpool indexer prefill on one packed chunk."""
    from types import SimpleNamespace

    import vllm.envs as envs
    import vllm.models.glm5next.nvidia.sparse_indexer as kpool_indexer
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadata,
        DeepseekV32IndexerPrefillChunkMetadata,
        DeepseekV32IndexerPrefillMetadata,
    )

    q, k, scales, weights, starts, ends, positions = inputs
    rows, keys = q.shape[0], k.shape[0]
    chunk = DeepseekV32IndexerPrefillChunkMetadata(
        block_table=torch.zeros(2, 1, device="cuda", dtype=torch.int32),
        cu_seqlen_ks=starts,
        cu_seqlen_ke=ends,
        cu_seq_lens=torch.zeros(3, device="cuda", dtype=torch.int32),
        token_to_seq=torch.zeros(keys, device="cuda", dtype=torch.int32),
        total_seq_lens=keys,
        max_seqlen_k=int((ends - starts).max()),
        token_start=0,
        token_end=rows,
        num_reqs=2,
        skip_kv_gather=True,  # K is already in the gather workspace
        local_cu_seq_lens=torch.zeros(3, device="cuda", dtype=torch.int32),
        local_total_seq_lens=keys,
        max_local_total_seq_lens=keys,
    )
    metadata = DeepseekV32IndexerMetadata(
        seq_lens=torch.zeros(2, device="cuda", dtype=torch.int32),
        max_seq_len=int(positions.max()) + 1,
        slot_mapping=torch.zeros(rows, device="cuda", dtype=torch.int64),
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=2,
        num_prefill_tokens=rows,
        prefill=DeepseekV32IndexerPrefillMetadata(
            chunks=[chunk], max_prefill_seq_len=int(positions.max()) + 1
        ),
    )
    workspace = SimpleNamespace(get_simultaneous=lambda *specs: (k, scales))
    monkeypatch.setattr(
        kpool_indexer,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"indexer": metadata}),
    )
    monkeypatch.setattr(kpool_indexer, "current_workspace_manager", lambda: workspace)
    monkeypatch.setattr(envs, "VLLM_USE_FUSED_MQA_TOPK", fused)
    monkeypatch.setattr(envs, "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", budget_mb)
    calls: list[torch.Size] = []
    fused_fn = fused_fn or kpool_indexer.fused_mqa_topk_prefill

    def spy(*a, **kw):
        calls.append(a[-1].shape)  # the output buffer: [rows, top-k pools]
        return fused_fn(*a, **kw)

    monkeypatch.setattr(kpool_indexer, "fused_mqa_topk_prefill", spy)
    out = torch.empty(rows, TOPK + KPOOL - 1, device="cuda", dtype=torch.int32)
    kpool_indexer.sparse_attn_indexer_kpool(
        hidden_states=torch.empty(rows, 1, device="cuda"),
        k_cache_prefix="indexer",
        kv_cache=torch.empty(0, device="cuda"),
        q_quant=q,
        q_scale=None,
        k=None,
        weights=weights,
        quant_block_size=128,
        scale_fmt="ue8m0",
        topk_tokens=TOPK,
        head_dim=128,
        max_pool_len=keys,
        total_seq_lens=keys,
        topk_indices_buffer=out,
        skip_k_cache_insert=True,
        index_kpool=KPOOL,
        positions=positions,
    )
    return out, calls


def _row_sets(out):
    return [set(r[r >= 0].tolist()) for r in out.cpu()]


@requires_fused
def test_kpool_indexer_fused_matches_dense(monkeypatch):
    """GLM-5.3-Flash kpool indexer: the fused kernel ranks the top-512 pools of
    a packed multi-request chunk like the dense logits + top-k path."""
    inputs = _kpool_inputs()
    fused, calls = _run_kpool_indexer(monkeypatch, inputs, fused=True)
    assert calls == [torch.Size([inputs[0].shape[0], TOPK // KPOOL])]
    dense, calls = _run_kpool_indexer(monkeypatch, inputs, fused=False)
    assert calls == []
    assert ((fused >= 0).sum(1) == (dense >= 0).sum(1)).all()
    # Same pools up to the dense path's bf16 boundary blur (see above); every
    # row also keeps its request's incomplete tail tokens.
    overlap = sum(len(a & b) for a, b in zip(_row_sets(fused), _row_sets(dense)))
    assert overlap >= 0.98 * int((dense >= 0).sum())


@requires_fused
def test_kpool_indexer_short_chunk_stays_dense(monkeypatch):
    """Chunks whose rows see fewer than fused_mqa_topk_min_keys pools keep the
    dense path (the fused kernel is not faster there)."""
    from vllm.model_executor.kernels.attention.dsa.fused_mqa_topk import (
        fused_mqa_topk_min_keys,
    )

    inputs = _kpool_inputs(pool_lens=(9000, 20000))
    assert int((inputs[5] - inputs[4]).max()) < fused_mqa_topk_min_keys(TOPK // KPOOL)
    _, calls = _run_kpool_indexer(monkeypatch, inputs, fused=True)
    assert calls == []


@requires_fused
def test_kpool_indexer_dense_fallback_splits_rows(monkeypatch):
    """Fused-planned chunks skip the logits budget; if the kernel declines,
    the dense fallback splits rows to stay within it and selects the same
    pools as one unbudgeted dense pass."""
    inputs = _kpool_inputs()
    whole, _ = _run_kpool_indexer(monkeypatch, inputs, fused=False)
    split, calls = _run_kpool_indexer(
        monkeypatch, inputs, fused=True, fused_fn=lambda *a, **kw: False, budget_mb=1
    )
    assert len(calls) == 1
    # top_k_per_row_prefill breaks bf16 ties at the k-th score arbitrarily (two
    # identical dense runs differ too), so compare the selected scores.
    from vllm.model_executor.layers.sparse_attn_indexer import as_fp32_logits
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits, mqa_logits_mx_inputs

    q, k, scales, weights, starts, ends, _ = inputs
    lens = ends - starts
    q_scale, w = mqa_logits_mx_inputs(q, None, weights)
    logits = as_fp32_logits(
        fp8_fp4_mqa_logits(
            (q, q_scale),
            (k, scales.view(torch.int32).squeeze(-1)),
            w,
            starts,
            ends,
            int(lens.max()),
        ),
        lens,
    )
    for r, (a, b) in enumerate(zip(_row_sets(split), _row_sets(whole))):
        # Request-relative token ids -> pools; the appended tail is not scored.
        pools = [
            sorted({t // KPOOL for t in s if t // KPOOL < int(lens[r])}) for s in (a, b)
        ]
        scores = [logits[r, p].sort().values for p in pools]
        assert torch.equal(scores[0], scores[1]), r
