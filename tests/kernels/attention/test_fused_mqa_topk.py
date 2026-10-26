# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-launch fused MQA-logits + top-k prefill vs the dense indexer path."""

from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
import vllm.models.glm5next.nvidia.sparse_indexer as kpool_indexer
from vllm import _custom_ops as ops
from vllm.model_executor.kernels.attention.dsa.fused_mqa_topk import (
    fused_mqa_topk_available,
    fused_mqa_topk_min_keys,
    fused_mqa_topk_prefill,
    get_fused_mqa_topk_workspace,
    use_fused_mqa_topk,
)
from vllm.model_executor.layers.sparse_attn_indexer import as_fp32_logits
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    mqa_logits_mx_inputs,
    mqa_logits_use_mx_scales,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerPrefillChunkMetadata,
    DeepseekV32IndexerPrefillMetadata,
)

requires_fused = pytest.mark.skipif(
    not torch.cuda.is_available() or not fused_mqa_topk_available(),
    reason="Requires the SM100 fused MQA+top-k extension",
)

TOPKS = (512, 1024, 2048)
TOPK = 2048
LOGITS_BUDGET = 512 * 1024**2


def _dense(q, k, scales, weights, starts, ends, topk=TOPK):
    """Upstream dense path: budgeted DeepGEMM logits + top_k_per_row_prefill."""
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
    fused_mqa_topk_prefill(q, k, scales, weights, starts, ends, out)
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
    fused_mqa_topk_prefill(q, k, scales, weights, starts, ends, out)
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
    keys = 65536
    q, k, scales, weights = _inputs(rows, keys, seed=921)
    starts = torch.zeros(rows, device="cuda", dtype=torch.int32)
    ends = torch.arange(keys - rows + 1, keys + 1, device="cuda", dtype=torch.int32)
    out = torch.empty(rows, TOPK, device="cuda", dtype=torch.int32)
    fused_mqa_topk_prefill(q, k, scales, weights, starts, ends, out)
    assert get_fused_mqa_topk_workspace(q.device, rows).counts.numel() >= rows
    _check(out, q, k, scales, weights, starts, ends)


@requires_fused
def test_rescan_scores_only_its_block():
    """The exact rescan of a block touches only that block's 8 rows.

    Regression test: the rescan's scheduler used a grid stride of 4096 blocks,
    so with more than 4096 blocks (rows > 32768) rescanning block v also scored
    block v + 4096, with row bounds read outside the rescanned block and output
    written without the relative-index conversion. Block 0 is forced into the
    rescan (a subnormal weight takes the exact path); block 4096 belongs to a
    second request (nonzero starts, relative=True) and must stay exact.
    """
    rows, keys_per_request = 33000, 8192  # 4125 blocks of 8 rows
    split = rows // 2
    q, k, scales, weights = _inputs(rows, 2 * keys_per_request, seed=927)
    weights[0, 0] = 1e-40  # subnormal fp32: block 0 takes the exact rescan
    starts = torch.zeros(rows, device="cuda", dtype=torch.int32)
    starts[split:] = keys_per_request
    ends = starts + keys_per_request
    out = torch.full((rows, TOPK), -7, device="cuda", dtype=torch.int32)
    fused_mqa_topk_prefill(q, k, scales, weights, starts, ends, out)
    _check(out, q, k, scales, weights, starts, ends, scores=False)
    # Scores of the rescanned block, the block 4096 blocks after it, and the
    # last block.
    check_rows = torch.cat(
        [
            torch.arange(0, 8),
            torch.arange(4096 * 8, 4097 * 8),
            torch.arange(rows - 8, rows),
        ]
    ).cuda()
    _check(
        out[check_rows],
        q[check_rows],
        k,
        scales,
        weights[check_rows],
        starts[check_rows],
        ends[check_rows],
    )


@requires_fused
def test_cuda_graph_replay():
    rows, keys = 64, 65536
    q, k, scales, weights = _inputs(rows, keys)
    starts = torch.zeros(rows, device="cuda", dtype=torch.int32)
    ends = torch.arange(keys - rows + 1, keys + 1, device="cuda", dtype=torch.int32)
    out = torch.empty(rows, TOPK, device="cuda", dtype=torch.int32)
    fn = lambda: fused_mqa_topk_prefill(  # noqa: E731
        q, k, scales, weights, starts, ends, out
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
def test_dispatch_predicate():
    """use_fused_mqa_topk: FP8 32x128 queries, top-k 512/1024/2048, full
    context, no FP4 cache / FP4 queries / candidate selection."""
    q = _inputs(16, 4096)[0]
    ok = dict(fp4_cache=False, full_context=True)
    assert use_fused_mqa_topk(q, None, 2048, **ok)
    assert not use_fused_mqa_topk(q, None, 768, **ok)
    assert not use_fused_mqa_topk(q[:, :16], None, 2048, **ok)
    assert not use_fused_mqa_topk(q.float(), None, 2048, **ok)
    assert not use_fused_mqa_topk(q, torch.ones(16, device="cuda"), 2048, **ok)
    assert not use_fused_mqa_topk(q, None, 2048, fp4_cache=True, full_context=True)
    assert not use_fused_mqa_topk(q, None, 2048, fp4_cache=False, full_context=False)
    assert not use_fused_mqa_topk(q, None, 2048, candidate_selection=True, **ok)


@requires_fused
def test_op_rejects_bad_inputs():
    """The launcher does not re-check; the op enforces its contract."""
    q, k, scales, weights = _inputs(16, 4096)
    starts = torch.zeros(16, device="cuda", dtype=torch.int32)
    ends = torch.full((16,), 4096, device="cuda", dtype=torch.int32)
    out768 = torch.empty(16, 768, device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="out must be"):
        fused_mqa_topk_prefill(q, k, scales, weights, starts, ends, out768)
    wide = torch.empty(16, 2 * TOPK, device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="must be contiguous"):
        fused_mqa_topk_prefill(q, k, scales, weights, starts, ends, wide[:, :TOPK])


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


def _run_kpool_indexer(monkeypatch, inputs, fused, budget_mb=512):
    """Run the GLM-5.3-Flash kpool indexer prefill on one packed chunk."""
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
    fused_fn = kpool_indexer.fused_mqa_topk_prefill

    def spy(*a, **kw):
        calls.append(a[-1].shape)  # the output buffer: [rows, top-k pools]
        fused_fn(*a, **kw)

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
    inputs = _kpool_inputs(pool_lens=(9000, 20000))
    assert int((inputs[5] - inputs[4]).max()) < fused_mqa_topk_min_keys(TOPK // KPOOL)
    _, calls = _run_kpool_indexer(monkeypatch, inputs, fused=True)
    assert calls == []


@requires_fused
def test_kpool_indexer_dense_fallback_splits_rows(monkeypatch):
    """Fused-planned chunks skip the logits budget; when such a chunk takes the
    dense path, its rows are split to stay within the budget and select the
    same pools as one unbudgeted dense pass."""
    inputs = _kpool_inputs()
    whole, _ = _run_kpool_indexer(monkeypatch, inputs, fused=False)
    split, calls = _run_kpool_indexer(monkeypatch, inputs, fused=False, budget_mb=1)
    assert calls == []
    # top_k_per_row_prefill breaks bf16 ties at the k-th score arbitrarily (two
    # identical dense runs differ too), so compare the selected scores.
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
