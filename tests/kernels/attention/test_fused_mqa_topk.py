# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused LiteTopK MQA-logits + top-k prefill vs the dense indexer path."""

import functools
from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.model_executor.kernels.attention.dsa.fused_mqa_topk import (
    FUSED_MQA_TOPK_MAX_KEYS,
    FUSED_MQA_TOPK_WINDOW_ROWS,
    fused_mqa_topk_available,
    fused_mqa_topk_chunk,
    fused_mqa_topk_prefill,
    get_fused_mqa_topk_workspaces,
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

requires_fused_mqa_topk = pytest.mark.skipif(
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
        q_scale, w = mqa_logits_mx_inputs(q[i:j], None, weights[i:j], torch.float32)
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

    `scores` compares the selected scores with the exact fp32 top-k; the
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


def _inputs(rows, keys, seed=913, heads=32):
    torch.manual_seed(seed)
    q = torch.randn(rows, heads, 128, device="cuda").to(torch.float8_e4m3fn)
    k = torch.randn(keys, 128, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.ones(keys, device="cuda")
    weights = torch.randn(rows, heads, device="cuda") / heads
    return q, k, scales, weights


def _call(q, k, scales, weights, starts, ends, out):
    """One op call; returns its workspaces (row_state is the fifth)."""
    workspaces = get_fused_mqa_topk_workspaces(
        q.device, q.shape[0], q.shape[1], k.shape[0]
    )
    torch.ops._C.fused_mqa_topk_prefill(
        q, k, scales, weights, starts, ends, out, *workspaces
    )
    return workspaces


@requires_fused_mqa_topk
@pytest.mark.parametrize("heads", [32, 64])
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
def test_single_request_matches_dense(keys, rows, case, topk, heads):
    q, k, scales, weights = _inputs(rows, keys, heads=heads)
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


@requires_fused_mqa_topk
@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("topk", TOPKS)
@pytest.mark.parametrize(
    "lens", [[3000, 5000, 1000], [40000, 1500, 70000, 9], [600000, 200000]]
)
def test_multi_request_chunk_matches_dense(lens, topk, heads):
    """Packed requests: rows of request r see [cu[r], cu[r] + causal_end)."""
    q_lens = [min(n, 1024) for n in lens]
    rows, keys = sum(q_lens), sum(lens)
    q, k, scales, weights = _inputs(rows, keys, seed=917, heads=heads)
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
    # (~98.6% overlap for top-512 over 600K keys; the fused_mqa_topk keys are
    # exact).
    overlap = sum(
        len(set(a[a >= 0].tolist()) & set(b[b >= 0].tolist()))
        for a, b in zip(out.cpu(), dense.cpu())
    )
    assert overlap >= 0.98 * int((dense >= 0).sum())


@requires_fused_mqa_topk
@pytest.mark.parametrize("rows", [32768, 32770])
def test_large_chunk_single_call(rows):
    """Prefill chunks above 16384 rows (e.g. --max-num-batched-tokens 32768)
    run in one call: op calls over row windows of FUSED_MQA_TOPK_WINDOW_ROWS
    sharing one set of workspaces."""
    keys = 65536
    q, k, scales, weights = _inputs(rows, keys, seed=921)
    starts = torch.zeros(rows, device="cuda", dtype=torch.int32)
    ends = torch.arange(keys - rows + 1, keys + 1, device="cuda", dtype=torch.int32)
    out = torch.empty(rows, TOPK, device="cuda", dtype=torch.int32)
    fused_mqa_topk_prefill(q, k, scales, weights, starts, ends, out)
    _check(out, q, k, scales, weights, starts, ends)


@requires_fused_mqa_topk
@pytest.mark.parametrize("topk", TOPKS)
@pytest.mark.parametrize("heads", [32, 64])
def test_recovery_with_period_aligned_keys(heads, topk):
    """Keys repeating with the period of the threshold sample (a stride of 65
    keys up to 512K keys of span) put the same key in every sample of a block,
    so most of its rows get a threshold far too high or too low: their
    candidates fall short of the top-k or overflow the slab, and the recovery
    rescores them exactly. The other rows of that request reach the selector
    with hundreds of equal scores at the k-th. The period-aligned request is
    the second of a packed chunk (nonzero starts: relative indices)."""
    rows, keys = 128, 16384  # per request
    q, k, scales, weights = _inputs(2 * rows, 2 * keys, seed=929, heads=heads)
    k = k.float()
    k[keys:] = k[keys : keys + 65].repeat(-(-keys // 65), 1)[:keys]
    k = k.to(torch.float8_e4m3fn)
    starts = torch.tensor([0] * rows + [keys] * rows, device="cuda", dtype=torch.int32)
    causal = torch.arange(keys - rows + 1, keys + 1, device="cuda", dtype=torch.int32)
    ends = starts + causal.repeat(2)
    out = torch.full((2 * rows, topk), -7, device="cuda", dtype=torch.int32)
    workspaces = _call(q, k, scales, weights, starts, ends, out)
    _check(out, q, k, scales, weights, starts, ends)
    counts, _, overflow = workspaces[4][:, rows:]
    recovered = (counts < topk) | (counts > workspaces[2].shape[1]) | (overflow != 0)
    assert recovered.any() and not recovered.all()


@requires_fused_mqa_topk
@pytest.mark.parametrize("heads", [32, 64])
def test_repeated_calls_select_the_same_keys(heads):
    """Small chunks split each block's scan into pieces that append their
    candidates in atomic order; rows without a tie at the k-th of the
    producer's candidate scores (the selector takes tied candidates in slab
    order) select the same keys in every call."""
    rows, keys = 64, 262144
    q, k, scales, weights = _inputs(rows, keys, seed=931, heads=heads)
    starts = torch.zeros(rows, device="cuda", dtype=torch.int32)
    ends = torch.arange(keys - rows + 1, keys + 1, device="cuda", dtype=torch.int32)
    selected = []
    for _ in range(3):
        out = torch.full((rows, TOPK), -7, device="cuda", dtype=torch.int32)
        workspaces = _call(q, k, scales, weights, starts, ends, out)
        _check(out, q, k, scales, weights, starts, ends, scores=False)
        selected.append(out.sort(dim=1).values)
    _check(out, q, k, scales, weights, starts, ends)
    values, counts = workspaces[2], workspaces[4][0]
    cap = values.shape[1]
    kept = torch.arange(cap, device="cuda") < counts[:, None]
    kth = values.masked_fill(~kept, -torch.inf).topk(TOPK + 1, dim=1).values
    untied = (counts >= TOPK) & (counts <= cap) & (kth[:, -2] != kth[:, -1])
    assert untied.float().mean() > 0.9
    assert all(torch.equal(selected[0][untied], s[untied]) for s in selected[1:])


@requires_fused_mqa_topk
@pytest.mark.parametrize("shared", [8.0, 256.0])
def test_gate_with_common_score_offset(shared):
    """Regression test: scores sharing a large common offset with a narrow top
    tail (top ranks within ~1 of each other at ~+68, or at ~+65540, past the
    fp16 range) must still get a tight threshold. The sample codes are fp16
    distances to a sampled score of the row; codes of the scores themselves
    would be 16 apart at ~+20K and saturate past 65504, and every row would
    overflow the candidate slab and take the recovery."""
    rows, keys, topk = 2048, 65536, TOPK
    q, k, scales, weights = _inputs(rows, keys, seed=5)
    # A shared component in dimension 0 adds shared * shared to every q.k.
    q = q.float()
    k = k.float()
    q[:, :, 0] = shared
    k[:, 0] = shared
    q, k = q.to(torch.float8_e4m3fn), k.to(torch.float8_e4m3fn)
    weights = torch.full_like(weights, 1.0 / 32)
    starts = torch.zeros(rows, device="cuda", dtype=torch.int32)
    ends = torch.arange(keys - rows + 1, keys + 1, device="cuda", dtype=torch.int32)
    out = torch.empty(rows, topk, device="cuda", dtype=torch.int32)
    workspaces = _call(q, k, scales, weights, starts, ends, out)
    _check(out, q, k, scales, weights, starts, ends)
    # Per-row candidate counts the producer left in its workspace: about the
    # sample's target (~4.2K for top-2048 at 64K keys), no row recovered.
    counts, _, overflow = workspaces[4]
    assert (counts >= topk).all() and (counts <= workspaces[2].shape[1]).all()
    assert (overflow == 0).all(), "rows recovered"
    assert counts.float().median() < 3 * topk


@requires_fused_mqa_topk
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


@requires_fused_mqa_topk
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
    # A misaligned candidate-key slab is rejected, not a device fault.
    workspaces = get_fused_mqa_topk_workspaces(q.device, 16, 32, 4096)
    rows, cap = workspaces[3].shape
    flat = torch.empty(rows * cap + 4, device="cuda", dtype=torch.int32)
    workspaces[3] = flat[1 : 1 + rows * cap].view(rows, cap)
    out = torch.empty(16, TOPK, device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="indices must be 16-byte aligned"):
        torch.ops._C.fused_mqa_topk_prefill(
            q, k, scales, weights, starts, ends, out, *workspaces
        )


@pytest.mark.parametrize(
    "max_seqlen_k,rows,heads,topk,fused",
    [
        (32768, 1480, 32, 2048, True),  # a short last wave of row blocks
        (1 << 20, 8, 32, 2048, True),
        (131072, 256, 64, 512, True),
        (28671, 16384, 32, 2048, False),  # below the top-2048 threshold
        (20480, 512, 32, 512, False),  # small chunks need 64K keys
        (49152, 256, 64, 512, False),
    ],
)
def test_fused_mqa_topk_chunk(max_seqlen_k, rows, heads, topk, fused):
    """Gate decisions at chunk shapes measured on B200 (cold L2)."""
    gate = functools.partial(fused_mqa_topk_chunk, heads=heads)
    assert gate(max_seqlen_k, max_seqlen_k, topk, rows) is fused
    assert not gate(max_seqlen_k, FUSED_MQA_TOPK_MAX_KEYS + 1, topk, rows)


@pytest.mark.parametrize(
    "topk,heads,min_keys",
    [
        (512, 32, 20480),
        (1024, 32, 24576),
        (2048, 32, 28672),
        (512, 64, 24576),
        (1024, 64, 28672),
        (2048, 64, 32768),
    ],
)
def test_fused_mqa_topk_chunk_boundaries(topk, heads, min_keys):
    """The gate's thresholds at 1024 rows and its 64K floor below 1024 rows."""
    gate = functools.partial(fused_mqa_topk_chunk, heads=heads)
    assert gate(min_keys, min_keys, topk, 1024)
    assert not gate(min_keys - 1, min_keys, topk, 1024)
    assert not gate(min_keys, min_keys, topk, 1023)
    assert gate(1 << 16, 1 << 16, topk, 1023)
    assert not gate((1 << 16) - 1, 1 << 16, topk, 1023)


@pytest.mark.parametrize("heads", [32, 64])
def test_profile_workspaces_cover_every_call(monkeypatch, heads):
    """The profile run reserves get_fused_mqa_topk_workspaces(device, tokens):
    one op call over at most FUSED_MQA_TOPK_WINDOW_ROWS rows at the largest
    head count and keys, within the dense prefill logits budget. No call over
    fewer rows or keys, or another head count, needs more of any."""
    monkeypatch.setattr(
        "vllm.model_executor.kernels.attention.dsa.fused_mqa_topk._sm_count",
        lambda index: 148,
    )

    def nbytes(*args):
        workspaces = get_fused_mqa_topk_workspaces(torch.device("meta", 0), *args)
        return [t.nbytes for t in workspaces]

    row_counts = (1, 3, 4, 5, 8, 9, 100, 1000, FUSED_MQA_TOPK_WINDOW_ROWS, 16384)
    reserved = [nbytes(rows) for rows in row_counts]
    for fewer, more in zip(reserved, reserved[1:]):
        assert all(a <= b for a, b in zip(fewer, more))
    assert reserved[-2] == reserved[-1] and sum(reserved[-1]) < LOGITS_BUDGET
    for rows, peak in zip(row_counts, reserved):
        for keys in (1, 4097, 65536, 262145, 786432, FUSED_MQA_TOPK_MAX_KEYS):
            assert all(a <= b for a, b in zip(nbytes(rows, heads, keys), peak))


KPOOL = 4


def _kpool_inputs(pool_lens=(20000, 70000), q_rows=96, tail=2, seed=919, heads=32):
    """Two packed GLM-5.3-Flash-style requests: K holds 4-token pools; the
    last `q_rows` tokens of each request score the pools they can see. Fewer
    than 32 heads are zero-padded to 32 (q and weights), like the indexer of
    the GLM-5.3-Flash checkpoint (16 heads)."""
    rows = q_rows * len(pool_lens)
    q, k, _, weights = _inputs(rows, sum(pool_lens), seed, heads)
    if heads < 32:
        q = torch.cat([q.float(), q.new_zeros(rows, 32 - heads, 128).float()], 1)
        q = q.to(torch.float8_e4m3fn)
        weights = torch.cat([weights, weights.new_zeros(rows, 32 - heads)], 1)
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


def _run_kpool_indexer(monkeypatch, inputs, fused_mqa_topk, budget_mb=512):
    """Run the GLM-5.3-Flash kpool indexer prefill on one packed chunk."""
    import vllm.models.glm5next.nvidia.sparse_indexer as kpool_indexer

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
    monkeypatch.setattr(envs, "VLLM_USE_FUSED_MQA_TOPK", fused_mqa_topk)
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


@requires_fused_mqa_topk
@pytest.mark.parametrize("heads", [16, 32, 64])
def test_kpool_indexer_fused_matches_dense(monkeypatch, heads):
    """GLM-5.3-Flash kpool indexer: the fused_mqa_topk kernel ranks the top-512 pools of
    a packed multi-request chunk like the dense logits + top-k path."""
    inputs = _kpool_inputs(heads=heads)
    # 192 rows take the kernel from 64K pools (70000 in the longest request).
    fused_mqa_topk_out, calls = _run_kpool_indexer(
        monkeypatch, inputs, fused_mqa_topk=True
    )
    assert calls == [torch.Size([inputs[0].shape[0], TOPK // KPOOL])]
    dense, calls = _run_kpool_indexer(monkeypatch, inputs, fused_mqa_topk=False)
    assert calls == []
    assert ((fused_mqa_topk_out >= 0).sum(1) == (dense >= 0).sum(1)).all()
    # Same pools up to the dense path's bf16 boundary blur (see above); every
    # row also keeps its request's incomplete tail tokens.
    overlap = sum(
        len(a & b) for a, b in zip(_row_sets(fused_mqa_topk_out), _row_sets(dense))
    )
    assert overlap >= 0.98 * int((dense >= 0).sum())


@requires_fused_mqa_topk
@pytest.mark.parametrize(
    "pool_lens,q_rows", [((9000, 20000), 512), ((9000, 40000), 96)]
)
def test_kpool_indexer_short_chunk_stays_dense(monkeypatch, pool_lens, q_rows):
    """Chunks the gate keeps dense take the dense path: 1024 rows below
    fused_mqa_topk_min_keys pools, and 192 rows below 64K pools."""
    inputs = _kpool_inputs(pool_lens=pool_lens, q_rows=q_rows)
    q, k, _, _, starts, ends, _ = inputs
    longest = int((ends - starts).max())
    assert not fused_mqa_topk_chunk(
        longest, k.shape[0], TOPK // KPOOL, q.shape[0], heads=32
    )
    _, calls = _run_kpool_indexer(monkeypatch, inputs, fused_mqa_topk=True)
    assert calls == []


@requires_fused_mqa_topk
def test_kpool_indexer_dense_fallback_splits_rows(monkeypatch):
    """Chunks planned for fused_mqa_topk skip the logits budget; when such a
    chunk takes the dense path, its rows are split to stay within the budget
    and select the
    same pools as one unbudgeted dense pass."""
    inputs = _kpool_inputs()
    whole, _ = _run_kpool_indexer(monkeypatch, inputs, fused_mqa_topk=False)
    split, calls = _run_kpool_indexer(
        monkeypatch, inputs, fused_mqa_topk=False, budget_mb=1
    )
    assert calls == []
    # top_k_per_row_prefill breaks bf16 ties at the k-th score arbitrarily (two
    # identical dense runs differ too), so compare the selected scores.
    q, k, scales, weights, starts, ends, _ = inputs
    lens = ends - starts
    q_scale, w = mqa_logits_mx_inputs(q, None, weights, torch.float32)
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
