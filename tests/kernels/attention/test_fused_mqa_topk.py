# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-launch fused MQA-logits + top-k prefill vs the dense indexer path."""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.kernels.attention.dsa.fused_mqa_topk import (
    fused_mqa_topk_available,
    fused_mqa_topk_prefill,
)
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder

requires_fused = pytest.mark.skipif(
    not torch.cuda.is_available() or not fused_mqa_topk_available(),
    reason="Requires the SM100 fused MQA+top-k extension",
)

TOPK = 2048
LOGITS_BUDGET = 512 * 1024**2


def _dense(q, k, scales, weights, starts, ends):
    """Upstream dense path: budgeted DeepGEMM logits + top_k_per_row_prefill."""
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

    rows = q.shape[0]
    out = torch.full((rows, TOPK), -1, device=q.device, dtype=torch.int32)
    step = max(1, LOGITS_BUDGET // (4 * k.shape[0]))
    for i in range(0, rows, step):
        j = min(rows, i + step)
        logits = fp8_fp4_mqa_logits(
            (q[i:j], None),
            (k, scales),
            weights[i:j],
            starts[i:j],
            ends[i:j],
            clean_logits=False,
        )
        ops.top_k_per_row_prefill(
            logits,
            starts[i:j],
            ends[i:j],
            out[i:j],
            j - i,
            logits.stride(0),
            logits.stride(1),
            TOPK,
        )
    return out


def _check(rel, q, k, scales, weights, starts, ends):
    """Indices relative to the row start: count, bounds, uniqueness, scores."""
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

    visible = (ends - starts).clamp(min=0)
    valid = rel >= 0
    assert (valid.sum(1) == visible.clamp(max=TOPK)).all(), "valid count"
    assert (rel[~valid] == -1).all(), "padding must be -1"
    idx = rel.long() + starts[:, None].long()
    assert ((~valid) | ((idx >= starts[:, None]) & (idx < ends[:, None]))).all()
    srt = idx.masked_fill(~valid, -1).sort().values
    assert ((srt[:, 1:] < 0) | (srt.diff(dim=1) > 0)).all(), "duplicates"
    # Check scores in row blocks so the reference logits stay small.
    step = max(1, LOGITS_BUDGET // (4 * k.shape[0]))
    for i in range(0, q.shape[0], step):
        j = min(q.shape[0], i + step)
        logits = fp8_fp4_mqa_logits(
            (q[i:j], None),
            (k, scales),
            weights[i:j],
            starts[i:j],
            ends[i:j],
            clean_logits=True,
        )
        ref = logits.topk(TOPK, dim=1).values
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
def test_single_request_matches_dense(keys, rows, case):
    q, k, scales, weights = _inputs(rows, keys)
    starts = torch.zeros(rows, device="cuda", dtype=torch.int32)
    ends = torch.arange(keys - rows + 1, keys + 1, device="cuda", dtype=torch.int32)
    if case == "early":  # rows with fewer than 2048 visible keys
        ends = torch.tensor(
            [
                0,
                1,
                31,
                127,
                1024,
                2047,
                2048,
                2049,
                4096,
                6144,
                8192,
                10000,
                12000,
                14000,
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
    out = torch.empty(rows, TOPK, device="cuda", dtype=torch.int32)
    assert fused_mqa_topk_prefill(q, k, scales, weights, starts, ends, out)
    _check(out, q, k, scales, weights, starts, ends)
    # Same convention and valid counts as the dense path (key sets may differ
    # only among equal scores at the top-k boundary).
    dense = _dense(q, k, scales, weights, starts, ends)
    _check(dense, q, k, scales, weights, starts, ends)
    assert ((out >= 0).sum(1) == (dense >= 0).sum(1)).all()


@requires_fused
@pytest.mark.parametrize(
    "lens", [[3000, 5000, 1000], [40000, 1500, 70000, 9], [600000, 200000]]
)
def test_multi_request_chunk_matches_dense(lens):
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
    out = torch.empty(rows, TOPK, device="cuda", dtype=torch.int32)
    assert fused_mqa_topk_prefill(q, k, scales, weights, starts, ends, out)
    _check(out, q, k, scales, weights, starts, ends)
    dense = _dense(q, k, scales, weights, starts, ends)
    _check(dense, q, k, scales, weights, starts, ends)
    assert ((out >= 0).sum(1) == (dense >= 0).sum(1)).all()
    # Both paths agree on (nearly) every selected key.
    overlap = sum(
        len(set(a[a >= 0].tolist()) & set(b[b >= 0].tolist()))
        for a, b in zip(out.cpu(), dense.cpu())
    )
    assert overlap >= 0.99 * int((dense >= 0).sum())


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
    assert not fused_mqa_topk_prefill(
        q, k, scales, weights, starts, ends, out[:, :1024]
    )
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


@pytest.mark.parametrize("fused,expected", [(False, 64), (True, 1)])
def test_planner_one_fused_call_per_16k_rows(fused, expected):
    chunks = _split([524288], [16384], fused)
    assert len(chunks) == expected
    assert sum(q.stop - q.start for _, q in chunks) == 16384


def test_planner_fused_any_length_and_short_queries():
    # No length threshold and no minimum query count on the fused path.
    assert _split([524288], [512], True) == [(slice(0, 1), slice(0, 512))]
    assert _split([4096], [4096], True) == [(slice(0, 1), slice(0, 4096))]
    # Long single-request steps are split at 16384 query rows.
    assert _split([65536], [40000], True) == [
        (slice(0, 1), slice(0, 16384)),
        (slice(0, 1), slice(16384, 32768)),
        (slice(0, 1), slice(32768, 40000)),
    ]


def test_planner_packs_requests_up_to_16k_rows():
    # Four 4K requests fit one fused call (16384 rows); dense needs a split.
    assert _split([4096] * 4, [4096] * 4, True) == [(slice(0, 4), slice(0, 16384))]
    assert len(_split([4096] * 4, [4096] * 4, False)) > 1
    chunks = _split([524288, 262144], [16384, 9], True, offset=2)
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
