# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the ZoomKV Top-K recall probe math."""

from __future__ import annotations

import math

import torch

from vllm.v1.attention.ops.zoomkv.recall_probe import compute_recall_record

BLOCK_SIZE = 16


def _make_case(
    seq_len: int = 160,
    num_kv_heads: int = 2,
    group: int = 4,
    head_dim: int = 32,
    hot_positions: tuple[int, ...] = (40, 55, 70),
):
    """Key cache where ``hot_positions`` strongly align with the query."""
    torch.manual_seed(0)
    num_blocks = seq_len // BLOCK_SIZE + 2
    key_cache = 0.01 * torch.randn(
        num_blocks, BLOCK_SIZE, num_kv_heads, head_dim, dtype=torch.float32
    )
    # Identity block table: logical block i -> physical block i.
    block_table = torch.arange(num_blocks, dtype=torch.int32)
    query = torch.randn(1, num_kv_heads * group, head_dim, dtype=torch.float32)

    q_per_kv = query[0].view(num_kv_heads, group, head_dim).mean(dim=1)
    for pos in hot_positions:
        b, off = divmod(pos, BLOCK_SIZE)
        for h in range(num_kv_heads):
            key_cache[b, off, h] = 10.0 * q_per_kv[h] / q_per_kv[h].norm()
    return query, key_cache, block_table


def test_perfect_retrieval_has_recall_one():
    query, key_cache, block_table = _make_case()
    seq_len = 160
    start_block, end_block = 1, 8  # zone tokens [16, 128)
    hot = (40, 55, 70)
    k = len(hot)
    topk = torch.tensor([list(hot)] * key_cache.shape[2], dtype=torch.int64)
    rec = compute_recall_record(
        query,
        key_cache,
        block_table,
        BLOCK_SIZE,
        seq_len,
        start_block,
        end_block,
        topk,
        scale=1.0 / math.sqrt(query.shape[-1]),
    )
    assert rec is not None
    assert rec["k"] == k
    assert rec["recall"] == [1.0] * key_cache.shape[2]
    assert rec["recall_mean"] == 1.0
    # The three planted tokens carry the bulk of the zone's attention mass.
    for cov, oracle in zip(rec["mass_coverage"], rec["oracle_mass_coverage"]):
        assert cov > 0.5
        assert oracle >= cov - 1e-6


def test_retrieval_query_ground_truth():
    query, key_cache, block_table = _make_case()
    kv = key_cache.shape[2]
    hot = (40, 55, 70)
    topk = torch.tensor([list(hot)] * kv, dtype=torch.int64)
    # The retriever's aggregated query: group mean, matching _make_case's
    # construction, so the hot tokens are also the exact q.K Top-K.
    rq = query[0].view(kv, -1, query.shape[-1]).mean(dim=1).unsqueeze(0)
    rec = compute_recall_record(
        query,
        key_cache,
        block_table,
        BLOCK_SIZE,
        160,
        1,
        8,
        topk,
        scale=1.0 / math.sqrt(query.shape[-1]),
        retrieval_query=rq,
    )
    assert rec is not None
    assert rec["recall_vs_rq"] == [1.0] * kv
    assert rec["recall_vs_rq_mean"] == 1.0


def test_disjoint_retrieval_has_recall_zero():
    query, key_cache, block_table = _make_case()
    # Retrieved tokens are in the zone but none of them is a hot token.
    topk = torch.tensor([[17, 18, 19]] * key_cache.shape[2], dtype=torch.int64)
    rec = compute_recall_record(
        query,
        key_cache,
        block_table,
        BLOCK_SIZE,
        160,
        1,
        8,
        topk,
        scale=1.0 / math.sqrt(query.shape[-1]),
    )
    assert rec is not None
    assert rec["recall"] == [0.0] * key_cache.shape[2]
    for cov in rec["mass_coverage"]:
        assert cov < 0.1


def test_padding_and_partial_hits():
    query, key_cache, block_table = _make_case()
    kv = key_cache.shape[2]
    # One hit (40), one miss (20), one -1 pad; k_eff stays 3.
    topk = torch.tensor([[40, 20, -1]] * kv, dtype=torch.int64)
    rec = compute_recall_record(
        query,
        key_cache,
        block_table,
        BLOCK_SIZE,
        160,
        1,
        8,
        topk,
        scale=1.0 / math.sqrt(query.shape[-1]),
    )
    assert rec is not None
    assert rec["k"] == 3
    for r in rec["recall"]:
        assert abs(r - 1.0 / 3.0) < 1e-6


def test_empty_zone_returns_none():
    query, key_cache, block_table = _make_case()
    rec = compute_recall_record(
        query,
        key_cache,
        block_table,
        BLOCK_SIZE,
        160,
        4,
        4,  # empty zone
        torch.full((key_cache.shape[2], 3), -1, dtype=torch.int64),
        scale=1.0,
    )
    assert rec is None


def test_zone_mass_fraction_sums_sanely():
    query, key_cache, block_table = _make_case()
    kv = key_cache.shape[2]
    topk = torch.tensor([[40, 55, 70]] * kv, dtype=torch.int64)
    rec = compute_recall_record(
        query,
        key_cache,
        block_table,
        BLOCK_SIZE,
        160,
        1,
        8,
        topk,
        scale=1.0 / math.sqrt(query.shape[-1]),
    )
    assert rec is not None
    for frac in rec["zone_mass_frac"]:
        assert 0.0 < frac <= 1.0
