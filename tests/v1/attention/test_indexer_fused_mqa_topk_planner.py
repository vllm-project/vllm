# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill chunk planning and dispatch predicates of the fused MQA+top-k
indexer path (CPU only; the kernel is tested in
tests/kernels/attention/test_fused_mqa_topk.py)."""

import pytest
import torch

import vllm.envs as envs
import vllm.model_executor.kernels.attention.dsa.fused_mqa_topk as fused_mqa_topk
from vllm.model_executor.kernels.attention.dsa.fused_mqa_topk import (
    FUSED_MQA_TOPK_MAX_KEYS,
    FUSED_MQA_TOPK_MAX_ROWS,
    fused_mqa_topk_chunk,
    fused_mqa_topk_min_keys,
    use_fused_mqa_topk,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadataBuilder,
    dense_logits_rows,
)

LOGITS_BUDGET = 512 * 1024**2


def _split(seq_lens, q_lens, fused, offset=0, workspace=1048576):
    return DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks(
        torch.tensor(seq_lens),
        torch.tensor(q_lens),
        workspace,
        LOGITS_BUDGET,
        request_offset=offset,
        plan_fused=fused,
    )


@pytest.mark.parametrize("q_rows", [16384, 32768])
@pytest.mark.parametrize("fused", [False, True])
def test_planner_one_fused_call_per_chunk(fused, q_rows):
    # One fused call per FUSED_MQA_TOPK_MAX_ROWS rows; dense splits by the
    # logits budget.
    chunks = _split([524288], [q_rows], fused)
    expected = -(-q_rows // FUSED_MQA_TOPK_MAX_ROWS) if fused else q_rows // 256
    assert len(chunks) == expected
    assert sum(q.stop - q.start for _, q in chunks) == q_rows


def test_planner_fused_any_length_and_short_queries():
    # No length threshold and no minimum query count on the fused path.
    assert _split([524288], [512], True) == [(slice(0, 1), slice(0, 512))]
    assert _split([4096], [4096], True) == [(slice(0, 1), slice(0, 4096))]
    # A 40000-row step is split at FUSED_MQA_TOPK_MAX_ROWS.
    cap = FUSED_MQA_TOPK_MAX_ROWS
    assert _split([65536], [40000], True) == [
        (slice(0, 1), slice(lo, min(lo + cap, 40000))) for lo in range(0, 40000, cap)
    ]


def test_planner_packs_requests_in_one_fused_call():
    # Requests share one fused call up to the row cap; dense needs a split.
    assert _split([4096] * 4, [4096] * 4, True) == [(slice(0, 4), slice(0, 16384))]
    assert len(_split([4096] * 4, [4096] * 4, False)) > 1
    # The second request would push the call past the row cap.
    assert _split([524288, 262144], [16384, 9], True, offset=2) == [
        (slice(2, 3), slice(0, 16384)),
        (slice(3, 4), slice(0, 9)),
    ]


def test_planner_fused_never_packs_past_key_limit():
    # Two 600K-key continuation steps of 50 rows: dense packs them into one
    # logits-budget chunk; the fused planner keeps one fused chunk per request so
    # no chunk reaches the dense path (whose logits buffer is not reserved).
    args = ([600_000, 600_000], [50, 50], False, 0, 4_000_000)
    assert _split(*args) == [(slice(0, 2), slice(0, 100))]
    fused = (*args[:2], True, *args[3:])
    assert _split(*fused) == [(slice(0, 1), slice(0, 50)), (slice(1, 2), slice(0, 50))]


def test_planner_keeps_budget_beyond_fused_key_limit():
    # A single request longer than the fused key limit is budget-planned.
    fused = _split([2_000_000], [256], True, workspace=4_000_000)
    dense = _split([2_000_000], [256], False, workspace=4_000_000)
    assert fused == dense and len(dense) == 4


@pytest.mark.parametrize("topk", [512, 1024, 2048])
def test_fused_chunk_thresholds(topk):
    min_keys = fused_mqa_topk_min_keys(topk)
    assert not fused_mqa_topk_chunk(min_keys - 1, min_keys, topk)
    assert fused_mqa_topk_chunk(min_keys, min_keys, topk)
    assert fused_mqa_topk_chunk(min_keys, FUSED_MQA_TOPK_MAX_KEYS, topk)
    # Packed gathered keys past the kernel's limit stay dense.
    assert not fused_mqa_topk_chunk(min_keys, FUSED_MQA_TOPK_MAX_KEYS + 1, topk)


@pytest.mark.parametrize(
    "heads, head_dim, topk, kpool, supported",
    [
        (32, 128, 2048, 1, True),  # GLM-5.2
        (32, 128, 512, 1, True),  # DeepSeek-V4.1-Flash
        (32, 128, 2048, 4, True),  # GLM-5.3-Flash: top-512 pools
        (64, 128, 2048, 1, False),  # DeepSeek-V3.2: 64 index heads
        (32, 64, 2048, 1, False),
        (32, 128, 256, 1, False),
        (32, 128, 2048, 3, False),  # 682 pools
    ],
)
def test_use_fused_mqa_topk(monkeypatch, heads, head_dim, topk, kpool, supported):
    """The single eligibility rule of the planner and the indexer layers."""
    monkeypatch.setattr(fused_mqa_topk, "fused_mqa_topk_available", lambda: True)
    monkeypatch.setattr(envs, "VLLM_USE_FUSED_MQA_TOPK", True)
    ok = dict(
        query_shape=(heads, head_dim),
        topk=topk // kpool,
        fp4_cache=False,
        full_context=True,
    )
    assert use_fused_mqa_topk(**ok) == supported
    if supported:
        assert not use_fused_mqa_topk(**{**ok, "fp4_cache": True})
        assert not use_fused_mqa_topk(**{**ok, "full_context": False})
        assert not use_fused_mqa_topk(**ok, candidate_selection=True)
        # A flattened [tokens, heads * head_dim] query does not match.
        assert not use_fused_mqa_topk(**{**ok, "query_shape": (heads * head_dim,)})
        monkeypatch.setattr(envs, "VLLM_USE_FUSED_MQA_TOPK", False)
        assert not use_fused_mqa_topk(**ok)
        monkeypatch.setattr(fused_mqa_topk, "fused_mqa_topk_available", lambda: False)
        monkeypatch.setattr(envs, "VLLM_USE_FUSED_MQA_TOPK", True)
        assert not use_fused_mqa_topk(**ok)


def test_dense_logits_rows_fits_budget(monkeypatch):
    """Dense row splitting keeps each call's fp32 [rows, max_seqlen_k] logits
    within the budget, and never splits a chunk the budget planner made."""
    monkeypatch.setattr(envs, "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", 512)
    for width in (1, 4096, 65536, 1 << 20):
        rows = dense_logits_rows(width)
        assert rows * width * 4 <= LOGITS_BUDGET
        assert (rows + 1) * width * 4 > LOGITS_BUDGET
    # Budget-planned chunks have rows * total_keys <= budget / 4 and every
    # row's max_seqlen_k <= total_keys, so they fit in one dense call.
    seq_lens = [131072, 65536, 4096]
    for req_slice, q_slice in _split(seq_lens, [4096, 4096, 4096], False):
        total_keys = sum(seq_lens[req_slice])
        assert q_slice.stop - q_slice.start <= dense_logits_rows(total_keys)
    assert dense_logits_rows(0) == LOGITS_BUDGET // 4
