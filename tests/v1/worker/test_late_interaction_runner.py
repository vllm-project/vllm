# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.entrypoints.pooling.scoring.utils import compute_maxsim_score
from vllm.pooling_params import LateInteractionParams, PoolingParams
from vllm.v1.pool.late_interaction import (
    LATE_INTERACTION_MODE_CACHE_QUERY,
    build_late_interaction_doc_params,
    build_late_interaction_query_params,
    compute_maxsim_score_batched,
)
from vllm.v1.pool.late_interaction_runner import LateInteractionRunner


def _make_pooling_params(
    late_interaction_params: LateInteractionParams,
) -> PoolingParams:
    return PoolingParams(
        task="token_embed",
        late_interaction_params=late_interaction_params,
    )


def test_postprocess_scores_and_releases_query_cache():
    runner = LateInteractionRunner()
    query_key = "query-0"
    query_emb = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    doc_emb = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=torch.float32)

    query_params = _make_pooling_params(
        build_late_interaction_query_params(query_key=query_key, query_uses=1)
    )
    query_output = runner.postprocess_pooler_output(
        raw_pooler_output=[query_emb],
        pooling_params=[query_params],
        req_ids=["query-req"],
        finished_mask=[True],
    )
    assert isinstance(query_output, list)
    assert query_output[0] is not None
    assert query_output[0].shape == torch.Size([])

    doc_params = _make_pooling_params(
        build_late_interaction_doc_params(query_key=query_key)
    )
    doc_output = runner.postprocess_pooler_output(
        raw_pooler_output=[doc_emb],
        pooling_params=[doc_params],
        req_ids=["doc-req"],
        finished_mask=[True],
    )
    assert isinstance(doc_output, list)
    assert doc_output[0] is not None
    assert torch.allclose(doc_output[0], compute_maxsim_score(query_emb, doc_emb))

    with pytest.raises(ValueError, match="query cache miss"):
        runner.postprocess_pooler_output(
            raw_pooler_output=[doc_emb],
            pooling_params=[doc_params],
            req_ids=["doc-req-2"],
            finished_mask=[True],
        )


def test_postprocess_scores_docs_in_batch():
    runner = LateInteractionRunner()
    query_key = "query-batch"
    query_emb = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    doc_emb_1 = torch.tensor([[1.0, 0.0], [0.5, 0.5]], dtype=torch.float32)
    doc_emb_2 = torch.tensor([[0.0, 1.0], [0.3, 0.7], [1.0, 0.0]], dtype=torch.float32)

    query_params = _make_pooling_params(
        build_late_interaction_query_params(query_key=query_key, query_uses=2)
    )
    runner.postprocess_pooler_output(
        raw_pooler_output=[query_emb],
        pooling_params=[query_params],
        req_ids=["query-req"],
        finished_mask=[True],
    )

    doc_params = _make_pooling_params(
        build_late_interaction_doc_params(query_key=query_key)
    )
    doc_output = runner.postprocess_pooler_output(
        raw_pooler_output=[doc_emb_1, doc_emb_2],
        pooling_params=[doc_params, doc_params],
        req_ids=["doc-req-1", "doc-req-2"],
        finished_mask=[True, True],
    )
    assert isinstance(doc_output, list)
    assert doc_output[0] is not None
    assert doc_output[1] is not None
    assert torch.allclose(doc_output[0], compute_maxsim_score(query_emb, doc_emb_1))
    assert torch.allclose(doc_output[1], compute_maxsim_score(query_emb, doc_emb_2))

    with pytest.raises(ValueError, match="query cache miss"):
        runner.postprocess_pooler_output(
            raw_pooler_output=[doc_emb_1],
            pooling_params=[doc_params],
            req_ids=["doc-req-3"],
            finished_mask=[True],
        )


def test_finished_request_releases_unscored_doc_use():
    runner = LateInteractionRunner()
    query_key = "query-cancel"
    query_emb = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    doc_emb = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    query_params = _make_pooling_params(
        build_late_interaction_query_params(query_key=query_key, query_uses=1)
    )
    runner.postprocess_pooler_output(
        raw_pooler_output=[query_emb],
        pooling_params=[query_params],
        req_ids=["query-req"],
        finished_mask=[True],
    )

    doc_params = _make_pooling_params(
        build_late_interaction_doc_params(query_key=query_key)
    )
    runner.register_request("doc-req", doc_params)
    runner.on_requests_finished({"doc-req"})

    with pytest.raises(ValueError, match="query cache miss"):
        runner.postprocess_pooler_output(
            raw_pooler_output=[doc_emb],
            pooling_params=[doc_params],
            req_ids=["doc-req-retry"],
            finished_mask=[True],
        )


def test_invalid_query_uses_raises():
    runner = LateInteractionRunner()
    bad_meta = LateInteractionParams(
        mode=LATE_INTERACTION_MODE_CACHE_QUERY,
        query_key="query-bad",
    )
    bad_meta.query_uses = "bad-int"  # type: ignore[assignment]
    bad_query_params = _make_pooling_params(bad_meta)

    with pytest.raises(ValueError, match="must be an integer value"):
        runner.postprocess_pooler_output(
            raw_pooler_output=[torch.ones((2, 2), dtype=torch.float32)],
            pooling_params=[bad_query_params],
            req_ids=["query-req"],
            finished_mask=[True],
        )


# ---------------------------------------------------------------------------
# Fused Triton scoring path (PR #40337).
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_score_flash_matches_fp64_oracle():
    """Shared-query scoring through the fused kernel must match a per-pair
    fp64 MaxSim oracle on ragged doc lengths, including length-1 docs."""
    torch.manual_seed(0)
    d = 128
    q = torch.randn(32, d, device="cuda", dtype=torch.float16)
    doc_lengths = [180, 37, 512, 1, 300, 64, 1030, 256]
    docs = [
        torch.randn(ld, d, device="cuda", dtype=torch.float16) for ld in doc_lengths
    ]

    runner = LateInteractionRunner()
    assert runner._flash_enabled
    scores = runner._score([q] * len(docs), docs)
    assert runner._flash_enabled, "kernel path must not have fallen back"

    for i, doc in enumerate(docs):
        ref = (q.double() @ doc.double().T).max(dim=1).values.sum().float()
        torch.testing.assert_close(
            scores[i].to(torch.float32), ref, atol=5e-2, rtol=1e-3
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_score_multi_query_uses_reference_path():
    """Distinct queries per pair route to the reference scorer and agree
    with it exactly (no kernel dispatch for the N:N pattern)."""
    torch.manual_seed(1)
    d = 128
    queries = [torch.randn(16, d, device="cuda") for _ in range(3)]
    docs = [torch.randn(ld, d, device="cuda") for ld in (50, 3, 77)]

    runner = LateInteractionRunner()
    scores = runner._score(queries, docs)
    ref = compute_maxsim_score_batched(queries, docs)
    for s, r in zip(scores, ref):
        torch.testing.assert_close(s, r)


def test_score_flash_failure_disables_and_falls_back(monkeypatch):
    """A kernel failure must serve the batch via the reference scorer and
    disable the kernel path for the rest of the process."""
    runner = LateInteractionRunner()
    runner._flash_enabled = True

    def _boom(query, docs):
        raise RuntimeError("synthetic kernel failure")

    monkeypatch.setattr(runner, "_score_flash_shared_query", _boom)
    q = torch.randn(8, 32)
    docs = [torch.randn(5, 32), torch.randn(7, 32)]
    # CPU tensors already skip the kernel; force the dispatch condition.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    scores = runner._score([q, q], docs)
    assert not runner._flash_enabled
    ref = compute_maxsim_score_batched([q, q], docs)
    for s, r in zip(scores, ref):
        torch.testing.assert_close(s, r)


def test_disabled_flag_uses_reference_path():
    runner = LateInteractionRunner(enable_flash=False)
    assert not runner._flash_enabled


# ---------------------------------------------------------------------------
# Autotune config pruning.
# ---------------------------------------------------------------------------
def _prune(monkeypatch, budget: int, named_args: dict):
    from vllm.v1.pool.flash_maxsim import _common

    monkeypatch.setattr(_common, "_smem_budget", lambda: budget)
    return _common._prune_configs(_common._get_configs(), named_args)


def _est(cfg, d_pad: int) -> int:
    bq, bd = cfg.kwargs["BLOCK_Q"], cfg.kwargs["BLOCK_D"]
    return (bq * d_pad + bd * d_pad) * 2 + bq * bd * 4


def test_prune_configs_uses_padded_dim(monkeypatch):
    """d=513 pads to 1024: survivors must fit the budget at d_pad, not at
    the un-padded d (which would admit ~2x-oversized tiles)."""
    budget = 166_912  # A100 opt-in
    survivors = _prune(monkeypatch, budget, {"Lq": 1024, "d": 513, "d_pad": 1024})
    assert survivors
    assert all(_est(c, 1024) <= budget for c in survivors)


def test_prune_configs_no_rejected_fallback(monkeypatch):
    """When nothing fits, the fallback is the single smallest-footprint
    config — never a slice of rejected configs."""
    from vllm.v1.pool.flash_maxsim import _common

    survivors = _prune(monkeypatch, 10_000, {"Lq": 1024, "d_pad": 1024})
    assert len(survivors) == 1
    smallest = min(_common._get_configs(), key=lambda c: _est(c, 1024))
    assert survivors[0].kwargs == smallest.kwargs
