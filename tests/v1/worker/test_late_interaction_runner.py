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
)
from vllm.v1.pool.late_interaction_runner import LateInteractionRunner


def _make_pooling_params(
    late_interaction_params: LateInteractionParams,
) -> PoolingParams:
    return PoolingParams(
        task="token_embed",
        late_interaction_params=late_interaction_params,
    )


def _make_emb(rows, dim=32):
    """Create a deterministic embedding tensor with realistic dimensions."""
    gen = torch.Generator().manual_seed(rows * 1000 + dim)
    return torch.randn(rows, dim, dtype=torch.float32, generator=gen)


def test_postprocess_scores_and_releases_query_cache():
    runner = LateInteractionRunner()
    query_key = "query-0"
    query_emb = _make_emb(8)
    doc_emb = _make_emb(12)

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
    query_emb = _make_emb(8)
    doc_emb_1 = _make_emb(10)
    doc_emb_2 = _make_emb(14)

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
    query_emb = _make_emb(8)
    doc_emb = _make_emb(10)

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
            raw_pooler_output=[_make_emb(8)],
            pooling_params=[bad_query_params],
            req_ids=["query-req"],
            finished_mask=[True],
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_score_zerocopy_pairs_matches_reference():
    """N:N scoring (multiple distinct queries) must match a per-pair fp64
    MaxSim oracle and the shared-query kernel path (PR #40337 review:
    single-launch pairwise kernel replaces per-unique-query launches).
    """
    torch.manual_seed(0)
    device = "cuda"
    d = 128

    # Ragged docs scattered in one projected batch tensor.
    doc_lengths = [180, 37, 512, 1, 300, 64, 1030, 256]
    doc_offsets = []
    total = 0
    for ld in doc_lengths:
        doc_offsets.append(total)
        total += ld
    batch = torch.randn(total, d, device=device, dtype=torch.float16)

    # Distinct query per pair except two pairs sharing one query (mixed N:N).
    q_lens = [32, 32, 1030, 16, 96, 5, 512, 32]
    queries = [torch.randn(lq, d, device=device, dtype=torch.float16) for lq in q_lens]
    queries[1] = queries[0]  # shared query across pairs 0 and 1

    scores = LateInteractionRunner._score_zerocopy(
        queries, batch, doc_offsets, doc_lengths
    )

    for i, (q, off, ld) in enumerate(zip(queries, doc_offsets, doc_lengths)):
        doc = batch[off : off + ld]
        ref = (q.double() @ doc.double().T).max(dim=1).values.sum().float()
        assert torch.isfinite(scores[i]), f"pair {i} score not finite"
        torch.testing.assert_close(
            scores[i].to(torch.float32), ref, atol=5e-2, rtol=1e-3
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_score_zerocopy_single_query_path_unchanged():
    """1:N (one distinct query) must still route through the shared-query
    kernel and agree with the fp64 oracle."""
    torch.manual_seed(1)
    device = "cuda"
    d = 128
    doc_lengths = [64, 200, 7, 128]
    doc_offsets = [0, 64, 264, 271]
    batch = torch.randn(sum(doc_lengths), d, device=device, dtype=torch.float16)
    q = torch.randn(32, d, device=device, dtype=torch.float16)
    queries = [q, q, q, q]

    scores = LateInteractionRunner._score_zerocopy(
        queries, batch, doc_offsets, doc_lengths
    )
    for i, (off, ld) in enumerate(zip(doc_offsets, doc_lengths)):
        doc = batch[off : off + ld]
        ref = (q.double() @ doc.double().T).max(dim=1).values.sum().float()
        torch.testing.assert_close(
            scores[i].to(torch.float32), ref, atol=5e-2, rtol=1e-3
        )
