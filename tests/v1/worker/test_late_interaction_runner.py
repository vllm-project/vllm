# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.pooling_params import LateInteractionParams, PoolingParams
from vllm.v1.pool.late_interaction import (
    LATE_INTERACTION_MODE_CACHE_QUERY,
    build_late_interaction_doc_params,
    build_late_interaction_query_params,
    compute_maxsim_score,
)
from vllm.v1.worker.gpu.pool.late_interaction_runner import LateInteractionRunner


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
