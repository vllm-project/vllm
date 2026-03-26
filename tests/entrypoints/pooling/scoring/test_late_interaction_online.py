# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online API tests for ColBERT late interaction scoring."""

import pytest
import requests

from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.pooling.score.protocol import RerankResponse, ScoreResponse

from .util import ColBERTScoringHfRunner

MODEL_NAME = "answerdotai/answerai-colbert-small-v1"
COLBERT_DIM = 96
MAX_MODEL_LEN = 512
LINEAR_WEIGHTS_KEY = "linear.weight"

TEXTS_1 = [
    "What is the capital of France?",
    "What is the capital of Germany?",
]

TEXTS_2 = [
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
]


@pytest.fixture(scope="module")
def server():
    args = [
        "--max-model-len",
        str(MAX_MODEL_LEN),
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def hf_model():
    return ColBERTScoringHfRunner(
        model_name=MODEL_NAME, linear_weights_key=LINEAR_WEIGHTS_KEY
    )


@pytest.mark.asyncio
async def test_score_api_queries_str_1_documents_str_1(
    hf_model, server: RemoteOpenAIServer
):
    score_response = requests.post(
        server.url_for("score"),
        json={
            "model": MODEL_NAME,
            "queries": TEXTS_1[0],
            "documents": TEXTS_2[0],
        },
    )
    score_response.raise_for_status()
    score = ScoreResponse.model_validate(score_response.json())

    assert score.id is not None
    assert score.data is not None
    assert len(score.data) == 1

    vllm_outputs = [d.score for d in score.data]
    hf_outputs = hf_model.predict([[TEXTS_1[0], TEXTS_2[0]]]).tolist()

    for i in range(len(vllm_outputs)):
        assert hf_outputs[i] == pytest.approx(vllm_outputs[i], rel=0.01)


@pytest.mark.asyncio
async def test_score_api_queries_str_1_documents_str_n(
    hf_model, server: RemoteOpenAIServer
):
    text_pairs = [
        [TEXTS_1[0], TEXTS_2[0]],
        [TEXTS_1[0], TEXTS_2[1]],
    ]

    score_response = requests.post(
        server.url_for("score"),
        json={
            "model": MODEL_NAME,
            "queries": TEXTS_1[0],
            "documents": TEXTS_2,
        },
    )
    score_response.raise_for_status()
    score = ScoreResponse.model_validate(score_response.json())

    assert score.id is not None
    assert score.data is not None
    assert len(score.data) == 2

    vllm_outputs = [d.score for d in score.data]
    hf_outputs = hf_model.predict(text_pairs).tolist()

    for i in range(len(vllm_outputs)):
        assert hf_outputs[i] == pytest.approx(vllm_outputs[i], rel=0.01)


@pytest.mark.asyncio
async def test_score_api_queries_str_n_documents_str_n(
    hf_model, server: RemoteOpenAIServer
):
    text_pairs = [
        [TEXTS_1[0], TEXTS_2[0]],
        [TEXTS_1[1], TEXTS_2[1]],
    ]

    score_response = requests.post(
        server.url_for("score"),
        json={
            "model": MODEL_NAME,
            "queries": TEXTS_1,
            "documents": TEXTS_2,
        },
    )
    score_response.raise_for_status()
    score = ScoreResponse.model_validate(score_response.json())

    assert score.id is not None
    assert score.data is not None
    assert len(score.data) == 2

    vllm_outputs = [d.score for d in score.data]
    hf_outputs = hf_model.predict(text_pairs).tolist()

    for i in range(len(vllm_outputs)):
        assert hf_outputs[i] == pytest.approx(vllm_outputs[i], rel=0.01)


@pytest.mark.asyncio
async def test_rerank_api_texts(server: RemoteOpenAIServer):
    """Test ColBERT rerank endpoint."""
    query = "What is the capital of France?"
    documents = [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris.",
    ]

    rerank_response = requests.post(
        server.url_for("rerank"),
        json={
            "model": MODEL_NAME,
            "query": query,
            "documents": documents,
        },
    )
    rerank_response.raise_for_status()
    rerank = RerankResponse.model_validate(rerank_response.json())

    assert rerank.id is not None
    assert rerank.results is not None
    assert len(rerank.results) == 2

    paris_result = next(r for r in rerank.results if r.index == 1)
    brazil_result = next(r for r in rerank.results if r.index == 0)

    assert paris_result.relevance_score > brazil_result.relevance_score


@pytest.mark.asyncio
async def test_rerank_api_top_n(server: RemoteOpenAIServer):
    """Test ColBERT rerank with top_n parameter."""
    query = "What is the capital of France?"
    documents = [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris.",
        "Machine learning is a field of AI.",
    ]

    rerank_response = requests.post(
        server.url_for("rerank"),
        json={
            "model": MODEL_NAME,
            "query": query,
            "documents": documents,
            "top_n": 2,
        },
    )
    rerank_response.raise_for_status()
    rerank = RerankResponse.model_validate(rerank_response.json())

    assert len(rerank.results) == 2
    assert rerank.results[0].index == 1


@pytest.mark.asyncio
async def test_token_embed(server: RemoteOpenAIServer):
    """Test ColBERT token_embed task via pooling endpoint."""
    text = "What is the capital of France?"

    pooling_response = requests.post(
        server.url_for("pooling"),
        json={
            "model": MODEL_NAME,
            "input": text,
            "task": "token_embed",
        },
    )
    pooling_response.raise_for_status()
    pooling = pooling_response.json()

    assert "data" in pooling
    assert len(pooling["data"]) == 1

    embeddings = pooling["data"][0]["data"]
    assert isinstance(embeddings, list)
    assert len(embeddings) > 0
    assert len(embeddings[0]) == COLBERT_DIM


@pytest.mark.asyncio
async def test_embed_not_supported(server: RemoteOpenAIServer):
    """Test that ColBERT model does not support 'embed' task."""
    task = "embed"
    text = "What is the capital of France?"

    response = requests.post(
        server.url_for("pooling"),
        json={
            "model": MODEL_NAME,
            "input": text,
            "task": task,
        },
    )

    assert response.json()["error"]["type"] == "BadRequestError"
    assert response.json()["error"]["message"].startswith(f"Unsupported task: {task!r}")
