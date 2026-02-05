# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online API tests for ColBERT late interaction scoring."""

import pytest
import requests

from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.pooling.score.protocol import RerankResponse, ScoreResponse

# ColBERT model - using answerai-colbert-small-v1 as it's a smaller model
MODEL_NAME = "answerdotai/answerai-colbert-small-v1"
COLBERT_DIM = 96  # This model uses 96-dimensional output
DTYPE = "half"
MAX_MODEL_LEN = 512


@pytest.fixture(scope="module")
def server():
    args = [
        "--max-model-len",
        str(MAX_MODEL_LEN),
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_colbert_rerank(server: RemoteOpenAIServer, model_name: str):
    """Test ColBERT rerank endpoint."""
    query = "What is the capital of France?"
    documents = [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris.",
    ]

    rerank_response = requests.post(
        server.url_for("rerank"),
        json={
            "model": model_name,
            "query": query,
            "documents": documents,
        },
    )
    rerank_response.raise_for_status()
    rerank = RerankResponse.model_validate(rerank_response.json())

    assert rerank.id is not None
    assert rerank.results is not None
    assert len(rerank.results) == 2

    # The relevant document (Paris) should have higher score
    paris_result = next(r for r in rerank.results if r.index == 1)
    brazil_result = next(r for r in rerank.results if r.index == 0)

    assert paris_result.relevance_score > brazil_result.relevance_score


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_colbert_rerank_top_n(server: RemoteOpenAIServer, model_name: str):
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
            "model": model_name,
            "query": query,
            "documents": documents,
            "top_n": 2,
        },
    )
    rerank_response.raise_for_status()
    rerank = RerankResponse.model_validate(rerank_response.json())

    assert len(rerank.results) == 2
    # Top result should be about Paris (index 1)
    assert rerank.results[0].index == 1


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_colbert_score(server: RemoteOpenAIServer, model_name: str):
    """Test ColBERT score endpoint."""
    text_1 = "What is the capital of France?"
    text_2 = ["The capital of France is Paris.", "Python is a language."]

    score_response = requests.post(
        server.url_for("score"),
        json={
            "model": model_name,
            "text_1": text_1,
            "text_2": text_2,
        },
    )
    score_response.raise_for_status()
    score = ScoreResponse.model_validate(score_response.json())

    assert score.id is not None
    assert score.data is not None
    assert len(score.data) == 2

    # The relevant document should have higher score
    assert score.data[0].score > score.data[1].score


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_colbert_token_embed(server: RemoteOpenAIServer, model_name: str):
    """Test ColBERT token_embed task via pooling endpoint."""
    text = "What is the capital of France?"

    pooling_response = requests.post(
        server.url_for("pooling"),
        json={
            "model": model_name,
            "input": text,
            "task": "token_embed",
        },
    )
    pooling_response.raise_for_status()
    pooling = pooling_response.json()

    assert "data" in pooling
    assert len(pooling["data"]) == 1

    # Token embeddings should be 2D
    embeddings = pooling["data"][0]["data"]
    assert isinstance(embeddings, list)
    assert len(embeddings) > 0  # Should have tokens
    assert len(embeddings[0]) == COLBERT_DIM


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_colbert_embed_not_supported(server: RemoteOpenAIServer, model_name: str):
    """Test that ColBERT model does not support 'embed' task."""
    text = "What is the capital of France?"

    pooling_response = requests.post(
        server.url_for("pooling"),
        json={
            "model": model_name,
            "input": text,
            "task": "embed",
        },
    )

    # Should return error
    assert pooling_response.status_code == 400
    assert "Task embed is not supported" in pooling_response.text
