# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online API tests for ColBERT late interaction scoring.

Tests are parametrized across multiple ColBERT backbones to ensure the
generic ColBERT support works with different encoder architectures.
"""

import pytest
import requests

from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.pooling.score.protocol import RerankResponse, ScoreResponse

COLBERT_MODELS = [
    {
        "name": "answerdotai/answerai-colbert-small-v1",
        "colbert_dim": 96,
        "max_model_len": 512,
        "extra_args": [],
    },
    {
        "name": "lightonai/GTE-ModernColBERT-v1",
        "colbert_dim": 128,
        "max_model_len": 299,
        "extra_args": [
            "--hf-overrides",
            '{"architectures": ["ColBERTModernBertModel"]}',
        ],
    },
    {
        "name": "jinaai/jina-colbert-v2",
        "colbert_dim": 128,
        "max_model_len": 8192,
        "extra_args": [
            "--hf-overrides",
            '{"architectures": ["ColBERTJinaRobertaModel"]}',
            "--trust-remote-code",
        ],
    },
]


@pytest.fixture(scope="class", params=COLBERT_MODELS, ids=lambda m: m["name"])
def model(request):
    yield request.param


@pytest.fixture(scope="class")
def server(model):
    args = [
        "--max-model-len",
        str(model["max_model_len"]),
        *model["extra_args"],
    ]

    with RemoteOpenAIServer(model["name"], args) as remote_server:
        yield remote_server


class TestColBERTOnline:
    def test_rerank(self, server: RemoteOpenAIServer, model: dict):
        """Test ColBERT rerank endpoint."""
        query = "What is the capital of France?"
        documents = [
            "The capital of Brazil is Brasilia.",
            "The capital of France is Paris.",
        ]

        rerank_response = requests.post(
            server.url_for("rerank"),
            json={
                "model": model["name"],
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

    def test_rerank_top_n(self, server: RemoteOpenAIServer, model: dict):
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
                "model": model["name"],
                "query": query,
                "documents": documents,
                "top_n": 2,
            },
        )
        rerank_response.raise_for_status()
        rerank = RerankResponse.model_validate(rerank_response.json())

        assert len(rerank.results) == 2
        assert rerank.results[0].index == 1

    def test_score(self, server: RemoteOpenAIServer, model: dict):
        """Test ColBERT score endpoint."""
        text_1 = "What is the capital of France?"
        text_2 = ["The capital of France is Paris.", "Python is a language."]

        score_response = requests.post(
            server.url_for("score"),
            json={
                "model": model["name"],
                "text_1": text_1,
                "text_2": text_2,
            },
        )
        score_response.raise_for_status()
        score = ScoreResponse.model_validate(score_response.json())

        assert score.id is not None
        assert score.data is not None
        assert len(score.data) == 2

        assert score.data[0].score > score.data[1].score

    def test_token_embed(self, server: RemoteOpenAIServer, model: dict):
        """Test ColBERT token_embed task via pooling endpoint."""
        text = "What is the capital of France?"

        pooling_response = requests.post(
            server.url_for("pooling"),
            json={
                "model": model["name"],
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
        assert len(embeddings[0]) == model["colbert_dim"]

    def test_embed_not_supported(self, server: RemoteOpenAIServer, model: dict):
        """Test that ColBERT model does not support 'embed' task."""
        task = "embed"
        text = "What is the capital of France?"

        response = requests.post(
            server.url_for("pooling"),
            json={
                "model": model["name"],
                "input": text,
                "task": task,
            },
        )

        assert response.json()["error"]["type"] == "BadRequestError"
        assert response.json()["error"]["message"].startswith(
            f"Unsupported task: {task!r}"
        )
