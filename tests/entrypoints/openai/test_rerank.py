# SPDX-License-Identifier: Apache-2.0

import pytest
import requests

from vllm.entrypoints.openai.protocol import RerankResponse

from ...utils import RemoteOpenAIServer

MODEL_NAME = "BAAI/bge-reranker-base"
DTYPE = "bfloat16"


@pytest.fixture(scope="module")
def server():
    args = ["--enforce-eager", "--max-model-len", "100", "--dtype", DTYPE]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_rerank_texts(server: RemoteOpenAIServer, model_name: str):
    query = "What is the capital of France?"
    documents = [
        "The capital of Brazil is Brasilia.", "The capital of France is Paris."
    ]

    rerank_response = requests.post(server.url_for("rerank"),
                                    json={
                                        "model": model_name,
                                        "query": query,
                                        "documents": documents,
                                    })
    rerank_response.raise_for_status()
    rerank = RerankResponse.model_validate(rerank_response.json())

    assert rerank.id is not None
    assert rerank.results is not None
    assert len(rerank.results) == 2
    assert rerank.results[0].relevance_score >= 0.9
    assert rerank.results[1].relevance_score <= 0.01


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_top_n(server: RemoteOpenAIServer, model_name: str):
    query = "What is the capital of France?"
    documents = [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris.", "Cross-encoder models are neat"
    ]

    rerank_response = requests.post(server.url_for("rerank"),
                                    json={
                                        "model": model_name,
                                        "query": query,
                                        "documents": documents,
                                        "top_n": 2
                                    })
    rerank_response.raise_for_status()
    rerank = RerankResponse.model_validate(rerank_response.json())

    assert rerank.id is not None
    assert rerank.results is not None
    assert len(rerank.results) == 2
    assert rerank.results[0].relevance_score >= 0.9
    assert rerank.results[1].relevance_score <= 0.01


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_rerank_max_model_len(server: RemoteOpenAIServer, model_name: str):

    query = "What is the capital of France?" * 100
    documents = [
        "The capital of Brazil is Brasilia.", "The capital of France is Paris."
    ]

    rerank_response = requests.post(server.url_for("rerank"),
                                    json={
                                        "model": model_name,
                                        "query": query,
                                        "documents": documents
                                    })
    assert rerank_response.status_code == 400
    # Assert just a small fragments of the response
    assert "Please reduce the length of the input." in \
        rerank_response.text
