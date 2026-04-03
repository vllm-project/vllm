# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for max_tokens_per_doc and max_tokens_per_query across reranker types:

1. Cross-encoder (jinaai/jina-reranker-v2-base-multilingual)
2. Chat template / decoder (BAAI/bge-reranker-v2-gemma)
3. Score template / decoder (Qwen/Qwen3-Reranker-0.6B)
"""

import json
import os
from dataclasses import dataclass

import pytest
import requests

from tests.utils import VLLM_PATH, RemoteOpenAIServer
from vllm.entrypoints.pooling.scoring.protocol import RerankResponse

os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

TEMPLATE_DIR = str(VLLM_PATH / "examples/pooling/score/template")


@dataclass
class RerankTestConfig:
    model: str
    args: list[str]


RERANK_CONFIGS = [
    RerankTestConfig(
        model="jinaai/jina-reranker-v2-base-multilingual",
        args=[
            "--enforce-eager",
            "--max-model-len", "256",
            "--gpu-memory-utilization", "0.3",
            "--trust-remote-code",
        ],
    ),
    RerankTestConfig(
        model="BAAI/bge-reranker-v2-gemma",
        args=[
            "--enforce-eager",
            "--max-model-len", "256",
            "--gpu-memory-utilization", "0.3",
            "--hf-overrides", json.dumps({
                "architectures": ["GemmaForSequenceClassification"],
                "classifier_from_token": ["Yes"],
                "method": "no_post_processing",
            }),
            "--chat-template",
            os.path.join(TEMPLATE_DIR, "bge-reranker-v2-gemma.jinja"),
        ],
    ),
    RerankTestConfig(
        model="Qwen/Qwen3-Reranker-0.6B",
        args=[
            "--enforce-eager",
            "--max-model-len", "256",
            "--gpu-memory-utilization", "0.3",
            "--hf-overrides", json.dumps({
                "architectures": ["Qwen3ForSequenceClassification"],
                "classifier_from_token": ["no", "yes"],
                "is_original_qwen3_reranker": True,
            }),
            "--chat-template",
            os.path.join(TEMPLATE_DIR, "qwen3_reranker.jinja"),
        ],
    ),
]


@pytest.fixture(scope="module", params=RERANK_CONFIGS, ids=lambda c: c.model)
def server(request):
    config: RerankTestConfig = request.param
    with RemoteOpenAIServer(config.model, config.args) as remote_server:
        yield config.model, remote_server


def test_max_tokens_per_doc(server):
    """Test that max_tokens_per_doc truncates documents correctly."""
    model_name, remote_server = server
    query = "What is the capital of France?"
    long_doc = "The capital of France is Paris. " * 20

    response = requests.post(
        remote_server.url_for("rerank"),
        json={
            "model": model_name,
            "query": query,
            "documents": [long_doc],
            "max_tokens_per_doc": 10,
        },
    )
    response.raise_for_status()
    rerank = RerankResponse.model_validate(response.json())

    assert rerank.id is not None
    assert rerank.results is not None
    assert len(rerank.results) == 1
    assert rerank.usage.prompt_tokens > 0


def test_max_tokens_per_doc_reduces_tokens(server):
    """Test that max_tokens_per_doc actually reduces the token count."""
    model_name, remote_server = server
    query = "What is the capital of France?"
    doc = "The capital of France is Paris. " * 5

    response_no_limit = requests.post(
        remote_server.url_for("rerank"),
        json={
            "model": model_name,
            "query": query,
            "documents": [doc],
        },
    )
    response_no_limit.raise_for_status()
    rerank_no_limit = RerankResponse.model_validate(response_no_limit.json())

    response_with_limit = requests.post(
        remote_server.url_for("rerank"),
        json={
            "model": model_name,
            "query": query,
            "documents": [doc],
            "max_tokens_per_doc": 5,
        },
    )
    response_with_limit.raise_for_status()
    rerank_with_limit = RerankResponse.model_validate(
        response_with_limit.json()
    )

    assert (
        rerank_with_limit.usage.prompt_tokens
        < rerank_no_limit.usage.prompt_tokens
    )
