# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for max_tokens_per_doc and max_tokens_per_query.
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

long_query = "What is the capital of France?" * 20
long_doc = "The capital of France is Paris. " * 20


@dataclass
class TestConfig:
    model: str
    args: list[str]
    without_truncated_prompt_tokens: int
    with_max_tokens_per_query_prompt_tokens: int
    with_max_tokens_per_doc_prompt_tokens: int
    with_max_tokens_per_query_and_doc_prompt_tokens: int


RERANK_CONFIGS = [
    # 1. cross-encoder
    TestConfig(
        model="jinaai/jina-reranker-v2-base-multilingual",
        args=[
            "--enforce-eager",
            "--max-model-len",
            "1024",
            "--trust-remote-code",
        ],
        without_truncated_prompt_tokens=284,
        with_max_tokens_per_query_prompt_tokens=154,
        with_max_tokens_per_doc_prompt_tokens=154,
        with_max_tokens_per_query_and_doc_prompt_tokens=24,
    ),
    # 2. cross-encoder + score template
    TestConfig(
        model="Qwen/Qwen3-Reranker-0.6B",
        args=[
            "--enforce-eager",
            "--max-model-len",
            "1024",
            "--hf-overrides",
            json.dumps(
                {
                    "architectures": ["Qwen3ForSequenceClassification"],
                    "classifier_from_token": ["no", "yes"],
                    "is_original_qwen3_reranker": True,
                }
            ),
            "--chat-template",
            os.path.join(TEMPLATE_DIR, "qwen3_reranker.jinja"),
        ],
        without_truncated_prompt_tokens=352,
        with_max_tokens_per_query_prompt_tokens=223,
        with_max_tokens_per_doc_prompt_tokens=221,
        with_max_tokens_per_query_and_doc_prompt_tokens=92,
    ),
    # 3. bi-encoder
    TestConfig(
        model="intfloat/multilingual-e5-small",
        args=[
            "--enforce-eager",
            "--max-model-len",
            "512",
            "--trust-remote-code",
        ],
        without_truncated_prompt_tokens=286,
        with_max_tokens_per_query_prompt_tokens=156,
        with_max_tokens_per_doc_prompt_tokens=155,
        with_max_tokens_per_query_and_doc_prompt_tokens=25,
    ),
    # 4. late-interaction
    TestConfig(
        model="answerdotai/answerai-colbert-small-v1",
        args=[
            "--enforce-eager",
            "--max-model-len",
            "512",
            "--trust-remote-code",
        ],
        without_truncated_prompt_tokens=285,
        with_max_tokens_per_query_prompt_tokens=155,
        with_max_tokens_per_doc_prompt_tokens=155,
        with_max_tokens_per_query_and_doc_prompt_tokens=25,
    ),
    # 5. jinaai/jina-reranker-v3
    TestConfig(
        model="jinaai/jina-reranker-v3",
        args=[
            "--enforce-eager",
            "--max-model-len",
            "1024",
            "--trust-remote-code",
        ],
        without_truncated_prompt_tokens=567,
        with_max_tokens_per_query_prompt_tokens=308,
        with_max_tokens_per_doc_prompt_tokens=436,
        with_max_tokens_per_query_and_doc_prompt_tokens=177,
    ),
]


@pytest.fixture(scope="module", params=RERANK_CONFIGS, ids=lambda c: c.model)
def server(request):
    config: TestConfig = request.param
    with RemoteOpenAIServer(config.model, config.args) as remote_server:
        yield config, remote_server


def test_without_truncated(server):
    """Test that max_tokens_per_doc truncates documents correctly."""
    config, remote_server = server

    response = requests.post(
        remote_server.url_for("rerank"),
        json={"model": config.model, "query": long_query, "documents": [long_doc]},
    )
    response.raise_for_status()
    rerank = RerankResponse.model_validate(response.json())

    assert rerank.id is not None
    assert rerank.results is not None
    assert len(rerank.results) == 1
    assert rerank.usage.prompt_tokens == config.without_truncated_prompt_tokens


def test_max_tokens_per_query(server):
    """Test that max_tokens_per_doc truncates documents correctly."""
    config, remote_server = server

    response = requests.post(
        remote_server.url_for("rerank"),
        json={
            "model": config.model,
            "query": long_query,
            "documents": [long_doc],
            "max_tokens_per_query": 10,
        },
    )
    response.raise_for_status()
    rerank = RerankResponse.model_validate(response.json())

    assert rerank.id is not None
    assert rerank.results is not None
    assert len(rerank.results) == 1
    assert rerank.usage.prompt_tokens == config.with_max_tokens_per_query_prompt_tokens


def test_max_tokens_per_doc(server):
    """Test that max_tokens_per_doc truncates documents correctly."""
    config, remote_server = server

    response = requests.post(
        remote_server.url_for("rerank"),
        json={
            "model": config.model,
            "query": long_query,
            "documents": [long_doc],
            "max_tokens_per_doc": 10,
        },
    )
    response.raise_for_status()
    rerank = RerankResponse.model_validate(response.json())

    assert rerank.id is not None
    assert rerank.results is not None
    assert len(rerank.results) == 1
    assert rerank.usage.prompt_tokens == config.with_max_tokens_per_doc_prompt_tokens


def test_max_tokens_per_query_and_doc(server):
    """Test that max_tokens_per_doc truncates documents correctly."""
    config, remote_server = server

    response = requests.post(
        remote_server.url_for("rerank"),
        json={
            "model": config.model,
            "query": long_query,
            "documents": [long_doc],
            "max_tokens_per_query": 10,
            "max_tokens_per_doc": 10,
        },
    )
    response.raise_for_status()
    rerank = RerankResponse.model_validate(response.json())

    assert rerank.id is not None
    assert rerank.results is not None
    assert len(rerank.results) == 1
    assert (
        rerank.usage.prompt_tokens
        == config.with_max_tokens_per_query_and_doc_prompt_tokens
    )
