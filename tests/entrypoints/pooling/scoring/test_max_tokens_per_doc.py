# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for max_tokens_per_doc across all cross-encoder code paths:

1. Cross-encoder with sep token (BAAI/bge-reranker-base)
   - Tested in test_cross_encoder_online.py

2. Cross-encoder with sep token, XLM-RoBERTa (jinaai/jina-reranker-v2-base-multilingual)
   - Multilingual encoder, CLS pooling

3. Chat template / Jinja path (BAAI/bge-reranker-v2-gemma)
   - LLM-as-reranker using a Jinja chat template

4. Score template path (Qwen/Qwen3-Reranker-0.6B)
   - Uses SupportsScoreTemplate + chat template
"""

import json
import os

import pytest
import requests

from tests.utils import VLLM_PATH, RemoteOpenAIServer
from vllm.entrypoints.pooling.scoring.protocol import RerankResponse

os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

TEMPLATE_DIR = str(VLLM_PATH / "examples/pooling/score/template")

# ── Model configs ──────────────────────────────────────────────────────

# Path 2: Cross-encoder with sep token, XLM-RoBERTa (Jina v2)
JINA_MODEL = "jinaai/jina-reranker-v2-base-multilingual"

# Path 3: Chat template (LLM-as-reranker, no sep token)
GEMMA_MODEL = "BAAI/bge-reranker-v2-gemma"
GEMMA_TEMPLATE = os.path.join(TEMPLATE_DIR, "bge-reranker-v2-gemma.jinja")
GEMMA_HF_OVERRIDES = {
    "architectures": ["GemmaForSequenceClassification"],
    "classifier_from_token": ["Yes"],
    "method": "no_post_processing",
}

# Path 3: Score template (Qwen3 reranker)
QWEN_MODEL = "Qwen/Qwen3-Reranker-0.6B"
QWEN_TEMPLATE = os.path.join(TEMPLATE_DIR, "qwen3_reranker.jinja")
QWEN_HF_OVERRIDES = {
    "architectures": ["Qwen3ForSequenceClassification"],
    "classifier_from_token": ["no", "yes"],
    "is_original_qwen3_reranker": True,
}


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def jina_server():
    args = [
        "--enforce-eager",
        "--max-model-len",
        "256",
        "--gpu-memory-utilization",
        "0.3",
        "--trust-remote-code",
    ]
    with RemoteOpenAIServer(JINA_MODEL, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def gemma_server():
    args = [
        "--enforce-eager",
        "--max-model-len",
        "256",
        "--gpu-memory-utilization",
        "0.3",
        "--hf-overrides",
        json.dumps(GEMMA_HF_OVERRIDES),
        "--chat-template",
        GEMMA_TEMPLATE,
    ]
    with RemoteOpenAIServer(GEMMA_MODEL, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def qwen_server():
    args = [
        "--enforce-eager",
        "--max-model-len",
        "256",
        "--gpu-memory-utilization",
        "0.3",
        "--hf-overrides",
        json.dumps(QWEN_HF_OVERRIDES),
        "--chat-template",
        QWEN_TEMPLATE,
    ]
    with RemoteOpenAIServer(QWEN_MODEL, args) as remote_server:
        yield remote_server


# ── Jina v2 (cross-encoder, XLM-RoBERTa, sep token path) ──────────────


@pytest.mark.asyncio
async def test_jina_max_tokens_per_doc(jina_server: RemoteOpenAIServer):
    """Test max_tokens_per_doc with jina-reranker-v2 (XLM-RoBERTa cross-encoder)."""
    query = "What is the capital of France?"
    long_doc = "The capital of France is Paris. " * 20

    response = requests.post(
        jina_server.url_for("rerank"),
        json={
            "model": JINA_MODEL,
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


@pytest.mark.asyncio
async def test_jina_max_tokens_per_doc_reduces_tokens(
    jina_server: RemoteOpenAIServer,
):
    """Verify truncation actually reduces tokens for Jina v2 reranker."""
    query = "What is the capital of France?"
    doc = "The capital of France is Paris. " * 5

    response_no_limit = requests.post(
        jina_server.url_for("rerank"),
        json={
            "model": JINA_MODEL,
            "query": query,
            "documents": [doc],
        },
    )
    response_no_limit.raise_for_status()
    rerank_no_limit = RerankResponse.model_validate(response_no_limit.json())

    response_with_limit = requests.post(
        jina_server.url_for("rerank"),
        json={
            "model": JINA_MODEL,
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


# ── Gemma (chat template / LLM-as-reranker path) ──────────────────────


@pytest.mark.asyncio
async def test_gemma_max_tokens_per_doc(gemma_server: RemoteOpenAIServer):
    """Test max_tokens_per_doc with bge-reranker-v2-gemma (chat template path)."""
    query = "What is the capital of France?"
    long_doc = "The capital of France is Paris. " * 20

    response = requests.post(
        gemma_server.url_for("rerank"),
        json={
            "model": GEMMA_MODEL,
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


@pytest.mark.asyncio
async def test_gemma_max_tokens_per_doc_reduces_tokens(
    gemma_server: RemoteOpenAIServer,
):
    """Verify truncation actually reduces tokens for gemma reranker."""
    query = "What is the capital of France?"
    doc = "The capital of France is Paris. " * 5

    response_no_limit = requests.post(
        gemma_server.url_for("rerank"),
        json={
            "model": GEMMA_MODEL,
            "query": query,
            "documents": [doc],
        },
    )
    response_no_limit.raise_for_status()
    rerank_no_limit = RerankResponse.model_validate(response_no_limit.json())

    response_with_limit = requests.post(
        gemma_server.url_for("rerank"),
        json={
            "model": GEMMA_MODEL,
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


# ── Qwen3 (score template path) ───────────────────────────────────────


@pytest.mark.asyncio
async def test_qwen_max_tokens_per_doc(qwen_server: RemoteOpenAIServer):
    """Test max_tokens_per_doc with Qwen3-Reranker (score template path)."""
    query = "What is the capital of France?"
    long_doc = "The capital of France is Paris. " * 20

    response = requests.post(
        qwen_server.url_for("rerank"),
        json={
            "model": QWEN_MODEL,
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


@pytest.mark.asyncio
async def test_qwen_max_tokens_per_doc_reduces_tokens(
    qwen_server: RemoteOpenAIServer,
):
    """Verify truncation actually reduces tokens for Qwen3 reranker."""
    query = "What is the capital of France?"
    doc = "The capital of France is Paris. " * 5

    response_no_limit = requests.post(
        qwen_server.url_for("rerank"),
        json={
            "model": QWEN_MODEL,
            "query": query,
            "documents": [doc],
        },
    )
    response_no_limit.raise_for_status()
    rerank_no_limit = RerankResponse.model_validate(response_no_limit.json())

    response_with_limit = requests.post(
        qwen_server.url_for("rerank"),
        json={
            "model": QWEN_MODEL,
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
