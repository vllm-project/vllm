# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import requests

from tests.entrypoints.pooling.scoring.util import (
    make_base64_image,
    make_image_mm_param,
)
from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.pooling.scoring.protocol import RerankResponse, ScoreResponse

MODEL_NAME = "vidore/colpali-v1.3-hf"


@pytest.fixture(scope="module")
def server():
    with RemoteOpenAIServer(MODEL_NAME, []) as remote_server:
        yield remote_server


@pytest.mark.asyncio
async def test_score_api_query_text_vs_docs_image(server: RemoteOpenAIServer):
    query = "Describe the red object"

    red_image = make_base64_image(64, 64, color=(255, 0, 0))
    blue_image = make_base64_image(64, 64, color=(0, 0, 255))

    documents = [
        make_image_mm_param(red_image),
        make_image_mm_param(blue_image),
    ]

    score_response = requests.post(
        server.url_for("score"),
        json={
            "model": MODEL_NAME,
            "queries": query,
            "documents": documents,
        },
    )
    score_response.raise_for_status()
    scores = ScoreResponse.model_validate(score_response.json())

    assert scores.id is not None
    assert scores.data is not None
    assert len(scores.data) == 2
    assert scores.data[0].score > scores.data[1].score


@pytest.mark.asyncio
async def test_score_api_query_text_vs_docs_mix(server: RemoteOpenAIServer):
    red_image = make_base64_image(64, 64, color=(255, 0, 0))
    query = "What is the capital of France?"
    documents: list = [
        "The capital of France is Paris.",
        make_image_mm_param(red_image),
    ]

    score_response = requests.post(
        server.url_for("score"),
        json={
            "model": MODEL_NAME,
            "queries": query,
            "documents": documents,
        },
    )
    score_response.raise_for_status()
    scores = ScoreResponse.model_validate(score_response.json())

    assert scores.id is not None
    assert scores.data is not None
    assert len(scores.data) == 2
    assert scores.data[0].score > scores.data[1].score


@pytest.mark.asyncio
async def test_score_api_query_image_vs_docs_text(server: RemoteOpenAIServer):
    red_image = make_base64_image(64, 64, color=(255, 0, 0))
    image_query = make_image_mm_param(red_image, text="red color")

    documents = [
        "Describe the red object.",
        "The capital of France is Paris.",
    ]

    score_response = requests.post(
        server.url_for("score"),
        json={
            "model": MODEL_NAME,
            "queries": image_query,
            "documents": documents,
        },
    )
    score_response.raise_for_status()
    scores = ScoreResponse.model_validate(score_response.json())

    assert scores.id is not None
    assert scores.data is not None
    assert len(scores.data) == 2
    assert scores.data[0].score > scores.data[1].score


@pytest.mark.asyncio
async def test_rerank_api_query_text_vs_docs_image(server: RemoteOpenAIServer):
    query = "Describe the red object"

    red_image = make_base64_image(64, 64, color=(255, 0, 0))
    blue_image = make_base64_image(64, 64, color=(0, 0, 255))

    documents = [
        make_image_mm_param(red_image),
        make_image_mm_param(blue_image),
    ]

    rerank_response = requests.post(
        server.url_for("rerank"),
        json={"model": MODEL_NAME, "query": query, "documents": documents},
    )

    rerank_response.raise_for_status()
    rerank = RerankResponse.model_validate(rerank_response.json())

    assert rerank.id is not None
    assert rerank.results is not None
    assert len(rerank.results) == 2

    red_result = next(r for r in rerank.results if r.index == 0)
    blue_result = next(r for r in rerank.results if r.index == 1)

    assert red_result.relevance_score > blue_result.relevance_score


@pytest.mark.asyncio
async def test_rerank_api_query_text_vs_docs_mix(server: RemoteOpenAIServer):
    red_image = make_base64_image(64, 64, color=(255, 0, 0))
    query = "What is the capital of France?"
    documents: list = [
        "The capital of France is Paris.",
        make_image_mm_param(red_image),
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

    result0 = next(r for r in rerank.results if r.index == 0)
    result1 = next(r for r in rerank.results if r.index == 1)

    assert result0.relevance_score > result1.relevance_score


@pytest.mark.asyncio
async def test_rerank_api_query_image_vs_docs_text(server: RemoteOpenAIServer):
    red_image = make_base64_image(64, 64, color=(255, 0, 0))
    image_query = make_image_mm_param(red_image, text="red color")

    documents = [
        "Describe the red object.",
        "The capital of France is Paris.",
    ]

    rerank_response = requests.post(
        server.url_for("rerank"),
        json={
            "model": MODEL_NAME,
            "query": image_query,
            "documents": documents,
        },
    )
    rerank_response.raise_for_status()
    rerank = RerankResponse.model_validate(rerank_response.json())

    assert rerank.id is not None
    assert rerank.results is not None
    assert len(rerank.results) == 2

    result0 = next(r for r in rerank.results if r.index == 0)
    result1 = next(r for r in rerank.results if r.index == 1)

    assert result0.relevance_score > result1.relevance_score
