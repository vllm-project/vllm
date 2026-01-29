# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import requests

from tests.utils import VLLM_PATH, RemoteOpenAIServer
from vllm.entrypoints.pooling.score.protocol import ScoreResponse
from vllm.multimodal.utils import encode_image_url, fetch_image

MODEL_NAME = "Qwen/Qwen3-VL-Reranker-2B"
HF_OVERRIDES = {
    "architectures": ["Qwen3VLForSequenceClassification"],
    "classifier_from_token": ["no", "yes"],
    "is_original_qwen3_reranker": True,
}

query = "A cat standing in the snow."
image_url = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/cat_snow.jpg"
documents = [
    {
        "type": "text",
        "text": query,
    },
    {
        "type": "image_url",
        "image_url": {"url": image_url},
    },
    {
        "type": "image_url",
        "image_url": {"url": encode_image_url(fetch_image(image_url))},
    },
]


@pytest.fixture(scope="module")
def server():
    args = [
        "--enforce-eager",
        "--max-model-len",
        "8192",
        "--chat-template",
        str(VLLM_PATH / "examples/pooling/score/template/qwen3_vl_reranker.jinja"),
    ]

    with RemoteOpenAIServer(
        MODEL_NAME, args, override_hf_configs=HF_OVERRIDES
    ) as remote_server:
        yield remote_server


def test_score_api_queries_str_documents_str(server: RemoteOpenAIServer):
    queries = "What is the capital of France?"
    documents = "The capital of France is Paris."

    score_response = requests.post(
        server.url_for("score"),
        json={
            "model": MODEL_NAME,
            "queries": queries,
            "documents": documents,
        },
    )
    score_response.raise_for_status()
    score = ScoreResponse.model_validate(score_response.json())

    assert score.id is not None
    assert score.data is not None
    assert len(score.data) == 1


def test_score_api_queries_str_documents_text_content(server: RemoteOpenAIServer):
    score_response = requests.post(
        server.url_for("score"),
        json={
            "model": MODEL_NAME,
            "queries": query,
            "documents": {"content": [documents[0]]},
        },
    )
    score_response.raise_for_status()
    score = ScoreResponse.model_validate(score_response.json())

    assert score.id is not None
    assert score.data is not None
    assert len(score.data) == 1


def test_score_api_queries_str_documents_image_url_content(server: RemoteOpenAIServer):
    score_response = requests.post(
        server.url_for("score"),
        json={
            "model": MODEL_NAME,
            "queries": query,
            "documents": {"content": [documents[1]]},
        },
    )
    score_response.raise_for_status()
    score = ScoreResponse.model_validate(score_response.json())

    assert score.id is not None
    assert score.data is not None
    assert len(score.data) == 1


def test_score_api_queries_str_documents_image_base64_content(
    server: RemoteOpenAIServer,
):
    score_response = requests.post(
        server.url_for("score"),
        json={
            "model": MODEL_NAME,
            "queries": query,
            "documents": {"content": [documents[2]]},
        },
    )
    score_response.raise_for_status()
    score = ScoreResponse.model_validate(score_response.json())

    assert score.id is not None
    assert score.data is not None
    assert len(score.data) == 1
