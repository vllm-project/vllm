# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import requests
import torch
import torch.nn.functional as F

from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.pooling.pooling.protocol import PoolingResponse
from vllm.entrypoints.pooling.score.protocol import RerankResponse
from vllm.platforms import current_platform

MODEL_NAME = "BAAI/bge-reranker-base"
DTYPE = "bfloat16"
input_text = "This product was excellent and exceeded my expectations"
input_tokens = [0, 3293, 12996, 509, 40881, 136, 204839, 297, 759, 202702, 2]


@pytest.fixture(scope="module")
def server():
    args = ["--enforce-eager", "--max-model-len", "100", "--dtype", DTYPE]

    # ROCm: Use Flex Attention to support encoder-only self-attention.
    if current_platform.is_rocm():
        args.extend(["--attention-backend", "FLEX_ATTENTION"])

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_basic(server: RemoteOpenAIServer, model_name: str):
    # test /v1/models
    response = requests.get(server.url_for("/v1/models"))
    served_model = response.json()["data"][0]["id"]
    assert served_model == MODEL_NAME

    # test /tokenize
    response = requests.post(
        server.url_for("/tokenize"),
        json={"model": model_name, "prompt": input_text},
    )
    assert response.json()["tokens"] == input_tokens


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_rerank_texts(server: RemoteOpenAIServer, model_name: str):
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
    assert rerank.results[0].relevance_score >= 0.9
    assert rerank.results[1].relevance_score <= 0.01


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_top_n(server: RemoteOpenAIServer, model_name: str):
    query = "What is the capital of France?"
    documents = [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris.",
        "Cross-encoder models are neat",
    ]

    rerank_response = requests.post(
        server.url_for("rerank"),
        json={"model": model_name, "query": query, "documents": documents, "top_n": 2},
    )
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
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris.",
    ]

    rerank_response = requests.post(
        server.url_for("rerank"),
        json={"model": model_name, "query": query, "documents": documents},
    )
    assert rerank_response.status_code == 400
    # Assert just a small fragments of the response
    assert "Please reduce the length of the input." in rerank_response.text


def test_invocations(server: RemoteOpenAIServer):
    query = "What is the capital of France?"
    documents = [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris.",
    ]

    request_args = {
        "model": MODEL_NAME,
        "query": query,
        "documents": documents,
    }

    rerank_response = requests.post(server.url_for("rerank"), json=request_args)
    rerank_response.raise_for_status()

    invocation_response = requests.post(
        server.url_for("invocations"), json=request_args
    )
    invocation_response.raise_for_status()

    rerank_output = rerank_response.json()
    invocation_output = invocation_response.json()

    assert rerank_output.keys() == invocation_output.keys()
    for rerank_result, invocations_result in zip(
        rerank_output["results"], invocation_output["results"]
    ):
        assert rerank_result.keys() == invocations_result.keys()
        assert rerank_result["relevance_score"] == pytest.approx(
            invocations_result["relevance_score"], rel=0.05
        )
        # TODO: reset this tolerance to 0.01 once we find
        # an alternative to flash_attn with bfloat16


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_use_activation(server: RemoteOpenAIServer, model_name: str):
    async def get_outputs(use_activation):
        query = "What is the capital of France?"
        documents = [
            "The capital of Brazil is Brasilia.",
            "The capital of France is Paris.",
        ]

        response = requests.post(
            server.url_for("rerank"),
            json={
                "model": model_name,
                "query": query,
                "documents": documents,
                "use_activation": use_activation,
            },
        )
        outputs = response.json()

        return torch.tensor([x["relevance_score"] for x in outputs["results"]])

    default = await get_outputs(use_activation=None)
    w_activation = await get_outputs(use_activation=True)
    wo_activation = await get_outputs(use_activation=False)

    assert torch.allclose(default, w_activation, atol=1e-2), (
        "Default should use activation."
    )
    assert not torch.allclose(w_activation, wo_activation, atol=1e-2), (
        "wo_activation should not use activation."
    )
    assert torch.allclose(F.sigmoid(wo_activation), w_activation, atol=1e-2), (
        "w_activation should be close to activation(wo_activation)."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_pooling_classify(server: RemoteOpenAIServer, model_name: str):
    response = requests.post(
        server.url_for("pooling"),
        json={
            "model": model_name,
            "input": input_text,
            "encoding_format": "float",
            "task": "classify",
        },
    )
    poolings = PoolingResponse.model_validate(response.json())
    assert len(poolings.data) == 1
    assert len(poolings.data[0].data) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_pooling_token_classify(server: RemoteOpenAIServer, model_name: str):
    response = requests.post(
        server.url_for("pooling"),
        json={"model": model_name, "input": input_text, "encoding_format": "float"},
    )

    poolings = PoolingResponse.model_validate(response.json())

    assert len(poolings.data) == 1
    assert len(poolings.data[0].data) == len(input_tokens)
    assert len(poolings.data[0].data[0]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("task", ["embed", "token_embed", "plugin"])
async def test_pooling_not_supported(
    server: RemoteOpenAIServer, model_name: str, task: str
):
    response = requests.post(
        server.url_for("pooling"),
        json={
            "model": model_name,
            "input": input_text,
            "encoding_format": "float",
            "task": task,
        },
    )
    assert response.json()["error"]["type"] == "BadRequestError"
    assert response.json()["error"]["message"].startswith(
        f"Task {task} is not supported"
    )


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_max_tokens_per_doc(server: RemoteOpenAIServer, model_name: str):
    """Test that max_tokens_per_doc truncates documents correctly."""
    query = "What is the capital of France?"
    # Create a long document that will be truncated
    long_doc = "The capital of France is Paris. " * 50
    short_doc = "The capital of Brazil is Brasilia."

    # Test with max_tokens_per_doc set
    rerank_response = requests.post(
        server.url_for("rerank"),
        json={
            "model": model_name,
            "query": query,
            "documents": [long_doc, short_doc],
            "max_tokens_per_doc": 10,
        },
    )
    rerank_response.raise_for_status()
    rerank = RerankResponse.model_validate(rerank_response.json())

    assert rerank.id is not None
    assert rerank.results is not None
    assert len(rerank.results) == 2

    # The truncated long doc should have fewer tokens used
    # Verify that the request succeeded with truncation
    assert rerank.usage.prompt_tokens > 0


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_max_tokens_per_doc_reduces_tokens(server: RemoteOpenAIServer, model_name: str):
    """Test that max_tokens_per_doc actually reduces the token count."""
    query = "What is the capital of France?"
    long_doc = "The capital of France is Paris. " * 20

    # Without max_tokens_per_doc
    response_no_limit = requests.post(
        server.url_for("rerank"),
        json={
            "model": model_name,
            "query": query,
            "documents": [long_doc],
        },
    )
    response_no_limit.raise_for_status()
    rerank_no_limit = RerankResponse.model_validate(response_no_limit.json())

    # With max_tokens_per_doc
    response_with_limit = requests.post(
        server.url_for("rerank"),
        json={
            "model": model_name,
            "query": query,
            "documents": [long_doc],
            "max_tokens_per_doc": 10,
        },
    )
    response_with_limit.raise_for_status()
    rerank_with_limit = RerankResponse.model_validate(response_with_limit.json())

    # The version with truncation should use fewer tokens
    assert rerank_with_limit.usage.prompt_tokens < rerank_no_limit.usage.prompt_tokens


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_max_tokens_per_doc_validation(server: RemoteOpenAIServer, model_name: str):
    """Test that max_tokens_per_doc validation works correctly."""
    query = "What is the capital of France?"
    documents = ["The capital of France is Paris."]

    # Test with invalid max_tokens_per_doc (0)
    response = requests.post(
        server.url_for("rerank"),
        json={
            "model": model_name,
            "query": query,
            "documents": documents,
            "max_tokens_per_doc": 0,
        },
    )
    assert response.status_code == 400
    assert "max_tokens_per_doc must be a positive integer" in response.text

    # Test with invalid max_tokens_per_doc (negative)
    response = requests.post(
        server.url_for("rerank"),
        json={
            "model": model_name,
            "query": query,
            "documents": documents,
            "max_tokens_per_doc": -5,
        },
    )
    assert response.status_code == 400
    assert "max_tokens_per_doc must be a positive integer" in response.text
