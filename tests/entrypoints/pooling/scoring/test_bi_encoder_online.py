# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import requests

from tests.entrypoints.pooling.scoring.util import EncoderScoringHfRunner
from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.pooling.pooling.protocol import PoolingResponse
from vllm.entrypoints.pooling.score.protocol import RerankResponse, ScoreResponse
from vllm.platforms import current_platform

MODEL_NAME = "BAAI/bge-base-en-v1.5"
input_text = "This product was excellent and exceeded my expectations"
DTYPE = "half"
EMBEDDING_SIZE = 768

TEXTS_1 = [
    "What is the capital of France?",
    "What is the capital of Germany?",
]

TEXTS_2 = [
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
]


def _assert_score_output_matches_hf(hf_model, server: RemoteOpenAIServer,
                                    payload: dict, text_pairs: list[list[str]]):
    score_response = requests.post(
        server.url_for("score"),
        json={"model": MODEL_NAME, **payload},
    )
    score_response.raise_for_status()
    score = ScoreResponse.model_validate(score_response.json())

    assert score.id is not None
    assert score.data is not None

    vllm_outputs = [d.score for d in score.data]
    hf_outputs = hf_model.predict(text_pairs).tolist()
    assert len(vllm_outputs) == len(hf_outputs)
    for hf_output, vllm_output in zip(hf_outputs, vllm_outputs):
        assert hf_output == pytest.approx(vllm_output, rel=0.01)


@pytest.fixture(scope="module")
def server():
    args = ["--enforce-eager", "--max-model-len", "100", "--dtype", DTYPE]

    # ROCm: Use Flex Attention to support encoder-only self-attention.
    if current_platform.is_rocm():
        args.extend(["--attention-backend", "FLEX_ATTENTION"])

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def hf_model():
    return EncoderScoringHfRunner(MODEL_NAME)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "text_pairs"),
    [
        pytest.param(
            {
                "queries": TEXTS_1[0],
                "documents": TEXTS_2[0],
            },
            [[TEXTS_1[0], TEXTS_2[0]]],
            id="queries-str-documents-str",
        ),
        pytest.param(
            {
                "queries": TEXTS_1[0],
                "documents": TEXTS_2,
            },
            [
                [TEXTS_1[0], TEXTS_2[0]],
                [TEXTS_1[0], TEXTS_2[1]],
            ],
            id="queries-str-documents-list",
        ),
        pytest.param(
            {
                "queries": TEXTS_1,
                "documents": TEXTS_2,
            },
            [
                [TEXTS_1[0], TEXTS_2[0]],
                [TEXTS_1[1], TEXTS_2[1]],
            ],
            id="queries-list-documents-list",
        ),
        pytest.param(
            {
                "queries": TEXTS_1,
                "items": TEXTS_2,
            },
            [
                [TEXTS_1[0], TEXTS_2[0]],
                [TEXTS_1[1], TEXTS_2[1]],
            ],
            id="queries-list-items-list",
        ),
        pytest.param(
            {
                "text_1": TEXTS_1,
                "text_2": TEXTS_2,
            },
            [
                [TEXTS_1[0], TEXTS_2[0]],
                [TEXTS_1[1], TEXTS_2[1]],
            ],
            id="text-1-vs-text-2",
        ),
        pytest.param(
            {
                "data_1": TEXTS_1,
                "data_2": TEXTS_2,
            },
            [
                [TEXTS_1[0], TEXTS_2[0]],
                [TEXTS_1[1], TEXTS_2[1]],
            ],
            id="data-1-vs-data-2",
        ),
    ],
)
async def test_score_api_request_formats(hf_model, server: RemoteOpenAIServer,
                                         payload: dict,
                                         text_pairs: list[list[str]]):
    _assert_score_output_matches_hf(hf_model, server, payload, text_pairs)


@pytest.mark.asyncio
async def test_rerank_api_texts(server: RemoteOpenAIServer):
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
    query = "What is the capital of France?"
    documents = [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris.",
        "Cross-encoder models are neat",
    ]

    rerank_response = requests.post(
        server.url_for("rerank"),
        json={"model": MODEL_NAME, "query": query, "documents": documents, "top_n": 2},
    )
    rerank_response.raise_for_status()
    rerank = RerankResponse.model_validate(rerank_response.json())

    assert rerank.id is not None
    assert rerank.results is not None
    assert len(rerank.results) == 2
    assert rerank.results[0].index == 1


@pytest.mark.asyncio
async def test_rerank_api_max_model_len(server: RemoteOpenAIServer):
    query = "What is the capital of France?" * 100
    documents = [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris.",
    ]

    rerank_response = requests.post(
        server.url_for("rerank"),
        json={"model": MODEL_NAME, "query": query, "documents": documents},
    )
    assert rerank_response.status_code == 400
    # Assert just a small fragments of the response
    assert "Please reduce the length of the input prompt" in rerank_response.text


@pytest.mark.asyncio
async def test_score_api_max_model_len(server: RemoteOpenAIServer):
    queries = "What is the capital of France?" * 20
    documents = [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris.",
    ]

    score_response = requests.post(
        server.url_for("score"),
        json={
            "model": MODEL_NAME,
            "queries": queries,
            "documents": documents,
        },
    )
    assert score_response.status_code == 400
    # Assert just a small fragments of the response
    assert "Please reduce the length of the input prompt" in score_response.text

    # Test truncation
    score_response = requests.post(
        server.url_for("score"),
        json={
            "model": MODEL_NAME,
            "queries": queries,
            "documents": documents,
            "truncate_prompt_tokens": 101,
        },
    )
    assert score_response.status_code == 400
    assert "Please request a smaller truncation size." in score_response.text


@pytest.mark.asyncio
async def test_invocations(server: RemoteOpenAIServer):
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
            invocations_result["relevance_score"], rel=0.01
        )


@pytest.mark.asyncio
async def test_pooling_embed(server: RemoteOpenAIServer):
    response = requests.post(
        server.url_for("pooling"),
        json={
            "model": MODEL_NAME,
            "input": input_text,
            "encoding_format": "float",
            "task": "embed",
        },
    )
    poolings = PoolingResponse.model_validate(response.json())
    assert len(poolings.data) == 1
    assert len(poolings.data[0].data) == EMBEDDING_SIZE


@pytest.mark.asyncio
@pytest.mark.parametrize("task", ["classify", "token_classify", "plugin"])
async def test_pooling_not_supported(server: RemoteOpenAIServer, task: str):
    response = requests.post(
        server.url_for("pooling"),
        json={
            "model": MODEL_NAME,
            "input": input_text,
            "encoding_format": "float",
            "task": task,
        },
    )
    assert response.json()["error"]["type"] == "BadRequestError"
    assert response.json()["error"]["message"].startswith(f"Unsupported task: {task!r}")
