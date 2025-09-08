# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest
import requests

from vllm.entrypoints.openai.protocol import RerankResponse, ScoreResponse

from ...utils import RemoteOpenAIServer


# Minimal CPU smoke test
# These are intentionally very lightweight to guard encoder-only /
# cross-encoder paths on CPU backend.
# Models are marked so they run in the CPU test matrix.

MODELS = [
    pytest.param(
        {"name": "BAAI/bge-base-en-v1.5", "is_cross_encoder": False},
        marks=[pytest.mark.core_model, pytest.mark.cpu_model],
    ),
    pytest.param(
        {
            "name": "sentence-transformers/stsb-roberta-base-v2",
            "is_cross_encoder": True,
        },
        marks=[pytest.mark.core_model, pytest.mark.cpu_model],
    ),
]

DTYPE = "float32"


@pytest.fixture(scope="class", params=MODELS)
def model(request):
    yield request.param


@pytest.fixture(scope="class")
def server(model: dict[str, Any]):
    # Force CPU backend explicitly
    args = [
        "--enforce-eager",
        "--max-model-len",
        "64",
        "--dtype",
        DTYPE,
        "--device",
        "cpu",
        "--runner",
        "pooling",
        "--task",
        "score",
    ]

    with RemoteOpenAIServer(model["name"], args) as remote_server:
        yield remote_server


@pytest.mark.timeout(60)
def test_score_smoke(server: RemoteOpenAIServer, model: dict[str, Any]):
    """Single /score call should not crash on CPU."""
    text_1 = "What is the capital of France?"
    text_2 = "The capital of France is Paris."

    resp = requests.post(
        server.url_for("score"),
        json={
            "model": model["name"],
            "text_1": text_1,
            "text_2": text_2,
        },
    )
    resp.raise_for_status()
    _ = ScoreResponse.model_validate(resp.json())


@pytest.fixture(scope="class",
                params=[pytest.param(
                    "sentence-transformers/stsb-roberta-base-v2",
                    marks=[pytest.mark.core_model, pytest.mark.cpu_model])])
def cx_model(request) -> str:
    return request.param


@pytest.fixture(scope="class")
def server_cx(cx_model: str):
    args = [
        "--enforce-eager",
        "--max-model-len",
        "64",
        "--dtype",
        DTYPE,
        "--device",
        "cpu",
        "--runner",
        "pooling",
        "--task",
        "score",
    ]
    with RemoteOpenAIServer(cx_model, args) as remote_server:
        yield remote_server


@pytest.mark.timeout(60)
def test_rerank_smoke(server_cx: RemoteOpenAIServer, cx_model: str):
    """Single /rerank call should not crash on CPU for cross-encoder."""
    query = "What is the capital of France?"
    documents = [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris.",
    ]

    resp = requests.post(
        server_cx.url_for("rerank"),
        json={
            "model": cx_model,
            "query": query,
            "documents": documents,
        },
    )
    resp.raise_for_status()
    _ = RerankResponse.model_validate(resp.json())
