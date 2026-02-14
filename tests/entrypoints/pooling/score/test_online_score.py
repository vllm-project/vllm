# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import pytest
import requests
import torch
import torch.nn.functional as F
from torch import tensor

from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.pooling.score.protocol import ScoreResponse
from vllm.platforms import current_platform

MODELS = [
    {"name": "BAAI/bge-reranker-v2-m3", "is_cross_encoder": True},
    {"name": "BAAI/bge-base-en-v1.5", "is_cross_encoder": False},
]
DTYPE = "half"


def run_transformers(hf_model, model, text_pairs):
    if model["is_cross_encoder"]:
        return hf_model.predict(text_pairs).tolist()
    else:
        hf_embeddings = [hf_model.encode(text_pair) for text_pair in text_pairs]
        return [
            F.cosine_similarity(tensor(pair[0]), tensor(pair[1]), dim=0)
            for pair in hf_embeddings
        ]


@pytest.fixture(scope="class", params=MODELS)
def model(request):
    yield request.param


@pytest.fixture(scope="class")
def server(model: dict[str, Any]):
    args = ["--enforce-eager", "--max-model-len", "100", "--dtype", DTYPE]

    # ROCm: Use Flex Attention to support encoder-only self-attention.
    if current_platform.is_rocm():
        args.extend(["--attention-backend", "FLEX_ATTENTION"])

    with RemoteOpenAIServer(model["name"], args) as remote_server:
        yield remote_server


@pytest.fixture(scope="class")
def runner(model: dict[str, Any], hf_runner):
    kwargs = {
        "dtype": DTYPE,
        "is_cross_encoder"
        if model["is_cross_encoder"]
        else "is_sentence_transformer": True,
    }

    with hf_runner(model["name"], **kwargs) as hf_model:
        yield hf_model


class TestModel:
    def test_queries_str_documents_str(
        self, server: RemoteOpenAIServer, model: dict[str, Any], runner
    ):
        queries = "What is the capital of France?"
        documents = "The capital of France is Paris."

        score_response = requests.post(
            server.url_for("score"),
            json={
                "model": model["name"],
                "queries": queries,
                "documents": documents,
            },
        )
        score_response.raise_for_status()
        score = ScoreResponse.model_validate(score_response.json())

        assert score.id is not None
        assert score.data is not None
        assert len(score.data) == 1

        vllm_outputs = [d.score for d in score.data]

        text_pairs = [[queries, documents]]
        hf_outputs = run_transformers(runner, model, text_pairs)

        for i in range(len(vllm_outputs)):
            assert hf_outputs[i] == pytest.approx(vllm_outputs[i], rel=0.01)

    def test_queries_str_items_str(
        self, server: RemoteOpenAIServer, model: dict[str, Any], runner
    ):
        queries = "What is the capital of France?"
        items = "The capital of France is Paris."

        score_response = requests.post(
            server.url_for("score"),
            json={
                "model": model["name"],
                "queries": queries,
                "items": items,
            },
        )
        score_response.raise_for_status()
        score = ScoreResponse.model_validate(score_response.json())

        assert score.id is not None
        assert score.data is not None
        assert len(score.data) == 1

        vllm_outputs = [d.score for d in score.data]

        text_pairs = [[queries, items]]
        hf_outputs = run_transformers(runner, model, text_pairs)

        for i in range(len(vllm_outputs)):
            assert hf_outputs[i] == pytest.approx(vllm_outputs[i], rel=0.01)

    def test_text_1_str_text_2_str(
        self, server: RemoteOpenAIServer, model: dict[str, Any], runner
    ):
        text_1 = "What is the capital of France?"
        text_2 = "The capital of France is Paris."

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
        assert len(score.data) == 1

        vllm_outputs = [d.score for d in score.data]

        text_pairs = [[text_1, text_2]]
        hf_outputs = run_transformers(runner, model, text_pairs)

        for i in range(len(vllm_outputs)):
            assert hf_outputs[i] == pytest.approx(vllm_outputs[i], rel=0.01)

    def test_data_1_str_data_2_str(
        self, server: RemoteOpenAIServer, model: dict[str, Any], runner
    ):
        data_1 = "What is the capital of France?"
        data_2 = "The capital of France is Paris."

        score_response = requests.post(
            server.url_for("score"),
            json={
                "model": model["name"],
                "data_1": data_1,
                "data_2": data_2,
            },
        )
        score_response.raise_for_status()
        score = ScoreResponse.model_validate(score_response.json())

        assert score.id is not None
        assert score.data is not None
        assert len(score.data) == 1

        vllm_outputs = [d.score for d in score.data]

        text_pairs = [[data_1, data_2]]
        hf_outputs = run_transformers(runner, model, text_pairs)

        for i in range(len(vllm_outputs)):
            assert hf_outputs[i] == pytest.approx(vllm_outputs[i], rel=0.01)

    def test_queries_str_documents_list(
        self, server: RemoteOpenAIServer, model: dict[str, Any], runner
    ):
        queries = "What is the capital of France?"
        documents = [
            "The capital of Brazil is Brasilia.",
            "The capital of France is Paris.",
        ]

        score_response = requests.post(
            server.url_for("score"),
            json={
                "model": model["name"],
                "queries": queries,
                "documents": documents,
            },
        )
        score_response.raise_for_status()
        score = ScoreResponse.model_validate(score_response.json())

        assert score.id is not None
        assert score.data is not None
        assert len(score.data) == 2

        vllm_outputs = [d.score for d in score.data]

        text_pairs = [[queries, documents[0]], [queries, documents[1]]]
        hf_outputs = run_transformers(runner, model, text_pairs)

        for i in range(len(vllm_outputs)):
            assert hf_outputs[i] == pytest.approx(vllm_outputs[i], rel=0.01)

    def test_queries_list_documents_list(
        self, server: RemoteOpenAIServer, model: dict[str, Any], runner
    ):
        queries = [
            "What is the capital of the United States?",
            "What is the capital of France?",
        ]
        documents = [
            "The capital of Brazil is Brasilia.",
            "The capital of France is Paris.",
        ]

        score_response = requests.post(
            server.url_for("score"),
            json={
                "model": model["name"],
                "queries": queries,
                "documents": documents,
            },
        )
        score_response.raise_for_status()
        score = ScoreResponse.model_validate(score_response.json())

        assert score.id is not None
        assert score.data is not None
        assert len(score.data) == 2

        vllm_outputs = [d.score for d in score.data]

        text_pairs = [[queries[0], documents[0]], [queries[1], documents[1]]]
        hf_outputs = run_transformers(runner, model, text_pairs)

        for i in range(len(vllm_outputs)):
            assert hf_outputs[i] == pytest.approx(vllm_outputs[i], rel=0.01)

    def test_score_max_model_len(
        self, server: RemoteOpenAIServer, model: dict[str, Any]
    ):
        queries = "What is the capital of France?" * 20
        documents = [
            "The capital of Brazil is Brasilia.",
            "The capital of France is Paris.",
        ]

        score_response = requests.post(
            server.url_for("score"),
            json={
                "model": model["name"],
                "queries": queries,
                "documents": documents,
            },
        )
        assert score_response.status_code == 400
        # Assert just a small fragments of the response
        assert "Please reduce the length of the input." in score_response.text

        # Test truncation
        score_response = requests.post(
            server.url_for("score"),
            json={
                "model": model["name"],
                "queries": queries,
                "documents": documents,
                "truncate_prompt_tokens": 101,
            },
        )
        assert score_response.status_code == 400
        assert "Please request a smaller truncation size." in score_response.text

    def test_invocations(self, server: RemoteOpenAIServer, model: dict[str, Any]):
        queries = "What is the capital of France?"
        documents = "The capital of France is Paris."

        request_args = {
            "model": model["name"],
            "queries": queries,
            "documents": documents,
        }

        score_response = requests.post(server.url_for("score"), json=request_args)
        score_response.raise_for_status()

        invocation_response = requests.post(
            server.url_for("invocations"), json=request_args
        )
        invocation_response.raise_for_status()

        score_output = score_response.json()
        invocation_output = invocation_response.json()

        assert score_output.keys() == invocation_output.keys()
        for score_data, invocation_data in zip(
            score_output["data"], invocation_output["data"]
        ):
            assert score_data.keys() == invocation_data.keys()
            assert score_data["score"] == pytest.approx(
                invocation_data["score"], rel=0.05
            )
            # TODO: reset this tolerance to 0.01 once we find
            # an alternative to flash_attn with bfloat16

    def test_use_activation(self, server: RemoteOpenAIServer, model: dict[str, Any]):
        def get_outputs(use_activation):
            queries = "What is the capital of France?"
            documents = "The capital of France is Paris."
            response = requests.post(
                server.url_for("score"),
                json={
                    "model": model["name"],
                    "queries": queries,
                    "documents": documents,
                    "use_activation": use_activation,
                },
            )
            outputs = response.json()
            return torch.tensor([x["score"] for x in outputs["data"]])

        default = get_outputs(use_activation=None)
        w_activation = get_outputs(use_activation=True)
        wo_activation = get_outputs(use_activation=False)

        if model["is_cross_encoder"]:
            assert torch.allclose(default, w_activation, atol=1e-2), (
                "Default should use activation."
            )
            assert not torch.allclose(w_activation, wo_activation, atol=1e-2), (
                "wo_activation should not use activation."
            )
            assert torch.allclose(F.sigmoid(wo_activation), w_activation, atol=1e-2), (
                "w_activation should be close to activation(wo_activation)."
            )
