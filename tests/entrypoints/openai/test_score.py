# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any

import pytest
import requests
import torch.nn.functional as F
from torch import tensor

from vllm.entrypoints.openai.protocol import ScoreResponse

from ...utils import RemoteOpenAIServer

MODELS = [
    {
        "name": "BAAI/bge-reranker-v2-m3",
        "is_cross_encoder": True
    },
    {
        "name": "BAAI/bge-base-en-v1.5",
        "is_cross_encoder": False
    },
]
DTYPE = "half"


def run_transformers(hf_model, model, text_pairs):
    if model["is_cross_encoder"]:
        return hf_model.predict(text_pairs).tolist()
    else:
        hf_embeddings = [
            hf_model.encode(text_pair) for text_pair in text_pairs
        ]
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

    with RemoteOpenAIServer(model["name"], args) as remote_server:
        yield remote_server


@pytest.fixture(scope="class")
def runner(model: dict[str, Any], hf_runner):
    kwargs = {
        "dtype": DTYPE,
        "is_cross_encoder" if model["is_cross_encoder"]\
              else "is_sentence_transformer": True
    }

    with hf_runner(model["name"], **kwargs) as hf_model:
        yield hf_model


class TestModel:

    def test_text_1_str_text_2_list(self, server: RemoteOpenAIServer,
                                    model: dict[str, Any], runner):
        text_1 = "What is the capital of France?"
        text_2 = [
            "The capital of Brazil is Brasilia.",
            "The capital of France is Paris."
        ]

        score_response = requests.post(server.url_for("score"),
                                       json={
                                           "model": model["name"],
                                           "text_1": text_1,
                                           "text_2": text_2,
                                       })
        score_response.raise_for_status()
        score = ScoreResponse.model_validate(score_response.json())

        assert score.id is not None
        assert score.data is not None
        assert len(score.data) == 2

        vllm_outputs = [d.score for d in score.data]

        text_pairs = [[text_1, text_2[0]], [text_1, text_2[1]]]
        hf_outputs = run_transformers(runner, model, text_pairs)

        for i in range(len(vllm_outputs)):
            assert math.isclose(hf_outputs[i], vllm_outputs[i], rel_tol=0.01)

    def test_text_1_list_text_2_list(self, server: RemoteOpenAIServer,
                                     model: dict[str, Any], runner):
        text_1 = [
            "What is the capital of the United States?",
            "What is the capital of France?"
        ]
        text_2 = [
            "The capital of Brazil is Brasilia.",
            "The capital of France is Paris."
        ]

        score_response = requests.post(server.url_for("score"),
                                       json={
                                           "model": model["name"],
                                           "text_1": text_1,
                                           "text_2": text_2,
                                       })
        score_response.raise_for_status()
        score = ScoreResponse.model_validate(score_response.json())

        assert score.id is not None
        assert score.data is not None
        assert len(score.data) == 2

        vllm_outputs = [d.score for d in score.data]

        text_pairs = [[text_1[0], text_2[0]], [text_1[1], text_2[1]]]
        hf_outputs = run_transformers(runner, model, text_pairs)

        for i in range(len(vllm_outputs)):
            assert math.isclose(hf_outputs[i], vllm_outputs[i], rel_tol=0.01)

    def test_text_1_str_text_2_str(self, server: RemoteOpenAIServer,
                                   model: dict[str, Any], runner):
        text_1 = "What is the capital of France?"
        text_2 = "The capital of France is Paris."

        score_response = requests.post(server.url_for("score"),
                                       json={
                                           "model": model["name"],
                                           "text_1": text_1,
                                           "text_2": text_2,
                                       })
        score_response.raise_for_status()
        score = ScoreResponse.model_validate(score_response.json())

        assert score.id is not None
        assert score.data is not None
        assert len(score.data) == 1

        vllm_outputs = [d.score for d in score.data]

        text_pairs = [[text_1, text_2]]
        hf_outputs = run_transformers(runner, model, text_pairs)

        for i in range(len(vllm_outputs)):
            assert math.isclose(hf_outputs[i], vllm_outputs[i], rel_tol=0.01)

    def test_score_max_model_len(self, server: RemoteOpenAIServer,
                                 model: dict[str, Any]):

        text_1 = "What is the capital of France?" * 20
        text_2 = [
            "The capital of Brazil is Brasilia.",
            "The capital of France is Paris."
        ]

        score_response = requests.post(server.url_for("score"),
                                       json={
                                           "model": model["name"],
                                           "text_1": text_1,
                                           "text_2": text_2,
                                       })
        assert score_response.status_code == 400
        # Assert just a small fragments of the response
        assert "Please reduce the length of the input." in \
            score_response.text

        # Test truncation
        score_response = requests.post(server.url_for("score"),
                                       json={
                                           "model": model["name"],
                                           "text_1": text_1,
                                           "text_2": text_2,
                                           "truncate_prompt_tokens": 101
                                       })
        assert score_response.status_code == 400
        assert "Please, select a smaller truncation size." in \
            score_response.text
