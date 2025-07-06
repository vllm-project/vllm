# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import pytest
import requests
import torch.nn.functional as F
from torch import tensor

from vllm.entrypoints.openai.protocol import RerankResponse, ScoreResponse

from ...utils import RemoteOpenAIServer


@pytest.fixture(autouse=True)
def v1(run_with_both_engines):
    # Simple autouse wrapper to run both engines for each test
    # This can be promoted up to conftest.py to run for every
    # test in a package
    pass


MODELS = [
    {
        "name": "BAAI/bge-reranker-v2-m3",
        "is_cross_encoder": True
    },
    {
        "name": "BAAI/bge-base-en-v1.5",
        "is_cross_encoder": False
    },
    {
        "name": "Qwen/Qwen3-Reranker-0.6B",
        "is_cross_encoder": True,
        "is_qwen3_reranker": True,
    },
]
DTYPE = "half"


def _run_qwen3_reranker_hf(hf_model, text_pairs, instruction):
    """Helper to run Qwen3 reranker with HF, applying the template."""
    prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    formatted_pairs = []
    for query, doc in text_pairs:
        q_formatted = f"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
        d_formatted = f"<Document>: {doc}{suffix}"
        formatted_pairs.append([q_formatted, d_formatted])

    return hf_model.predict(formatted_pairs).tolist()


def run_transformers(hf_model, model, text_pairs):
    if model.get("is_qwen3_reranker"):
        # The default instruction used in the server fixture.
        default_instruction = "Given a web search query, retrieve relevant passages that answer the query"
        return _run_qwen3_reranker_hf(hf_model, text_pairs,
                                      default_instruction)
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
    args = ["--enforce-eager", "--max-model-len", "256", "--dtype", DTYPE]
    if model.get("is_qwen3_reranker"):
        import json
        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        default_instruction = "Given a web search query, retrieve relevant passages that answer the query"

        hf_overrides = {
            "architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
            "score_template": {
                "query_template":
                f"{prefix}<Instruct>: {{instruction}}\n<Query>: {{query}}\n",
                "document_template": f"<Document>: {{document}}{suffix}",
                "default_context": {
                    "instruction": default_instruction
                }
            }
        }
        args.extend(["--hf-overrides", json.dumps(hf_overrides)])

    with RemoteOpenAIServer(model["name"], args) as remote_server:
        yield remote_server


@pytest.fixture(scope="class")
def runner(model: dict[str, Any], hf_runner):
    model_name = model["name"]
    kwargs = {"dtype": DTYPE}
    if model.get("is_qwen3_reranker"):
        # For the HF reference, use the pre-converted Sequence Classification
        # model to simplify the runner logic.
        model_name = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"
        hf_runner_kwargs = {
            "dtype": DTYPE,
            "is_cross_encoder": True,
            "trust_remote_code": True,
        }
    elif model["is_cross_encoder"]:
        hf_runner_kwargs = {"dtype": DTYPE, "is_cross_encoder": True}
    else:
        hf_runner_kwargs = {"dtype": DTYPE, "is_sentence_transformer": True}

    with hf_runner(model_name, **hf_runner_kwargs) as hf_model:
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
            assert hf_outputs[i] == pytest.approx(vllm_outputs[i], rel=0.01)

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
            assert hf_outputs[i] == pytest.approx(vllm_outputs[i], rel=0.01)

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
            assert hf_outputs[i] == pytest.approx(vllm_outputs[i], rel=0.01)

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

    def test_rerank_with_template(self, server: RemoteOpenAIServer,
                                  model: dict[str, Any], runner):
        if not model.get("is_qwen3_reranker"):
            pytest.skip("Test only for Qwen3 Reranker with template support.")

        instruction = "Find the document that is most relevant to the query about national capitals."
        query = "What is the capital of China?"
        documents = [
            "The capital of France is Paris.",
            "The capital of China is Beijing."
        ]

        # vLLM run with custom instruction via kwargs
        rerank_response = requests.post(
            server.url_for("rerank"),
            json={
                "model": model["name"],
                "query": query,
                "documents": documents,
                "score_template_kwargs": {
                    "instruction": instruction
                }
            })
        rerank_response.raise_for_status()
        response_data = RerankResponse.model_validate(rerank_response.json())
        vllm_outputs = {
            res.document.text: res.relevance_score
            for res in response_data.results
        }

        # HF reference run with the same custom instruction
        text_pairs = [[query, doc] for doc in documents]
        hf_outputs = _run_qwen3_reranker_hf(runner, text_pairs, instruction)

        for i, doc in enumerate(documents):
            assert vllm_outputs[doc] == pytest.approx(hf_outputs[i],
                                                      rel=0.01)

    def test_score_with_template(self, server: RemoteOpenAIServer,
                                 model: dict[str, Any], runner):
        if not model.get("is_qwen3_reranker"):
            pytest.skip("Test only for Qwen3 Reranker with template support.")

        instruction = "Find the document that is most relevant to the query about national capitals."
        text_1 = "What is the capital of China?"
        text_2 = [
            "The capital of France is Paris.",
            "The capital of China is Beijing."
        ]

        # vLLM run with custom instruction via kwargs
        score_response = requests.post(
            server.url_for("score"),
            json={
                "model": model["name"],
                "text_1": text_1,
                "text_2": text_2,
                "score_template_kwargs": {
                    "instruction": instruction
                }
            })
        score_response.raise_for_status()
        response_data = ScoreResponse.model_validate(score_response.json())
        vllm_outputs = [res.score for res in response_data.data]

        # HF reference run with the same custom instruction
        text_pairs = [[text_1, doc] for doc in text_2]
        hf_outputs = _run_qwen3_reranker_hf(runner, text_pairs, instruction)

        for i in range(len(vllm_outputs)):
            assert vllm_outputs[i] == pytest.approx(hf_outputs[i], rel=0.01)
