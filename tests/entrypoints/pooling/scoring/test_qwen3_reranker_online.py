# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import requests

from tests.utils import VLLM_PATH, RemoteOpenAIServer
from vllm.entrypoints.chat_utils import load_chat_template
from vllm.entrypoints.pooling.scoring.protocol import RerankResponse, ScoreResponse
from vllm.tokenizers import get_tokenizer

MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
HF_OVERRIDES = {
    "architectures": ["Qwen3ForSequenceClassification"],
    "classifier_from_token": ["no", "yes"],
    "is_original_qwen3_reranker": True,
}
CHAT_TEMPLATE_PATH = VLLM_PATH / "examples/pooling/score/template/qwen3_reranker.jinja"

QUERY = "What is the capital of France?"
DOCUMENT = "The capital of France is Paris."


@pytest.fixture(scope="module")
def server():
    args = [
        "--runner",
        "pooling",
        "--enforce-eager",
        "--max-model-len",
        "512",
        "--gpu-memory-utilization",
        "0.2",
        "--chat-template",
        str(CHAT_TEMPLATE_PATH),
    ]

    with RemoteOpenAIServer(
        MODEL_NAME,
        args,
        override_hf_configs=HF_OVERRIDES,
    ) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def expected_template_token_count():
    tokenizer = get_tokenizer(tokenizer_name=MODEL_NAME)
    chat_template = load_chat_template(CHAT_TEMPLATE_PATH)
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "query", "content": QUERY},
            {"role": "document", "content": DOCUMENT},
        ],
        chat_template=chat_template,
        tools=None,
        tokenize=False,
    )
    templated_tokens = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    no_template_tokens = tokenizer(
        text=QUERY,
        text_pair=DOCUMENT,
        add_special_tokens=True,
    )["input_ids"]

    assert "<Instruct>:" in prompt
    assert len(templated_tokens) != len(no_template_tokens)
    return len(templated_tokens)


def test_score_and_rerank_use_qwen3_reranker_template(
    server: RemoteOpenAIServer,
    expected_template_token_count: int,
):
    score_response = requests.post(
        server.url_for("score"),
        json={
            "model": MODEL_NAME,
            "queries": QUERY,
            "documents": DOCUMENT,
        },
    )
    score_response.raise_for_status()
    score = ScoreResponse.model_validate(score_response.json())

    rerank_response = requests.post(
        server.url_for("rerank"),
        json={
            "model": MODEL_NAME,
            "query": QUERY,
            "documents": [DOCUMENT],
        },
    )
    rerank_response.raise_for_status()
    rerank = RerankResponse.model_validate(rerank_response.json())

    assert score.usage.prompt_tokens == expected_template_token_count
    assert rerank.usage.prompt_tokens == expected_template_token_count
    assert len(score.data) == 1
    assert len(rerank.results) == 1
    assert score.data[0].score == pytest.approx(
        rerank.results[0].relevance_score,
        rel=0.01,
    )
