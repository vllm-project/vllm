# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.generate.structured_decisions.api_router import (
    register_structured_decisions_api_router,
)
from vllm.entrypoints.generate.structured_decisions.question_types import LABELS
from vllm.entrypoints.generate.structured_decisions.serving import state_text
from vllm.entrypoints.generate.structured_decisions.strategies import (
    NextTokenStrategy,
    reply_label_ids,
    select_read_strategy,
)
from vllm.entrypoints.serve.exception_handling.register import (
    init_exception_handler,
)


def model(architecture: str, logprobs_mode: str = "raw_logprobs") -> Any:
    return SimpleNamespace(architecture=architecture, logprobs_mode=logprobs_mode)


def test_strategy_selection():
    qwen = "Qwen3ForCausalLM"
    assert select_read_strategy(model(qwen)) is NextTokenStrategy
    assert select_read_strategy(model(qwen, "processed_logprobs")) is NextTokenStrategy
    with pytest.raises(ValueError, match="does not support LlamaForCausalLM"):
        select_read_strategy(model("LlamaForCausalLM"))
    with pytest.raises(ValueError, match="not raw_logits"):
        select_read_strategy(model(qwen, "raw_logits"))


def test_route_is_registered_by_default():
    app = FastAPI()
    register_structured_decisions_api_router(app)
    paths = {getattr(route, "path", None) for route in app.routes}
    assert "/v1/systemone" in paths


def test_unsupported_model_returns_501():
    app = FastAPI()
    app.state.args = SimpleNamespace(log_error_stack=False)
    app.state.serving_structured_decisions = None
    init_exception_handler(app)
    register_structured_decisions_api_router(app)

    with TestClient(app) as client:
        response = client.post(
            "/v1/systemone",
            json={
                "model": "unsupported",
                "state": "x",
                "questions": {"answer": {"type": "choice", "criteria": {"yes": None}}},
            },
        )
    assert response.status_code == 501


def test_structured_state_keeps_unicode_in_prompt():
    state = {"message": "Français 日本語"}
    assert state_text(state) == '{"message": "Français 日本語"}'


@pytest.fixture(scope="module")
def qwen():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "state"}],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=False,
    )
    return tokenizer, prompt_ids


def test_labels_start_the_reply(qwen):
    tokenizer, prompt_ids = qwen
    tail, ids = reply_label_ids(tokenizer, prompt_ids)
    assert tokenizer.decode(tail) == "\n\n"
    assert [tokenizer.decode([i]) for i in ids] == list(LABELS)
    # After a colon, Qwen writes ":A" as one token, so "A" is not one token.
    with pytest.raises(ValueError, match="not one distinct token"):
        reply_label_ids(tokenizer, tokenizer.encode("team:"))
