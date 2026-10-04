# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI

from vllm.entrypoints.generate.structured_decisions.api_router import (
    register_structured_decisions_api_router,
)
from vllm.entrypoints.generate.structured_decisions.question_types import (
    StructuredDecisionError,
    build_question,
)
from vllm.entrypoints.generate.structured_decisions.strategies import (
    NextTokenStrategy,
    label_token_ids,
    select_read_strategy,
    single_token_labels,
)


def model(architecture: str, logprobs_mode: str = "raw_logprobs") -> Any:
    return SimpleNamespace(architecture=architecture, logprobs_mode=logprobs_mode)


def test_strategy_selection():
    assert select_read_strategy(model("Qwen3ForCausalLM")) is NextTokenStrategy
    assert select_read_strategy(model("LlamaForCausalLM")) is None
    qwen = "Qwen3ForCausalLM"
    assert select_read_strategy(model(qwen, "processed_logprobs")) is NextTokenStrategy
    assert select_read_strategy(model(qwen, "raw_logits")) is None


def test_route_needs_the_flag():
    for enabled in (False, True):
        app = FastAPI()
        app.state.args = SimpleNamespace(enable_structured_decisions=enabled)
        register_structured_decisions_api_router(app)
        paths = {getattr(route, "path", None) for route in app.routes}
        assert ("/v1/systemone" in paths) == enabled


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
    q = build_question("q", "choice", "", dict.fromkeys(["x", "y", "z"]), 128)
    ids = label_token_ids(tokenizer, prompt_ids, q)
    assert [tokenizer.decode([i]) for i in ids] == ["A", "B", "C"]
    with pytest.raises(StructuredDecisionError, match="not one distinct token"):
        label_token_ids(tokenizer, prompt_ids, replace(q, labels=("A", "A", "B")))


def test_label_pool_skips_labels_that_are_not_one_token(qwen):
    tokenizer, prompt_ids = qwen
    pool = single_token_labels(tokenizer, prompt_ids)
    assert pool[:26] == tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    # "BQ" splits into two tokens after Qwen's generation prompt.
    assert "BQ" not in pool and len(pool) >= 128
    wide = build_question(
        "q", "choice", "", {str(i): None for i in range(128)}, 128, pool
    )
    label_token_ids(tokenizer, prompt_ids, wide)
