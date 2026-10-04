# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI

from vllm.entrypoints.generate.structured_decisions.api_router import (
    register_structured_decisions_api_router,
)
from vllm.entrypoints.generate.structured_decisions.strategies import (
    NextTokenStrategy,
    select_read_strategy,
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
