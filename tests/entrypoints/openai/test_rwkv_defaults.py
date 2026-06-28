# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from typing import Any

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.rwkv_defaults import (
    RWKV_DEFAULT_STOP_TOKEN_IDS,
    RWKV_DEFAULT_STOPS,
    apply_rwkv_default_sampling_params,
)


@dataclass
class _HFConfig:
    model_type: str = "rwkv7"


@dataclass
class _ModelConfig:
    tokenizer_mode: str = "rwkv"
    hf_config: _HFConfig = field(default_factory=_HFConfig)


def _chat_request(**kwargs: Any) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="rwkv-test",
        messages=[{"role": "user", "content": "hi"}],
        **kwargs,
    )


def test_rwkv_chat_defaults_stop_string_and_token_id() -> None:
    default_sampling_params: dict[str, Any] = {}
    apply_rwkv_default_sampling_params(default_sampling_params, _ModelConfig())

    sampling_params = _chat_request().to_sampling_params(128, default_sampling_params)

    assert sampling_params.stop == list(RWKV_DEFAULT_STOPS)
    assert sampling_params.stop_token_ids == list(RWKV_DEFAULT_STOP_TOKEN_IDS)


def test_rwkv_defaults_do_not_apply_to_other_models() -> None:
    default_sampling_params: dict[str, Any] = {}
    model_config = _ModelConfig(tokenizer_mode="auto", hf_config=_HFConfig("llama"))

    apply_rwkv_default_sampling_params(default_sampling_params, model_config)

    assert default_sampling_params == {}


def test_rwkv_chat_defaults_do_not_override_explicit_empty_stops() -> None:
    default_sampling_params: dict[str, Any] = {}
    apply_rwkv_default_sampling_params(default_sampling_params, _ModelConfig())

    sampling_params = _chat_request(
        stop=[],
        stop_token_ids=[],
    ).to_sampling_params(128, default_sampling_params)

    assert sampling_params.stop == []
    assert sampling_params.stop_token_ids == []


def test_rwkv_responses_defaults_stop_string_and_token_id() -> None:
    default_sampling_params: dict[str, Any] = {}
    apply_rwkv_default_sampling_params(default_sampling_params, _ModelConfig())

    request = ResponsesRequest(model="rwkv-test", input="hi")
    sampling_params = request.to_sampling_params(128, default_sampling_params)

    assert sampling_params.stop == list(RWKV_DEFAULT_STOPS)
    assert sampling_params.stop_token_ids == list(RWKV_DEFAULT_STOP_TOKEN_IDS)
