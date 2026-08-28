# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.exceptions import VLLMValidationError


@pytest.mark.parametrize("raw_value", [-2, 0.6, 10.5])
def test_chat_completion_request_rejects_invalid_thinking_token_budget(raw_value):
    with pytest.raises(VLLMValidationError, match="thinking_token_budget"):
        ChatCompletionRequest.model_validate(
            {
                "model": "qwen",
                "messages": [{"role": "user", "content": "hello"}],
                "thinking_token_budget": raw_value,
            }
        )


def test_chat_completion_request_accepts_valid_thinking_token_budget():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "thinking_token_budget": 10,
        }
    )
    assert request.thinking_token_budget == 10


def test_chat_completion_request_accepts_minus_one_as_unlimited():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "thinking_token_budget": -1,
        }
    )
    assert request.thinking_token_budget is None


@pytest.mark.parametrize("raw_value", [0.6, 3.14, -2])
def test_completion_request_rejects_invalid_thinking_token_budget(raw_value):
    with pytest.raises(VLLMValidationError, match="thinking_token_budget"):
        CompletionRequest.model_validate(
            {
                "model": "qwen",
                "prompt": "hello",
                "thinking_token_budget": raw_value,
            }
        )


def test_completion_request_accepts_valid_thinking_token_budget():
    request = CompletionRequest.model_validate(
        {
            "model": "qwen",
            "prompt": "hello",
            "thinking_token_budget": 5,
        }
    )
    assert request.thinking_token_budget == 5


def test_completion_request_accepts_minus_one_as_unlimited():
    request = CompletionRequest.model_validate(
        {
            "model": "qwen",
            "prompt": "hello",
            "thinking_token_budget": -1,
        }
    )
    assert request.thinking_token_budget is None


def test_chat_completion_request_post_thinking_to_sampling_params():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 1.0,
            "post_thinking": {"temperature": 0.4, "top_p": 0.95, "top_k": 20},
        }
    )
    params = request.to_sampling_params(max_tokens=16, default_sampling_params={})
    assert params.temperature == 1.0
    assert params.post_thinking is not None
    assert params.post_thinking.temperature == 0.4
    assert params.post_thinking.top_p == 0.95
    assert params.post_thinking.top_k == 20


def test_chat_completion_request_inherits_server_default_post_thinking():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
        }
    )
    params = request.to_sampling_params(
        max_tokens=16,
        default_sampling_params={"post_thinking": {"temperature": 0.4}},
    )
    assert params.post_thinking is not None
    assert params.post_thinking.temperature == 0.4


def test_chat_completion_request_rejects_invalid_post_thinking_temperature():
    with pytest.raises(VLLMValidationError, match="temperature"):
        ChatCompletionRequest.model_validate(
            {
                "model": "qwen",
                "messages": [{"role": "user", "content": "hello"}],
                "post_thinking": {"temperature": 3.5},
            }
        )
