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


def test_chat_completion_request_accepts_reasoning_eos_policy():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning_eos_policy": "force_end",
        }
    )
    assert request.reasoning_eos_policy == "force_end"


def test_chat_completion_request_rejects_invalid_reasoning_eos_policy():
    with pytest.raises(Exception, match="reasoning_eos_policy"):
        ChatCompletionRequest.model_validate(
            {
                "model": "qwen",
                "messages": [{"role": "user", "content": "hello"}],
                "reasoning_eos_policy": "drop",
            }
        )


def test_chat_completion_request_to_sampling_params_forwards_policy():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning_eos_policy": "force_end",
        }
    )
    params = request.to_sampling_params(max_tokens=16, default_sampling_params={})
    assert params.reasoning_eos_policy == "force_end"


def test_completion_request_accepts_reasoning_eos_policy():
    request = CompletionRequest.model_validate(
        {
            "model": "qwen",
            "prompt": "hello",
            "reasoning_eos_policy": "force_end",
        }
    )
    assert request.reasoning_eos_policy == "force_end"
    params = request.to_sampling_params(max_tokens=16, default_sampling_params={})
    assert params.reasoning_eos_policy == "force_end"


def test_completion_request_rejects_invalid_reasoning_eos_policy():
    with pytest.raises(Exception, match="reasoning_eos_policy"):
        CompletionRequest.model_validate(
            {
                "model": "qwen",
                "prompt": "hello",
                "reasoning_eos_policy": "drop",
            }
        )


def test_completion_request_accepts_minus_one_as_unlimited():
    request = CompletionRequest.model_validate(
        {
            "model": "qwen",
            "prompt": "hello",
            "thinking_token_budget": -1,
        }
    )
    assert request.thinking_token_budget is None
