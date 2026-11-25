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


@pytest.mark.parametrize(
    "request_cls,payload",
    [
        (ChatCompletionRequest, {"messages": [{"role": "user", "content": "hi"}]}),
        (CompletionRequest, {"prompt": "hi"}),
    ],
)
@pytest.mark.parametrize(
    "use_beam_search,rejected",
    [(False, False), ("false", False), ("0", False), (True, True)],
)
def test_reasoning_end_str_is_rejected_with_beam_search(
    request_cls, payload, use_beam_search, rejected
):
    """Beam search drops the thinking budget; the override must not look applied.

    The check sees `use_beam_search` after coercion, so string falsy values pass.
    """
    data = {
        "model": "qwen",
        **payload,
        "use_beam_search": use_beam_search,
        "thinking_token_budget": 8,
        "reasoning_end_str": "Time to answer.</think>",
    }
    if rejected:
        with pytest.raises(VLLMValidationError, match="reasoning_end_str"):
            request_cls.model_validate(data)
    else:
        assert request_cls.model_validate(data).use_beam_search is False
