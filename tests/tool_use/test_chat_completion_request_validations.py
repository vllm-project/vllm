# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest


def test_chat_completion_request_with_no_tools():
    # tools key is not present
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "facebook/opt-125m",
        }
    )
    assert request.tool_choice == "none"

    # tools key is None
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "facebook/opt-125m",
            "tools": None,
        }
    )
    assert request.tool_choice == "none"

    # tools key present but empty -- should be rejected
    with pytest.raises(ValueError, match="must not be an empty array"):
        ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "facebook/opt-125m",
                "tools": [],
            }
        )


@pytest.mark.parametrize("tool_choice", ["auto", "required"])
def test_chat_completion_request_with_tool_choice_but_no_tools(tool_choice):
    with pytest.raises(
        ValueError, match="When using `tool_choice`, `tools` must be set."
    ):
        ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "facebook/opt-125m",
                "tool_choice": tool_choice,
            }
        )

    with pytest.raises(
        ValueError, match="When using `tool_choice`, `tools` must be set."
    ):
        ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "facebook/opt-125m",
                "tool_choice": tool_choice,
                "tools": None,
            }
        )


def test_reasoning_content_normalized_to_reasoning():
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {
                    "role": "assistant",
                    "content": "4",
                    "reasoning_content": "2+2 equals 4",
                },
                {"role": "user", "content": "Are you sure?"},
            ],
            "model": "facebook/opt-125m",
        }
    )
    assistant_msg = request.messages[1]
    assert assistant_msg.get("reasoning") == "2+2 equals 4"
    assert "reasoning_content" not in assistant_msg


def test_reasoning_takes_precedence_over_reasoning_content():
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {
                    "role": "assistant",
                    "content": "4",
                    "reasoning": "from reasoning field",
                    "reasoning_content": "from reasoning_content field",
                },
            ],
            "model": "facebook/opt-125m",
        }
    )
    assistant_msg = request.messages[1]
    assert assistant_msg.get("reasoning") == "from reasoning field"
    assert "reasoning_content" not in assistant_msg


def test_no_reasoning_fields_unchanged():
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
            "model": "facebook/opt-125m",
        }
    )
    assistant_msg = request.messages[1]
    assert assistant_msg.get("reasoning") is None
    assert "reasoning_content" not in assistant_msg
