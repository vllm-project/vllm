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

    # tools key present but empty -- accepted per OpenAI spec
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "facebook/opt-125m",
            "tools": [],
        }
    )
    assert request.tool_choice == "none"


@pytest.mark.parametrize("tool_choice", ["none", "auto", "required"])
def test_chat_completion_request_with_tool_choice_but_no_tools(tool_choice):
    with pytest.raises(ValueError, match="`tool_choice` is only allowed when"):
        ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "facebook/opt-125m",
                "tool_choice": tool_choice,
            }
        )

    with pytest.raises(ValueError, match="`tool_choice` is only allowed when"):
        ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "facebook/opt-125m",
                "tool_choice": tool_choice,
                "tools": None,
            }
        )


@pytest.mark.parametrize("tool_choice", ["none", "auto", "required"])
def test_chat_completion_request_with_empty_tools_and_string_tool_choice(
    tool_choice,
):
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "facebook/opt-125m",
            "tools": [],
            "tool_choice": tool_choice,
        }
    )
    assert request.tool_choice == tool_choice


def test_chat_completion_request_with_empty_tools_and_named_tool_choice():
    with pytest.raises(ValueError, match="does not match any"):
        ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "facebook/opt-125m",
                "tools": [],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "foo"},
                },
            }
        )


def test_chat_completion_request_named_tool_choice_without_tools():
    with pytest.raises(ValueError, match="`tool_choice` is only allowed when"):
        ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "facebook/opt-125m",
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "foo"},
                },
            }
        )
