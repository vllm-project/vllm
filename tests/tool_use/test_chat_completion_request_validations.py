# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.exceptions import VLLMValidationError


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
    with pytest.raises(VLLMValidationError, match="must not be an empty array"):
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
        VLLMValidationError, match="When using `tool_choice`, `tools` must be set."
    ):
        ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "facebook/opt-125m",
                "tool_choice": tool_choice,
            }
        )

    with pytest.raises(
        VLLMValidationError, match="When using `tool_choice`, `tools` must be set."
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


SAMPLE_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
        },
    },
}


def test_structured_outputs_with_named_tool_choice_rejected():
    """structured_outputs cannot be combined with a named tool_choice."""
    with pytest.raises(
        VLLMValidationError,
        match="structured outputs or tools, not both",
    ):
        ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "facebook/opt-125m",
                "tools": [SAMPLE_TOOL],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "get_weather"},
                },
                "structured_outputs": {"json": {"type": "object"}},
            }
        )


def test_structured_outputs_with_auto_tool_choice_allowed():
    """structured_outputs with tool_choice 'auto' should be allowed."""
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "facebook/opt-125m",
            "tools": [SAMPLE_TOOL],
            "tool_choice": "auto",
            "structured_outputs": {"json": {"type": "object"}},
        }
    )
    assert request.tool_choice == "auto"


def test_multiple_structured_outputs_rejected():
    """Only one kind of structured output constraint is allowed."""
    with pytest.raises(
        VLLMValidationError,
        match="You can only use one kind of constraints",
    ):
        ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "facebook/opt-125m",
                "structured_outputs": {
                    "json": {"type": "object"},
                    "regex": ".*",
                },
            }
        )


TOOL_DECLARING_ROLES = ["developer", "system"]


def _message_level_tools_request(role="developer", **kwargs):
    """A request whose only tool is declared on a message."""
    return {
        "messages": [
            {"role": role, "content": "", "tools": [SAMPLE_TOOL]},
            {"role": "user", "content": "What is the weather?"},
        ],
        "model": "facebook/opt-125m",
        **kwargs,
    }


@pytest.mark.parametrize("role", TOOL_DECLARING_ROLES)
def test_message_level_tools_do_not_change_tool_choice_default(role):
    """A message declaration must not switch the request into tool mode."""
    request = ChatCompletionRequest.model_validate(_message_level_tools_request(role))
    assert request.tools is None
    assert request.tool_choice == "none"


@pytest.mark.parametrize("role", TOOL_DECLARING_ROLES)
def test_message_level_tools_allow_auto_tool_choice(role):
    """`auto` needs no tool schema, so a message declaration satisfies it."""
    request = ChatCompletionRequest.model_validate(
        _message_level_tools_request(role, tool_choice="auto")
    )
    assert request.tools is None
    assert request.tool_choice == "auto"


@pytest.mark.parametrize("role", TOOL_DECLARING_ROLES)
def test_message_level_tools_with_tool_choice_none(role):
    request = ChatCompletionRequest.model_validate(
        _message_level_tools_request(role, tool_choice="none")
    )
    assert request.tool_choice == "none"


@pytest.mark.parametrize("role", TOOL_DECLARING_ROLES)
@pytest.mark.parametrize(
    "tool_choice",
    [
        "required",
        {"type": "function", "function": {"name": "get_weather"}},
    ],
)
def test_message_level_tools_reject_constrained_tool_choice(tool_choice, role):
    """These need a grammar built from the request-level tool schemas."""
    with pytest.raises(VLLMValidationError, match='Only `tool_choice` "auto"'):
        ChatCompletionRequest.model_validate(
            _message_level_tools_request(role, tool_choice=tool_choice)
        )


@pytest.mark.parametrize("role", ["user", "assistant", "tool"])
def test_tools_on_unsupported_role_ignored(role):
    """Only roles forwarded to the chat template declare tools."""
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [
                {"role": role, "content": "Hello", "tools": [SAMPLE_TOOL]},
                {"role": "user", "content": "Hello"},
            ],
            "model": "facebook/opt-125m",
        }
    )
    assert request.tool_choice == "none"


@pytest.mark.parametrize("role", TOOL_DECLARING_ROLES)
def test_empty_message_level_tools_do_not_enable_tools(role):
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [
                {"role": role, "content": "", "tools": []},
                {"role": "user", "content": "Hello"},
            ],
            "model": "facebook/opt-125m",
        }
    )
    assert request.tool_choice == "none"


@pytest.mark.parametrize("role", TOOL_DECLARING_ROLES)
@pytest.mark.parametrize(
    "tools",
    [
        "not-a-list",
        ["not-a-dict"],
        [{"type": "function"}],
        [{"type": "function", "function": {}}],
        [{"type": "function", "function": {"name": 5}}],
        [{"type": "not-a-function", "function": {"name": "get_weather"}}],
    ],
)
def test_malformed_message_level_tools_rejected(tools, role):
    """Message-level tools follow the request-level tool schema."""
    with pytest.raises(VLLMValidationError, match="Invalid tool declared on a message"):
        ChatCompletionRequest.model_validate(
            {
                "messages": [
                    {"role": role, "content": "", "tools": tools},
                    {"role": "user", "content": "Hello"},
                ],
                "model": "facebook/opt-125m",
            }
        )
