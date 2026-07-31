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
    # Finer-grained rules: "auto" without tools is a harmless no-op;
    # "required"/named without any tool source (request level OR message
    # level) is rejected.
    expected = None if tool_choice == "auto" else VLLMValidationError
    if expected is None:
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "facebook/opt-125m",
                "tool_choice": tool_choice,
            }
        )
        assert request.tool_choice == tool_choice
    else:
        with pytest.raises(expected, match="tools must be declared"):
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


_SAMPLE_TOOL_DICT = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather",
        "parameters": {"type": "object", "properties": {}},
    },
}


def test_request_level_tools_still_default_tool_choice_auto():
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "facebook/opt-125m",
            "tools": [_SAMPLE_TOOL_DICT],
        }
    )
    assert request.tool_choice == "auto"


@pytest.mark.parametrize("role", ["system", "developer"])
def test_message_level_tools_default_tool_choice_auto(role):
    # Message-level tool declarations must also default tool_choice to
    # "auto"; the field default "none" would tell the model not to call the
    # tools the client just declared.
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [
                {"role": role, "content": "instructions", "tools": [_SAMPLE_TOOL_DICT]},
                {"role": "user", "content": "Hello"},
            ],
            "model": "facebook/opt-125m",
        }
    )
    assert request.tool_choice == "auto"


def test_message_level_tools_do_not_override_explicit_tool_choice():
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [
                {
                    "role": "system",
                    "content": "instructions",
                    "tools": [_SAMPLE_TOOL_DICT],
                },
                {"role": "user", "content": "Hello"},
            ],
            "model": "facebook/opt-125m",
            "tool_choice": "required",
        }
    )
    assert request.tool_choice == "required"


def test_tools_on_other_roles_do_not_default_tool_choice_auto():
    # tools on roles that are not passed through to the template (user,
    # assistant, tool) must not influence the tool_choice default.
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [
                {"role": "user", "content": "Hello", "tools": [_SAMPLE_TOOL_DICT]},
            ],
            "model": "facebook/opt-125m",
        }
    )
    assert request.tool_choice == "none"


def test_auto_tool_choice_without_tools_allowed():
    # "auto" without any tools is a harmless no-op: the model simply answers.
    for tools_kwarg in ({}, {"tools": None}):
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "facebook/opt-125m",
                "tool_choice": "auto",
                **tools_kwarg,
            }
        )
        assert request.tool_choice == "auto"


def test_required_tool_choice_without_any_tools_rejected():
    # "required" is contradictory without any tool source: reject unless
    # tools are declared at the request level OR on a message.
    with pytest.raises(VLLMValidationError, match="tools must be declared"):
        ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "facebook/opt-125m",
                "tool_choice": "required",
            }
        )


def test_required_tool_choice_with_message_level_tools_allowed():
    # Tools declared on a system message satisfy the requirement.
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [
                {"role": "system", "content": "", "tools": [SAMPLE_TOOL]},
                {"role": "user", "content": "Hello"},
            ],
            "model": "facebook/opt-125m",
            "tool_choice": "required",
        }
    )
    assert request.tool_choice == "required"


def test_named_tool_choice_without_any_tools_rejected():
    # A named tool_choice with no tool declared anywhere is rejected.
    with pytest.raises(VLLMValidationError, match="tools must be declared"):
        ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "facebook/opt-125m",
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "get_weather"},
                },
            }
        )


def test_named_tool_choice_with_message_level_tools_allowed():
    # A named tool_choice without request-level tools is allowed when the
    # named tool may be declared at the message level (not cross-checked).
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [
                {"role": "system", "content": "", "tools": [SAMPLE_TOOL]},
                {"role": "user", "content": "Hello"},
            ],
            "model": "facebook/opt-125m",
            "tool_choice": {
                "type": "function",
                "function": {"name": "get_weather"},
            },
        }
    )
    assert request.tool_choice.function.name == "get_weather"


def test_named_tool_choice_must_match_declared_tools():
    # When request-level tools ARE present, a named tool_choice must still
    # match one of them.
    with pytest.raises(
        VLLMValidationError, match="does not match any of the specified `tools`"
    ):
        ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "facebook/opt-125m",
                "tools": [SAMPLE_TOOL],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "nondefined_function_name"},
                },
            }
        )
