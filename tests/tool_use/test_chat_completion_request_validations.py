# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.exceptions import VLLMValidationError

OMITTED = object()
NAMED_TOOL_CHOICE = {
    "type": "function",
    "function": {"name": "get_weather"},
}
MISSING_TOOL_CHOICE = {
    "type": "function",
    "function": {"name": "missing_tool"},
}

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


def _request_payload(tools, tool_choice):
    payload = {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "facebook/opt-125m",
    }
    if tools is not OMITTED:
        payload["tools"] = tools
    if tool_choice is not OMITTED:
        payload["tool_choice"] = tool_choice
    return payload


TOOLS_CASES = [
    ("omitted", OMITTED),
    ("null", None),
    ("empty", []),
    ("provided", [SAMPLE_TOOL]),
]
TOOL_CHOICE_CASES = [
    ("omitted", OMITTED),
    ("null", None),
    ("none", "none"),
    ("auto", "auto"),
    ("required", "required"),
    ("named", NAMED_TOOL_CHOICE),
    ("missing", MISSING_TOOL_CHOICE),
]
EXPECTED: dict[str, dict[str, tuple[bool, str | None] | None]] = {
    "omitted": {
        "omitted": (False, "none"),
        "null": (False, None),
        "none": (False, "none"),
        "auto": (False, "none"),
        "required": None,
        "named": None,
        "missing": None,
    },
    "null": {
        "omitted": (False, "none"),
        "null": (False, None),
        "none": (False, "none"),
        "auto": (False, "none"),
        "required": None,
        "named": None,
        "missing": None,
    },
    "empty": {
        "omitted": (False, "none"),
        "null": (False, None),
        "none": (False, "none"),
        "auto": (False, "none"),
        "required": None,
        "named": None,
        "missing": None,
    },
    "provided": {
        "omitted": (True, "auto"),
        "null": (True, "auto"),
        "none": (True, "none"),
        "auto": (True, "auto"),
        "required": (True, "required"),
        "named": (True, "named"),
        "missing": None,
    },
}


@pytest.mark.parametrize("tools_name,tools", TOOLS_CASES)
@pytest.mark.parametrize("tool_choice_name,tool_choice", TOOL_CHOICE_CASES)
def test_chat_completion_request_tool_choice_matrix(
    tools_name, tools, tool_choice_name, tool_choice
):
    """Validate tool_choice defaults and requirements for every tool state."""
    expected = EXPECTED[tools_name][tool_choice_name]
    payload = _request_payload(tools, tool_choice)
    if expected is None:
        with pytest.raises(VLLMValidationError):
            ChatCompletionRequest.model_validate(payload)
        return

    expected_has_tools, expected_tool_choice = expected
    request = ChatCompletionRequest.model_validate(payload)
    assert (request.tools is not None) is expected_has_tools
    if expected_has_tools:
        assert len(request.tools) == 1

    if expected_tool_choice == "named":
        assert isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam)
        assert request.tool_choice.function.name == "get_weather"
    else:
        assert request.tool_choice == expected_tool_choice


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
