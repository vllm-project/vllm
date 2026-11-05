# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedFunction,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.exceptions import VLLMValidationError
from vllm.tool_parsers.utils import get_json_schema_from_tools


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


@pytest.mark.parametrize("tool_choice", [None, "none", "auto", "required"])
def test_chat_completion_request_with_empty_tools(tool_choice):
    """OpenAI treats `tools: []` like an omitted field, with no tool to call."""
    payload = {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "facebook/opt-125m",
        "tools": [],
    }
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    request = ChatCompletionRequest.model_validate(payload)
    assert request.tools is None
    assert request.tool_choice == "none"


def test_chat_completion_request_with_empty_tools_and_named_tool_choice():
    with pytest.raises(VLLMValidationError, match="The tool `f` specified"):
        ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "facebook/opt-125m",
                "tools": [],
                "tool_choice": {"type": "function", "function": {"name": "f"}},
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


def test_typed_named_tool_choice_accepted():
    choice = ChatCompletionNamedToolChoiceParam(
        function=ChatCompletionNamedFunction(name="get_weather")
    )
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "Hello"}],
        model="facebook/opt-125m",
        tools=[SAMPLE_TOOL],
        tool_choice=choice,
    )
    assert isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam)
    assert request.tool_choice.function.name == "get_weather"


def test_typed_tools_accepted():
    tool = ChatCompletionToolsParam.model_validate(SAMPLE_TOOL)
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "Hello"}],
        model="facebook/opt-125m",
        tools=[tool],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )
    assert request.tools is not None and len(request.tools) == 1
    assert request.tools[0].function.name == "get_weather"
    assert request.tools[0] == tool


def test_typed_tools_and_typed_tool_choice_accepted():
    tool = ChatCompletionToolsParam.model_validate(SAMPLE_TOOL)
    choice = ChatCompletionNamedToolChoiceParam(
        function=ChatCompletionNamedFunction(name="get_weather")
    )
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "Hello"}],
        model="facebook/opt-125m",
        tools=[tool],
        tool_choice=choice,
    )
    assert isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam)
    assert request.tool_choice.function.name == "get_weather"
    assert request.tools is not None and len(request.tools) == 1
    assert request.tools[0].function.name == "get_weather"


def test_typed_named_tool_choice_mismatch_rejected():
    choice = ChatCompletionNamedToolChoiceParam(
        function=ChatCompletionNamedFunction(name="unknown_fn")
    )
    with pytest.raises(
        VLLMValidationError, match="does not match any of the specified `tools`"
    ):
        ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="facebook/opt-125m",
            tools=[SAMPLE_TOOL],
            tool_choice=choice,
        )


def test_typed_tools_mismatch_rejected():
    tool = ChatCompletionToolsParam.model_validate(SAMPLE_TOOL)
    with pytest.raises(
        VLLMValidationError, match="does not match any of the specified `tools`"
    ):
        ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="facebook/opt-125m",
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": "unknown_fn"}},
        )


def test_both_typed_mismatch_rejected():
    tool = ChatCompletionToolsParam.model_validate(SAMPLE_TOOL)
    choice = ChatCompletionNamedToolChoiceParam(
        function=ChatCompletionNamedFunction(name="unknown_fn")
    )
    with pytest.raises(
        VLLMValidationError, match="does not match any of the specified `tools`"
    ):
        ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="facebook/opt-125m",
            tools=[tool],
            tool_choice=choice,
        )


def test_typed_tools_and_tool_choice_end_to_end_schema():
    tool = ChatCompletionToolsParam.model_validate(SAMPLE_TOOL)
    choice = ChatCompletionNamedToolChoiceParam(
        function=ChatCompletionNamedFunction(name="get_weather")
    )
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "Hello"}],
        model="facebook/opt-125m",
        tools=[tool],
        tool_choice=choice,
    )
    schema = get_json_schema_from_tools(request.tool_choice, request.tools)
    assert schema == {"type": "object", "properties": {"location": {"type": "string"}}}


def test_empty_tools_with_typed_named_tool_choice_mismatch_rejected():
    choice = ChatCompletionNamedToolChoiceParam(
        function=ChatCompletionNamedFunction(name="get_weather")
    )
    with pytest.raises(
        VLLMValidationError, match="does not match any of the specified `tools`"
    ):
        ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="facebook/opt-125m",
            tools=[],
            tool_choice=choice,
        )
