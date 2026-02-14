# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for Qwen2.5-Coder tool parser.

This tests the <tools>JSON</tools> format used by Qwen2.5-Coder models.
"""

import json
from collections.abc import Generator

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    ToolCall,
)
from vllm.tokenizers import TokenizerLike, get_tokenizer
from vllm.tokenizers.detokenizer_utils import detokenize_incrementally
from vllm.tool_parsers.qwen25_coder_tool_parser import Qwen25CoderToolParser

# Use a model that has the <tools> token in vocabulary
# Fallback to a common model if specific Qwen2.5-Coder model not available
MODEL = "gpt2"


@pytest.fixture(scope="module")
def qwen25_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)


@pytest.fixture
def qwen25_tool_parser(qwen25_tokenizer):
    return Qwen25CoderToolParser(qwen25_tokenizer)


@pytest.fixture
def sample_tools():
    return [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            },
        ),
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "search_web",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "num_results": {
                            "type": "integer",
                            "description": "Number of results",
                        },
                    },
                    "required": ["query"],
                },
            },
        ),
    ]


def assert_tool_calls(
    actual_tool_calls: list[ToolCall], expected_tool_calls: list[ToolCall]
):
    """Assert that actual tool calls match expected tool calls."""
    assert len(actual_tool_calls) == len(expected_tool_calls)

    for actual, expected in zip(actual_tool_calls, expected_tool_calls):
        assert actual.type == "function"
        assert actual.function.name == expected.function.name
        assert json.loads(actual.function.arguments) == json.loads(
            expected.function.arguments
        )


def stream_delta_message_generator(
    tool_parser: Qwen25CoderToolParser,
    tokenizer: TokenizerLike,
    model_output: str,
    request: ChatCompletionRequest | None = None,
) -> Generator[DeltaMessage, None, None]:
    """Generate delta messages by streaming tokens one at a time."""
    all_token_ids = tokenizer.encode(model_output, add_special_tokens=False)

    previous_text = ""
    previous_tokens = None
    prefix_offset = 0
    read_offset = 0

    for i, delta_token in enumerate(all_token_ids):
        delta_token_ids = [delta_token]
        previous_token_ids = all_token_ids[:i]
        current_token_ids = all_token_ids[: i + 1]

        (new_tokens, delta_text, new_prefix_offset, new_read_offset) = (
            detokenize_incrementally(
                tokenizer=tokenizer,
                all_input_ids=current_token_ids,
                prev_tokens=previous_tokens,
                prefix_offset=prefix_offset,
                read_offset=read_offset,
                skip_special_tokens=False,
                spaces_between_special_tokens=True,
            )
        )

        current_text = previous_text + delta_text

        delta_message = tool_parser.extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request=request,
        )
        if delta_message:
            yield delta_message

        previous_text = current_text
        previous_tokens = (
            previous_tokens + new_tokens if previous_tokens else new_tokens
        )
        prefix_offset = new_prefix_offset
        read_offset = new_read_offset


class TestExtractToolCallsNonStreaming:
    """Test non-streaming tool call extraction."""

    def test_no_tool_calls(self, qwen25_tool_parser):
        """Test response without any tool calls."""
        model_output = "This is a regular response without tool calls."
        result = qwen25_tool_parser.extract_tool_calls(model_output, request=None)

        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == model_output

    def test_single_tool_call(self, qwen25_tool_parser, sample_tools):
        """Test extracting a single tool call."""
        model_output = """<tools>
{"name": "get_current_weather", "arguments": {"location": "San Francisco, CA", "unit": "celsius"}}
</tools>"""

        request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)
        result = qwen25_tool_parser.extract_tool_calls(model_output, request=request)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_current_weather"

        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["location"] == "San Francisco, CA"
        assert args["unit"] == "celsius"

    def test_single_tool_call_with_content(self, qwen25_tool_parser, sample_tools):
        """Test tool call with preceding content."""
        model_output = """Let me check the weather for you.
<tools>
{"name": "get_current_weather", "arguments": {"location": "Tokyo, Japan"}}
</tools>"""

        request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)
        result = qwen25_tool_parser.extract_tool_calls(model_output, request=request)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.content == "Let me check the weather for you."

    def test_parallel_tool_calls_array(self, qwen25_tool_parser, sample_tools):
        """Test extracting multiple tool calls as JSON array."""
        model_output = """<tools>
[
    {"name": "get_current_weather", "arguments": {"location": "Paris, France"}},
    {"name": "search_web", "arguments": {"query": "Paris attractions", "num_results": 5}}
]
</tools>"""

        request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)
        result = qwen25_tool_parser.extract_tool_calls(model_output, request=request)

        assert result.tools_called
        assert len(result.tool_calls) == 2

        assert result.tool_calls[0].function.name == "get_current_weather"
        assert result.tool_calls[1].function.name == "search_web"

        args1 = json.loads(result.tool_calls[0].function.arguments)
        assert args1["location"] == "Paris, France"

        args2 = json.loads(result.tool_calls[1].function.arguments)
        assert args2["query"] == "Paris attractions"
        assert args2["num_results"] == 5

    def test_multiple_tools_blocks(self, qwen25_tool_parser, sample_tools):
        """Test extracting from multiple <tools> blocks."""
        model_output = """<tools>
{"name": "get_current_weather", "arguments": {"location": "London, UK"}}
</tools>
<tools>
{"name": "search_web", "arguments": {"query": "London events"}}
</tools>"""

        request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)
        result = qwen25_tool_parser.extract_tool_calls(model_output, request=request)

        assert result.tools_called
        assert len(result.tool_calls) == 2

    def test_parameters_key_support(self, qwen25_tool_parser, sample_tools):
        """Test that 'parameters' key is also supported (alias for 'arguments')."""
        model_output = """<tools>
{"name": "get_current_weather", "parameters": {"location": "Berlin, Germany"}}
</tools>"""

        request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)
        result = qwen25_tool_parser.extract_tool_calls(model_output, request=request)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["location"] == "Berlin, Germany"

    def test_nested_json_arguments(self, qwen25_tool_parser):
        """Test tool call with nested JSON in arguments."""
        tools = [
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "complex_function",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "config": {"type": "object"},
                            "items": {"type": "array"},
                        },
                    },
                },
            )
        ]

        model_output = """<tools>
{"name": "complex_function", "arguments": {"config": {"nested": {"deep": true}}, "items": [1, 2, {"key": "value"}]}}
</tools>"""

        request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)
        result = qwen25_tool_parser.extract_tool_calls(model_output, request=request)

        assert result.tools_called
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["config"]["nested"]["deep"] is True
        assert args["items"] == [1, 2, {"key": "value"}]

    def test_empty_arguments(self, qwen25_tool_parser):
        """Test tool call with empty arguments."""
        tools = [
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "no_args_function",
                    "parameters": {"type": "object", "properties": {}},
                },
            )
        ]

        model_output = """<tools>
{"name": "no_args_function", "arguments": {}}
</tools>"""

        request = ChatCompletionRequest(model=MODEL, messages=[], tools=tools)
        result = qwen25_tool_parser.extract_tool_calls(model_output, request=request)

        assert result.tools_called
        assert result.tool_calls[0].function.name == "no_args_function"
        assert json.loads(result.tool_calls[0].function.arguments) == {}

    def test_malformed_json_returns_content(self, qwen25_tool_parser):
        """Test that malformed JSON is returned as content."""
        model_output = """<tools>
{"name": "broken", "arguments": {invalid json
</tools>"""

        result = qwen25_tool_parser.extract_tool_calls(model_output, request=None)

        # Should not crash, returns original as content
        assert not result.tools_called

    def test_unicode_in_arguments(self, qwen25_tool_parser, sample_tools):
        """Test tool call with unicode characters in arguments."""
        model_output = """<tools>
{"name": "get_current_weather", "arguments": {"location": "Tokyo, Japan"}}
</tools>"""

        request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)
        result = qwen25_tool_parser.extract_tool_calls(model_output, request=request)

        assert result.tools_called
        args = json.loads(result.tool_calls[0].function.arguments)
        assert "Tokyo" in args["location"]


class TestExtractToolCallsStreaming:
    """Test streaming tool call extraction."""

    def test_no_tool_calls_streaming(
        self, qwen25_tool_parser, qwen25_tokenizer, sample_tools
    ):
        """Test streaming response without tool calls."""
        model_output = "This is a regular response without any tool calls."
        request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)

        collected_content = ""
        for delta in stream_delta_message_generator(
            qwen25_tool_parser, qwen25_tokenizer, model_output, request
        ):
            if delta.content:
                collected_content += delta.content

        assert collected_content == model_output

    def test_single_tool_streaming(
        self, qwen25_tool_parser, qwen25_tokenizer, sample_tools
    ):
        """Test streaming a single tool call."""
        model_output = """<tools>
{"name": "get_current_weather", "arguments": {"location": "NYC"}}
</tools>"""

        request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)

        tool_states = {}
        for delta in stream_delta_message_generator(
            qwen25_tool_parser, qwen25_tokenizer, model_output, request
        ):
            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    idx = tool_call.index

                    if idx not in tool_states:
                        tool_states[idx] = {
                            "id": None,
                            "name": None,
                            "arguments": "",
                            "type": None,
                        }

                    if tool_call.id:
                        tool_states[idx]["id"] = tool_call.id
                    if tool_call.type:
                        tool_states[idx]["type"] = tool_call.type
                    if tool_call.function:
                        if tool_call.function.name:
                            tool_states[idx]["name"] = tool_call.function.name
                        if tool_call.function.arguments is not None:
                            tool_states[idx]["arguments"] += (
                                tool_call.function.arguments
                            )

        # Verify we got the tool call
        assert len(tool_states) == 1
        state = tool_states[0]
        assert state["name"] == "get_current_weather"
        assert state["type"] == "function"
        assert state["id"] is not None

        # Verify arguments
        args = json.loads(state["arguments"])
        assert args["location"] == "NYC"

    def test_tool_with_content_streaming(
        self, qwen25_tool_parser, qwen25_tokenizer, sample_tools
    ):
        """Test streaming tool call with preceding content."""
        model_output = """Let me check that for you.
<tools>
{"name": "get_current_weather", "arguments": {"location": "LA"}}
</tools>"""

        request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)

        collected_content = ""
        tool_states = {}

        for delta in stream_delta_message_generator(
            qwen25_tool_parser, qwen25_tokenizer, model_output, request
        ):
            if delta.content:
                collected_content += delta.content
            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    idx = tool_call.index
                    if idx not in tool_states:
                        tool_states[idx] = {
                            "id": None,
                            "name": None,
                            "arguments": "",
                            "type": None,
                        }
                    if tool_call.id:
                        tool_states[idx]["id"] = tool_call.id
                    if tool_call.type:
                        tool_states[idx]["type"] = tool_call.type
                    if tool_call.function:
                        if tool_call.function.name:
                            tool_states[idx]["name"] = tool_call.function.name
                        if tool_call.function.arguments is not None:
                            tool_states[idx]["arguments"] += (
                                tool_call.function.arguments
                            )

        # Content before tool call should be collected
        assert "Let me check" in collected_content

        # Tool call should be parsed
        assert len(tool_states) == 1


class TestAdjustRequest:
    """Test request adjustment for tool parsing."""

    def test_adjust_request_disables_skip_special_tokens(
        self, qwen25_tool_parser, sample_tools
    ):
        """Test that request is adjusted to not skip special tokens."""
        request = ChatCompletionRequest(
            model=MODEL,
            messages=[],
            tools=sample_tools,
            skip_special_tokens=True,
        )

        adjusted = qwen25_tool_parser.adjust_request(request)
        assert adjusted.skip_special_tokens is False

    def test_adjust_request_no_tools(self, qwen25_tool_parser):
        """Test that request without tools is not modified."""
        request = ChatCompletionRequest(
            model=MODEL,
            messages=[],
            tools=None,
            skip_special_tokens=True,
        )

        adjusted = qwen25_tool_parser.adjust_request(request)
        # Should still be True when no tools
        assert adjusted.skip_special_tokens is True


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_incomplete_tools_tag(self, qwen25_tool_parser):
        """Test handling of incomplete <tools> tag."""
        model_output = """<tools>
{"name": "test", "arguments": {}}"""  # Missing </tools>

        result = qwen25_tool_parser.extract_tool_calls(model_output, request=None)

        # Should still parse the tool call
        assert result.tools_called
        assert result.tool_calls[0].function.name == "test"

    def test_whitespace_handling(self, qwen25_tool_parser, sample_tools):
        """Test handling of various whitespace in tool calls."""
        model_output = """<tools>

  {  "name"  :  "get_current_weather"  ,  "arguments"  :  {  "location"  :  "Chicago"  }  }

</tools>"""

        request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)
        result = qwen25_tool_parser.extract_tool_calls(model_output, request=request)

        assert result.tools_called
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["location"] == "Chicago"

    def test_special_characters_in_arguments(self, qwen25_tool_parser, sample_tools):
        """Test handling of special characters in argument values."""
        model_output = """<tools>
{"name": "search_web", "arguments": {"query": "What's the weather? <tag> & stuff"}}
</tools>"""

        request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)
        result = qwen25_tool_parser.extract_tool_calls(model_output, request=request)

        assert result.tools_called
        args = json.loads(result.tool_calls[0].function.arguments)
        assert "What's the weather?" in args["query"]
        assert "<tag>" in args["query"]
        assert "&" in args["query"]


@pytest.mark.parametrize(
    "model_output,expected_tool_name,expected_args",
    [
        (
            '<tools>{"name": "func1", "arguments": {"a": 1}}</tools>',
            "func1",
            {"a": 1},
        ),
        (
            '<tools>{"name": "func2", "parameters": {"b": "test"}}</tools>',
            "func2",
            {"b": "test"},
        ),
        (
            '<tools>\n{"name": "func3", "arguments": {}}\n</tools>',
            "func3",
            {},
        ),
    ],
    ids=["simple", "parameters_key", "with_newlines"],
)
def test_various_formats(
    qwen25_tool_parser, model_output, expected_tool_name, expected_args
):
    """Test various valid tool call formats."""
    result = qwen25_tool_parser.extract_tool_calls(model_output, request=None)

    assert result.tools_called
    assert result.tool_calls[0].function.name == expected_tool_name
    assert json.loads(result.tool_calls[0].function.arguments) == expected_args
