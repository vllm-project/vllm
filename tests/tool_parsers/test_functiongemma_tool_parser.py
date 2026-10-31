# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.functiongemma_tool_parser import FunctionGemmaToolParser


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]
    tokenizer.get_vocab.return_value = {}
    return tokenizer


@pytest.fixture
def parser(mock_tokenizer):
    return FunctionGemmaToolParser(mock_tokenizer)


@pytest.fixture
def mock_request():
    request = MagicMock(spec=ChatCompletionRequest)
    request.tools = []
    request.tool_choice = "auto"
    request.parallel_tool_calls = True
    return request


class TestExtractToolCalls:
    def test_no_tool_calls(self, parser, mock_request):
        model_output = "Hello, how can I help you today?"
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == model_output

    def test_single_tool_call(self, parser, mock_request):
        model_output = (
            "<start_function_call>call:get_weather{location:<escape>London<escape>}"
            "<end_function_call>"
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        assert '"location": "London"' in result.tool_calls[0].function.arguments

    def test_multiple_arguments(self, parser, mock_request):
        model_output = (
            "<start_function_call>call:get_weather{"
            "location:<escape>San Francisco<escape>,"
            "unit:<escape>celsius<escape>}"
            "<end_function_call>"
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        args = result.tool_calls[0].function.arguments
        assert "San Francisco" in args
        assert "celsius" in args

    def test_text_before_tool_call(self, parser, mock_request):
        model_output = (
            "Let me check the weather for you. "
            "<start_function_call>call:get_weather{location:<escape>Paris<escape>}"
            "<end_function_call>"
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert result.content == "Let me check the weather for you."

    def test_multiple_tool_calls(self, parser, mock_request):
        model_output = (
            "<start_function_call>call:get_weather{location:<escape>London<escape>}"
            "<end_function_call>"
            "<start_function_call>call:get_time{timezone:<escape>UTC<escape>}"
            "<end_function_call>"
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[1].function.name == "get_time"


class TestParseArguments:
    def test_empty_arguments(self, parser, mock_request):
        result = parser._parse_arguments("", mock_request, "lookup")
        assert result == {}

    def test_single_string_argument(self, parser, mock_request):
        result = parser._parse_arguments(
            "city:<escape>Tokyo<escape>", mock_request, "lookup"
        )
        assert result == {"city": "Tokyo"}

    def test_multiple_arguments(self, parser, mock_request):
        args_str = "city:<escape>Tokyo<escape>,country:<escape>Japan<escape>"
        result = parser._parse_arguments(args_str, mock_request, "lookup")
        assert result == {"city": "Tokyo", "country": "Japan"}

    def test_numeric_string_argument(self, parser, mock_request):
        result = parser._parse_arguments(
            "count:<escape>42<escape>", mock_request, "lookup"
        )
        assert result == {"count": "42"}

    def test_boolean_string_argument(self, parser, mock_request):
        result = parser._parse_arguments(
            "enabled:<escape>true<escape>", mock_request, "lookup"
        )
        assert result == {"enabled": "true"}

    def test_argument_with_spaces(self, parser, mock_request):
        result = parser._parse_arguments(
            "message:<escape>Hello World<escape>", mock_request, "lookup"
        )
        assert result == {"message": "Hello World"}


@pytest.mark.parametrize(
    "value",
    ["123", "true", "false", "null", "1e3", "[1,2]", '{"x":1}', '"quoted"', ""],
)
@pytest.mark.parametrize("streaming", [False, True])
def test_escape_delimited_arguments_remain_strings(value, streaming):
    """String delimiters preserve literal content even when it is valid JSON."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import TokenizersBackend

    tokenizer = TokenizersBackend(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    request = ChatCompletionRequest(
        model="functiongemma",
        messages=[{"role": "user", "content": "Pass the value to lookup."}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                    },
                },
            }
        ],
        tool_choice="auto",
    )
    parser = FunctionGemmaToolParser(tokenizer, tools=request.tools)
    chunks = [
        "<start_function_call>",
        "call:lookup{",
        f"value:<escape>{value}<escape>",
        "}<end_function_call>",
    ]
    if streaming:
        previous = ""
        fragments = []
        for chunk in chunks:
            current = previous + chunk
            delta = parser.extract_tool_calls_streaming(
                previous, current, chunk, [], [], [], request
            )
            previous = current
            if delta and delta.tool_calls:
                for call in delta.tool_calls:
                    if call.function and call.function.arguments:
                        fragments.append(call.function.arguments)
        arguments = "".join(fragments)
    else:
        result = parser.extract_tool_calls("".join(chunks), request)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        arguments = result.tool_calls[0].function.arguments
    assert json.loads(arguments) == {"value": value}


class TestAdjustRequest:
    def test_skip_special_tokens_disabled(self, parser, mock_request):
        mock_request.tools = [{"type": "function", "function": {"name": "test"}}]
        mock_request.tool_choice = "auto"
        mock_request.skip_special_tokens = True

        result = parser.adjust_request(mock_request)
        assert result.skip_special_tokens is False

    def test_skip_special_tokens_when_tool_choice_none(self, parser, mock_request):
        mock_request.tools = [{"type": "function", "function": {"name": "test"}}]
        mock_request.tool_choice = "none"
        mock_request.skip_special_tokens = True

        result = parser.adjust_request(mock_request)
        assert result.skip_special_tokens is True


class TestBufferDeltaText:
    def test_regular_text_not_buffered(self, parser):
        result = parser._buffer_delta_text("hello")
        assert result == "hello"
        assert parser.buffered_delta_text == ""

    def test_complete_tag_flushed(self, parser):
        parser.buffered_delta_text = "<start_function_"
        result = parser._buffer_delta_text("call>")
        assert "<start_function_call>" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


@pytest.mark.parametrize(
    ("schema", "value", "expected"),
    [
        ({"type": "integer"}, "42", 42),
        ({"type": "number"}, "1.5", 1.5),
        ({"type": "boolean"}, "true", True),
        ({"type": "array", "items": {"type": "integer"}}, "[1,2]", [1, 2]),
        ({"type": ["string", "null"]}, "null", None),
        ({"type": "integer"}, "not a number", "not a number"),
        ({}, "42", "42"),
    ],
)
def test_escape_delimited_arguments_follow_declared_type(schema, value, expected):
    """A non-string type declared in the tool schema still converts the value,
    e.g. when the bundled chat template renders an earlier integer argument
    inside <escape> and the model echoes that form."""
    request = ChatCompletionRequest(
        model="functiongemma",
        messages=[{"role": "user", "content": "Pass the value to lookup."}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object", "properties": {"value": schema}},
                },
            }
        ],
    )
    parser = FunctionGemmaToolParser(MagicMock(), tools=request.tools)
    result = parser.extract_tool_calls(
        "<start_function_call>call:lookup{"
        f"value:<escape>{value}<escape>}}<end_function_call>",
        request,
    )
    assert result.tools_called
    assert json.loads(result.tool_calls[0].function.arguments) == {"value": expected}
