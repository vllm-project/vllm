# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest

from tests.tool_parsers.utils import run_tool_extraction
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import ExtractedToolCallInformation
from vllm.tool_parsers import ToolParserManager


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {}
    tokenizer.tokenize.side_effect = lambda text: list(text)
    return tokenizer


@pytest.fixture
def mock_request():
    request = MagicMock(spec=ChatCompletionRequest)
    request.tools = []
    request.tool_choice = "auto"
    return request


@pytest.fixture
def parser(mock_tokenizer):
    parser_cls = ToolParserManager.get_tool_parser("nemotron_nano_v2")
    return parser_cls(mock_tokenizer, tools=[])


def test_nemotron_nano_v2_registered_and_accepts_tools(mock_tokenizer):
    parser_cls = ToolParserManager.get_tool_parser("nemotron_nano_v2")

    parser = parser_cls(mock_tokenizer, tools=[])

    assert parser.tool_call_start_token == "<TOOLCALL>"


def test_extract_tool_calls_returns_content_without_tool_call(parser, mock_request):
    model_output = "No tool call here."

    result = parser.extract_tool_calls(model_output, mock_request)

    assert isinstance(result, ExtractedToolCallInformation)
    assert result.tools_called is False
    assert result.tool_calls == []
    assert result.content == model_output


def test_extract_tool_calls_from_nemotron_array(parser, mock_request):
    model_output = (
        "Let me check that."
        '<TOOLCALL>[{"name": "get_weather", '
        '"arguments": {"city": "Tokyo", "unit": "celsius"}}]</TOOLCALL>'
    )

    result = parser.extract_tool_calls(model_output, mock_request)

    assert result.tools_called is True
    assert result.content == "Let me check that."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].type == "function"
    assert result.tool_calls[0].function.name == "get_weather"
    assert result.tool_calls[0].function.arguments == (
        '{"city": "Tokyo", "unit": "celsius"}'
    )


def test_extract_tool_calls_wraps_single_object(parser, mock_request):
    model_output = (
        '<TOOLCALL>{"name": "lookup", "arguments": {"query": "vllm"}}</TOOLCALL>'
    )

    result = parser.extract_tool_calls(model_output, mock_request)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "lookup"
    assert result.tool_calls[0].function.arguments == '{"query": "vllm"}'


def test_extract_tool_calls_supports_string_arguments(parser, mock_request):
    model_output = (
        '<TOOLCALL>[{"name": "run_query", '
        '"arguments": "{\\"sql\\": \\"select 1\\"}"}]</TOOLCALL>'
    )

    result = parser.extract_tool_calls(model_output, mock_request)

    assert result.tools_called is True
    assert result.tool_calls[0].function.name == "run_query"
    assert result.tool_calls[0].function.arguments == '{"sql": "select 1"}'


def test_extract_tool_calls_returns_original_for_malformed(parser, mock_request):
    model_output = '<TOOLCALL>[{"name": "broken", "arguments": {}</TOOLCALL>'

    result = parser.extract_tool_calls(model_output, mock_request)

    assert result.tools_called is False
    assert result.tool_calls == []
    assert result.content == model_output


def test_streaming_reconstructs_tool_call(parser, mock_request):
    model_output = (
        "Let me check."
        '<TOOLCALL>[{"name": "get_weather", '
        '"arguments": {"city": "Tokyo", "unit": "celsius"}}]</TOOLCALL>'
    )

    content, tool_calls = run_tool_extraction(
        parser,
        list(model_output),
        request=mock_request,
        streaming=True,
    )

    assert content == "Let me check."
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"
    assert tool_calls[0].function.arguments == ('{"city": "Tokyo", "unit": "celsius"}')


def test_streaming_handles_nested_json_arguments(parser, mock_request):
    model_output = (
        '<TOOLCALL>[{"name": "search", '
        '"arguments": {"filters": {"city": "Tokyo"}, '
        '"items": [{"name": "rain", "value": true}]}}]</TOOLCALL>'
    )

    content, tool_calls = run_tool_extraction(
        parser,
        list(model_output),
        request=mock_request,
        streaming=True,
    )

    assert content is None
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "search"
    assert tool_calls[0].function.arguments == (
        '{"filters": {"city": "Tokyo"}, "items": [{"name": "rain", "value": true}]}'
    )


def test_extract_tool_calls_keeps_think_block_as_content(parser, mock_request):
    model_output = (
        "<think>\nI need the weather for Tokyo.\n</think>\n"
        '<TOOLCALL>[{"name": "get_weather", "arguments": {"city": "Tokyo"}}]</TOOLCALL>'
    )

    result = parser.extract_tool_calls(model_output, mock_request)

    assert result.tools_called is True
    assert result.content == "<think>\nI need the weather for Tokyo.\n</think>\n"
    assert result.tool_calls[0].function.name == "get_weather"
    assert result.tool_calls[0].function.arguments == '{"city": "Tokyo"}'


def test_streaming_keeps_think_block_as_content(parser, mock_request):
    model_output = (
        "<think>\nI need the weather for Tokyo.\n</think>\n"
        '<TOOLCALL>[{"name": "get_weather", "arguments": {"city": "Tokyo"}}]</TOOLCALL>'
    )

    content, tool_calls = run_tool_extraction(
        parser,
        list(model_output),
        request=mock_request,
        streaming=True,
    )

    assert content == "<think>\nI need the weather for Tokyo.\n</think>\n"
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"
    assert tool_calls[0].function.arguments == '{"city": "Tokyo"}'


def test_streaming_handles_multiple_tool_calls(parser, mock_request):
    model_output = (
        '<TOOLCALL>[{"name": "get_weather", '
        '"arguments": {"city": "Tokyo"}}, '
        '{"name": "lookup_timezone", '
        '"arguments": {"city": "Tokyo"}}]</TOOLCALL>'
    )

    content, tool_calls = run_tool_extraction(
        parser,
        list(model_output),
        request=mock_request,
        streaming=True,
        assert_one_tool_per_delta=False,
    )

    assert content is None
    assert len(tool_calls) == 2
    assert tool_calls[0].function.name == "get_weather"
    assert tool_calls[0].function.arguments == '{"city": "Tokyo"}'
    assert tool_calls[1].function.name == "lookup_timezone"
    assert tool_calls[1].function.arguments == '{"city": "Tokyo"}'


def test_streaming_single_delta_handles_content_and_tool_call(parser, mock_request):
    model_output = (
        "Let me check."
        '<TOOLCALL>[{"name": "get_weather", '
        '"arguments": {"city": "Tokyo"}}]</TOOLCALL>'
    )

    content, tool_calls = run_tool_extraction(
        parser,
        [model_output],
        request=mock_request,
        streaming=True,
    )

    assert content == "Let me check."
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"
    assert tool_calls[0].function.arguments == '{"city": "Tokyo"}'


def test_streaming_single_delta_handles_multiple_tool_calls(parser, mock_request):
    model_output = (
        '<TOOLCALL>[{"name": "get_weather", '
        '"arguments": {"city": "Tokyo"}}, '
        '{"name": "lookup_timezone", '
        '"arguments": {"city": "Tokyo"}}]</TOOLCALL>'
    )

    content, tool_calls = run_tool_extraction(
        parser,
        [model_output],
        request=mock_request,
        streaming=True,
        assert_one_tool_per_delta=False,
    )

    assert content is None
    assert len(tool_calls) == 2
    assert tool_calls[0].function.name == "get_weather"
    assert tool_calls[0].function.arguments == '{"city": "Tokyo"}'
    assert tool_calls[1].function.name == "lookup_timezone"
    assert tool_calls[1].function.arguments == '{"city": "Tokyo"}'
