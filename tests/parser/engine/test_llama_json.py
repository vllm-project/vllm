# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for semicolon-separated Llama JSON tool calls."""

import json

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from vllm.entrypoints.generate.base.protocol import ExtractedToolCallInformation
from vllm.parser.llama_json import (
    TOOL_START,
    LlamaJsonParser,
    _llama_arg_converter,
)
from vllm.tool_parsers import ToolParserManager

# =============================================================================
# Migrated legacy non-streaming tests
# =============================================================================

_TOOL_START_ID = 128010


@pytest.fixture
def mock_tokenizer():
    return make_mock_tokenizer({TOOL_START: _TOOL_START_ID})


@pytest.fixture
def parser(mock_tokenizer):
    return LlamaJsonParser(mock_tokenizer)


def test_extract_tool_calls_with_arguments(parser):
    model_output = (
        '{"name": "searchTool", "parameters": {"query": "test query", "limit": 10}}'
    )
    result = parser.extract_tool_calls(model_output, None)

    assert isinstance(result, ExtractedToolCallInformation)
    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "searchTool"
    assert '"query": "test query"' in result.tool_calls[0].function.arguments
    assert '"limit": 10' in result.tool_calls[0].function.arguments


def test_extract_tool_calls_no_json(parser):
    model_output = "This is just some text without any tool calls"
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is False
    assert len(result.tool_calls) == 0
    assert result.content == model_output


def test_extract_tool_calls_invalid_json(parser):
    model_output = '{"name": "invalidTool", "parameters": {invalid json}'
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is False
    assert len(result.tool_calls) == 0
    assert result.content == model_output


def test_extract_tool_calls_with_arguments_key(parser):
    model_output = '{"name": "searchTool", "arguments": {"query": "test"}}'
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "searchTool"
    assert '"query": "test"' in result.tool_calls[0].function.arguments


def test_extract_tool_calls_multiple_json(parser):
    model_output = (
        '{"name": "searchTool", "parameters": {"query": "test1"}}; '
        '{"name": "getOpenIncidentsTool", "parameters": {}}; '
        '{"name": "searchTool", "parameters": {"query": "test2"}}'
    )
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is True
    assert len(result.tool_calls) == 3
    assert result.tool_calls[0].function.name == "searchTool"
    assert '"query": "test1"' in result.tool_calls[0].function.arguments
    assert result.tool_calls[1].function.name == "getOpenIncidentsTool"
    assert result.tool_calls[1].function.arguments == "{}"
    assert result.tool_calls[2].function.name == "searchTool"
    assert '"query": "test2"' in result.tool_calls[2].function.arguments


def test_extract_tool_calls_multiple_json_with_whitespace(parser):
    model_output = (
        '{"name": "searchTool", "parameters": {"query": "test1"}} ; '
        '{"name": "getOpenIncidentsTool", "parameters": {}} ; '
        '{"name": "searchTool", "parameters": {"query": "test2"}}'
    )
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is True
    assert len(result.tool_calls) == 3
    assert result.tool_calls[0].function.name == "searchTool"
    assert result.tool_calls[1].function.name == "getOpenIncidentsTool"
    assert result.tool_calls[2].function.name == "searchTool"


def test_extract_tool_calls_deeply_nested_json(parser):
    model_output = (
        '{"name": "complexTool", '
        '"parameters": {'
        '"level1": {'
        '"level2": {'
        '"level3": {'
        '"level4": {'
        '"value": "deep"'
        "}}}}}}"
    )
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "complexTool"
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args["level1"]["level2"]["level3"]["level4"]["value"] == "deep"


def test_extract_tool_calls_multiple_with_deep_nesting(parser):
    model_output = (
        '{"name": "simpleTool", "parameters": {"value": "test"}}; '
        '{"name": "complexTool", "parameters": '
        '{"config": {"database": {"connection": {"pool": {"size": 10}}}}}}'
    )
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is True
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].function.name == "simpleTool"
    args0 = json.loads(result.tool_calls[0].function.arguments)
    assert args0["value"] == "test"
    assert result.tool_calls[1].function.name == "complexTool"
    args1 = json.loads(result.tool_calls[1].function.arguments)
    assert args1["config"]["database"]["connection"]["pool"]["size"] == 10


def test_extract_tool_calls_with_quotes_and_brackets_in_string(parser):
    model_output = (
        '{"name": "searchTool", '
        '"parameters": {'
        '"query": "test {value} [complex]",'
        '"nested": {"inner": "more {brackets}"}'
        "}}"
    )
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "searchTool"
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args["query"] == "test {value} [complex]"
    assert args["nested"]["inner"] == "more {brackets}"


def test_extract_tool_calls_with_escaped_quotes_in_nested_json(parser):
    model_output = (
        '{"name": "parserTool", "parameters": {"text": "He said \\"Hello {world}\\""}}'
    )
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "parserTool"
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args["text"] == 'He said "Hello {world}"'


def test_extract_tool_calls_missing_name_key(parser):
    model_output = '{"parameters": {}}'
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is False
    assert len(result.tool_calls) == 0
    assert result.content == model_output


def test_extract_tool_calls_missing_parameters_and_arguments_key(parser):
    model_output = '{"name": "toolWithoutParams"}'
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is False
    assert len(result.tool_calls) == 0
    assert result.content == model_output


# =============================================================================
# Engine-specific streaming and chunk-invariance tests
# =============================================================================


def _tokenize(text: str) -> list[tuple[int, str]]:
    tokens: list[tuple[int, str]] = []
    i = 0
    while i < len(text):
        if text.startswith(TOOL_START, i):
            tokens.append((_TOOL_START_ID, TOOL_START))
            i += len(TOOL_START)
        else:
            tokens.append((ord(text[i]), text[i]))
            i += 1
    return tokens


def _stream(parser, request, text: str, chunk_size: int):
    tokens = _tokenize(text)
    results = []
    previous_text = ""
    previous_token_ids: list[int] = []
    for start in range(0, len(tokens), chunk_size):
        batch = tokens[start : start + chunk_size]
        delta_text = "".join(token_text for _, token_text in batch)
        delta_token_ids = [token_id for token_id, _ in batch]
        current_text = previous_text + delta_text
        current_token_ids = previous_token_ids + delta_token_ids
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=tuple(previous_token_ids),
            current_token_ids=tuple(current_token_ids),
            delta_token_ids=tuple(delta_token_ids),
            request=request,
        )
        results.append(delta)
        previous_text = current_text
        previous_token_ids = current_token_ids
    finish = parser.finish_streaming()
    if finish is not None:
        results.append(finish)
    return results


def _collect_names(results) -> list[str]:
    names: dict[int, str] = {}
    for delta in results:
        if not (delta and delta.tool_calls):
            continue
        for tool_call in delta.tool_calls:
            if tool_call.function and tool_call.function.name:
                names[tool_call.index] = (
                    names.get(tool_call.index, "") + tool_call.function.name
                )
    return [names[index] for index in sorted(names)]


def _collect_args(results) -> dict[int, str]:
    args: dict[int, str] = {}
    for delta in results:
        if not (delta and delta.tool_calls):
            continue
        for tool_call in delta.tool_calls:
            if tool_call.function and tool_call.function.arguments:
                args[tool_call.index] = (
                    args.get(tool_call.index, "") + tool_call.function.arguments
                )
    return args


def test_arg_converter_accepts_arguments_and_parameters():
    assert (
        _llama_arg_converter(
            '{"name":"a","arguments":{"value":1}}',
            False,
        )
        == '{"value":1}'
    )
    assert (
        _llama_arg_converter(
            '{"name":"a","parameters":{"value":2}}',
            False,
        )
        == '{"value":2}'
    )


@pytest.mark.parametrize("with_marker", [False, True])
def test_non_streaming_parallel_calls(parser, mock_request, with_marker):
    prefix = TOOL_START if with_marker else ""
    text = (
        f'{prefix}{{"name":"a","arguments":{{"x":1}}}}; '
        '{"name":"b","parameters":{"y":2}}'
    )
    _, content, tools = parser.parse(text, mock_request)
    assert content is None
    assert [tool.name for tool in tools] == ["a", "b"]
    assert [json.loads(tool.arguments) for tool in tools] == [{"x": 1}, {"y": 2}]


def test_brace_in_plain_text_is_not_a_tool_call(parser, mock_request):
    text = 'Example: {"name":"a","arguments":{}}'
    _, content, tools = parser.parse(text, mock_request)
    assert content == text
    assert tools is None


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 11])
@pytest.mark.parametrize("with_marker", [False, True])
def test_streaming_parallel_calls_are_chunk_invariant(
    parser,
    mock_request,
    chunk_size,
    with_marker,
):
    prefix = TOOL_START if with_marker else ""
    text = (
        f'{prefix}{{"name":"a","arguments":{{"text":"hello; world"}}}}; '
        '{"name":"b","arguments":{"items":[1,2]}}'
    )
    results = _stream(parser, mock_request, text, chunk_size)
    assert _collect_names(results) == ["a", "b"]
    args = _collect_args(results)
    assert json.loads(args[0]) == {"text": "hello; world"}
    assert json.loads(args[1]) == {"items": [1, 2]}


@pytest.mark.parametrize("name", ["llama3_json", "llama4_json"])
def test_registered_tool_parser_aliases(name):
    parser_class = ToolParserManager.get_tool_parser(name)
    assert parser_class.__name__ == "LlamaJsonEngineToolParser"
