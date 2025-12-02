# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest

from vllm.entrypoints.openai.protocol import ExtractedToolCallInformation
from vllm.entrypoints.openai.tool_parsers.llama_tool_parser import Llama3JsonToolParser
from vllm.tokenizers import TokenizerLike


@pytest.fixture
def parser(default_tokenizer: TokenizerLike):
    return Llama3JsonToolParser(default_tokenizer)


def test_extract_tool_calls_simple(parser):
    # Test with a simple tool call
    model_output = (
        'Here is the result: {"name": "getOpenIncidentsTool", '
        '"parameters": {}} Would you like to know more?'
    )
    result = parser.extract_tool_calls(model_output, None)

    assert isinstance(result, ExtractedToolCallInformation)
    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].type == "function"
    assert result.tool_calls[0].function.name == "getOpenIncidentsTool"
    assert result.tool_calls[0].function.arguments == "{}"
    assert result.content is None


def test_extract_tool_calls_with_arguments(parser):
    # Test with a tool call that has arguments
    model_output = (
        '{"name": "searchTool", "parameters": {"query": "test query", "limit": 10}}'
    )
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "searchTool"
    assert '"query": "test query"' in result.tool_calls[0].function.arguments
    assert '"limit": 10' in result.tool_calls[0].function.arguments


def test_extract_tool_calls_no_json(parser):
    # Test with text that doesn't contain a JSON object
    model_output = "This is just some text without any tool calls"
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is False
    assert len(result.tool_calls) == 0
    assert result.content == model_output


def test_extract_tool_calls_invalid_json(parser):
    # Test with invalid JSON
    model_output = '{"name": "invalidTool", "parameters": {invalid json}'
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is False
    assert len(result.tool_calls) == 0
    assert result.content == model_output


def test_extract_tool_calls_with_arguments_key(parser):
    # Test with a tool call that uses "arguments" instead of "parameters"
    model_output = '{"name": "searchTool", "arguments": {"query": "test"}}'
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "searchTool"
    assert '"query": "test"' in result.tool_calls[0].function.arguments


def test_extract_tool_calls_multiple_json(parser):
    # Test with multiple JSONs separated by semicolons
    model_output = (
        '{"name": "searchTool", "parameters": {"query": "test1"}}; '
        '{"name": "getOpenIncidentsTool", "parameters": {}}; '
        '{"name": "searchTool", "parameters": {"query": "test2"}}'
    )
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is True
    assert len(result.tool_calls) == 3

    # Check first tool call
    assert result.tool_calls[0].function.name == "searchTool"
    assert '"query": "test1"' in result.tool_calls[0].function.arguments

    # Check second tool call
    assert result.tool_calls[1].function.name == "getOpenIncidentsTool"
    assert result.tool_calls[1].function.arguments == "{}"

    # Check third tool call
    assert result.tool_calls[2].function.name == "searchTool"
    assert '"query": "test2"' in result.tool_calls[2].function.arguments


def test_extract_tool_calls_multiple_json_with_whitespace(parser):
    # Test with multiple JSONs separated by semicolons and extra whitespace
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


def test_extract_tool_calls_multiple_json_with_surrounding_text(parser):
    # Test with multiple JSONs and surrounding text
    model_output = (
        "Here are the results: "
        '{"name": "searchTool", "parameters": {"query": "test1"}}; '
        '{"name": "getOpenIncidentsTool", "parameters": {}}; '
        '{"name": "searchTool", "parameters": {"query": "test2"}} '
        "Would you like to know more?"
    )
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is True
    assert len(result.tool_calls) == 3
    assert result.tool_calls[0].function.name == "searchTool"
    assert result.tool_calls[1].function.name == "getOpenIncidentsTool"
    assert result.tool_calls[2].function.name == "searchTool"


def test_extract_tool_calls_deeply_nested_json(parser):
    # Test with deeply nested JSON parameters (5 levels)
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
    # Verify the nested structure is preserved in the arguments
    import json

    args = json.loads(result.tool_calls[0].function.arguments)
    assert args["level1"]["level2"]["level3"]["level4"]["value"] == "deep"


def test_extract_tool_calls_multiple_with_deep_nesting(parser):
    # Test with multiple tool calls where some have deeply nested parameters
    model_output = (
        '{"name": "simpleTool", "parameters": {"value": "test"}}; '
        '{"name": "complexTool", "parameters": '
        '{"config": {"database": {"connection": {"pool": {"size": 10}}}}}}'
    )
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is True
    assert len(result.tool_calls) == 2

    # Check first tool call
    assert result.tool_calls[0].function.name == "simpleTool"
    import json

    args0 = json.loads(result.tool_calls[0].function.arguments)
    assert args0["value"] == "test"

    # Check second tool call with deep nesting
    assert result.tool_calls[1].function.name == "complexTool"
    args1 = json.loads(result.tool_calls[1].function.arguments)
    assert args1["config"]["database"]["connection"]["pool"]["size"] == 10


def test_extract_tool_calls_with_quotes_and_brackets_in_string(parser):
    # Test with quotes and brackets inside quoted string values
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
    # Verify the string values are preserved including brackets and quotes
    import json

    args = json.loads(result.tool_calls[0].function.arguments)
    assert args["query"] == "test {value} [complex]"
    assert args["nested"]["inner"] == "more {brackets}"


def test_extract_tool_calls_with_escaped_quotes_in_nested_json(parser):
    # Test with escaped quotes in deeply nested JSON
    model_output = (
        '{"name": "parserTool", "parameters": {"text": "He said \\"Hello {world}\\""}}'
    )
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "parserTool"
    # Verify escaped quotes are preserved
    import json

    args = json.loads(result.tool_calls[0].function.arguments)
    assert args["text"] == 'He said "Hello {world}"'


def test_extract_tool_calls_missing_name_key(parser):
    # Test that missing "name" key returns content
    model_output = '{"parameters": {}}'
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is False
    assert len(result.tool_calls) == 0
    assert result.content == model_output


def test_extract_tool_calls_missing_parameters_and_arguments_key(parser):
    # Test that missing both "parameters" and "arguments" keys returns content
    model_output = '{"name": "toolWithoutParams"}'
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is False
    assert len(result.tool_calls) == 0
    assert result.content == model_output


def test_regex_timeout_handling(parser):
    """Test regex timeout is handled gracefully"""
    fake_problematic_input = "{hello world[A(A=" + "\t)A(A=,\t" * 2

    # create a mock regex that raises TimeoutError
    mock_regex = MagicMock()
    mock_regex.finditer.side_effect = TimeoutError("Regex timeout")

    with patch.object(parser, "tool_call_start_regex", mock_regex):
        result = parser.extract_tool_calls(fake_problematic_input, None)

        # should treat as regular text when regex times out
        assert result.content == fake_problematic_input
        assert result.tools_called is False
        assert len(result.tool_calls) == 0
        mock_regex.finditer.assert_called_once()
