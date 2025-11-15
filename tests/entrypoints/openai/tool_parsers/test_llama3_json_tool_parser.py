# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.protocol import ExtractedToolCallInformation
from vllm.entrypoints.openai.tool_parsers.llama_tool_parser import Llama3JsonToolParser
from vllm.transformers_utils.tokenizer import AnyTokenizer


@pytest.fixture
def parser(default_tokenizer: AnyTokenizer):
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
    # Test with deeply nested JSON (more than 2 levels)
    # This is a regression test for the regex pattern bug
    model_output = (
        '{"name": "get_current_conditions", '
        '"parameters": {"location": {"city": "San Francisco", "state": "CA"}, '
        '"unit": "Fahrenheit"}}'
    )
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "get_current_conditions"

    # Verify the entire parameters object is captured
    import json
    args = json.loads(result.tool_calls[0].function.arguments)
    assert "location" in args
    assert args["location"]["city"] == "San Francisco"
    assert args["location"]["state"] == "CA"
    assert args["unit"] == "Fahrenheit"


def test_extract_tool_calls_very_deeply_nested_json(parser):
    # Test with very deeply nested JSON (3+ levels)
    model_output = (
        '{"name": "complex_tool", '
        '"parameters": {"level1": {"level2": {"level3": {"value": "deep"}}}}}'
    )
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "complex_tool"

    # Verify the entire nested structure is captured
    import json
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args["level1"]["level2"]["level3"]["value"] == "deep"


def test_extract_tool_calls_with_braces_in_strings(parser):
    # Test with braces inside string values
    # This is a regression test for string-awareness in JSON extraction
    model_output = (
        '{"name": "search", '
        '"parameters": {"query": "find users with status {active}"}}'
    )
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "search"

    # Verify the string with braces is captured correctly
    import json
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args["query"] == "find users with status {active}"


def test_extract_tool_calls_with_code_snippets(parser):
    # Test with code snippets containing braces
    model_output = (
        '{"name": "code_tool", '
        '"parameters": {"snippet": "function() { return {}; }"}}'
    )
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "code_tool"

    # Verify the code snippet is captured correctly
    import json
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args["snippet"] == "function() { return {}; }"


def test_extract_tool_calls_with_escaped_quotes(parser):
    # Test with escaped quotes in strings
    model_output = (
        '{"name": "test", '
        '"parameters": {"text": "He said \\"hello {world}\\""}}'
    )
    result = parser.extract_tool_calls(model_output, None)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "test"

    # Verify escaped quotes are handled correctly
    import json
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args["text"] == 'He said "hello {world}"'
