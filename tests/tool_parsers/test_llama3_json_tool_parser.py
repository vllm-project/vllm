# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Adapter-level contract tests for ``Llama3JsonToolParser.extract_tool_calls``.

The parser only needs the ``<|python_tag|>`` / end-of-turn markers from the
tokenizer, so these tests use a mock vocabulary and never download a model.
Engine-level cases (streaming, chunk boundaries, forced tool choice) live in
``tests/parser/engine/test_llama_json.py``.
"""

import json
import time
from unittest.mock import MagicMock

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from vllm.entrypoints.generate.base.protocol import ExtractedToolCallInformation
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.tool_parsers.llama_tool_parser import Llama3JsonToolParser

LLAMA_VOCAB = {
    "<|python_tag|>": 128010,
    "<|eot_id|>": 128009,
    "<|eom_id|>": 128008,
}


@pytest.fixture
def parser():
    return Llama3JsonToolParser(make_mock_tokenizer(LLAMA_VOCAB))


@pytest.fixture
def request_auto():
    req = MagicMock(spec=ChatCompletionRequest)
    req.tools = []
    req.tool_choice = "auto"
    req.include_reasoning = True
    return req


def test_extract_tool_calls_simple(parser, request_auto):
    # Text before a tool call is returned as content; text after the last
    # call is not returned.
    model_output = (
        'Here is the result: {"name": "getOpenIncidentsTool", '
        '"parameters": {}} Would you like to know more?'
    )
    result = parser.extract_tool_calls(model_output, request_auto)

    assert isinstance(result, ExtractedToolCallInformation)
    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].type == "function"
    assert result.tool_calls[0].function.name == "getOpenIncidentsTool"
    assert result.tool_calls[0].function.arguments == "{}"
    assert result.content == "Here is the result:"


def test_extract_tool_calls_with_arguments(parser, request_auto):
    # Test with a tool call that has arguments
    model_output = (
        '{"name": "searchTool", "parameters": {"query": "test query", "limit": 10}}'
    )
    result = parser.extract_tool_calls(model_output, request_auto)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "searchTool"
    # The arguments are the model's own text for the object, not a re-dump.
    args = result.tool_calls[0].function.arguments
    assert args == '{"query": "test query", "limit": 10}'


def test_extract_tool_calls_no_json(parser, request_auto):
    # Test with text that doesn't contain a JSON object
    model_output = "This is just some text without any tool calls"
    result = parser.extract_tool_calls(model_output, request_auto)

    assert result.tools_called is False
    assert len(result.tool_calls) == 0
    assert result.content == model_output


def test_extract_tool_calls_invalid_json(parser, request_auto):
    # Test with invalid JSON
    model_output = '{"name": "invalidTool", "parameters": {invalid json}'
    result = parser.extract_tool_calls(model_output, request_auto)

    assert result.tools_called is False
    assert len(result.tool_calls) == 0
    assert result.content == model_output


def test_extract_tool_calls_with_arguments_key(parser, request_auto):
    # Test with a tool call that uses "arguments" instead of "parameters"
    model_output = '{"name": "searchTool", "arguments": {"query": "test"}}'
    result = parser.extract_tool_calls(model_output, request_auto)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "searchTool"
    assert result.tool_calls[0].function.arguments == '{"query": "test"}'


def test_extract_tool_calls_multiple_json(parser, request_auto):
    # Test with multiple JSONs separated by semicolons
    model_output = (
        '{"name": "searchTool", "parameters": {"query": "test1"}}; '
        '{"name": "getOpenIncidentsTool", "parameters": {}}; '
        '{"name": "searchTool", "parameters": {"query": "test2"}}'
    )
    result = parser.extract_tool_calls(model_output, request_auto)

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


def test_extract_tool_calls_multiple_json_with_whitespace(parser, request_auto):
    # Test with multiple JSONs separated by semicolons and extra whitespace
    model_output = (
        '{"name": "searchTool", "parameters": {"query": "test1"}} ; '
        '{"name": "getOpenIncidentsTool", "parameters": {}} ; '
        '{"name": "searchTool", "parameters": {"query": "test2"}}'
    )
    result = parser.extract_tool_calls(model_output, request_auto)

    assert result.tools_called is True
    assert len(result.tool_calls) == 3
    assert result.tool_calls[0].function.name == "searchTool"
    assert result.tool_calls[1].function.name == "getOpenIncidentsTool"
    assert result.tool_calls[2].function.name == "searchTool"


def test_extract_tool_calls_multiple_json_with_surrounding_text(parser, request_auto):
    # Test with multiple JSONs and surrounding text
    model_output = (
        "Here are the results: "
        '{"name": "searchTool", "parameters": {"query": "test1"}}; '
        '{"name": "getOpenIncidentsTool", "parameters": {}}; '
        '{"name": "searchTool", "parameters": {"query": "test2"}} '
        "Would you like to know more?"
    )
    result = parser.extract_tool_calls(model_output, request_auto)

    assert result.tools_called is True
    assert len(result.tool_calls) == 3
    assert result.tool_calls[0].function.name == "searchTool"
    assert result.tool_calls[1].function.name == "getOpenIncidentsTool"
    assert result.tool_calls[2].function.name == "searchTool"
    # Leading text is content; the trailing question is not returned.
    assert result.content == "Here are the results:"


def test_extract_tool_calls_deeply_nested_json(parser, request_auto):
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
    result = parser.extract_tool_calls(model_output, request_auto)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "complexTool"
    # Verify the nested structure is preserved in the arguments
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args["level1"]["level2"]["level3"]["level4"]["value"] == "deep"


def test_extract_tool_calls_multiple_with_deep_nesting(parser, request_auto):
    # Test with multiple tool calls where some have deeply nested parameters
    model_output = (
        '{"name": "simpleTool", "parameters": {"value": "test"}}; '
        '{"name": "complexTool", "parameters": '
        '{"config": {"database": {"connection": {"pool": {"size": 10}}}}}}'
    )
    result = parser.extract_tool_calls(model_output, request_auto)

    assert result.tools_called is True
    assert len(result.tool_calls) == 2

    # Check first tool call
    assert result.tool_calls[0].function.name == "simpleTool"
    args0 = json.loads(result.tool_calls[0].function.arguments)
    assert args0["value"] == "test"

    # Check second tool call with deep nesting
    assert result.tool_calls[1].function.name == "complexTool"
    args1 = json.loads(result.tool_calls[1].function.arguments)
    assert args1["config"]["database"]["connection"]["pool"]["size"] == 10


def test_extract_tool_calls_with_quotes_and_brackets_in_string(parser, request_auto):
    # Test with quotes and brackets inside quoted string values
    model_output = (
        '{"name": "searchTool", '
        '"parameters": {'
        '"query": "test {value} [complex]",'
        '"nested": {"inner": "more {brackets}"}'
        "}}"
    )
    result = parser.extract_tool_calls(model_output, request_auto)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "searchTool"
    # Verify the string values are preserved including brackets and quotes
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args["query"] == "test {value} [complex]"
    assert args["nested"]["inner"] == "more {brackets}"


def test_extract_tool_calls_with_escaped_quotes_in_nested_json(parser, request_auto):
    # Test with escaped quotes in deeply nested JSON
    model_output = (
        '{"name": "parserTool", "parameters": {"text": "He said \\"Hello {world}\\""}}'
    )
    result = parser.extract_tool_calls(model_output, request_auto)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "parserTool"
    # Verify escaped quotes are preserved
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args["text"] == 'He said "Hello {world}"'


def test_extract_tool_calls_missing_name_key(parser, request_auto):
    # Test that missing "name" key returns content
    model_output = '{"parameters": {}}'
    result = parser.extract_tool_calls(model_output, request_auto)

    assert result.tools_called is False
    assert len(result.tool_calls) == 0
    assert result.content == model_output


def test_extract_tool_calls_missing_parameters_and_arguments_key(parser, request_auto):
    # Test that missing both "parameters" and "arguments" keys returns content
    model_output = '{"name": "toolWithoutParams"}'
    result = parser.extract_tool_calls(model_output, request_auto)

    assert result.tools_called is False
    assert len(result.tool_calls) == 0
    assert result.content == model_output


@pytest.mark.parametrize("repeat", [2, 100_000])
def test_pathological_brace_prose_is_content(parser, request_auto, repeat):
    """Brace-and-tab prose that looks like the start of JSON stays content.

    The regex scan that used to need a timeout guard is gone, so this checks
    the behavior directly: no tool call, the text comes back unchanged, and
    the work stays small even for a long repetition.
    """
    model_output = "{hello world[A(A=" + "\t)A(A=,\t" * repeat
    start = time.monotonic()
    result = parser.extract_tool_calls(model_output, request_auto)
    elapsed = time.monotonic() - start

    assert result.tools_called is False
    assert result.tool_calls == []
    assert result.content == model_output
    assert elapsed < 30
