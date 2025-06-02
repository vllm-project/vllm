# SPDX-License-Identifier: Apache-2.0

import pytest
from transformers import AutoTokenizer

from vllm.entrypoints.openai.protocol import ExtractedToolCallInformation
from vllm.entrypoints.openai.tool_parsers.llama_tool_parser import (
    Llama3JsonToolParser)


@pytest.fixture
def parser():
    # Use a small tokenizer for testing
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return Llama3JsonToolParser(tokenizer)


def test_extract_tool_calls_simple(parser):
    # Test with a simple tool call
    model_output = ('Here is the result: {"name": "getOpenIncidentsTool", '
                    '"parameters": {}} Would you like to know more?')
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
        '{"name": "searchTool", "parameters": {"query": "test query", '
        '"limit": 10}}')
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
