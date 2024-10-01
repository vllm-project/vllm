import json
from typing import Dict, List

import pytest

from vllm.entrypoints.openai.protocol import ToolCall, FunctionCall
from vllm.entrypoints.openai.tool_parsers import JambaToolParser
from vllm.transformers_utils.tokenizer import get_tokenizer

MODEL = "ai21labs/Jamba-tiny-dev"


@pytest.fixture(scope="module")
def jamba_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)


@pytest.fixture
def jamba_tool_parser(jamba_tokenizer):
    return JambaToolParser(jamba_tokenizer)


def assert_tool_calls(actual_tool_calls: List[ToolCall],
                      expected_tool_calls: List[ToolCall]):
    assert len(actual_tool_calls) == len(expected_tool_calls)

    for actual_tool_call, expected_tool_call in zip(actual_tool_calls, expected_tool_calls):
        assert isinstance(actual_tool_call.id, str)
        assert len(actual_tool_call.id) > 16

        assert actual_tool_call.type == "function"
        assert actual_tool_call.function == expected_tool_call.function


def test_extract_tool_calls_no_tools(jamba_tool_parser):
    model_output = "This is a test"
    extracted_tool_calls = jamba_tool_parser.extract_tool_calls(model_output)
    assert extracted_tool_calls.tools_called == False
    assert extracted_tool_calls.tool_calls == []
    assert extracted_tool_calls.content == model_output

@pytest.mark.parametrize(
    ids=[
        "single_tool",
        "single_tool_with_content",
        "parallel_tools",
    ],
    argnames=["model_output", "expected_tool_calls", "expected_content"],
    argvalues=[
        (
                '''<tool_calls> [\n{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}] </tool_calls>''',
                [ToolCall(function=FunctionCall(name="get_current_weather", arguments=json.dumps({"city": "Dallas", "state": "TX", "unit": "fahrenheit"})))],
                None
         ),
        (
                '''Sure! let me call the tool for you. <tool_calls> [\n{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}] </tool_calls>''',
                [ToolCall(function=FunctionCall(name="get_current_weather", arguments=json.dumps({"city": "Dallas", "state": "TX", "unit": "fahrenheit"})))],
                "Sure! let me call the tool for you. "
         ),
        (
                '''<tool_calls> [\n{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}, \n{"name": "get_current_weather", "arguments": {"city": "Orlando", "state": "FL", "unit": "fahrenheit"}}] </tool_calls>''',
                [ToolCall(function=FunctionCall(name="get_current_weather", arguments=json.dumps({"city": "Dallas", "state": "TX", "unit": "fahrenheit"}))),
                 ToolCall(function=FunctionCall(name="get_current_weather", arguments=json.dumps({"city": "Orlando", "state": "FL", "unit": "fahrenheit"})))],
                None
        )
    ],
)
def test_extract_tool_calls(jamba_tool_parser, model_output, expected_tool_calls, expected_content):
    extracted_tool_calls = jamba_tool_parser.extract_tool_calls(model_output)
    assert extracted_tool_calls.tools_called == True

    assert_tool_calls(extracted_tool_calls.tool_calls, expected_tool_calls)

    assert extracted_tool_calls.content == expected_content
