# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from vllm.entrypoints.openai.protocol import FunctionCall, ToolCall
from vllm.entrypoints.openai.tool_parsers import xLAMToolParser
from vllm.transformers_utils.tokenizer import get_tokenizer

# Use a common model that is likely to be available
MODEL = "Salesforce/Llama-xLAM-2-8B-fc-r"


@pytest.fixture(scope="module")
def xlam_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)


@pytest.fixture
def xlam_tool_parser(xlam_tokenizer):
    return xLAMToolParser(xlam_tokenizer)


def assert_tool_calls(actual_tool_calls: list[ToolCall],
                      expected_tool_calls: list[ToolCall]):
    assert len(actual_tool_calls) == len(expected_tool_calls)

    for actual_tool_call, expected_tool_call in zip(actual_tool_calls,
                                                    expected_tool_calls):
        assert isinstance(actual_tool_call.id, str)
        assert len(actual_tool_call.id) > 16

        assert actual_tool_call.type == "function"
        assert actual_tool_call.function == expected_tool_call.function


def test_extract_tool_calls_no_tools(xlam_tool_parser):
    model_output = "This is a test"
    extracted_tool_calls = xlam_tool_parser.extract_tool_calls(
        model_output, request=None)  # type: ignore[arg-type]
    assert not extracted_tool_calls.tools_called
    assert extracted_tool_calls.tool_calls == []
    assert extracted_tool_calls.content == model_output


@pytest.mark.parametrize(
    ids=[
        "parallel_tool_calls",
        "single_tool_with_think_tag",
        "single_tool_with_json_code_block",
        "single_tool_with_tool_calls_tag",
    ],
    argnames=["model_output", "expected_tool_calls", "expected_content"],
    argvalues=[
        (
            '''[{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}, {"name": "get_current_weather", "arguments": {"city": "Orlando", "state": "FL", "unit": "fahrenheit"}}]''',  # noqa: E501
            [
                ToolCall(function=FunctionCall(name="get_current_weather",
                                               arguments=json.dumps(
                                                   {
                                                       "city": "Dallas",
                                                       "state": "TX",
                                                       "unit": "fahrenheit"
                                                   }))),
                ToolCall(function=FunctionCall(name="get_current_weather",
                                               arguments=json.dumps(
                                                   {
                                                       "city": "Orlando",
                                                       "state": "FL",
                                                       "unit": "fahrenheit"
                                                   })))
            ],
            None),
        (
            '''<think>I'll help you with that.</think>[{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}]''',  # noqa: E501
            [
                ToolCall(function=FunctionCall(name="get_current_weather",
                                               arguments=json.dumps(
                                                   {
                                                       "city": "Dallas",
                                                       "state": "TX",
                                                       "unit": "fahrenheit"
                                                   })))
            ],
            "<think>I'll help you with that.</think>"),
        (
            '''I'll help you with that.\n```json\n[{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}]\n```''',  # noqa: E501
            [
                ToolCall(function=FunctionCall(name="get_current_weather",
                                               arguments=json.dumps(
                                                   {
                                                       "city": "Dallas",
                                                       "state": "TX",
                                                       "unit": "fahrenheit"
                                                   })))
            ],
            "I'll help you with that."),
        (
            '''I'll check the weather for you.[TOOL_CALLS][{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}]''',  # noqa: E501
            [
                ToolCall(function=FunctionCall(name="get_current_weather",
                                               arguments=json.dumps(
                                                   {
                                                       "city": "Dallas",
                                                       "state": "TX",
                                                       "unit": "fahrenheit"
                                                   })))
            ],
            "I'll check the weather for you."),
    ],
)
def test_extract_tool_calls(xlam_tool_parser, model_output,
                            expected_tool_calls, expected_content):
    extracted_tool_calls = xlam_tool_parser.extract_tool_calls(
        model_output, request=None)  # type: ignore[arg-type]
    assert extracted_tool_calls.tools_called

    assert_tool_calls(extracted_tool_calls.tool_calls, expected_tool_calls)

    assert extracted_tool_calls.content == expected_content
