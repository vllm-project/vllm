# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
            """[{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}, {"name": "get_current_weather", "arguments": {"city": "Orlando", "state": "FL", "unit": "fahrenheit"}}]""",  # noqa: E501
            [
                ToolCall(function=FunctionCall(
                    name="get_current_weather",
                    arguments=json.dumps({
                        "city": "Dallas",
                        "state": "TX",
                        "unit": "fahrenheit",
                    }),
                )),
                ToolCall(function=FunctionCall(
                    name="get_current_weather",
                    arguments=json.dumps({
                        "city": "Orlando",
                        "state": "FL",
                        "unit": "fahrenheit",
                    }),
                )),
            ],
            None,
        ),
        (
            """<think>I'll help you with that.</think>[{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}]""",  # noqa: E501
            [
                ToolCall(function=FunctionCall(
                    name="get_current_weather",
                    arguments=json.dumps({
                        "city": "Dallas",
                        "state": "TX",
                        "unit": "fahrenheit",
                    }),
                ))
            ],
            "<think>I'll help you with that.</think>",
        ),
        (
            """I'll help you with that.\n```json\n[{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}]\n```""",  # noqa: E501
            [
                ToolCall(function=FunctionCall(
                    name="get_current_weather",
                    arguments=json.dumps({
                        "city": "Dallas",
                        "state": "TX",
                        "unit": "fahrenheit",
                    }),
                ))
            ],
            "I'll help you with that.",
        ),
        (
            """I'll check the weather for you.[TOOL_CALLS][{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}]""",  # noqa: E501
            [
                ToolCall(function=FunctionCall(
                    name="get_current_weather",
                    arguments=json.dumps({
                        "city": "Dallas",
                        "state": "TX",
                        "unit": "fahrenheit",
                    }),
                ))
            ],
            "I'll check the weather for you.",
        ),
    ],
)
def test_extract_tool_calls(xlam_tool_parser, model_output,
                            expected_tool_calls, expected_content):
    extracted_tool_calls = xlam_tool_parser.extract_tool_calls(
        model_output, request=None)  # type: ignore[arg-type]
    assert extracted_tool_calls.tools_called

    assert_tool_calls(extracted_tool_calls.tool_calls, expected_tool_calls)

    assert extracted_tool_calls.content == expected_content


@pytest.mark.parametrize(
    ids=["list_structured_tool_call"],
    argnames=["model_output", "expected_tool_calls", "expected_content"],
    argvalues=[
        (
            """[{"name": "get_current_weather", "arguments": {"city": "Seattle", "state": "WA", "unit": "celsius"}}]""",  # noqa: E501
            [
                ToolCall(function=FunctionCall(
                    name="get_current_weather",
                    arguments=json.dumps({
                        "city": "Seattle",
                        "state": "WA",
                        "unit": "celsius",
                    }),
                ))
            ],
            None,
        ),
    ],
)
def test_extract_tool_calls_list_structure(xlam_tool_parser, model_output,
                                           expected_tool_calls,
                                           expected_content):
    """Test extraction of tool calls when the model outputs a list-structured tool call."""  # noqa: E501
    extracted_tool_calls = xlam_tool_parser.extract_tool_calls(
        model_output, request=None)  # type: ignore[arg-type]
    assert extracted_tool_calls.tools_called

    assert_tool_calls(extracted_tool_calls.tool_calls, expected_tool_calls)

    assert extracted_tool_calls.content == expected_content


# Test for preprocess_model_output method
def test_preprocess_model_output(xlam_tool_parser):
    # Test with list structure
    model_output = """[{"name": "get_current_weather", "arguments": {"city": "Seattle"}}]"""  # noqa: E501
    content, potential_tool_calls = xlam_tool_parser.preprocess_model_output(
        model_output)
    assert content is None
    assert potential_tool_calls == model_output

    # Test with thinking tag
    model_output = """<think>I'll help you with that.</think>[{"name": "get_current_weather", "arguments": {"city": "Seattle"}}]"""  # noqa: E501
    content, potential_tool_calls = xlam_tool_parser.preprocess_model_output(
        model_output)
    assert content == "<think>I'll help you with that.</think>"
    assert (
        potential_tool_calls ==
        '[{"name": "get_current_weather", "arguments": {"city": "Seattle"}}]')

    # Test with JSON code block
    model_output = """I'll help you with that.
```json
[{"name": "get_current_weather", "arguments": {"city": "Seattle"}}]
```"""
    content, potential_tool_calls = xlam_tool_parser.preprocess_model_output(
        model_output)
    assert content == "I'll help you with that."
    assert "get_current_weather" in potential_tool_calls

    # Test with no tool calls
    model_output = """I'll help you with that."""
    content, potential_tool_calls = xlam_tool_parser.preprocess_model_output(
        model_output)
    assert content == model_output
    assert potential_tool_calls is None


# Simulate streaming to test extract_tool_calls_streaming
def test_streaming_with_list_structure(xlam_tool_parser):
    # Reset streaming state
    xlam_tool_parser.prev_tool_calls = []
    xlam_tool_parser.current_tools_sent = []
    xlam_tool_parser.streamed_args = []
    xlam_tool_parser.current_tool_id = -1

    # Simulate receiving a message with list structure
    current_text = """[{"name": "get_current_weather", "arguments": {"city": "Seattle"}}]"""  # noqa: E501

    # First call to set up the tool
    xlam_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=current_text,
        delta_text="]",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=None,
    )

    # Make sure the tool is set up correctly
    assert (xlam_tool_parser.current_tool_id
            >= 0), "Tool index should be initialized"

    # Manually set up the state for sending the tool name
    xlam_tool_parser.current_tools_sent = [False]

    # Call to send the function name
    result = xlam_tool_parser.extract_tool_calls_streaming(
        previous_text=current_text,
        current_text=current_text,
        delta_text="",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=None,
    )

    # Check that we get a result with the proper tool call
    if result is not None:
        assert hasattr(result, "tool_calls")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_current_weather"
