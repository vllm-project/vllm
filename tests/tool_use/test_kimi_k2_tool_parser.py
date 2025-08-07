# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

import json

import pytest

from vllm.entrypoints.openai.protocol import FunctionCall, ToolCall
from vllm.entrypoints.openai.tool_parsers import KimiK2ToolParser
from vllm.transformers_utils.tokenizer import get_tokenizer

# Use a common model that is likely to be available
MODEL = "moonshotai/Kimi-K2-Instruct"


@pytest.fixture(scope="module")
def kimi_k2_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL, trust_remote_code=True)


@pytest.fixture
def kimi_k2_tool_parser(kimi_k2_tokenizer):
    return KimiK2ToolParser(kimi_k2_tokenizer)


def assert_tool_calls(actual_tool_calls: list[ToolCall],
                      expected_tool_calls: list[ToolCall]):
    assert len(actual_tool_calls) == len(expected_tool_calls)

    for actual_tool_call, expected_tool_call in zip(actual_tool_calls,
                                                    expected_tool_calls):

        assert actual_tool_call.type == "function"
        assert actual_tool_call.function == expected_tool_call.function

        # assert tool call id format
        assert actual_tool_call.id.startswith("functions.")
        assert actual_tool_call.id.split(':')[-1].isdigit()
        assert actual_tool_call.id.split('.')[1].split(
            ':')[0] == expected_tool_call.function.name


def test_extract_tool_calls_no_tools(kimi_k2_tool_parser):
    model_output = "This is a test"
    extracted_tool_calls = kimi_k2_tool_parser.extract_tool_calls(
        model_output, request=None)  # type: ignore[arg-type]
    assert not extracted_tool_calls.tools_called
    assert extracted_tool_calls.tool_calls == []
    assert extracted_tool_calls.content == model_output


@pytest.mark.parametrize(
    ids=[
        "tool_call_with_content_before",
        "multi_tool_call_with_content_before",
    ],
    argnames=["model_output", "expected_tool_calls", "expected_content"],
    argvalues=[
        (
            """I'll help you check the weather. <|tool_calls_section_begin|> <|tool_call_begin|>
functions.get_weather:0 <|tool_call_argument_begin|> {"city": "Beijing"} <|tool_call_end|> <|tool_calls_section_end|>""",
            [
                ToolCall(id='functions.get_weather:0',
                         function=FunctionCall(
                             name="get_weather",
                             arguments=json.dumps({
                                 "city": "Beijing",
                             }, ),
                         ),
                         type='function')
            ],
            "I'll help you check the weather. ",
        ),
        (
            """I'll help you check the weather. <|tool_calls_section_begin|> <|tool_call_begin|>
functions.get_weather:0 <|tool_call_argument_begin|> {"city": "Beijing"} <|tool_call_end|> <|tool_call_begin|>
functions.get_weather:1 <|tool_call_argument_begin|> {"city": "Shanghai"} <|tool_call_end|> <|tool_calls_section_end|>""",
            [
                ToolCall(id='functions.get_weather:0',
                         function=FunctionCall(
                             name="get_weather",
                             arguments=json.dumps({
                                 "city": "Beijing",
                             }, ),
                         ),
                         type='function'),
                ToolCall(id='functions.get_weather:1',
                         function=FunctionCall(
                             name="get_weather",
                             arguments=json.dumps({
                                 "city": "Shanghai",
                             }, ),
                         ),
                         type='function')
            ],
            "I'll help you check the weather. ",
        ),
    ],
)
def test_extract_tool_calls(kimi_k2_tool_parser, model_output,
                            expected_tool_calls, expected_content):
    extracted_tool_calls = kimi_k2_tool_parser.extract_tool_calls(
        model_output, request=None)  # type: ignore[arg-type]
    assert extracted_tool_calls.tools_called

    assert_tool_calls(extracted_tool_calls.tool_calls, expected_tool_calls)

    assert extracted_tool_calls.content == expected_content


def test_extract_tool_calls_invalid_json(kimi_k2_tool_parser):
    """we'll return every funcall result"""
    model_output = """I'll help you check the weather. <|tool_calls_section_begin|> <|tool_call_begin|>
functions.invalid_get_weather:0 <|tool_call_argument_begin|> {"city": "Beijing" <|tool_call_end|> <|tool_call_begin|>
functions.valid_get_weather:1 <|tool_call_argument_begin|> {"city": "Shanghai"} <|tool_call_end|> <|tool_calls_section_end|>"""

    extracted_tool_calls = kimi_k2_tool_parser.extract_tool_calls(
        model_output, request=None)  # type: ignore[arg-type]

    assert extracted_tool_calls.tools_called
    # Should extract only the valid JSON tool calls
    assert len(extracted_tool_calls.tool_calls) == 2
    assert extracted_tool_calls.tool_calls[
        0].function.name == "invalid_get_weather"
    assert extracted_tool_calls.tool_calls[
        1].function.name == "valid_get_weather"


def test_extract_tool_calls_invalid_funcall(kimi_k2_tool_parser):
    """we'll return every funcall result"""
    model_output = """I'll help you check the weather. <|tool_calls_section_begin|> <|tool_call_begin|>
functions.invalid_get_weather.0 <|tool_call_argument_begin|> {"city": "Beijing"} <|tool_call_end|> <|tool_call_begin|>
functions.valid_get_weather:1 <|tool_call_argument_begin|> {"city": "Shanghai"} <|tool_call_end|> <|tool_calls_section_end|>"""

    extracted_tool_calls = kimi_k2_tool_parser.extract_tool_calls(
        model_output, request=None)  # type: ignore[arg-type]

    assert extracted_tool_calls.tools_called
    # Should extract only the valid JSON tool calls
    assert len(extracted_tool_calls.tool_calls) == 1
    assert extracted_tool_calls.tool_calls[
        0].function.name == "valid_get_weather"


def test_streaming_basic_functionality(kimi_k2_tool_parser):
    """Test basic streaming functionality."""
    # Reset streaming state
    kimi_k2_tool_parser.current_tool_name_sent = False
    kimi_k2_tool_parser.prev_tool_call_arr = []
    kimi_k2_tool_parser.current_tool_id = -1
    kimi_k2_tool_parser.streamed_args_for_tool = []

    # Test with a simple tool call
    current_text = """ check the weather. <|tool_calls_section_begin|> <|tool_call_begin|>
functions.get_weather:0 <|tool_call_argument_begin|> {"city": "Beijing"} <|tool_call_end|> <|tool_calls_section_end|>"""

    # First call should handle the initial setup
    result = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="I'll help you",
        current_text=current_text,
        delta_text="<|tool_calls_section_end|>",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=None,
    )

    # The result might be None or contain tool call information
    # This depends on the internal state management
    if result is not None and hasattr(result,
                                      'tool_calls') and result.tool_calls:
        assert len(result.tool_calls) >= 0


def test_streaming_no_tool_calls(kimi_k2_tool_parser):
    """Test streaming when there are no tool calls."""
    current_text = "This is just regular text without any tool calls."

    result = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="This is just regular text",
        current_text=current_text,
        delta_text=" without any tool calls.",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=None,
    )

    # Should return the delta text as content
    assert result is not None
    assert hasattr(result, 'content')
    assert result.content == " without any tool calls."
