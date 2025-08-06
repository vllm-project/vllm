# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections import defaultdict

import pytest
from openai_harmony import (HarmonyEncodingName, Message, Role,
                            load_harmony_encoding)

from vllm.entrypoints.openai.protocol import FunctionCall, ToolCall
from vllm.entrypoints.openai.tool_parsers import OpenAIToolParser
from vllm.transformers_utils.tokenizer import get_tokenizer

MODEL = "gpt2"


@pytest.fixture(scope="module")
def openai_tokenizer():
    # The parser does not use the tokenizer, but the constructor requires it.
    return get_tokenizer(MODEL)


@pytest.fixture
def openai_tool_parser(openai_tokenizer):
    return OpenAIToolParser(openai_tokenizer)


@pytest.fixture(scope="module")
def harmony_encoding():
    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def assert_tool_calls(actual_tool_calls: list[ToolCall],
                      expected_tool_calls: list[ToolCall]):
    assert len(actual_tool_calls) == len(expected_tool_calls)

    for actual_tool_call, expected_tool_call in zip(actual_tool_calls,
                                                    expected_tool_calls):
        assert isinstance(actual_tool_call.id, str)
        assert len(actual_tool_call.id) > 16  # Default from protocol.py
        assert actual_tool_call.type == "function"
        assert actual_tool_call.function == expected_tool_call.function


def test_extract_tool_calls_no_tools(openai_tool_parser, harmony_encoding):
    msg = Message.from_role_and_content(Role.ASSISTANT,
                                        "This is a test").with_channel("final")
    stop_token = harmony_encoding.token_from_string("<|return|>")
    token_ids = harmony_encoding.render_message(msg) + [stop_token]

    extracted_info = openai_tool_parser.extract_tool_calls("",
                                                           request=None,
                                                           token_ids=token_ids)
    assert not extracted_info.tools_called
    assert extracted_info.tool_calls == []
    assert extracted_info.content == "This is a test"


def test_extract_tool_calls_single_tool(openai_tool_parser, harmony_encoding):
    msg = Message.from_role_and_content(
        Role.ASSISTANT, '{"city": "Dallas"}').with_channel("commentary"). \
        with_recipient("functions.get_current_weather").with_content_type("json")
    stop_token = harmony_encoding.token_from_string("<|call|>")
    token_ids = harmony_encoding.render_message(msg) + [stop_token]

    extracted_info = openai_tool_parser.extract_tool_calls("",
                                                           request=None,
                                                           token_ids=token_ids)
    assert extracted_info.tools_called
    expected_tool_calls = [
        ToolCall(
            function=FunctionCall(name="get_current_weather",
                                  arguments=json.dumps({"city": "Dallas"})))
    ]
    assert_tool_calls(extracted_info.tool_calls, expected_tool_calls)
    assert extracted_info.content is None


def test_extract_tool_calls_multiple_tools(openai_tool_parser,
                                           harmony_encoding):
    msg1 = Message.from_role_and_content(
        Role.ASSISTANT, '{"city": "Dallas"}').with_channel("commentary"). \
        with_recipient("functions.get_current_weather").with_content_type("json")
    msg2 = Message.from_role_and_content(
        Role.ASSISTANT, '{}').with_channel("commentary"). \
        with_recipient("functions.get_user_location").with_content_type("json")
    stop_token = harmony_encoding.token_from_string("<|call|>")
    token_ids = harmony_encoding.render_message(
        msg1) + harmony_encoding.render_message(msg2) + [stop_token]

    extracted_info = openai_tool_parser.extract_tool_calls("",
                                                           request=None,
                                                           token_ids=token_ids)
    assert extracted_info.tools_called
    expected_tool_calls = [
        ToolCall(
            function=FunctionCall(name="get_current_weather",
                                  arguments=json.dumps({"city": "Dallas"}))),
        ToolCall(function=FunctionCall(name="get_user_location",
                                       arguments=json.dumps({})))
    ]
    assert_tool_calls(extracted_info.tool_calls, expected_tool_calls)
    assert extracted_info.content is None


def test_extract_tool_calls_with_reasoning(openai_tool_parser,
                                           harmony_encoding):
    msg1 = Message.from_role_and_content(
        Role.ASSISTANT, "Thinking about the weather.").with_channel("analysis")
    msg2 = Message.from_role_and_content(
        Role.ASSISTANT, '{"city": "Dallas"}').with_channel("commentary"). \
        with_recipient("functions.get_current_weather").with_content_type("json")
    msg3 = Message.from_role_and_content(
        Role.ASSISTANT, "The weather is nice.").with_channel("final")

    stop_token = harmony_encoding.token_from_string("<|return|>")
    token_ids = harmony_encoding.render_message(
        msg1) + harmony_encoding.render_message(
            msg2) + harmony_encoding.render_message(msg3) + [stop_token]

    extracted_info = openai_tool_parser.extract_tool_calls("",
                                                           request=None,
                                                           token_ids=token_ids)
    assert extracted_info.tools_called
    assert extracted_info.reasoning_content == "Thinking about the weather."
    expected_tool_calls = [
        ToolCall(
            function=FunctionCall(name="get_current_weather",
                                  arguments=json.dumps({"city": "Dallas"})))
    ]
    assert_tool_calls(extracted_info.tool_calls, expected_tool_calls)
    assert extracted_info.content == "The weather is nice."


def test_extract_tool_calls_streaming(openai_tool_parser, harmony_encoding):
    msg1 = Message.from_role_and_content(
        Role.ASSISTANT, "Thinking...").with_channel("analysis")
    msg2 = Message.from_role_and_content(
        Role.ASSISTANT, '{"location": "Tokyo"}').with_channel("commentary"). \
        with_recipient("functions.get_current_weather").with_content_type("json")

    stop_token = harmony_encoding.token_from_string("<|call|>")
    token_ids = harmony_encoding.render_message(
        msg1) + harmony_encoding.render_message(msg2) + [stop_token]

    reasoning_content = ""
    tool_calls = defaultdict(lambda: {"name": "", "arguments": ""})

    for i in range(len(token_ids)):
        delta_message = openai_tool_parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="",
            delta_text="",
            previous_token_ids=token_ids[:i],
            current_token_ids=token_ids[:i + 1],
            delta_token_ids=[token_ids[i]],
            request=None,
        )

        if delta_message:
            if delta_message.reasoning_content:
                reasoning_content += delta_message.reasoning_content
            if delta_message.tool_calls:
                for tool_call_delta in delta_message.tool_calls:
                    if tool_call_delta.function.name:
                        tool_calls[
                            tool_call_delta.index]["name"] = \
                                tool_call_delta.function.name
                    if tool_call_delta.function.arguments:
                        tool_calls[
                            tool_call_delta.index]["arguments"] += \
                                tool_call_delta.function.arguments

    assert reasoning_content == "Thinking..."
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "get_current_weather"
    assert tool_calls[0]["arguments"] == '{"location": "Tokyo"}'
