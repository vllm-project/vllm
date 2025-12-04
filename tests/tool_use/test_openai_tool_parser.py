# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
from openai_harmony import (
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    Role,
    SystemContent,
    load_harmony_encoding,
)

from vllm.entrypoints.openai.protocol import FunctionCall, ToolCall
from vllm.entrypoints.openai.tool_parsers.openai_tool_parser import OpenAIToolParser
from vllm.tokenizers import get_tokenizer

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


def assert_tool_calls(
    actual_tool_calls: list[ToolCall],
    expected_tool_calls: list[ToolCall],
):
    assert len(actual_tool_calls) == len(expected_tool_calls)

    for actual_tool_call, expected_tool_call in zip(
        actual_tool_calls, expected_tool_calls
    ):
        assert isinstance(actual_tool_call.id, str)
        assert len(actual_tool_call.id) > 16  # Default from protocol.py
        assert actual_tool_call.type == "function"
        assert actual_tool_call.function == expected_tool_call.function


def test_extract_tool_calls_no_tools(openai_tool_parser, harmony_encoding):
    convo = Conversation.from_messages(
        [
            Message.from_role_and_content(
                Role.SYSTEM,
                SystemContent.new(),
            ),
            Message.from_role_and_content(
                Role.DEVELOPER,
                DeveloperContent.new().with_instructions("Talk like a pirate!"),
            ),
            Message.from_role_and_content(Role.USER, "Arrr, how be you?"),
            Message.from_role_and_content(
                Role.ASSISTANT, "This is a test"
            ).with_channel("final"),
        ]
    )
    token_ids = harmony_encoding.render_conversation_for_completion(
        convo, Role.ASSISTANT
    )
    extracted_info = openai_tool_parser.extract_tool_calls(
        "",
        request=None,
        token_ids=token_ids,
    )
    assert not extracted_info.tools_called
    assert extracted_info.tool_calls == []
    assert extracted_info.content == "This is a test"


@pytest.mark.parametrize(
    "tool_args",
    [
        '{"location": "Tokyo"}',
        '{\n"location": "Tokyo"\n}',
    ],
)
def test_extract_tool_calls_single_tool(
    openai_tool_parser, harmony_encoding, tool_args
):
    convo = Conversation.from_messages(
        [
            Message.from_role_and_content(Role.USER, "What is the weather in Tokyo?"),
            Message.from_role_and_content(
                Role.ASSISTANT,
                'User asks: "What is the weather in Tokyo?" We need to use get_current_weather tool.',  #  noqa: E501
            ).with_channel("analysis"),
            Message.from_role_and_content(Role.ASSISTANT, tool_args)
            .with_channel("commentary")
            .with_recipient("functions.get_current_weather")
            .with_content_type("json"),
        ]
    )
    token_ids = harmony_encoding.render_conversation_for_completion(
        convo, Role.ASSISTANT
    )

    extracted_info = openai_tool_parser.extract_tool_calls(
        "",
        request=None,
        token_ids=token_ids,
    )
    assert extracted_info.tools_called
    expected_tool_calls = [
        ToolCall(
            function=FunctionCall(
                name="get_current_weather",
                arguments=json.dumps({"location": "Tokyo"}),
            )
        )
    ]
    assert_tool_calls(extracted_info.tool_calls, expected_tool_calls)
    assert extracted_info.content is None


def test_extract_tool_calls_multiple_tools(
    openai_tool_parser,
    harmony_encoding,
):
    convo = Conversation.from_messages(
        [
            Message.from_role_and_content(
                Role.USER, "What is the weather in Tokyo based on where I'm at?"
            ),
            Message.from_role_and_content(
                Role.ASSISTANT,
                'User asks: "What is the weather in Tokyo?" based on their location. We need to use get_current_weather tool and get_user_location tool.',  #  noqa: E501
            ).with_channel("analysis"),
            Message.from_role_and_content(Role.ASSISTANT, '{"location": "Tokyo"}')
            .with_channel("commentary")
            .with_recipient("functions.get_current_weather")
            .with_content_type("json"),
            Message.from_role_and_content(Role.ASSISTANT, '{"location": "Tokyo"}')
            .with_channel("commentary")
            .with_recipient("functions.get_user_location")
            .with_content_type("json"),
            Message.from_role_and_content(Role.ASSISTANT, '{"location": "Tokyo"}')
            .with_channel("commentary")
            .with_recipient("functions.no_content_type"),
            Message.from_role_and_content(Role.ASSISTANT, "foo")
            .with_channel("commentary")
            .with_recipient("functions.not_json_no_content_type"),
            Message.from_role_and_content(Role.ASSISTANT, "{}")
            .with_channel("commentary")
            .with_recipient("functions.empty_args")
            .with_content_type("json"),
            Message.from_role_and_content(Role.ASSISTANT, "")
            .with_channel("commentary")
            .with_recipient("functions.no_args")
            .with_content_type("json"),
        ]
    )
    token_ids = harmony_encoding.render_conversation_for_completion(
        convo,
        Role.ASSISTANT,
    )

    extracted_info = openai_tool_parser.extract_tool_calls(
        "",
        request=None,
        token_ids=token_ids,
    )
    assert extracted_info.tools_called
    expected_tool_calls = [
        ToolCall(
            function=FunctionCall(
                name="get_current_weather",
                arguments=json.dumps({"location": "Tokyo"}),
            )
        ),
        ToolCall(
            function=FunctionCall(
                name="get_user_location",
                arguments=json.dumps({"location": "Tokyo"}),
            )
        ),
        ToolCall(
            function=FunctionCall(
                name="no_content_type",
                arguments=json.dumps({"location": "Tokyo"}),
            )
        ),
        ToolCall(
            function=FunctionCall(
                name="not_json_no_content_type",
                arguments="foo",
            )
        ),
        ToolCall(
            function=FunctionCall(
                name="empty_args",
                arguments=json.dumps({}),
            )
        ),
        ToolCall(
            function=FunctionCall(
                name="no_args",
                arguments="",
            )
        ),
    ]
    assert_tool_calls(extracted_info.tool_calls, expected_tool_calls)
    assert extracted_info.content is None


def test_extract_tool_calls_with_content(
    openai_tool_parser,
    harmony_encoding,
):
    final_content = "This tool call will get the weather."
    convo = Conversation.from_messages(
        [
            Message.from_role_and_content(
                Role.USER, "What is the weather in Tokyo based on where I'm at?"
            ),
            Message.from_role_and_content(
                Role.ASSISTANT,
                'User asks: "What is the weather in Tokyo?" based on their location. We need to use get_current_weather tool and get_user_location tool.',  #  noqa: E501
            ).with_channel("analysis"),
            Message.from_role_and_content(Role.ASSISTANT, '{"location": "Tokyo"}')
            .with_channel("commentary")
            .with_recipient("functions.get_current_weather")
            .with_content_type("json"),
            Message.from_role_and_content(Role.ASSISTANT, final_content).with_channel(
                "final"
            ),
        ]
    )
    token_ids = harmony_encoding.render_conversation_for_completion(
        convo,
        Role.ASSISTANT,
    )

    extracted_info = openai_tool_parser.extract_tool_calls(
        "",
        request=None,
        token_ids=token_ids,
    )
    assert extracted_info.tools_called
    expected_tool_calls = [
        ToolCall(
            function=FunctionCall(
                name="get_current_weather",
                arguments=json.dumps({"location": "Tokyo"}),
            )
        ),
    ]
    assert_tool_calls(extracted_info.tool_calls, expected_tool_calls)
    assert extracted_info.content == final_content
