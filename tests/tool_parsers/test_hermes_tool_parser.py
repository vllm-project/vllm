# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import ToolParser
from vllm.tool_parsers.granite4_tool_parser import Granite4ToolParser
from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser

CONFIGS = {
    "llama": {
        "tool_parser": Hermes2ProToolParser,
    },
    "granite4": {
        "tool_parser": Granite4ToolParser,
    },
}


@pytest.fixture
def qwen_tokenizer() -> TokenizerLike:
    from vllm.tokenizers import get_tokenizer

    return get_tokenizer("Qwen/Qwen3-32B")


@pytest.fixture(params=CONFIGS.keys())
def hermes_parser(request, qwen_tokenizer: TokenizerLike) -> ToolParser:
    config = CONFIGS[request.param]
    return config["tool_parser"](qwen_tokenizer)


@pytest.fixture
def any_chat_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        seed=42,
        model="Qwen/Qwen3-32B",
        messages=[],
    )


def test_hermes_parser_streaming_just_forward_text(
    qwen_tokenizer: TokenizerLike,
    hermes_parser: ToolParser,
    any_chat_request: ChatCompletionRequest,
) -> None:
    text = """This is some prior text that has nothing to do with tool calling."""
    tokens = qwen_tokenizer.encode(text)
    previous_text = ""
    delta_messages = []
    for token in tokens:
        delta_text = qwen_tokenizer.decode([token])
        current_text = previous_text + delta_text
        delta = hermes_parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=any_chat_request,
        )
        previous_text = current_text
        delta_messages.append(delta)

    for delta in delta_messages:
        assert delta is not None
        assert not delta.tool_calls

    print(delta_messages)
    assert "".join([delta.content for delta in delta_messages]) == text


def test_hermes_parser_streaming_failure_case_bug_19056(
    qwen_tokenizer: TokenizerLike,
    hermes_parser: ToolParser,
    any_chat_request: ChatCompletionRequest,
) -> None:
    text = """<tool_call>
{"name": "final_answer", "arguments": {"trigger": true}}
</tool_call>"""
    tokens = qwen_tokenizer.encode(text)
    previous_text = ""
    delta_messages = []
    for token in tokens:
        text = qwen_tokenizer.decode([token])
        current_text = previous_text + text
        delta = hermes_parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=any_chat_request,
        )
        previous_text = current_text
        if delta is not None:
            delta_messages.append(delta)

    assert delta_messages[0].tool_calls[0].function.name == "final_answer"
    tool_call_args = "".join(
        delta.tool_calls[0].function.arguments or "" for delta in delta_messages
    )
    assert tool_call_args == '{"trigger": true}'


def test_hermes_parser_streaming(
    qwen_tokenizer: TokenizerLike,
    hermes_parser: ToolParser,
    any_chat_request: ChatCompletionRequest,
) -> None:
    text = '<tool_call>\
{"name": "get_current_temperature",\
"arguments": {"location":\
"San Francisco, California, United States", "unit": "celsius"}}\
</tool_call>'

    tokens = qwen_tokenizer.encode(text)
    previous_text = ""
    delta_messages = []
    for token in tokens:
        text = qwen_tokenizer.decode([token])
        current_text = previous_text + text
        delta = hermes_parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=any_chat_request,
        )
        previous_text = current_text
        if delta is not None:
            delta_messages.append(delta)
    print(delta_messages)
    assert delta_messages[0].tool_calls[0].function.name == "get_current_temperature"
    # load to normalize whitespace
    tool_call_args = json.loads(
        "".join(
            delta.tool_calls[0].function.arguments or "" for delta in delta_messages
        )
    )
    assert tool_call_args == {
        "location": "San Francisco, California, United States",
        "unit": "celsius",
    }


def test_hermes_parser_non_streaming_no_tool_call(
    hermes_parser: ToolParser,
    any_chat_request: ChatCompletionRequest,
) -> None:
    text = """This is not a tool call."""
    tool_call = hermes_parser.extract_tool_calls(
        model_output=text,
        request=any_chat_request,
    )

    assert tool_call is not None
    assert not tool_call.tools_called


def test_hermes_parser_non_streaming_tool_call_between_tags(
    hermes_parser: ToolParser,
    any_chat_request: ChatCompletionRequest,
) -> None:
    text = """<tool_call>
{"name": "final_answer", "arguments": {"trigger": true}}
</tool_call>"""
    tool_call = hermes_parser.extract_tool_calls(
        model_output=text,
        request=any_chat_request,
    )

    assert tool_call is not None
    assert tool_call.tools_called
    assert tool_call.tool_calls[0].function.name == "final_answer"
    assert tool_call.tool_calls[0].function.arguments == '{"trigger": true}'


def test_hermes_parser_non_streaming_tool_call_until_eos(
    hermes_parser: ToolParser,
    any_chat_request: ChatCompletionRequest,
) -> None:
    if isinstance(hermes_parser, Granite4ToolParser):
        pytest.skip(reason="The Granite4 tool parser enforces a complete response")

    text = """<tool_call>
{"name": "final_answer", "arguments": {"trigger": true}}"""
    tool_call = hermes_parser.extract_tool_calls(
        model_output=text,
        request=any_chat_request,
    )

    assert tool_call is not None
    assert tool_call.tools_called
    assert tool_call.tool_calls[0].function.name == "final_answer"
    assert tool_call.tool_calls[0].function.arguments == '{"trigger": true}'


def test_hermes_parser_non_streaming_tool_call_invalid_json(
    hermes_parser: ToolParser,
    any_chat_request: ChatCompletionRequest,
) -> None:
    # Missing closing brace to trigger exception
    text = """<tool_call>
{"name": "final_answer", "arguments": {"trigger": true}"""
    tool_call = hermes_parser.extract_tool_calls(
        model_output=text,
        request=any_chat_request,
    )

    assert tool_call is not None
    assert not tool_call.tools_called
