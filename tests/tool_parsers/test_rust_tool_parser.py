# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from unittest.mock import MagicMock

from vllm import _rust_tool_parser
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.tool_parsers.rust_tool_parser import RustToolParser

MOCK_TOKENIZER = MagicMock()
MOCK_TOKENIZER.get_vocab.return_value = {}

TC_START = "<｜DSML｜tool_calls>"
TC_END = "</｜DSML｜tool_calls>"
INV_START = '<｜DSML｜invoke name="'
INV_END = "</｜DSML｜invoke>"
PARAM_START = '<｜DSML｜parameter name="'
PARAM_END = "</｜DSML｜parameter>"


class DeepSeekV4RustToolParser(RustToolParser):
    rust_parser_name = "DeepSeekV4ToolParser"
    tool_call_start_token = TC_START


def sample_tool() -> ChatCompletionToolsParam:
    return ChatCompletionToolsParam(
        type="function",
        function={
            "name": "create_order",
            "description": "Create an order",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer"},
                    "shipping": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "zip": {"type": "integer"},
                        },
                    },
                },
            },
        },
    )


def build_tool_call() -> str:
    return (
        f"{TC_START}\n"
        f'{INV_START}create_order">\n'
        f'{PARAM_START}user_id" string="false">42{PARAM_END}\n'
        f'{PARAM_START}shipping" string="false">'
        f'{{"city":"Singapore","zip":18956}}'
        f"{PARAM_END}\n"
        f"{INV_END}\n"
        f"{TC_END}"
    )


def parse_streaming(
    parser: DeepSeekV4RustToolParser,
    text: str,
    chunk_size: int,
) -> list:
    deltas = []
    previous_text = ""
    for start in range(0, len(text), chunk_size):
        delta_text = text[start : start + chunk_size]
        current_text = previous_text + delta_text
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[1],
            request=MagicMock(),
        )
        previous_text = current_text
        if delta is not None:
            deltas.append(delta)

    delta = parser.extract_tool_calls_streaming(
        previous_text=previous_text,
        current_text=previous_text,
        delta_text="",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[2],
        request=MagicMock(),
    )
    if delta is not None:
        deltas.append(delta)

    return deltas


def collect_streamed_arguments(deltas: Sequence, tool_index: int = 0) -> str:
    return "".join(
        tool_call.function.arguments
        for delta in deltas
        for tool_call in delta.tool_calls or []
        if (
            tool_call.index == tool_index
            and tool_call.function is not None
            and tool_call.function.arguments is not None
        )
    )


def test_rust_tool_parser_extension_typed_api() -> None:
    tool = _rust_tool_parser.Tool(
        "create_order",
        "Create an order",
        sample_tool().function.parameters,
        None,
    )
    parser = _rust_tool_parser.ToolParser("DeepSeekV4ToolParser", [tool])
    output = _rust_tool_parser.ToolParserOutput()

    parser.parse_into(build_tool_call(), output)
    output.append(parser.finish())
    output = output.coalesce_calls()

    assert parser.preserve_special_tokens()
    assert output.normal_text == ""
    assert len(output.calls) == 1
    assert output.calls[0].name == "create_order"
    assert json.loads(output.calls[0].arguments) == {
        "user_id": 42,
        "shipping": {"city": "Singapore", "zip": 18956},
    }


def test_rust_tool_parser_adapter_extracts_complete_output() -> None:
    tool = sample_tool()
    parser = DeepSeekV4RustToolParser(MOCK_TOKENIZER, tools=[tool])

    result = parser.extract_tool_calls(
        "Let me create it. " + build_tool_call(),
        ChatCompletionRequest(messages=[], model="m", tools=[tool]),
    )

    assert result.tools_called
    assert result.content == "Let me create it. "
    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.function.name == "create_order"
    assert json.loads(tool_call.function.arguments) == {
        "user_id": 42,
        "shipping": {"city": "Singapore", "zip": 18956},
    }


def test_rust_tool_parser_adapter_streaming_flushes_final_call() -> None:
    parser = DeepSeekV4RustToolParser(MOCK_TOKENIZER, tools=[sample_tool()])

    deltas = parse_streaming(parser, build_tool_call(), chunk_size=5)

    names = [
        tool_call.function.name
        for delta in deltas
        for tool_call in delta.tool_calls or []
        if tool_call.function is not None and tool_call.function.name is not None
    ]
    assert names == ["create_order"]
    assert json.loads(collect_streamed_arguments(deltas)) == {
        "user_id": 42,
        "shipping": {"city": "Singapore", "zip": 18956},
    }


def test_rust_tool_parser_adapter_ignores_midstream_empty_delta() -> None:
    parser = DeepSeekV4RustToolParser(MOCK_TOKENIZER, tools=[sample_tool()])
    text = build_tool_call()
    split_at = len(TC_START) + 8
    deltas = []
    previous_text = ""

    for delta_text in (text[:split_at], "", text[split_at:], ""):
        current_text = previous_text + delta_text
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[1],
            request=MagicMock(),
        )
        previous_text = current_text
        if delta is not None:
            deltas.append(delta)

    names = [
        tool_call.function.name
        for delta in deltas
        for tool_call in delta.tool_calls or []
        if tool_call.function is not None and tool_call.function.name is not None
    ]
    assert names == ["create_order"]
    assert json.loads(collect_streamed_arguments(deltas)) == {
        "user_id": 42,
        "shipping": {"city": "Singapore", "zip": 18956},
    }


def test_rust_tool_parser_adapter_adjust_request_is_opaque() -> None:
    tool = sample_tool()
    parser = DeepSeekV4RustToolParser(MOCK_TOKENIZER, tools=[tool])
    request = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=[tool],
        tool_choice="required",
        skip_special_tokens=True,
    )

    adjusted = parser.adjust_request(request)

    assert adjusted is request
    assert adjusted.skip_special_tokens is False
    assert adjusted.structured_outputs is None
