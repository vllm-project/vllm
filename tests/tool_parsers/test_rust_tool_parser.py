# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.tool_parsers.rust_tool_parser import RustToolParser

# The PyO3 extension is an optional build artifact; skip when absent.
_rust_tool_parser = pytest.importorskip("vllm._rust_tool_parser")

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


class KimiK2RustToolParser(RustToolParser):
    rust_parser_name = "KimiK2ToolParser"
    tool_call_start_token = "<|tool_calls_section_begin|>"


def sample_tools() -> list[ChatCompletionToolsParam]:
    return [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "date": {"type": "string"},
                    },
                },
            },
        ),
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "add",
                "description": "Add two integers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                    },
                },
            },
        ),
    ]


EXPECTED_CALLS = [
    ("get_weather", {"location": "SF", "date": "2024-01-16"}),
    ("add", {"x": 3, "y": 5}),
]


def build_invoke(
    function_name: str,
    params: Sequence[tuple[str, str, bool]],
) -> str:
    param_text = "\n".join(
        f'{PARAM_START}{name}" string="{str(is_string).lower()}">{value}{PARAM_END}'
        for name, value, is_string in params
    )
    return f'{INV_START}{function_name}">\n{param_text}\n{INV_END}\n'


def build_tool_call() -> str:
    weather = build_invoke(
        "get_weather",
        [
            ("location", "SF", True),
            ("date", "2024-01-16", True),
        ],
    )
    add = build_invoke(
        "add",
        [
            ("x", "3", False),
            ("y", "5", False),
        ],
    )
    return f"{TC_START}\n{weather}{add}{TC_END}"


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
    tools = [
        _rust_tool_parser.Tool(
            tool.function.name,
            tool.function.description,
            tool.function.parameters,
            None,
        )
        for tool in sample_tools()
    ]
    parser = _rust_tool_parser.ToolParser("DeepSeekV4ToolParser", tools)
    output = _rust_tool_parser.ToolParserOutput()

    parser.parse_into(build_tool_call(), output)
    output.append(parser.finish())
    output = output.coalesce_calls()

    assert parser.preserve_special_tokens()
    assert output.normal_text == ""
    assert len(output.calls) == 2
    for call, (name, arguments) in zip(output.calls, EXPECTED_CALLS):
        assert call.name == name
        assert json.loads(call.arguments) == arguments


def test_rust_tool_parser_adapter_extracts_complete_output() -> None:
    tools = sample_tools()
    parser = DeepSeekV4RustToolParser(MOCK_TOKENIZER, tools=tools)

    result = parser.extract_tool_calls(
        "Let me create it. " + build_tool_call(),
        ChatCompletionRequest(messages=[], model="m", tools=tools),
    )

    assert result.tools_called
    assert result.content == "Let me create it. "
    assert len(result.tool_calls) == 2
    for tool_call, (name, arguments) in zip(result.tool_calls, EXPECTED_CALLS):
        assert tool_call.function.name == name
        assert json.loads(tool_call.function.arguments) == arguments


def test_rust_tool_parser_adapter_streaming_handles_multiple_calls() -> None:
    parser = DeepSeekV4RustToolParser(MOCK_TOKENIZER, tools=sample_tools())

    deltas = parse_streaming(parser, build_tool_call(), chunk_size=5)

    names = [
        tool_call.function.name
        for delta in deltas
        for tool_call in delta.tool_calls or []
        if tool_call.function is not None and tool_call.function.name is not None
    ]
    assert names == [name for name, _ in EXPECTED_CALLS]
    for index, (_, arguments) in enumerate(EXPECTED_CALLS):
        assert json.loads(collect_streamed_arguments(deltas, index)) == arguments


def test_rust_tool_parser_adapter_ignores_midstream_empty_delta() -> None:
    parser = DeepSeekV4RustToolParser(MOCK_TOKENIZER, tools=sample_tools())
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
    assert names == [name for name, _ in EXPECTED_CALLS]
    for index, (_, arguments) in enumerate(EXPECTED_CALLS):
        assert json.loads(collect_streamed_arguments(deltas, index)) == arguments


KIMI_EXPECTED_IDS = ["functions.get_weather:0", "functions.add:1"]


def build_kimi_tool_call() -> str:
    return (
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>"
        '{"location": "SF", "date": "2024-01-16"}<|tool_call_end|>'
        "<|tool_call_begin|>functions.add:1<|tool_call_argument_begin|>"
        '{"x": 3, "y": 5}<|tool_call_end|>'
        "<|tool_calls_section_end|>"
    )


def test_rust_tool_parser_adapter_complete_prefers_model_tool_call_ids() -> None:
    tools = sample_tools()
    parser = KimiK2RustToolParser(MOCK_TOKENIZER, tools=tools)

    result = parser.extract_tool_calls(
        "Let me check. " + build_kimi_tool_call(),
        ChatCompletionRequest(messages=[], model="m", tools=tools),
    )

    assert result.tools_called
    assert [tool_call.id for tool_call in result.tool_calls] == KIMI_EXPECTED_IDS
    for tool_call, (name, arguments) in zip(result.tool_calls, EXPECTED_CALLS):
        assert tool_call.function.name == name
        assert json.loads(tool_call.function.arguments) == arguments


def test_rust_tool_parser_adapter_streaming_prefers_model_tool_call_ids() -> None:
    parser = KimiK2RustToolParser(MOCK_TOKENIZER, tools=sample_tools())

    deltas = parse_streaming(parser, build_kimi_tool_call(), chunk_size=5)

    ids = [
        tool_call.id
        for delta in deltas
        for tool_call in delta.tool_calls or []
        if tool_call.id is not None
    ]
    assert ids == KIMI_EXPECTED_IDS
    for index, (_, arguments) in enumerate(EXPECTED_CALLS):
        assert json.loads(collect_streamed_arguments(deltas, index)) == arguments


def test_rust_tool_parser_adapter_streaming_generates_ids_as_fallback() -> None:
    # DeepSeekV4 never emits model tool call IDs, so the bridge mints them.
    parser = DeepSeekV4RustToolParser(MOCK_TOKENIZER, tools=sample_tools())

    deltas = parse_streaming(parser, build_tool_call(), chunk_size=5)

    ids = [
        tool_call.id
        for delta in deltas
        for tool_call in delta.tool_calls or []
        if tool_call.function is not None and tool_call.function.name is not None
    ]
    assert len(ids) == len(EXPECTED_CALLS)
    assert all(ids)
    assert len(set(ids)) == len(ids)


def test_rust_tool_parser_adapter_adjust_request_is_opaque() -> None:
    tools = sample_tools()
    parser = DeepSeekV4RustToolParser(MOCK_TOKENIZER, tools=tools)
    request = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=tools,
        tool_choice="required",
        skip_special_tokens=True,
    )

    adjusted = parser.adjust_request(request)

    assert adjusted is request
    assert adjusted.skip_special_tokens is False
    assert adjusted.structured_outputs is None
