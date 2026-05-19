# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

import json
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.engine.protocol import FunctionCall, ToolCall
from vllm.tool_parsers import ToolParser, ToolParserManager
from vllm.tool_parsers.minicpm5xml_tool_parser import MiniCPM5XMLToolParser


def _tool(name: str, parameters: dict) -> ChatCompletionToolsParam:
    return ChatCompletionToolsParam(
        type="function",
        function={
            "name": name,
            "parameters": parameters,
        },
    )


def make_tools_weather() -> list[ChatCompletionToolsParam]:
    return [
        _tool(
            "get_weather",
            {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "date": {"type": "string"},
                },
                "required": ["city"],
            },
        )
    ]


def make_tools_sum() -> list[ChatCompletionToolsParam]:
    return [
        _tool(
            "sum_values",
            {
                "type": "object",
                "properties": {
                    "nums": {"type": "array"},
                    "exact": {"type": "boolean"},
                },
                "required": ["nums"],
            },
        )
    ]


def make_tools_no_required() -> list[ChatCompletionToolsParam]:
    return [
        _tool(
            "noop",
            {
                "type": "object",
                "properties": {"note": {"type": "string"}},
                "required": [],
            },
        )
    ]


def make_request(
    tools: list[ChatCompletionToolsParam],
    tool_choice: str = "auto",
) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
        tool_choice=tool_choice,
    )


def make_tool_call(name: str, arguments: dict) -> ToolCall:
    return ToolCall(
        type="function",
        function=FunctionCall(
            name=name,
            arguments=json.dumps(arguments, ensure_ascii=False),
        ),
    )


def assert_tool_calls(
    actual: list[ToolCall],
    expected: list[ToolCall],
) -> None:
    assert len(actual) == len(expected)
    for act, exp in zip(actual, expected):
        assert act.type == "function"
        assert act.function.name == exp.function.name
        assert json.loads(act.function.arguments) == json.loads(
            exp.function.arguments)


@pytest.fixture
def parser() -> ToolParser:
    mock_tokenizer = MagicMock()
    return MiniCPM5XMLToolParser(mock_tokenizer)


def test_registered_in_tool_parser_manager() -> None:
    cls = ToolParserManager.get_tool_parser("minicpm5")
    assert cls is MiniCPM5XMLToolParser


def test_adjust_request_skip_special_tokens(parser: ToolParser) -> None:
    request = make_request(make_tools_weather())
    assert request.skip_special_tokens is True
    adjusted = parser.adjust_request(request)
    assert adjusted.skip_special_tokens is False


def test_adjust_request_tool_choice_none(parser: ToolParser) -> None:
    request = make_request(make_tools_weather(), tool_choice="none")
    adjusted = parser.adjust_request(request)
    assert adjusted.skip_special_tokens is True


def test_no_tool_call(parser: ToolParser) -> None:
    request = make_request(make_tools_weather())
    out = parser.extract_tool_calls("How can I help you?", request)
    assert not out.tools_called
    assert out.tool_calls == []
    assert out.content == "How can I help you?"


def test_single_call_with_surrounding_text(parser: ToolParser) -> None:
    request = make_request(make_tools_weather())
    text = (
        "Intro before.\n"
        '<function name="get_weather">'
        '<param name="city">上海</param>'
        '<param name="date">2024-06-27</param>'
        "</function>\n"
        "Outro after.\n"
    )
    out = parser.extract_tool_calls(text, request)
    assert out.tools_called
    assert_tool_calls(
        out.tool_calls,
        [make_tool_call("get_weather", {
            "city": "上海",
            "date": "2024-06-27",
        })],
    )
    assert "Intro before." in (out.content or "")
    assert "Outro after." in (out.content or "")


def test_cdata_multiline(parser: ToolParser) -> None:
    request = make_request(make_tools_weather())
    text = (
        '<function name="get_weather">'
        '<param name="city"><![CDATA[北\n京]]></param>'
        '<param name="date">2024-06-27</param>'
        "</function>\n"
    )
    out = parser.extract_tool_calls(text, request)
    assert len(out.tool_calls) == 1
    args = json.loads(out.tool_calls[0].function.arguments)
    assert args["city"] == "北\n京"
    assert args["date"] == "2024-06-27"


def test_unknown_tool_block_preserved(parser: ToolParser) -> None:
    request = make_request(make_tools_weather())
    text = (
        '<function name="unknown">'
        '<param name="x">1</param>'
        "</function>\n"
    )
    out = parser.extract_tool_calls(text, request)
    assert not out.tools_called
    assert "unknown" in (out.content or "")


def test_non_string_types(parser: ToolParser) -> None:
    request = make_request(make_tools_sum())
    text = (
        '<function name="sum_values">'
        '<param name="nums">[1, 2, 3]</param>'
        '<param name="exact">true</param>'
        "</function>\n"
    )
    out = parser.extract_tool_calls(text, request)
    assert len(out.tool_calls) == 1
    args = json.loads(out.tool_calls[0].function.arguments)
    assert args["nums"] == [1, 2, 3]
    assert args["exact"] is True


def test_multiple_calls_interleaved_text(parser: ToolParser) -> None:
    tools = make_tools_weather() + make_tools_sum()
    request = make_request(tools)
    text = (
        "Head\n"
        '<function name="get_weather"><param name="city">北京</param></function>\n'
        "TXT\n"
        '<function name="sum_values"><param name="nums">[7,8,9]</param>'
        '<param name="exact">false</param></function>\n'
        "Tail\n"
    )
    out = parser.extract_tool_calls(text, request)
    assert len(out.tool_calls) == 2
    args0 = json.loads(out.tool_calls[0].function.arguments)
    assert args0["city"] == "北京"
    args1 = json.loads(out.tool_calls[1].function.arguments)
    assert args1["nums"] == [7, 8, 9]
    assert args1["exact"] is False
    assert "Head" in (out.content or "")
    assert "TXT" in (out.content or "")
    assert "Tail" in (out.content or "")


def test_incomplete_missing_function_end(parser: ToolParser) -> None:
    request = make_request(make_tools_weather())
    text = (
        '<function name="get_weather">'
        '<param name="city">北京</param>'
    )
    out = parser.extract_tool_calls(text, request)
    assert not out.tools_called
    assert "get_weather" in (out.content or "")


def test_param_missing_name_invalid(parser: ToolParser) -> None:
    request = make_request(make_tools_weather())
    text = (
        '<function name="get_weather">'
        "<param>北京</param>"
        '<param name="date">2024-06-27</param>'
        "</function>\n"
    )
    out = parser.extract_tool_calls(text, request)
    assert not out.tools_called
    assert "<param>北京</param>" in (out.content or "")


def test_duplicate_param_names_invalid(parser: ToolParser) -> None:
    request = make_request(make_tools_weather())
    text = (
        '<function name="get_weather">'
        '<param name="city">北京</param>'
        '<param name="city">上海</param>'
        "</function>\n"
    )
    out = parser.extract_tool_calls(text, request)
    assert not out.tools_called


def test_case_sensitive_param_name_invalid(parser: ToolParser) -> None:
    request = make_request(make_tools_weather())
    text = (
        '<function name="get_weather">'
        '<param name="City">北京</param>'
        "</function>\n"
    )
    out = parser.extract_tool_calls(text, request)
    assert not out.tools_called


def test_no_required_and_zero_param_valid(parser: ToolParser) -> None:
    request = make_request(make_tools_no_required())
    text = '<function name="noop"></function>\n'
    out = parser.extract_tool_calls(text, request)
    assert len(out.tool_calls) == 1
    args = json.loads(out.tool_calls[0].function.arguments)
    assert args == {}
