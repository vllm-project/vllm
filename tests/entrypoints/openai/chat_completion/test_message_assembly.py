# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.entrypoints.generate.base.protocol import FunctionCall
from vllm.entrypoints.openai.chat_completion.message_assembly import (
    assemble_chat_message_fields,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

TOOLS = [
    {
        "type": "function",
        "function": {"name": "get_weather", "parameters": {"type": "object"}},
    }
]
WITH_TOOL_PARSER: Any = SimpleNamespace(tool_parser_cls=object)
WITHOUT_TOOL_PARSER: Any = SimpleNamespace(tool_parser_cls=None)


def _request(**kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "hi"}], **kwargs
    )


def _calls(*ids: str | None) -> list[FunctionCall]:
    return [FunctionCall(id=i, name="get_weather", arguments="{}") for i in ids]


def _assemble(request, *, enable=True, parser_cls=WITH_TOOL_PARSER, calls=None):
    return assemble_chat_message_fields(
        request,
        "think",
        "text",
        _calls("a") if calls is None else calls,
        enable_auto_tools=enable,
        parser_cls=parser_cls,
    )


@pytest.mark.parametrize(
    "kwargs",
    [{"enable": False}, {"parser_cls": WITHOUT_TOOL_PARSER}, {"parser_cls": None}],
)
def test_auto_tools_off_drops_tool_calls(kwargs):
    fields = _assemble(_request(tools=TOOLS), **kwargs)
    assert fields.tool_calls == []
    assert fields.content == "text"


@pytest.mark.parametrize("tool_choice", ["none", None])
def test_none_tool_choice_drops_tool_calls(tool_choice):
    request = _request(tools=TOOLS)
    request.tool_choice = tool_choice
    fields = _assemble(request)
    assert fields.tool_calls == []
    assert fields.content == "text"


def test_auto_with_tools_keeps_tool_calls():
    fields = _assemble(_request(tools=TOOLS, tool_choice="auto"))
    assert [tc.id for tc in fields.tool_calls] == ["a"]
    assert fields.content == "text"


def test_auto_without_tools_drops_tool_calls():
    request = _request(tools=TOOLS, tool_choice="auto")
    request.tools = []
    assert _assemble(request).tool_calls == []


@pytest.mark.parametrize("enable", [True, False])
@pytest.mark.parametrize(
    "tool_choice",
    ["required", {"type": "function", "function": {"name": "get_weather"}}],
)
def test_required_and_named_keep_tool_calls_and_default_content(enable, tool_choice):
    fields = assemble_chat_message_fields(
        _request(tools=TOOLS, tool_choice=tool_choice),
        None,
        None,
        _calls("a"),
        enable_auto_tools=enable,
        parser_cls=None,
    )
    assert fields.content == ""
    assert [tc.id for tc in fields.tool_calls] == ["a"]


def test_generates_missing_tool_call_id():
    fields = _assemble(_request(tools=TOOLS, tool_choice="auto"), calls=_calls(None))
    assert fields.tool_calls[0].id


def test_parallel_tool_calls_false_keeps_first():
    request = _request(tools=TOOLS, tool_choice="auto", parallel_tool_calls=False)
    fields = _assemble(request, calls=_calls("a", "b"))
    assert [tc.id for tc in fields.tool_calls] == ["a"]


@pytest.mark.parametrize("include_reasoning", [True, False])
def test_include_reasoning(include_reasoning):
    fields = _assemble(_request(include_reasoning=include_reasoning))
    assert fields.reasoning == ("think" if include_reasoning else None)
