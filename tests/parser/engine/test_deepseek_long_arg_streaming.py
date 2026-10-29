# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming regression tests for long DeepSeek DSML parameter values.

A long ``string="true"`` value carries no tag boundary inside it, so the
engine's ``arg_structural_chars`` gate used to suppress every delta of that
value: it was buffered until its closing tag arrived, and the client received
no bytes for the whole generation. For multi-KB values that silence exceeds a
client's read timeout, so the request looks like a dropped stream.

These tests assert the value trickles out while it is generated.
"""

import json
from dataclasses import dataclass

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.streaming_helpers import (
    collect_tool_arguments,
    simulate_tool_streaming,
)
from vllm.parser.deepseek_v32 import (
    DSML_FUNC_END,
    DSML_FUNC_START,
    DeepSeekV32Parser,
)
from vllm.parser.deepseek_v4 import (
    DSML_INVOKE_END,
    DSML_INVOKE_PREFIX,
    DSML_PARAM_CLOSE,
    DSML_PARAM_START,
    DSML_TOOL_END,
    DSML_TOOL_START,
    DeepSeekV4Parser,
)
from vllm.parser.deepseek_v41 import (
    DSML_INVOKE_END as V41_INVOKE_END,
    DSML_INVOKE_PREFIX as V41_INVOKE_PREFIX,
    DSML_PARAM_CLOSE as V41_PARAM_CLOSE,
    DSML_PARAM_START as V41_PARAM_START,
    DSML_TOOL_END as V41_TOOL_END,
    DSML_TOOL_START as V41_TOOL_START,
    DeepSeekV41Parser,
)

_VOCAB = {
    "<think>": 50,
    "</think>": 51,
    "<｜DSML｜tool_calls>": 52,
    "</｜DSML｜tool_calls>": 53,
    "<｜DSML｜function_calls>": 54,
    "</｜DSML｜function_calls>": 55,
    "<｜DSML｜ calls>": 56,
    "</｜DSML｜ calls>": 57,
}

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": "Write a file to the given path",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    }
]

# Long enough that buffering it would hold the client silent, and free of
# ``'>'`` so the structural-char gate never opens inside the value.
_BODY = (
    "研究报告正文，包含标点、数字 123 与换行，"
    "但没有任何标签尖括号。\n"
) * 40

# A value that contains ``'<'`` and a partial close-tag prefix on purpose.
_BODY_WITH_TAGS = "abc<def</ghi</｜DSML｜par xyz" * 40


@dataclass(frozen=True)
class _Dialect:
    name: str
    parser_cls: type
    tool_start: str
    tool_end: str
    invoke_prefix: str
    invoke_end: str
    param_start: str
    param_close: str


_DIALECTS = [
    _Dialect(
        name="deepseek_v4",
        parser_cls=DeepSeekV4Parser,
        tool_start=DSML_TOOL_START,
        tool_end=DSML_TOOL_END,
        invoke_prefix=DSML_INVOKE_PREFIX,
        invoke_end=DSML_INVOKE_END,
        param_start=DSML_PARAM_START,
        param_close=DSML_PARAM_CLOSE,
    ),
    _Dialect(
        name="deepseek_v41",
        parser_cls=DeepSeekV41Parser,
        tool_start=V41_TOOL_START,
        tool_end=V41_TOOL_END,
        invoke_prefix=V41_INVOKE_PREFIX,
        invoke_end=V41_INVOKE_END,
        param_start=V41_PARAM_START,
        param_close=V41_PARAM_CLOSE,
    ),
    _Dialect(
        name="deepseek_v32",
        parser_cls=DeepSeekV32Parser,
        tool_start=DSML_FUNC_START,
        tool_end=DSML_FUNC_END,
        invoke_prefix=DSML_INVOKE_PREFIX,
        invoke_end=DSML_INVOKE_END,
        param_start=DSML_PARAM_START,
        param_close=DSML_PARAM_CLOSE,
    ),
]


def _dialect_id(dialect: _Dialect) -> str:
    return dialect.name


def _write_call(dialect: _Dialect, body: str) -> str:
    return (
        f"{dialect.tool_start}{dialect.invoke_prefix}Write\">\n"
        f'{dialect.param_start} name="path" string="true">/tmp/a.md'
        f"{dialect.param_close}\n"
        f'{dialect.param_start} name="content" string="true">{body}'
        f"{dialect.param_close}\n"
        f"{dialect.invoke_end}{dialect.tool_end}"
    )


def _stream(dialect: _Dialect, body: str, request, chunk_size: int = 24):
    parser = dialect.parser_cls(make_mock_tokenizer(_VOCAB), tools=_TOOLS)
    text = _write_call(dialect, body)
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    results = simulate_tool_streaming(parser, request, chunks)
    fragments = [
        tc.function.arguments
        for delta, _ in results
        if delta and delta.tool_calls
        for tc in delta.tool_calls
        if tc.function and tc.function.arguments
    ]
    return collect_tool_arguments(results), fragments


@pytest.mark.parametrize("dialect", _DIALECTS, ids=_dialect_id)
def test_long_value_streams_while_generated(dialect, mock_request):
    mock_request.tools = _TOOLS
    arguments, fragments = _stream(dialect, _BODY, mock_request)

    assert json.loads(arguments) == {"path": "/tmp/a.md", "content": _BODY}
    # The value must trickle out while it is generated instead of arriving as
    # a single block once the closing tag lands.
    assert len(fragments) > 20
    assert max(len(fragment) for fragment in fragments) < len(_BODY) // 4


@pytest.mark.parametrize("dialect", _DIALECTS, ids=_dialect_id)
def test_value_with_angle_brackets_is_not_corrupted(dialect, mock_request):
    mock_request.tools = _TOOLS
    arguments, fragments = _stream(dialect, _BODY_WITH_TAGS, mock_request)

    assert json.loads(arguments) == {"path": "/tmp/a.md", "content": _BODY_WITH_TAGS}
    assert all("<｜DSML｜parameter" not in fragment for fragment in fragments)
