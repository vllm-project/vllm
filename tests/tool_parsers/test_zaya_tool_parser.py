# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the ZAYA1 parser: Qwen3 grammar with `<zyphra_tool_call>`."""

import json

import pytest
import regex as re

from tests.parser.engine.conftest import make_mock_tokenizer
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.parser.parser_manager import ParserManager

_VOCAB = {
    "<think>": 10,
    "</think>": 11,
    "<|im_end|>": 12,
    "<zyphra_tool_call>": 13,
    "</zyphra_tool_call>": 14,
}

_TOOLS = [
    ChatCompletionToolsParam(
        type="function",
        function={
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                },
            },
        },
    )
]

_TOOL_CALL = (
    "<zyphra_tool_call>\n<function=get_weather>\n"
    "<parameter=city>\nParis\n</parameter>\n"
    "<parameter=days>\n3\n</parameter>\n"
    "</function>\n</zyphra_tool_call>"
)


@pytest.fixture
def parser():
    """The parser the server builds for `--reasoning-parser zaya
    --tool-call-parser zaya --enable-auto-tool-choice`."""
    parser_cls = ParserManager.get_parser(
        tool_parser_name="zaya",
        reasoning_parser_name="zaya",
        enable_auto_tools=True,
    )
    assert parser_cls is not None
    return parser_cls(make_mock_tokenizer(_VOCAB), _TOOLS)


@pytest.fixture
def request_obj():
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        tools=_TOOLS,
    )


@pytest.mark.parametrize(
    "output",
    [
        f"Check the forecast.\n</think>\n\n{_TOOL_CALL}",
        # The tool call wrapper also ends reasoning when `</think>` is missing
        f"Check the forecast.\n{_TOOL_CALL}",
    ],
)
def test_reasoning_then_tool_call(parser, request_obj, output):
    reasoning, content, tool_calls = parser.parse(
        output, request_obj, enable_auto_tools=True
    )

    assert reasoning is not None and reasoning.strip() == "Check the forecast."
    assert not (content or "").strip()
    assert tool_calls is not None and len(tool_calls) == 1
    assert tool_calls[0].name == "get_weather"
    assert json.loads(tool_calls[0].arguments) == {"city": "Paris", "days": 3}


def test_streaming_matches_non_streaming(parser, request_obj):
    output = f"Check the forecast.\n</think>\n\n{_TOOL_CALL}"
    # Special tokens arrive whole; everything else a few characters at a time
    pieces = [
        chunk
        for part in re.split(r"(<[^>]*>)", output)
        for chunk in ([part] if part in _VOCAB else re.findall(r".{1,3}", part, re.S))
        if chunk
    ]

    reasoning, name, arguments = "", None, ""
    for i, piece in enumerate(pieces):
        delta = parser.parse_delta(
            piece,
            [_VOCAB[piece]] if piece in _VOCAB else [],
            request_obj,
            finished=i == len(pieces) - 1,
        )
        if delta is None:
            continue
        reasoning += delta.reasoning or ""
        for tool_call in delta.tool_calls or []:
            name = tool_call.function.name or name
            arguments += tool_call.function.arguments or ""

    assert reasoning.strip() == "Check the forecast."
    assert name == "get_weather"
    assert json.loads(arguments) == {"city": "Paris", "days": 3}
