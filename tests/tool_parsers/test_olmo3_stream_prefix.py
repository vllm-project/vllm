# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for olmo3 dropping a streamed prefix of <function_calls>."""

from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.abstract_parser import DelegatingParser
from vllm.tool_parsers.olmo3_tool_parser import Olmo3PythonicToolParser

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
]


class _Olmo3Parser(DelegatingParser):
    tool_parser_cls = Olmo3PythonicToolParser


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        tools=_TOOLS,
    )


def _stream(text: str, chunk_size: int = 1) -> tuple[str, int]:
    parser = _Olmo3Parser(MagicMock())
    request = _request()
    content: list[str] = []
    tool_deltas = 0
    for start in range(0, len(text), chunk_size):
        piece = text[start : start + chunk_size]
        finished = start + chunk_size >= len(text)
        delta = parser.parse_delta(
            piece,
            [start + 1],
            request,
            prompt_token_ids=[0],
            finished=finished,
        )
        if delta is None:
            continue
        if delta.content:
            content.append(delta.content)
        if delta.tool_calls:
            tool_deltas += len(delta.tool_calls)
    return "".join(content), tool_deltas


def _nonstreaming_content(text: str) -> str | None:
    parser = _Olmo3Parser(MagicMock())
    _reasoning, content, _calls = parser.parse(
        text, _request(), enable_auto_tools=True
    )
    return content


@pytest.mark.parametrize("chunk_size", [1, 64])
@pytest.mark.parametrize(
    "text",
    [
        "<function",
        "<function_calls>",
        "<",
    ],
)
def test_streamed_function_calls_prefix_is_released(text: str, chunk_size: int):
    """A reply that never leaves the <function_calls> prefix is still content.

    The parser withholds that prefix in case the rest of the tag arrives.
    When the stream ends first, the withheld text has to be emitted. The
    non-streaming path already returns it as content.
    """
    streamed, tool_deltas = _stream(text, chunk_size)
    assert tool_deltas == 0
    assert streamed == _nonstreaming_content(text)


@pytest.mark.parametrize("chunk_size", [1, 8])
@pytest.mark.parametrize(
    "text",
    [
        "hello world",
        "<hello world",
        "use <function if you want",
    ],
)
def test_streamed_plain_text_still_passes_through(text: str, chunk_size: int):
    streamed, tool_deltas = _stream(text, chunk_size)
    assert tool_deltas == 0
    assert streamed == text
