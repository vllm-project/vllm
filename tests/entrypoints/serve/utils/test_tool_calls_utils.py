# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatMessage,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    DeltaToolCall,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.serve.utils.tool_calls_utils import (
    maybe_filter_parallel_tool_calls,
)


def _make_tool_calls(n: int) -> list[ToolCall]:
    return [
        ToolCall(function=FunctionCall(name=f"tool_{i}", arguments="{}"))
        for i in range(n)
    ]


def _make_delta_tool_calls(n: int) -> list[DeltaToolCall]:
    return [DeltaToolCall(index=i) for i in range(n)]


def _make_non_stream_choice(tool_calls: list[ToolCall]) -> ChatCompletionResponseChoice:
    return ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", tool_calls=tool_calls),
        finish_reason="tool_calls",
    )


def _make_stream_choice(
    delta_tool_calls: list[DeltaToolCall],
) -> ChatCompletionResponseStreamChoice:
    return ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(tool_calls=delta_tool_calls),
        finish_reason=None,
    )


def _make_request(parallel_tool_calls) -> ChatCompletionRequest:
    req = ChatCompletionRequest.model_construct(parallel_tool_calls=parallel_tool_calls)
    return req


@pytest.mark.parametrize(
    "parallel_tool_calls,expected_count",
    [
        (True, 2),  # explicit True → keep all
        (None, 2),  # explicit null → treat as default (True) → keep all
        (False, 1),  # explicit False → trim to first
    ],
)
def test_maybe_filter_parallel_tool_calls_non_streaming(
    parallel_tool_calls, expected_count
):
    request = _make_request(parallel_tool_calls)
    choice = _make_non_stream_choice(_make_tool_calls(2))
    result = maybe_filter_parallel_tool_calls(choice, request)
    assert len(result.message.tool_calls) == expected_count


@pytest.mark.parametrize(
    "parallel_tool_calls,expected_count",
    [
        (True, 2),
        (None, 2),
        (False, 1),
    ],
)
def test_maybe_filter_parallel_tool_calls_streaming(
    parallel_tool_calls, expected_count
):
    request = _make_request(parallel_tool_calls)
    choice = _make_stream_choice(_make_delta_tool_calls(2))
    result = maybe_filter_parallel_tool_calls(choice, request)
    assert len(result.delta.tool_calls) == expected_count
