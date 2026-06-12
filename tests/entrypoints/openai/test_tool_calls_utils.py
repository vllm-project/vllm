# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for maybe_filter_parallel_tool_calls (issue #44948).

``parallel_tool_calls`` defaults to ``true``; an explicit ``null`` means
"unspecified" and must resolve to that default, the same as omitting the
field. Only an explicit ``false`` trims the response to the first tool call.
"""

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatMessage,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.serve.utils.tool_calls_utils import (
    maybe_filter_parallel_tool_calls,
)


def _make_request(**kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        model="test-model",
        **kwargs,
    )


def _make_choice(num_tool_calls: int) -> ChatCompletionResponseChoice:
    tool_calls = [
        ToolCall(function=FunctionCall(name=f"fn_{i}", arguments="{}"))
        for i in range(num_tool_calls)
    ]
    return ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", tool_calls=tool_calls),
    )


def _make_stream_choice(num_tool_calls: int) -> ChatCompletionResponseStreamChoice:
    tool_calls = [
        DeltaToolCall(index=i, function=DeltaFunctionCall(name=f"fn_{i}"))
        for i in range(num_tool_calls)
    ]
    return ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(tool_calls=tool_calls),
    )


@pytest.mark.cpu_test
def test_omitted_defaults_to_true_keeps_all_tool_calls():
    request = _make_request()
    assert request.parallel_tool_calls is True

    choice = maybe_filter_parallel_tool_calls(_make_choice(3), request)
    assert len(choice.message.tool_calls) == 3


@pytest.mark.cpu_test
def test_explicit_true_keeps_all_tool_calls():
    request = _make_request(parallel_tool_calls=True)
    choice = maybe_filter_parallel_tool_calls(_make_choice(3), request)
    assert len(choice.message.tool_calls) == 3


@pytest.mark.cpu_test
def test_explicit_false_trims_to_first_tool_call():
    request = _make_request(parallel_tool_calls=False)
    choice = maybe_filter_parallel_tool_calls(_make_choice(3), request)
    assert len(choice.message.tool_calls) == 1
    assert choice.message.tool_calls[0].function.name == "fn_0"


@pytest.mark.cpu_test
def test_explicit_null_resolves_to_default_keeps_all_tool_calls():
    """Regression test for #44948: explicit ``null`` was treated as falsy and
    filtered like ``false``; it must behave like an omitted field instead."""
    request = _make_request(parallel_tool_calls=None)
    assert request.parallel_tool_calls is None

    choice = maybe_filter_parallel_tool_calls(_make_choice(3), request)
    assert len(choice.message.tool_calls) == 3


@pytest.mark.cpu_test
def test_streaming_explicit_false_trims_to_index_zero():
    request = _make_request(parallel_tool_calls=False)
    choice = maybe_filter_parallel_tool_calls(_make_stream_choice(3), request)
    assert [tc.index for tc in choice.delta.tool_calls] == [0]


@pytest.mark.cpu_test
def test_streaming_explicit_null_keeps_all_tool_calls():
    """Streaming variant of the #44948 regression."""
    request = _make_request(parallel_tool_calls=None)
    choice = maybe_filter_parallel_tool_calls(_make_stream_choice(3), request)
    assert [tc.index for tc in choice.delta.tool_calls] == [0, 1, 2]
