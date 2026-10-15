# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TypeVar

from vllm.entrypoints.generate.base.protocol import DeltaMessage, FunctionCall
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

# Used internally
_ToolCallsContainerT = TypeVar(
    "_ToolCallsContainerT",
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    DeltaMessage,
    list[FunctionCall],
)


def maybe_filter_parallel_tool_calls(
    item: _ToolCallsContainerT,
    request: ChatCompletionRequest | ResponsesRequest,
) -> _ToolCallsContainerT:
    """Filter to first tool call only when parallel_tool_calls is explicitly False."""
    if request.parallel_tool_calls is not False:
        return item

    if isinstance(item, list):
        return item[:1]
    if isinstance(item, ChatCompletionResponseChoice):
        if item.message.tool_calls:
            item.message.tool_calls = item.message.tool_calls[:1]
        return item

    delta = item.delta if isinstance(item, ChatCompletionResponseStreamChoice) else item
    if delta.tool_calls:
        delta.tool_calls = [
            tool_call for tool_call in delta.tool_calls if tool_call.index == 0
        ]
    return item


def resolve_finish_reason(
    engine_finish_reason: str | None,
    request: ChatCompletionRequest | None,
    tool_calls_made: bool,
) -> str:
    """Map the engine finish reason to the OpenAI chat ``finish_reason``.

    ``tool_calls`` is reported only when the choice actually carries tool
    calls, the engine stopped on its own (a ``length`` cut stays visible so
    clients can tell a truncated call from a complete one) and the caller did
    not force a single named tool, for which OpenAI reports ``stop``.
    """
    finish_reason = engine_finish_reason or "stop"
    forced_named_tool = request is not None and isinstance(
        request.tool_choice, ChatCompletionNamedToolChoiceParam
    )
    if tool_calls_made and finish_reason == "stop" and not forced_named_tool:
        return "tool_calls"
    return finish_reason
