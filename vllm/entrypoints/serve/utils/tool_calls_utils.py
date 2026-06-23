# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TypeVar

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
)
from vllm.entrypoints.openai.engine.protocol import (
    FunctionCall,
    ToolCall,
)

# Used internally
_ChatCompletionResponseChoiceT = TypeVar(
    "_ChatCompletionResponseChoiceT",
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
)


def maybe_filter_parallel_tool_calls(
    choice: _ChatCompletionResponseChoiceT, request: ChatCompletionRequest
) -> _ChatCompletionResponseChoiceT:
    """Filter to first tool call only when parallel_tool_calls is explicitly False."""

    if request.parallel_tool_calls is not False:
        return choice

    if isinstance(choice, ChatCompletionResponseChoice) and choice.message.tool_calls:
        choice.message.tool_calls = choice.message.tool_calls[:1]
    elif (
        isinstance(choice, ChatCompletionResponseStreamChoice)
        and choice.delta.tool_calls
    ):
        choice.delta.tool_calls = [
            tool_call for tool_call in choice.delta.tool_calls if tool_call.index == 0
        ]

    return choice


def make_tool_call_items(
    tool_calls: list[FunctionCall] | None,
) -> list[ToolCall]:
    return [
        ToolCall(id=tc.id, function=tc) if tc.id else ToolCall(function=tc)
        for tc in (tool_calls or [])
    ]
