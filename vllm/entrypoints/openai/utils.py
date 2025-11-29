# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TypeVar

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
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
    """Filter to first tool call only when parallel_tool_calls is False."""

    if request.parallel_tool_calls:
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
