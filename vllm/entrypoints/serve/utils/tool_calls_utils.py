# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TypeVar

from vllm.entrypoints.generate.base.protocol import DeltaMessage, FunctionCall
from vllm.entrypoints.openai.chat_completion.protocol import (
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
