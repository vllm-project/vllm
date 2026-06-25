# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence

from openai.types.responses import ResponseFunctionToolCall

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import (
    ResponseInputOutputItem,
    ResponsesRequest,
)


def count_tool_calls(tool_calls: object) -> int:
    if tool_calls is None:
        return 0
    if isinstance(tool_calls, (str, bytes, dict)):
        return 1
    if isinstance(tool_calls, Iterable):
        return sum(1 for _ in tool_calls)
    return 1


def count_chat_history_tool_calls(
    messages: Sequence[ChatCompletionMessageParam],
) -> int:
    return sum(
        count_tool_calls(msg.get("tool_calls"))
        for msg in messages
        if isinstance(msg, dict) and msg.get("role") == "assistant"
    )


def count_response_history_tool_calls(
    response_items: Sequence[ResponseInputOutputItem],
) -> int:
    count = 0
    for item in response_items:
        if isinstance(item, ResponseFunctionToolCall):
            count += 1
            continue

        if isinstance(item, dict):
            item_type = item.get("type")
            if item_type == "function_call":
                count += 1
            elif item.get("role") == "assistant":
                count += count_tool_calls(item.get("tool_calls"))

    return count


def count_history_tool_calls(
    request: ChatCompletionRequest | ResponsesRequest,
) -> int:
    if isinstance(request, ChatCompletionRequest):
        return count_chat_history_tool_calls(request.messages)

    request_input = request.input
    if isinstance(request_input, str):
        return 0

    return count_response_history_tool_calls(request_input)
