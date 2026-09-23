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


_JSON_WS = " \t\r\n"


def scan_json_object_end(raw: str, start: int) -> int | None:
    """Return the end index (exclusive) of the JSON object starting at
    ``raw[start]`` (which must be ``{``), or ``None`` when it is still
    unterminated. String- and escape-aware so braces inside strings do not
    mislead the depth count."""
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if escape:
            escape = False
            continue
        if in_string:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def extract_json_member_object(raw: str, key: str) -> str | None:
    """Extract the raw text span of the top-level object-valued member ``key``
    from a (possibly incomplete) ``{...}`` JSON wrapper.

    Returns the verbatim substring (prefix-stable as the input grows, which
    the engine's argument-delta diffing relies on), possibly an unterminated
    object prefix; ``None`` when the value has not started. Raises
    ``ValueError`` when the member is present but its value is not a JSON
    object. A verbatim substring (not re-serialized JSON) is required because
    the engine diffs successive converter outputs and expects each to extend
    the previous one; the scanner is string/escape-aware so a matching literal
    inside another value cannot mislead it, and it recovers a partial span when
    the wrapper is truncated (where ``json.loads`` would fail)."""
    depth = 0
    in_string = False
    escape = False
    string_start = -1
    last_string: str | None = None
    for i, ch in enumerate(raw):
        if escape:
            escape = False
            continue
        if in_string:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
                if depth == 1:
                    last_string = raw[string_start + 1 : i]
            continue
        if ch == '"':
            in_string = True
            string_start = i
        elif ch == ":" and depth == 1 and last_string == key:
            value_start = i + 1
            while value_start < len(raw) and raw[value_start] in _JSON_WS:
                value_start += 1
            if value_start >= len(raw):
                return None
            if raw[value_start] != "{":
                raise ValueError(f"member {key!r} must be a JSON object")
            value_end = scan_json_object_end(raw, value_start)
            if value_end is None:
                return raw[value_start:]
            return raw[value_start:value_end]
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
    return None
