# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from typing import TYPE_CHECKING

import partial_json_parser
from partial_json_parser.core.options import Allow

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.tool_parsers.utils import partial_json_loads
from vllm.utils.mistral import is_mistral_tokenizer

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
else:
    TokenizerLike = object


def _scan_string_end(text: str, start: int) -> int | None:
    """Index just past the closing quote of the string starting at ``start``.

    Returns ``None`` when the string is unterminated.
    """
    i = start + 1
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch == '"':
            return i + 1
        i += 1
    return None


def _scan_value_text(text: str, start: int) -> str:
    """Raw text of the JSON value starting at ``start``.

    For a partially written value, returns the text written so far. Braces
    and quotes inside string literals are ignored.
    """
    i = start
    n = len(text)
    depth = 0
    in_string = False
    escaped = False
    while i < n:
        ch = text[i]
        if escaped:
            escaped = False
        elif in_string:
            if ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            if depth == 0:
                break
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
        elif ch == "," and depth == 0:
            break
        i += 1
    return text[start:i]


def _top_level_object_span(text: str, call_index: int) -> tuple[int, int | None] | None:
    """Span of the ``call_index``-th top-level ``{...}`` object in ``text``.

    Returns ``(start, end)`` with ``end`` exclusive, or ``None`` when fewer
    objects were written. ``end`` is ``None`` when the object is not closed
    yet. String-aware: braces inside string literals are ignored.
    """
    depth = 0
    seen = 0
    start = -1
    in_string = False
    escaped = False
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if escaped:
            escaped = False
        elif in_string:
            if ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                if seen == call_index:
                    start = i
                seen += 1
            depth += 1
        elif ch == "}":
            depth -= 1
            if start >= 0 and depth == 0:
                return start, i + 1
        i += 1
    if start >= 0:
        return start, None
    return None


def _tool_parameters_text(text: str, span: tuple[int, int | None]) -> str:
    """Raw ``parameters`` text of the tool call at ``span``.

    Returns the raw characters of the ``"parameters"`` value written so far,
    or ``""`` when the key has not been written yet. String-aware: a
    ``"parameters"`` key nested inside the parameters value itself is ignored.
    """
    start, end = span
    body = text[start : end if end is not None else len(text)]
    depth = 0
    in_string = False
    escaped = False
    i, n = 0, len(body)
    while i < n:
        ch = body[i]
        if escaped:
            escaped = False
        elif in_string:
            if ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            if depth == 1:
                key_end = _scan_string_end(body, i)
                if key_end is not None and body[i + 1 : key_end - 1] == ("parameters"):
                    j = key_end
                    while j < n and body[j] in " \t\n\r":
                        j += 1
                    if j < n and body[j] == ":":
                        j += 1
                        while j < n and body[j] in " \t\n\r":
                            j += 1
                        return _scan_value_text(body, j)
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    return ""


def extract_named_tool_call_streaming(
    *,
    delta_text: str,
    function_name: str,
    function_name_returned: bool,
    tool_call_idx: int | None,
    tool_call_id_type: str,
    tokenizer: "TokenizerLike",
    tool_call_array_index: int = 0,
) -> tuple[DeltaMessage | None, bool]:
    """Build a streaming tool-call delta for forced named tool choice."""
    if function_name_returned:
        delta_tool_call = DeltaToolCall(
            function=DeltaFunctionCall(arguments=delta_text),
            index=tool_call_array_index,
        )
    else:
        if is_mistral_tokenizer(tokenizer):
            # Import mistral_common only if we need it.
            from vllm.parser.mistral import MistralToolCall

            tool_call_id = MistralToolCall.generate_random_id()
        else:
            tool_call_id = make_tool_call_id(
                id_type=tool_call_id_type,
                func_name=function_name,
                idx=tool_call_idx,
            )
        delta_tool_call = DeltaToolCall(
            id=tool_call_id,
            type="function",
            function=DeltaFunctionCall(
                name=function_name,
                arguments=delta_text,
            ),
            index=tool_call_array_index,
        )
        function_name_returned = True
    return (
        DeltaMessage(tool_calls=[delta_tool_call]),
        function_name_returned,
    )


def extract_required_tool_call_streaming(
    *,
    current_text: str | None,
    sent_args: dict[int, int],
    tool_call_idx: int | None,
    tool_call_id_type: str,
) -> tuple[DeltaMessage | None, dict[int, int]]:
    """Build a streaming tool-call delta for ``tool_choice="required"``.

    The model writes a JSON array of tool calls. The helper emits one
    ``DeltaToolCall`` per array index that has something new — the
    ``(id, name)`` chunk the first time the name is known to be complete,
    then ``arguments``-only chunks — and returns them in a single
    ``DeltaMessage``, or ``None`` when there is nothing new to send.

    Args:
        current_text: The full tool-call text generated so far.
        sent_args: Maps each tool-call index to the number of
            ``"parameters"`` characters already streamed for it; key presence
            also records that the ``(id, name)`` chunk for that index went
            out. Updated in place. It is cleared when the text transiently
            fails to parse, so the next successful parse re-streams from
            scratch (the serving layer pins tool-call ids per index to
            de-duplicate the replay).
        tool_call_idx: Position of the tool call in the request history,
            used for id generation.
        tool_call_id_type: Id generation strategy for new tool calls.

    Returns:
        A ``(delta_message, sent_args)`` tuple carrying the updated progress
        map.

    """
    if current_text is None or current_text == "":
        # if the current text is empty, we cannot parse it
        return None, sent_args
    try:
        flags = Allow.ALL
        obj, _ = partial_json_loads(current_text, flags)
    except (
        partial_json_parser.core.exceptions.MalformedJSON,
        json.JSONDecodeError,
    ):
        obj = None

    # check if the current text is a valid array
    # containing partial tool calling objects
    # if not, reset and wait for the next delta
    if obj is None or not isinstance(obj, list) or not len(obj) > 0:
        sent_args.clear()
        return None, sent_args

    last_index = len(obj) - 1
    delta_tool_calls: list[DeltaToolCall] = []
    for index, tool_call in enumerate(obj):
        if not isinstance(tool_call, dict):
            continue
        span = _top_level_object_span(current_text, index)
        if span is None:
            continue
        parameters_text = _tool_parameters_text(current_text, span)
        if index in sent_args:
            arguments = parameters_text[sent_args[index] :]
            if arguments == "":
                continue
            sent_args[index] += len(arguments)
            delta_tool_calls.append(
                DeltaToolCall(
                    function=DeltaFunctionCall(
                        # OpenAI API returns None
                        # instead of name every time
                        name=None,
                        arguments=arguments,
                    ),
                    index=index,
                )
            )
            continue
        name = tool_call.get("name")
        if not isinstance(name, str):
            continue
        if index == last_index and (
            # The trailing object's name is only known to be complete once
            # its "parameters" started (it always follows the name), or the
            # object itself is already closed.
            "parameters" not in tool_call and span[1] is None
        ):
            continue
        tool_call_id = make_tool_call_id(
            id_type=tool_call_id_type,
            func_name=name,
            idx=tool_call_idx,
        )
        sent_args[index] = len(parameters_text)
        delta_tool_calls.append(
            DeltaToolCall(
                id=tool_call_id,
                type="function",
                function=DeltaFunctionCall(name=name, arguments=parameters_text),
                index=index,
            )
        )

    if not delta_tool_calls:
        return None, sent_args
    return DeltaMessage(tool_calls=delta_tool_calls), sent_args
