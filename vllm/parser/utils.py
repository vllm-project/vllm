# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Iterable, Sequence

from openai.types.responses import ResponseFunctionToolCall

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import (
    ResponseInputOutputItem,
    ResponsesRequest,
)
from vllm.parser.json_scan import JsonWalker, array_items, object_members
from vllm.renderers.chat_utils import ChatCompletionMessageParam


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
    end: list[int] = []

    def on_outside(i: int, ch: str, depth: int) -> bool:
        if ch == "}" and depth == 0:
            end.append(i + 1)
            return False
        return True

    JsonWalker(start=start, brackets="object", key_depth=None).advance(
        raw, on_outside_char=on_outside
    )
    return end[0] if end else None


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
    result: list[str | None] = []

    def on_outside(i: int, ch: str, depth: int) -> bool:
        if ch != ":" or depth != 1 or walker.last_key != key:
            return True
        value_start = i + 1
        while value_start < len(raw) and raw[value_start] in _JSON_WS:
            value_start += 1
        if value_start >= len(raw):
            result.append(None)
            return False
        if raw[value_start] != "{":
            raise ValueError(f"member {key!r} must be a JSON object")
        value_end = scan_json_object_end(raw, value_start)
        if value_end is None:
            result.append(raw[value_start:])
        else:
            result.append(raw[value_start:value_end])
        return False

    walker = JsonWalker(brackets="any", key_depth=1)
    walker.advance(raw, on_outside_char=on_outside)
    return result[0] if result else None


def _collect_type_edits(
    raw: str,
    start: int,
    end: int,
    schema: dict,
    edits: list[tuple[int, int, str]],
) -> None:
    """Record ``(start, end, replacement)`` for every scalar span in
    ``raw[start:end]`` whose literal type disagrees with *schema*.

    Mirrors ``ParserEngine._coerce_dict``/``_coerce_value`` but works on
    text spans (objects recurse through ``properties``, arrays ``items``),
    and is stricter: a value already valid for one of several declared
    types (e.g. a string under ``{"type": ["string", "null"]}``) is left
    exactly as the model wrote it, where ``_coerce_value``'s best-effort
    priority order (null > integer > ... > string) would still prefer
    reinterpreting it. Splicing a verbatim model span demands that
    stricter guarantee; re-serialising callers don't need it.
    """
    # Deferred import: vllm.parser.engine.parser_engine imports
    # vllm.parser.abstract_parser, which imports this module.
    from vllm.parser.engine.parser_engine import ParserEngine
    from vllm.tool_parsers.utils import is_valid_for_schema_type

    if raw[start] == "{":
        properties = schema.get("properties")
        if isinstance(properties, dict):
            for key, value_start, value_end in object_members(raw, start, end):
                sub_schema = properties.get(key)
                if isinstance(sub_schema, dict):
                    _collect_type_edits(raw, value_start, value_end, sub_schema, edits)
        return
    if raw[start] == "[":
        items = schema.get("items")
        if isinstance(items, dict):
            for value_start, value_end in array_items(raw, start, end):
                _collect_type_edits(raw, value_start, value_end, items, edits)
        return
    try:
        value = json.loads(raw[start:end])
    except (json.JSONDecodeError, ValueError):
        return
    if is_valid_for_schema_type(value, schema):
        return
    try:
        coerced, changed = ParserEngine._coerce_value(value, schema)
    except RecursionError:
        # json.loads above only covers this span; _coerce_value re-enters
        # json.loads on the decoded string via coerce_to_schema_type, whose
        # deep nesting can raise RecursionError too -- left uncoerced, as legacy did.
        return
    if changed:
        edits.append((start, end, json.dumps(coerced, ensure_ascii=False)))


def splice_types(raw: str, start: int, end: int, schema: dict) -> str:
    """``raw[start:end]`` with schema-coerced scalars substituted in place,
    every other byte kept verbatim.

    Re-serialising the decoded object (what ``ParserEngine._fix_arg_types``
    does by default) rewrites the model's separators, so a parser whose
    argument stream is append-only (verbatim, prefix-stable) would have its
    correction dropped. Splicing only rewrites values that actually change,
    keeping the result prefix-compatible.
    """
    edits: list[tuple[int, int, str]] = []
    _collect_type_edits(raw, start, end, schema, edits)
    if not edits:
        return raw[start:end]
    # Edits arrive in document order and never overlap, so the result is
    # assembled in one pass instead of copying the whole string per edit.
    out: list[str] = []
    cursor = start
    for edit_start, edit_end, text in edits:
        out.append(raw[cursor:edit_start])
        out.append(text)
        cursor = edit_end
    out.append(raw[cursor:end])
    return "".join(out)
