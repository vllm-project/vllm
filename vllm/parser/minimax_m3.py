# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax M3 parser for namespace-delimited XML-style tool calls.

MiniMax M3 prefixes every structural tag with the ``]<]minimax[>[`` marker::

    ]<]minimax[>[<tool_call>
    ]<]minimax[>[<invoke name="create_order">
    ]<]minimax[>[<user_id>42]<]minimax[>[</user_id>
    ]<]minimax[>[<items>
    ]<]minimax[>[<item>]<]minimax[>[<sku>book-001]<]minimax[>[</sku>]<]minimax[>[</item>
    ]<]minimax[>[</items>
    ]<]minimax[>[</invoke>
    ]<]minimax[>[</tool_call>

Each ``<invoke>`` becomes one tool call. Parameters are elements named after
the argument and may nest: an element holding child elements becomes an
object, or an array when its schema says so (the chat template renders list
items as ``<item>``). Arguments are emitted once the whole ``<invoke>`` is
parsed. Reasoning is handled by ``MiniMaxM3ReasoningParser``.
"""

from __future__ import annotations

import functools
import json
from typing import TYPE_CHECKING, Any

from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.tool_parsers.utils import (
    coerce_to_schema_type,
    extract_types_from_schema,
    find_tool_properties,
)

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

NAMESPACE = "]<]minimax[>["
TOOL_CALL_START = f"{NAMESPACE}<tool_call>"
TOOL_CALL_END = f"{NAMESPACE}</tool_call>"
INVOKE_PREFIX_DQ = f'{NAMESPACE}<invoke name="'
INVOKE_PREFIX_SQ = f"{NAMESPACE}<invoke name='"
INVOKE_PREFIX_UNQUOTED = f"{NAMESPACE}<invoke name="
INVOKE_END = f"{NAMESPACE}</invoke>"
NAME_END_DQ = '">'
NAME_END_SQ = "'>"
NAME_END_UNQUOTED = ">"
ELEMENT_START = f"{NAMESPACE}<"
ELEMENT_END_START = f"{NAMESPACE}</"
MIXED_TEXT_FIELD = "$text"
MAX_ELEMENT_DEPTH = 128

# An element value: its text, or its ``(name, value)`` children in order.
_Element = str | list[tuple[str, "_Element"]]


def _parse_element(text: str, pos: int, depth: int) -> tuple[str, _Element, int] | None:
    """Parse the element starting at ``pos``; ``None`` if malformed."""
    if depth > MAX_ELEMENT_DEPTH:
        return None
    name_start = pos + len(ELEMENT_START)
    name_end = text.find(">", name_start)
    if name_end <= name_start:
        return None
    name = text[name_start:name_end]
    if name.startswith("/") or not name.strip():
        return None

    close_tag = f"{ELEMENT_END_START}{name}>"
    pos = name_end + 1
    text_parts: list[str] = []
    children: list[tuple[str, _Element]] = []
    while True:
        marker = text.find(NAMESPACE, pos)
        if marker < 0:
            return None
        text_parts.append(text[pos:marker])
        if text.startswith(close_tag, marker):
            pos = marker + len(close_tag)
            break
        if not text.startswith(ELEMENT_START, marker):
            return None
        child = _parse_element(text, marker, depth + 1)
        if child is None:
            return None
        child_name, child_value, pos = child
        children.append((child_name, child_value))

    body_text = "".join(text_parts)
    if not children:
        return name, body_text, pos
    if body_text.strip():
        # Keep mixed text under a reserved field that no child uses.
        field = MIXED_TEXT_FIELD
        while any(child_name == field for child_name, _ in children):
            field = "$" + field
        children.append((field, body_text))
    return name, children, pos


def _parse_invoke_params(raw_args: str) -> list[tuple[str, _Element]]:
    """Parse the parameter elements of one ``<invoke>`` body.

    Parsing stops at the first malformed or unterminated element, keeping the
    parameters before it.
    """
    params: list[tuple[str, _Element]] = []
    pos = 0
    while True:
        while pos < len(raw_args) and raw_args[pos].isspace():
            pos += 1
        if not raw_args.startswith(ELEMENT_START, pos):
            return params
        element = _parse_element(raw_args, pos, depth=1)
        if element is None:
            return params
        name, value, pos = element
        params.append((name, value))


def _container_schema(schema: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """Return the first ``array`` or ``object`` alternative of ``schema``."""
    types = schema.get("type")
    for kind in ("array", "object"):
        if types == kind or (isinstance(types, list) and kind in types):
            return kind, schema
    if types is None:
        if "items" in schema:
            return "array", schema
        if "properties" in schema or "additionalProperties" in schema:
            return "object", schema
    for choice_field in ("anyOf", "oneOf"):
        for choice in schema.get(choice_field) or ():
            if isinstance(choice, dict) and (found := _container_schema(choice)):
                return found
    return None


def _convert_element(value: _Element, schema: Any) -> Any:
    if not isinstance(schema, dict):
        schema = None
    container = _container_schema(schema) if schema is not None else None

    if isinstance(value, str):
        if schema is None:
            return value
        types = extract_types_from_schema(schema)
        # The template renders an empty list or dict as an empty element.
        if container is not None and "string" not in types and not value.strip():
            return [] if container[0] == "array" else {}
        return coerce_to_schema_type(value, types)

    if container is not None and container[0] == "array":
        items = container[1].get("items")
        return [_convert_element(child, items) for _, child in value]
    return _convert_object(value, container[1] if container is not None else None)


def _convert_object(
    elements: list[tuple[str, _Element]], schema: dict[str, Any] | None
) -> dict[str, Any]:
    """Convert named elements to an object; repeated names collect into a list."""
    properties = (schema or {}).get("properties") or {}
    additional = (schema or {}).get("additionalProperties")
    result: dict[str, Any] = {}
    repeated: set[str] = set()
    for name, element in elements:
        converted = _convert_element(element, properties.get(name, additional))
        if name not in result:
            result[name] = converted
        elif name in repeated:
            result[name].append(converted)
        else:
            result[name] = [result[name], converted]
            repeated.add(name)
    return result


def _minimax_m3_arg_converter(
    raw_args: str,
    partial: bool,
    properties: dict[str, Any] | None = None,
) -> str:
    params = _parse_invoke_params(raw_args)
    arguments = _convert_object(params, {"properties": properties or {}})
    return json.dumps(arguments, ensure_ascii=False)


@functools.cache
def minimax_m3_config() -> ParserEngineConfig:
    return ParserEngineConfig(
        name="minimax_m3",
        initial_state=ParserState.CONTENT,
        terminals={
            "TOOL_START": TOOL_CALL_START,
            "TOOL_END": TOOL_CALL_END,
            "INVOKE_PREFIX_DQ": INVOKE_PREFIX_DQ,
            "INVOKE_PREFIX_SQ": INVOKE_PREFIX_SQ,
            "INVOKE_PREFIX_UNQUOTED": INVOKE_PREFIX_UNQUOTED,
            "INVOKE_END": INVOKE_END,
            "NAME_END_DQ": NAME_END_DQ,
            "NAME_END_SQ": NAME_END_SQ,
            "NAME_END_UNQUOTED": NAME_END_UNQUOTED,
        },
        transitions={
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (),
            ),
            **{
                (state, terminal): Transition(
                    ParserState.TOOL_NAME,
                    (EventType.TOOL_CALL_START,),
                )
                for state in (ParserState.TOOL_PREAMBLE, ParserState.TOOL_BETWEEN)
                for terminal in (
                    "INVOKE_PREFIX_DQ",
                    "INVOKE_PREFIX_SQ",
                    "INVOKE_PREFIX_UNQUOTED",
                )
            },
            **{
                (ParserState.TOOL_NAME, terminal): Transition(
                    ParserState.TOOL_ARGS,
                    (),
                )
                for terminal in ("NAME_END_DQ", "NAME_END_SQ", "NAME_END_UNQUOTED")
            },
            # Nested parameter elements are argument text; only the invoke
            # closer ends the call.
            (ParserState.TOOL_ARGS, "INVOKE_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
            ),
            # The tool block ends the message: stay in TOOL_BETWEEN so any
            # text after it is dropped.
            (ParserState.TOOL_PREAMBLE, "TOOL_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (),
            ),
            (ParserState.TOOL_BETWEEN, "TOOL_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (),
            ),
        },
        arg_converter=_minimax_m3_arg_converter,
        # Emit arguments once per `<invoke>`: converted values are not
        # prefix-stable while elements arrive, e.g. a repeated parameter name
        # turns an already streamed scalar into a list.
        stream_arg_deltas=False,
        tool_args_json=False,
        strip_content_whitespace_with_tools=False,
    )


class MinimaxM3Parser(ParserEngine):
    """MiniMax M3 tool-call parser backed by the declarative parser engine."""

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        kwargs.setdefault("parser_engine_config", minimax_m3_config())
        super().__init__(tokenizer, tools, **kwargs)
        self._arg_converter = self._convert_args

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        # The M3 markers are non-special added tokens, so they decode with
        # special tokens skipped.
        return request

    def _convert_args(self, raw_args: str, partial: bool) -> str:
        func_name = next(
            (slot.name for slot in self._tool_slots if slot.args == raw_args), ""
        )
        properties = find_tool_properties(self._tools, func_name.strip())
        return _minimax_m3_arg_converter(raw_args, partial, properties)
