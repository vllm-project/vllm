# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ricursive Ripple parser for reasoning and tool calls.

Ripple decides per turn whether to think; the generation prompt does not open
`<think>`. Tool calls are JSON objects in Hermes-style tags::

    <think>
    ...reasoning...
    </think>
    ...content...
    <tool_call>
    {"name": "get_weather", "arguments": {"city": "Paris"}}
    </tool_call>

The markers are plain text, not special tokens.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ArgValueExtender,
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.parser.utils import extract_json_member_object

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

THINK_START = "<think>"
THINK_END = "</think>"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"


def _framed(marker: str, before: bool = True, after: bool = True) -> tuple[str, ...]:
    """Spellings of `marker` with its optional framing newlines, longest first."""
    heads = ("\n", "") if before else ("",)
    tails = ("\n", "") if after else ("",)
    spellings = {head + marker + tail for head in heads for tail in tails}
    return tuple(sorted(spellings, key=len, reverse=True))


def _ripple_arg_converter(raw_args: str, partial: bool) -> str:
    args = extract_json_member_object(raw_args, "arguments")
    return args if args is not None else ("" if partial else "{}")


def _open_string_value_escape(span: str) -> bool | None:
    """Scan the open JSON object prefix `span` (starting at `{`).

    Returns whether a backslash escape is pending when `span` ends inside a
    string that is not one of the object's own keys, and `None` otherwise.
    Strings nested deeper than the object's members are always treated as
    values: nothing streams for them until they close.
    """
    depth = 0
    in_string = False
    escape = False
    expects_value = False
    string_is_value = False
    for ch in span:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
                if depth == 1:
                    expects_value = False
            continue
        if ch == '"':
            in_string = True
            string_is_value = depth > 1 or expects_value
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 1:
                expects_value = False
        elif ch == ":" and depth == 1:
            expects_value = True
    if depth >= 1 and in_string and string_is_value:
        return escape
    return None


def _ripple_arg_value_extender(raw_args: str) -> ArgValueExtender | None:
    """Extend an open string value inside the `arguments` object.

    `_ripple_arg_converter` returns a verbatim span of the raw text, so the
    raw characters of the open string up to its closing quote are what the
    converter would add. Streaming them directly keeps a long string argument
    from rescanning the whole call on every delta.
    """
    try:
        span = extract_json_member_object(raw_args, "arguments")
    except ValueError:
        return None
    if span is None:
        return None
    escape = _open_string_value_escape(span)
    if escape is None:
        return None

    def extend(delta: str) -> str | None:
        nonlocal escape
        for ch in delta:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                return None
        return delta

    return extend


@functools.cache
def ripple_config() -> ParserEngineConfig:
    return ParserEngineConfig(
        name="ripple",
        initial_state=ParserState.CONTENT,
        # Reasoning is optional, so structured output applies from the first
        # token and the model answers directly.
        wait_for_reasoning=False,
        terminals={
            # The chat template frames each marker with newlines
            # ("<think>\n...\n</think>\n", "\n<tool_call>\n...\n</tool_call>"),
            # so they belong to the markers rather than the content.
            "THINK_START": _framed(THINK_START, before=False),
            "THINK_END": _framed(THINK_END),
            "TOOL_START": _framed(TOOL_CALL_START),
            "TOOL_END": _framed(TOOL_CALL_END, after=False),
        },
        transitions={
            (ParserState.CONTENT, "THINK_START"): Transition(
                ParserState.REASONING,
                (EventType.REASONING_START,),
            ),
            (ParserState.REASONING, "THINK_START"): Transition(ParserState.REASONING),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                (EventType.REASONING_END,),
            ),
            (ParserState.CONTENT, "THINK_END"): Transition(ParserState.CONTENT),
            (ParserState.REASONING, "TOOL_START"): Transition(
                ParserState.TOOL_ARGS,
                (EventType.REASONING_END, EventType.TOOL_CALL_START),
            ),
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_ARGS,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
        },
        arg_converter=_ripple_arg_converter,
        arg_value_extender=_ripple_arg_value_extender,
        # Streaming cannot retract text it already sent, so keep the
        # non-streaming content the same.
        strip_content_whitespace_with_tools=False,
    )


class RippleParser(ParserEngine):
    CONFIG_NAME = "ripple"

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        # The chat template has no thinking switch.
        kwargs.pop("chat_template_kwargs", None)
        kwargs.setdefault("parser_engine_config", ripple_config())
        super().__init__(tokenizer, tools, **kwargs)

    @property
    def reasoning_end_str(self) -> str:
        # The thinking budget matches this as token ids. The template's leading
        # newline merges into the previous token, so leave it out.
        return THINK_END + "\n"
