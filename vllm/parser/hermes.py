# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hermes 2 Pro tool-call parser, backed by the declarative parser engine.

Tool calls repeat as back-to-back pairs::

    <tool_call>{"name": "get_weather", "arguments": {"city": "SF"}}</tool_call>

The body is a JSON wrapper, so the name comes from the engine's
name-from-args path and :func:`_hermes_arg_converter` carves out the
``arguments`` value -- the same shape as Inkling, with the key spelled
``arguments``.
"""

from __future__ import annotations

import functools
import json
from typing import TYPE_CHECKING

import regex as re

from vllm.entrypoints.openai.engine.protocol import ExtractedToolCallInformation
from vllm.logger import init_logger
from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.utils.mistral import is_mistral_tokenizer

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

logger = init_logger(__name__)

TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"

_WS = " \t\r\n"


def _scan_json_value(raw: str, start: int) -> int | None:
    """Return the end index (exclusive) of the JSON object starting at
    ``raw[start]``, or ``None`` when the object is still unterminated."""
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


def _args_value_span(raw: str) -> str | None:
    """Return the verbatim text of the top-level ``"arguments"`` value,
    possibly an unterminated prefix, or ``None`` if it has not started.

    Raises:
        ValueError: the value is not a JSON object.
    """
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
        elif ch == ":" and depth == 1 and last_string == "arguments":
            value_start = i + 1
            while value_start < len(raw) and raw[value_start] in _WS:
                value_start += 1
            if value_start >= len(raw):
                return None
            if raw[value_start] != "{":
                raise ValueError("Hermes tool call arguments must be a JSON object")
            value_end = _scan_json_value(raw, value_start)
            if value_end is None:
                return raw[value_start:]
            return raw[value_start:value_end]
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
    return None


def _hermes_arg_converter(raw_args: str, partial: bool) -> str:
    """Carve the ``arguments`` object out of the tool-call JSON wrapper.

    The engine treats the whole ``TOOL_ARGS`` span as the arguments, and
    skips ``_fix_arg_types`` when there is no converter, so omitting one
    makes streaming and non-streaming disagree on argument types.

    The span is returned verbatim rather than re-serialised: the engine
    requires each converter output to extend the previous one, and
    ``json.dumps`` reflows whitespace between ticks, silently dropping
    argument deltas.
    """
    span = _args_value_span(raw_args)
    if span is None:
        return "" if partial else "{}"
    return span


@functools.cache
def hermes_config(
    tool_call_start: str = TOOL_CALL_START,
    tool_call_end: str = TOOL_CALL_END,
) -> ParserEngineConfig:
    return ParserEngineConfig(
        name="hermes",
        initial_state=ParserState.CONTENT,
        terminals={
            "TOOL_START": tool_call_start,
            "TOOL_END": tool_call_end,
        },
        # Text-matched only: the engine stops trusting text matches for
        # token-id terminals once real token IDs arrive, which strands
        # tokenizers whose delimiter is not a single token. Inkling opts
        # out the same way.
        token_id_terminals={},
        transitions={
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_ARGS,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
        },
        arg_converter=_hermes_arg_converter,
        stream_arg_deltas=True,
        tool_args_json=True,
        strip_trailing_reasoning_whitespace=True,
        drop_whitespace_only_content_before_tools=True,
        strip_content_whitespace_with_tools=False,
        validate_tool_names=False,
    )


class HermesParser(ParserEngine):
    """Hermes 2 Pro parser backed by the declarative parser engine.

    Delimiters are class attributes so subclasses can reuse this grammar
    with different tags; see :class:`~vllm.parser.longcat.LongcatParser`.
    """

    TOOL_CALL_START: str = TOOL_CALL_START
    TOOL_CALL_END: str = TOOL_CALL_END

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        if is_mistral_tokenizer(tokenizer):
            logger.error("Detected Mistral tokenizer when using a Hermes model")
            tokenizer = tokenizer.tokenizer
        kwargs.setdefault(
            "parser_engine_config",
            hermes_config(self.TOOL_CALL_START, self.TOOL_CALL_END),
        )
        super().__init__(tokenizer, tools, **kwargs)

    @functools.cached_property
    def _tool_call_regex(self) -> re.Pattern:
        start, end = re.escape(self.TOOL_CALL_START), re.escape(self.TOOL_CALL_END)
        return re.compile(f"{start}(.*?){end}|{start}(.*)", re.DOTALL)

    def extract_tool_calls_from_content(
        self,
        content: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Reject the response when a tool-call body is not valid JSON.

        Matches the legacy parser: non-streaming is all-or-nothing, while
        streaming keeps the engine's best-effort behaviour.
        """
        for match in self._tool_call_regex.finditer(content):
            body = match.group(1) if match.group(1) is not None else match.group(2)
            try:
                json.loads(body)
            except (json.JSONDecodeError, ValueError):
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=content
                )
        return super().extract_tool_calls_from_content(content, request)
