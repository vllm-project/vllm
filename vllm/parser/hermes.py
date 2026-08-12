# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hermes 2 Pro tool-call parser, backed by the declarative parser engine.

Hermes tool-call format::

    <tool_call>
    {"name": "get_weather", "arguments": {"city": "SF"}}
    </tool_call>

Tool calls repeat as separate ``<tool_call>...</tool_call>`` pairs back to
back (not a single JSON array). Each call's body is a JSON object whose
``name`` is extracted by the engine's name-from-args path and whose
``arguments`` value is carved out of the wrapper by
:func:`_hermes_arg_converter` -- the same envelope shape as Inkling
(``{"name":...,"args":{...}}``), just with the key spelled ``arguments``
and delimited by ``<tool_call>``/``</tool_call>`` instead of special tokens.

Hermes has no closing-tag requirement: if ``</tool_call>`` never arrives
before EOS, the remaining text is still treated as that call's body, and
the engine's own EOS-completion path (best-effort extraction of whatever
arrived) applies with no separate state needed for it here.

Legacy Hermes is stricter than that generic best-effort convention for the
single-shot, non-streaming ``extract_tool_calls`` path specifically: it
requires the *whole* ``{"name":...,"arguments":{...}}`` body to parse as
valid JSON, and reports no tool call at all (falling back to raw content)
otherwise -- unlike incremental streaming, which always did best-effort
extraction via ``is_complete_json`` on partial text. See
:meth:`HermesParser.extract_tool_calls_from_content`.
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
    """Extract the raw text span of the top-level ``"arguments"`` value
    from a (possibly incomplete) ``{"name":...,"arguments":{...}}`` body.

    Returns the verbatim substring (prefix-stable across growing input,
    which the engine's argument-delta diffing relies on), possibly an
    unterminated object prefix; ``None`` when the value has not started.
    Raises ``ValueError`` when the value is not a JSON object.
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
    """Carve the ``arguments`` object out of the tool-call JSON body.

    Why a converter at all: the engine's ``tool_args_json`` machinery
    treats the *entire* TOOL_ARGS text as the tool arguments, but Hermes's
    payload is the ``{"name":...,"arguments":{...}}`` wrapper -- without a
    converter, ``_compute_arg_delta`` streams the wrapper verbatim into the
    OpenAI ``arguments`` field (``converter is None -> raw delta``,
    unconditionally), and non-streaming extraction never runs
    ``_fix_arg_types`` on it either.

    Why a hand-rolled scanner instead of (partial) ``json.loads`` +
    ``json.dumps``: the engine diffs successive converter outputs and
    requires each to extend the previous one (``startswith``); a violation
    silently drops argument deltas. Re-serialization changes whitespace and
    closes unterminated structures differently across ticks, so the only
    prefix-stable output is a verbatim substring of the input. The scanner
    is also string/escape-aware so an ``"arguments"`` literal inside the
    name or a string value cannot mislead it (this is the exact bug class
    that motivated this migration: an argument literally named ``"name"``
    must not be mistaken for the tool's own name field), and it still
    recovers a partial span when EOS truncates the wrapper (where
    ``json.loads`` would fail).
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
        # Left empty (not mirroring `terminals`) so the engine always trusts
        # the text lexer for these terminals, even after real token IDs have
        # started flowing -- see StreamingParserEngine._process_lex_tokens's
        # "strict" token-id mode, which otherwise refuses to accept a
        # text-matched terminal once it has seen any token IDs, unless that
        # terminal was also pre-lexed by an actual matching token ID. Hermes
        # delimiters reliably arrive as literal text (ParserEngine.adjust_request
        # forces skip_special_tokens=False), so token-ID pre-lexing buys nothing
        # here and only risks starving text matching for models/tokenizers where
        # the delimiter is a synthesized string rather than one true token
        # (Inkling opts out the same way, for the same reason).
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

    Hermes has no dedicated reasoning parser (no ``vllm/reasoning/*hermes*``
    module exists), so this engine config has no ``THINK_*`` terminals and
    ``initial_state=ParserState.CONTENT`` -- ``ParserEngine.__init__``
    detects the absence of reasoning terminals and leaves
    ``_reasoning_ended`` true from the start, matching that there is no
    separate reasoning phase to bridge.

    ``TOOL_CALL_START``/``TOOL_CALL_END`` are class attributes rather than
    hardcoded so subclasses can reuse this same wrapper grammar with a
    different pair of delimiters (e.g. Longcat's ``<longcat_tool_call>``),
    the same way the legacy implementation let subclasses override
    ``tool_call_start_token``/``tool_call_end_token`` instance attributes.
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
        """Reject the whole response if any tool-call body is not valid
        JSON, matching the legacy Hermes parser's single-shot
        ``json.loads`` requirement (see module docstring). Streaming keeps
        the engine's normal best-effort behavior; this override only
        affects this non-streaming entry point.
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
