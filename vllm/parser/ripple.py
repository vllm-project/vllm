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

from vllm.parser.engine.events import EventType, SemanticEvent
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine
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
        # Streaming cannot retract text it already sent, so keep the
        # non-streaming content the same.
        strip_content_whitespace_with_tools=False,
    )


class RippleStreamingParserEngine(StreamingParserEngine):
    def reset(self, initial_state: ParserState | None = None) -> None:
        super().reset(initial_state)
        self._content_started = False

    def _emit_for_state(self, text: str, token_count: int = 0) -> list[SemanticEvent]:
        # A tool call forwarded as text by the reasoning pass is not the answer.
        if (
            not self._content_started
            and self.state is ParserState.CONTENT
            and not self._in_skipped_tool_span
            and text.strip()
        ):
            self._content_started = True
        return super()._emit_for_state(text, token_count)

    def _on_terminal(
        self, terminal: str, value: str, token_count: int = 0
    ) -> list[SemanticEvent]:
        if (
            self._content_started
            and self.state is ParserState.CONTENT
            and terminal in ("THINK_START", "THINK_END")
        ):
            return self._emit_for_state(value, token_count)
        content_started = self._content_started
        events = super()._on_terminal(terminal, value, token_count)
        if self.skip_tool_parsing and terminal in self._tool_terminals:
            # Forwarded tool-call markers are not answer content either.
            self._content_started = content_started
        return events


class RippleParser(ParserEngine):
    CONFIG_NAME = "ripple"
    _streaming_engine_cls = RippleStreamingParserEngine

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
