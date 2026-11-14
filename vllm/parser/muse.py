# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark parser: recipient-routed messages parsed by one state machine.

Muse Spark writes an assistant turn as a sequence of messages::

    to=self<|message|>...reasoning...<|eom|><|start|>assistant
    to=get_weather<|message|><atem:function_calls>
    <atem:invoke name="get_weather">
    <atem:parameter name="city">Paris</atem:parameter>
    </atem:invoke>
    </atem:function_calls><|eom|><|start|>assistant
    to=user<|message|>...final answer...<|eot|>

The recipient before `<|message|>` routes each body: `self` or none is
reasoning, `user` and `commentary` are content, `developer` is dropped, and
any other recipient is a tool. The model may write `to=NAME<|message|>` right
after a body without `<|eom|>`, so a trailing `to=NAME` in a body is held back.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import regex as re

from vllm.parser.engine.events import EventType, SemanticEvent
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine
from vllm.tool_parsers.schema_types import ParameterTypesCache, convert_value
from vllm.tool_parsers.utils import find_tool_name
from vllm.v1.token_buffer import rfind_tokens

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

START = "<|start|>"
MESSAGE = "<|message|>"
EOM = "<|eom|>"
EOT = "<|eot|>"

_RECIPIENT_RE = re.compile(r"to=([A-Za-z0-9_.\-]+)\s*$")
_HEADER_TAIL_RE = re.compile(r"(?:t|to|to=[A-Za-z0-9_.\-]{0,256})$")
_INVOKE_RE = re.compile(
    r'<atem:invoke\b[^<>]*?\bname="([^"]+)"[^<>]*>(.*?)</atem:invoke>', re.DOTALL
)
_PARAMETER_RE = re.compile(
    r'<atem:parameter\b[^<>]*?\bname="([^"]+)"[^<>]*?>(.*?)</atem:parameter>',
    re.DOTALL,
)
_INVOKE_END = "</atem:invoke>"
_BODY_STATES = frozenset({ParserState.CONTENT, ParserState.REASONING})
_MAX_HEADER_TOKENS = 64


def _route_header(header: str) -> tuple[str, str | None]:
    match = _RECIPIENT_RE.search(header)
    recipient = match.group(1) if match else None
    if recipient in (None, "self"):
        return "REASONING_HEADER", recipient
    if recipient in ("user", "commentary"):
        return "CONTENT_HEADER", recipient
    if recipient == "developer":
        return "DROPPED_HEADER", recipient
    return "TOOL_HEADER", recipient


def _decode_value(raw: str) -> object:
    try:
        return json.loads(raw)
    except ValueError:
        return raw


@functools.cache
def muse_config() -> ParserEngineConfig:
    routes = {
        "REASONING_HEADER": Transition(ParserState.REASONING),
        "CONTENT_HEADER": Transition(ParserState.CONTENT),
        # The dropped body is buffered as header text and discarded with it.
        "DROPPED_HEADER": Transition(ParserState.MESSAGE_HEADER),
        "TOOL_HEADER": Transition(
            ParserState.TOOL_PREAMBLE, (EventType.REASONING_END,)
        ),
        "START": Transition(ParserState.MESSAGE_HEADER),
        # Content keeps the bodies after a tool message when the tool pass
        # re-parses the reasoning pass's content.
        "EOM": Transition(ParserState.CONTENT),
        "EOT": Transition(ParserState.CONTENT),
    }
    markers = {"START": START, "MSG": MESSAGE, "EOM": EOM, "EOT": EOT}
    return ParserEngineConfig(
        name="muse",
        terminals=markers,
        token_id_terminals=markers,
        transitions={
            (state, terminal): transition
            for state in (
                ParserState.MESSAGE_HEADER,
                ParserState.REASONING,
                ParserState.CONTENT,
                ParserState.TOOL_PREAMBLE,
            )
            for terminal, transition in routes.items()
        },
        initial_state=ParserState.MESSAGE_HEADER,
        stream_arg_deltas=False,
        tool_args_json=False,
        strip_trailing_reasoning_whitespace=False,
    )


class MuseStreamingParserEngine(StreamingParserEngine):
    def reset(self, initial_state: ParserState | None = None) -> None:
        super().reset(initial_state)
        self._header_tail = ""
        self._header_tail_token_count = 0
        self._tool_body = ""
        self._tool_body_pos = 0
        self._routed_header = False

    def finish(self) -> list[SemanticEvent]:
        events = self._process_scanner_items(self._scanner.flush_pending())
        events.extend(self._process_lex_tokens(self._lexer.flush()))
        events.extend(self._flush_header_tail())
        events.extend(self._flush_unframed_text())
        self._message_header_buffer = ""
        self._message_header_token_count = 0
        self._tool_body = ""
        self._tool_body_pos = 0
        self._record_reasoning_tokens(events)
        return events

    def _on_terminal(
        self, terminal: str, value: str, token_count: int = 0
    ) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        if terminal == "MSG":
            self._routed_header = True
            terminal, recipient = _route_header(self._take_header(events))
            # The tool pass routes the forwarded header again.
            if recipient is not None:
                value = f"to={recipient}{value}"
        else:
            events.extend(self._flush_unframed_text())
        events.extend(self._flush_header_tail())
        if (
            self.state is ParserState.TOOL_PREAMBLE
            and (ParserState.TOOL_PREAMBLE, terminal) in self.config.transitions
        ):
            self._tool_body = ""
            self._tool_body_pos = 0
        events.extend(super()._on_terminal(terminal, value, token_count))
        return events

    def _emit_for_state(self, text: str, token_count: int = 0) -> list[SemanticEvent]:
        if self.state is ParserState.TOOL_PREAMBLE:
            return self._feed_tool_body(text)
        if self.state not in _BODY_STATES:
            return super()._emit_for_state(text, token_count)
        text = self._header_tail + text
        token_count += self._header_tail_token_count
        tail = _HEADER_TAIL_RE.search(text)
        split = tail.start() if tail is not None else len(text)
        self._header_tail = text[split:]
        if not split:
            self._header_tail_token_count = token_count
            return []
        self._header_tail_token_count = 0
        return super()._emit_for_state(text[:split], token_count)

    def _take_header(self, events: list[SemanticEvent]) -> str:
        if self.state is ParserState.MESSAGE_HEADER:
            return self._message_header_buffer
        if self.state is ParserState.TOOL_PREAMBLE:
            return self._tool_body
        header = self._header_tail
        if not _RECIPIENT_RE.fullmatch(header):
            return ""
        events.extend(super()._emit_for_state("", self._header_tail_token_count))
        self._header_tail = ""
        self._header_tail_token_count = 0
        return header

    def _flush_unframed_text(self) -> list[SemanticEvent]:
        if (
            self.state is not ParserState.MESSAGE_HEADER
            or self._routed_header
            or not self._message_header_buffer
        ):
            return []
        text = self._message_header_buffer
        token_count = self._message_header_token_count
        self._message_header_buffer = ""
        self._message_header_token_count = 0
        self.state = ParserState.CONTENT
        return self._emit_for_state(text, token_count) + self._flush_header_tail()

    def _flush_header_tail(self) -> list[SemanticEvent]:
        tail, token_count = self._header_tail, self._header_tail_token_count
        if not tail:
            return []
        self._header_tail = ""
        self._header_tail_token_count = 0
        return super()._emit_for_state(tail, token_count)

    def _feed_tool_body(self, text: str) -> list[SemanticEvent]:
        searched = len(self._tool_body)
        self._tool_body += text
        events: list[SemanticEvent] = []
        if self._tool_body.find(_INVOKE_END, max(searched - len(_INVOKE_END), 0)) < 0:
            return events
        for match in _INVOKE_RE.finditer(self._tool_body, self._tool_body_pos):
            self._tool_body_pos = match.end()
            self.tool_index += 1
            events.extend(
                SemanticEvent(event_type, value=event_value, tool_index=self.tool_index)
                for event_type, event_value in (
                    (EventType.TOOL_CALL_START, ""),
                    (EventType.TOOL_NAME, match.group(1)),
                    (EventType.ARG_VALUE_CHUNK, match.group(0)),
                    (EventType.TOOL_CALL_END, ""),
                )
            )
        return events


class MuseParser(ParserEngine):
    _streaming_engine_cls = MuseStreamingParserEngine

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        kwargs.setdefault("parser_engine_config", muse_config())
        super().__init__(tokenizer, tools, **kwargs)
        self._has_reasoning = True
        self._reasoning_ended = False
        self._arg_converter = self._convert_args
        self._parameter_types = ParameterTypesCache()
        vocab = self.vocab
        self._message_ids = (vocab.get(MESSAGE),)
        self._start_id = vocab.get(START)
        self._boundary_ids = tuple(
            vocab.get(token) for token in (START, MESSAGE, EOM, EOT)
        )

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        pos = len(input_ids)
        while (pos := rfind_tokens(input_ids, self._message_ids, stop=pos)) >= 0:
            ended = self._header_ends_reasoning(input_ids, pos)
            if ended is not None:
                return ended
        return False

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        pos = len(input_ids)
        start = max(pos - len(list(delta_ids)), 0)
        while (pos := rfind_tokens(input_ids, self._message_ids, start, pos)) >= 0:
            if self._header_ends_reasoning(input_ids, pos):
                return True
        return False

    def _header_ends_reasoning(self, input_ids: Sequence[int], pos: int) -> bool | None:
        """True for a tool message, False for another role's message, else None."""
        start = max(pos - _MAX_HEADER_TOKENS, 0)
        boundary = rfind_tokens(input_ids, self._boundary_ids, start, pos)
        begin = boundary + 1 if boundary >= 0 else start
        header = self.model_tokenizer.decode(list(input_ids[begin:pos]))
        if (
            boundary >= 0
            and input_ids[boundary] == self._start_id
            and not header.startswith("assistant")
        ):
            return False
        return True if _route_header(header)[0] == "TOOL_HEADER" else None

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return []

    def _handle_tool_name(self, event: SemanticEvent) -> None:
        self._tool_slots[event.tool_index].name += self._registered_name(event.value)

    def _registered_name(self, name: str) -> str:
        if not self._tools or find_tool_name(self._tools, name):
            return name
        head, sep, tail = name.partition(".")
        if sep and head == tail and find_tool_name(self._tools, head):
            return head
        return name

    def _convert_args(self, raw_args: str, partial: bool) -> str:
        match = _INVOKE_RE.match(raw_args)
        if match is None:
            return "{}"
        properties, extra = self._parameter_types.for_tools(self._tools).get(
            self._registered_name(match.group(1)), ({}, None)
        )
        arguments = {
            key: convert_value(raw, properties.get(key, extra), _decode_value)
            for key, raw in _PARAMETER_RE.findall(match.group(2))
        }
        return json.dumps(arguments, ensure_ascii=False)
