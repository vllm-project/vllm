# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 parser: ``<think>``/``</think>``
reasoning plus DSML tool calls in a single state machine.

DeepSeek V4 output format::

    <think>
    ...reasoning...
    </think>
    <｜DSML｜tool_calls>
    <｜DSML｜invoke name="func_name">
    <｜DSML｜parameter name="location" string="true">杭州</｜DSML｜parameter>
    <｜DSML｜parameter name="count" string="false">5</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>
"""

from __future__ import annotations

import contextlib
import functools
import json
from typing import TYPE_CHECKING

import regex as re

from vllm.parser.engine.events import EventType, SemanticEvent
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ArgValueExtender,
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine
from vllm.parser.engine.token_id_scanner import DROP_TERMINAL
from vllm.tool_parsers.utils import find_tool_properties

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

_DSML = "｜DSML｜"

DSML_THINK_START = "<think>"
DSML_THINK_END = "</think>"
DSML_TOOL_START = f"<{_DSML}tool_calls>"
DSML_TOOL_END = f"</{_DSML}tool_calls>"
DSML_INVOKE_PREFIX = f'<{_DSML}invoke name="'
DSML_INVOKE_NAME_END = '">'
DSML_INVOKE_END = f"</{_DSML}invoke>"
DSML_PARAM_START = f"<{_DSML}parameter"
DSML_PARAM_CLOSE = f"</{_DSML}parameter>"

# Spellings variants of ``DSML_TOOL_START`` observed in production.
DSML_TOOL_START_VARIANTS: tuple[str, ...] = (
    f"<{_DSML}toolcalls>",
    f"<{_DSML}tool>",
)


_WHITESPACE_RE = re.compile(r"\s*")


@functools.cache
def _param_head_patterns(param_start: str) -> tuple[re.Pattern, re.Pattern, re.Pattern]:
    """Parameter head, next-head lookahead and unfinished-head regexes.

    The ``string`` attribute is optional: the model sometimes omits it, and
    dropping such a parameter hands the client a tool call with no arguments.
    """
    start = re.escape(param_start)
    return (
        re.compile(rf'{start}\s+name="([^"]+)"(?:\s+string="([^"]*)")?\s*>'),
        re.compile(rf"{start}\s+name="),
        re.compile(rf"{start}(?:\s[^>]*)?\Z"),
    )


# (name, string attribute, value); an open param adds ``held_back``.
_Param = tuple[str, str | None, str]
_OpenParam = tuple[str, str | None, str, bool]


def _split_params(
    raw_args: str, partial: bool, param_start: str, param_close: str
) -> tuple[list[_Param], _OpenParam | None]:
    """Split raw DSML arguments into closed params and the trailing open one.

    DSML has no escaping, so a value can contain ``param_close`` itself. A
    closer ends the value only when what follows is blank or the next
    parameter head (or, while streaming, a head still being written);
    otherwise it is part of the value. A value whose closer is missing ends
    at the next head. The open param is ``(name, is_str, value, held_back)``:
    an unfinished next parameter tag is cut from its value and ``held_back``
    reports whether that happened.
    """
    head_re, next_head_re, open_head_re = _param_head_patterns(param_start)
    closed: list[_Param] = []
    pos = 0
    while m := head_re.search(raw_args, pos):
        name, is_str, value_start = m.group(1), m.group(2), m.end()
        next_head = next_head_re.search(raw_args, value_start)
        search = value_start
        while True:
            close = raw_args.find(param_close, search)
            if next_head is not None and (close < 0 or next_head.start() < close):
                closed.append((name, is_str, raw_args[value_start : next_head.start()]))
                pos = next_head.start()
                break
            if close < 0:
                value = raw_args[value_start:]
                cut = value.find(param_start, value.rfind(">") + 1)
                if cut >= 0:
                    return closed, (name, is_str, value[:cut], True)
                return closed, (name, is_str, value, False)
            after = close + len(param_close)
            follow = _WHITESPACE_RE.match(raw_args, after).end()
            if (
                follow == len(raw_args)
                or next_head_re.match(raw_args, follow)
                or (
                    partial
                    and (
                        open_head_re.match(raw_args, follow)
                        or param_start.startswith(raw_args[follow:])
                    )
                )
            ):
                closed.append((name, is_str, raw_args[value_start:close]))
                pos = after
                break
            search = after
    return closed, None


def _dsml_arg_converter(
    raw_args: str,
    partial: bool,
    *,
    param_start: str = DSML_PARAM_START,
    param_close: str = DSML_PARAM_CLOSE,
) -> str:
    params: dict[str, object] = {}
    closed, open_param = _split_params(raw_args, partial, param_start, param_close)

    # A missing ``string`` attribute is treated like ``string="false"``: try
    # JSON, fall back to the literal text.
    for name, is_str, value in closed:
        if is_str == "true":
            params[name] = value
        else:
            try:
                params[name] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                params[name] = value

    if partial and open_param:
        name, is_str, value, _ = open_param
        if is_str == "true":
            params[name] = value
        else:
            with contextlib.suppress(json.JSONDecodeError, ValueError):
                params[name] = json.loads(value)

    return json.dumps(params, ensure_ascii=False)


def _dsml_arg_value_extender(
    raw_args: str,
    *,
    param_start: str = DSML_PARAM_START,
    param_close: str = DSML_PARAM_CLOSE,
) -> ArgValueExtender | None:
    # ``arg_structural_chars`` skips every delta without ``>``, so a long
    # string value would wait for its closing tag. Extend the open trailing
    # ``string="true"`` value instead. The lexer can flush a held partial
    # tag as text, so keep a tail to catch a tag split across deltas.
    closed, open_param = _split_params(raw_args, True, param_start, param_close)
    if open_param is None:
        return None
    name, is_str, _, held_back = open_param
    if is_str != "true" or held_back or any(p[0] == name for p in closed):
        return None

    keep = max(len(param_start), len(param_close)) - 1
    tail = raw_args[-keep:]

    def extend(delta: str) -> str | None:
        nonlocal tail
        text = tail + delta
        if param_start in text or param_close in text:
            return None
        tail = text[-keep:]
        return json.dumps(delta, ensure_ascii=False)[1:-1]

    return extend


def _unwrap_wrapper_args(
    args_json: str,
    tools: list[Tool] | None,
    func_name: str | None,
) -> str:
    if not tools or not func_name:
        return args_json
    try:
        args = json.loads(args_json)
    except (json.JSONDecodeError, ValueError):
        return args_json
    if not isinstance(args, dict):
        return args_json
    properties = find_tool_properties(tools, func_name)
    if not properties:
        return args_json
    allowed = set(properties.keys())
    for wrapper in ("arguments", "input"):
        if set(args.keys()) != {wrapper} or wrapper in allowed:
            continue
        inner = args[wrapper]
        if isinstance(inner, str):
            try:
                inner = json.loads(inner)
            except json.JSONDecodeError:
                return args_json
        if isinstance(inner, dict) and set(inner.keys()).issubset(allowed):
            return json.dumps(inner, ensure_ascii=False)
    return args_json


class DsmlStreamingEngine(StreamingParserEngine):
    """Keeps a DSML call open while its last parameter value is unclosed.

    A value can contain ``INVOKE_END`` or ``TOOL_END`` text. Inside an open
    value such a closer is held: a following ``INVOKE_PREFIX``, ``TOOL_END``
    or the end of output makes it structural (closing the value first), and
    any other text makes it part of the value.
    """

    _HELD_TERMINALS = frozenset({"INVOKE_END", "TOOL_END"})
    _RESOLVING_TERMINALS = frozenset({"INVOKE_PREFIX", "TOOL_END"})

    def reset(self, initial_state: ParserState | None = None) -> None:
        super().reset(initial_state)
        self._dsml_args = ""
        self._dsml_held: tuple[str, str, int] | None = None
        self._dsml_held_ws = ""
        self._dsml_held_ws_tokens = 0

    def _value_open(self) -> bool:
        param_start = self.config.terminal_literal("PARAM_START")
        param_close = self.config.terminal_literal("PARAM_CLOSE")
        assert param_start is not None and param_close is not None
        _, open_param = _split_params(self._dsml_args, True, param_start, param_close)
        return open_param is not None

    def _emit_args(self, text: str, token_count: int = 0) -> list[SemanticEvent]:
        self._dsml_args += text
        return super()._emit_for_state(text, token_count)

    def _release_held(self) -> list[SemanticEvent]:
        assert self._dsml_held is not None
        terminal, value, token_count = self._dsml_held
        text = (value or self.config.terminal_literal(terminal) or "") + (
            self._dsml_held_ws
        )
        token_count += self._dsml_held_ws_tokens
        self._clear_held()
        return self._emit_args(text, token_count)

    def _clear_held(self) -> None:
        self._dsml_held = None
        self._dsml_held_ws = ""
        self._dsml_held_ws_tokens = 0

    def _end_held_call(self) -> list[SemanticEvent]:
        assert self._dsml_held is not None
        terminal, value, token_count = self._dsml_held
        # The whitespace after a structural closer is dropped, as before.
        self._clear_held()
        events: list[SemanticEvent] = []
        if self._value_open():
            param_close = self.config.terminal_literal("PARAM_CLOSE")
            assert param_close is not None
            events.extend(self._emit_args(param_close))
        events.extend(super()._on_terminal(terminal, value, token_count))
        return events

    def _apply_transition(
        self, transition: Transition, value: str, token_count: int = 0
    ) -> list[SemanticEvent]:
        if (
            transition.next_state == ParserState.TOOL_ARGS
            and self.state != ParserState.TOOL_ARGS
        ):
            self._dsml_args = ""
        return super()._apply_transition(transition, value, token_count)

    def _emit_for_state(self, text: str, token_count: int = 0) -> list[SemanticEvent]:
        if self.state != ParserState.TOOL_ARGS:
            return super()._emit_for_state(text, token_count)
        events: list[SemanticEvent] = []
        if self._dsml_held is not None:
            if not text.strip():
                self._dsml_held_ws += text
                self._dsml_held_ws_tokens += token_count
                return events
            events.extend(self._release_held())
        events.extend(self._emit_args(text, token_count))
        return events

    def _on_terminal(
        self, terminal: str, value: str, token_count: int = 0
    ) -> list[SemanticEvent]:
        if (
            self.state != ParserState.TOOL_ARGS
            or self.skip_tool_parsing
            or terminal == DROP_TERMINAL
        ):
            return super()._on_terminal(terminal, value, token_count)
        events: list[SemanticEvent] = []
        if self._dsml_held is not None:
            if terminal in self._RESOLVING_TERMINALS:
                events.extend(self._end_held_call())
                events.extend(self._on_terminal(terminal, value, token_count))
                return events
            events.extend(self._release_held())
        if terminal in self._HELD_TERMINALS and self._value_open():
            self._dsml_held = (terminal, value, token_count)
            return events
        events.extend(super()._on_terminal(terminal, value, token_count))
        return events

    def finish(self) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        if self._dsml_held is not None:
            events.extend(self._process_scanner_items(self._scanner.flush_pending()))
            events.extend(self._process_lex_tokens(self._lexer.flush()))
            if self._dsml_held is not None:
                events.extend(self._end_held_call())
            self._record_reasoning_tokens(events)
        events.extend(super().finish())
        return events


@functools.cache
def deepseek_v4_config(thinking: bool = False) -> ParserEngineConfig:
    return ParserEngineConfig(
        name="deepseek_v4",
        initial_state=ParserState.REASONING if thinking else ParserState.CONTENT,
        wait_for_reasoning=thinking,
        terminals={
            "THINK_START": DSML_THINK_START,
            "THINK_END": DSML_THINK_END,
            "TOOL_START": (DSML_TOOL_START, *DSML_TOOL_START_VARIANTS),
            "TOOL_END": DSML_TOOL_END,
            "INVOKE_PREFIX": DSML_INVOKE_PREFIX,
            "INVOKE_NAME_END": DSML_INVOKE_NAME_END,
            "INVOKE_END": DSML_INVOKE_END,
            "PARAM_START": DSML_PARAM_START,
            "PARAM_CLOSE": DSML_PARAM_CLOSE,
        },
        token_id_terminals={
            "THINK_START": DSML_THINK_START,
            "THINK_END": DSML_THINK_END,
            "TOOL_START": DSML_TOOL_START,
            "TOOL_END": DSML_TOOL_END,
        },
        transitions={
            (ParserState.CONTENT, "THINK_START"): Transition(
                ParserState.REASONING,
                (EventType.REASONING_START,),
            ),
            # Absorb a bare </think> with no prior <think>
            (ParserState.CONTENT, "THINK_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
            # Absorb a duplicate <think> while already reasoning
            (ParserState.REASONING, "THINK_START"): Transition(
                ParserState.REASONING,
                (),
            ),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                (EventType.REASONING_END,),
            ),
            # Tool call beginning while still inside <think>
            (ParserState.REASONING, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.REASONING_END,),
            ),
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (),
            ),
            (ParserState.TOOL_PREAMBLE, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.CONTENT, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_NAME, "INVOKE_NAME_END"): Transition(
                ParserState.TOOL_ARGS,
                (),
            ),
            (ParserState.TOOL_ARGS, "INVOKE_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
            ),
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
            ),
            # Parallel tool calls
            (ParserState.TOOL_BETWEEN, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            # A tool call ends the turn: stay in TOOL_BETWEEN so any text
            # after the block is dropped.
            (ParserState.TOOL_BETWEEN, "TOOL_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (),
            ),
            (ParserState.TOOL_BETWEEN, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (),
            ),
        },
        content_events={
            ParserState.CONTENT: EventType.TEXT_CHUNK,
            ParserState.REASONING: EventType.REASONING_CHUNK,
            ParserState.TOOL_NAME: EventType.TOOL_NAME,
            ParserState.TOOL_ARGS: EventType.ARG_VALUE_CHUNK,
        },
        arg_converter=_dsml_arg_converter,
        arg_structural_chars=frozenset(">"),
        arg_value_extender=_dsml_arg_value_extender,
        strip_content_whitespace_with_tools=False,
        tool_args_json=False,
    )


class DeepSeekV4Parser(ParserEngine):
    parser_config = staticmethod(deepseek_v4_config)
    streaming_engine_class = DsmlStreamingEngine

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        chat_kwargs = kwargs.pop("chat_template_kwargs", None) or {}
        thinking = bool(
            chat_kwargs.get("thinking") or chat_kwargs.get("enable_thinking")
        )
        if "thinking" not in chat_kwargs and "enable_thinking" not in chat_kwargs:
            thinking = True
        thinking = thinking and chat_kwargs.get("reasoning_effort") != "none"
        super().__init__(
            tokenizer,
            tools,
            parser_engine_config=self.parser_config(thinking=thinking),
            **kwargs,
        )
        self._arg_converter = self._convert_args

    def _convert_args(self, raw_args: str, partial: bool) -> str:
        converter = self.parser_engine_config.arg_converter
        assert converter is not None
        result = converter(raw_args, partial)
        if not self._tools:
            return result
        func_name = next((s.name for s in self._tool_slots if s.args == raw_args), None)
        return _unwrap_wrapper_args(result, self._tools, func_name)
