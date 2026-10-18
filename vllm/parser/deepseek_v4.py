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

from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ArgValueExtender,
    ParserEngineConfig,
    ParserState,
    Transition,
)
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


def _param_patterns(
    param_start: str, param_close: str
) -> tuple[re.Pattern, re.Pattern, re.Pattern]:
    """Complete, trailing-partial, and value-end parameter regexes for one DSML dialect.

    The ``string`` attribute is optional: the model sometimes omits it, and
    dropping such a parameter hands the client a tool call with no arguments.
    """
    start, close = re.escape(param_start), re.escape(param_close)
    head = rf'{start}\s+name="([^"]+)"(?:\s+string="([^"]*)")?\s*>'
    value_end = rf"{close}|{start}\s+name="
    return (
        re.compile(rf"{head}(.*?)(?:{close}|(?={start}\s+name=))", re.DOTALL),
        re.compile(rf"{head}(.*)$", re.DOTALL),
        re.compile(value_end),
    )


_PARAM_RE, _PARTIAL_PARAM_RE, _PARAM_VALUE_END_RE = _param_patterns(
    DSML_PARAM_START, DSML_PARAM_CLOSE
)


def _value_end_window(text: str, value_end_re: re.Pattern) -> str:
    """Return the suffix of *text* that may still grow into a value end."""
    start = text.rfind("<")
    if start == -1:
        return ""
    tail = text[start:]
    return tail if value_end_re.fullmatch(tail, partial=True) else ""


def _dsml_parse_args(
    raw_args: str,
    partial: bool,
    *,
    param_re: re.Pattern = _PARAM_RE,
    partial_param_re: re.Pattern = _PARTIAL_PARAM_RE,
    value_end_re: re.Pattern = _PARAM_VALUE_END_RE,
) -> tuple[dict[str, object], re.Match | None]:
    params: dict[str, object] = {}

    last_end = 0
    # A missing ``string`` attribute is treated like ``string="false"``: try
    # JSON, fall back to the literal text.
    for m in param_re.finditer(raw_args):
        name, is_str, value = m.group(1), m.group(2), m.group(3)
        if is_str == "true":
            params[name] = value
        else:
            try:
                params[name] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                params[name] = value
        last_end = m.end()

    open_match = None
    pm = partial_param_re.search(raw_args, last_end)
    # A repeated name must not replace the value already closed for it.
    if pm and pm.group(1) in params:
        pm = None
    if pm:
        name, is_str, value = pm.group(1), pm.group(2), pm.group(3)
        # The trailing parameter is unclosed, so a suffix that can only be the
        # start of its close marker (or of the next parameter) is not part of
        # the value, including at the end of a truncated output.
        window = _value_end_window(value, value_end_re)
        if window:
            value = value[: -len(window)]
        if is_str == "true":
            params[name] = value
            open_match = pm
        elif partial:
            with contextlib.suppress(json.JSONDecodeError, ValueError):
                params[name] = json.loads(value)
        else:
            try:
                params[name] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                params[name] = value

    return params, open_match


def _dsml_arg_converter(
    raw_args: str,
    partial: bool,
    *,
    param_re: re.Pattern = _PARAM_RE,
    partial_param_re: re.Pattern = _PARTIAL_PARAM_RE,
    value_end_re: re.Pattern = _PARAM_VALUE_END_RE,
) -> str:
    params, _ = _dsml_parse_args(
        raw_args,
        partial,
        param_re=param_re,
        partial_param_re=partial_param_re,
        value_end_re=value_end_re,
    )
    return json.dumps(params, ensure_ascii=False)


def _dsml_arg_value_extender(
    raw_args: str,
    *,
    param_re: re.Pattern = _PARAM_RE,
    partial_param_re: re.Pattern = _PARTIAL_PARAM_RE,
    value_end_re: re.Pattern = _PARAM_VALUE_END_RE,
) -> ArgValueExtender | None:
    params, match = _dsml_parse_args(
        raw_args,
        partial=True,
        param_re=param_re,
        partial_param_re=partial_param_re,
        value_end_re=value_end_re,
    )
    if match is None or next(reversed(params)) != match.group(1):
        return None
    if set(params.keys()) <= {"arguments", "input"}:
        return None

    window = _value_end_window(match.group(3), value_end_re)

    def extend(delta: str) -> str | None:
        nonlocal window
        text = window + delta
        if value_end_re.search(text):
            return None
        new_window = _value_end_window(text, value_end_re)
        emit = text[: len(text) - len(new_window)]
        window = new_window
        return json.dumps(emit, ensure_ascii=False)[1:-1]

    return extend


def _unwrap_wrapper_args(
    args_json: str,
    tools: list[Tool] | None,
    func_name: str | None,
    *,
    partial: bool = False,
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
                return "{}" if partial else args_json
        if isinstance(inner, dict) and set(inner.keys()).issubset(allowed):
            return json.dumps(inner, ensure_ascii=False)
    return args_json


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
        arg_value_extender=_dsml_arg_value_extender,
        arg_structural_chars=frozenset(">"),
        strip_content_whitespace_with_tools=False,
        tool_args_json=False,
    )


class DeepSeekV4Parser(ParserEngine):
    parser_config = staticmethod(deepseek_v4_config)

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
        return _unwrap_wrapper_args(result, self._tools, func_name, partial=partial)
