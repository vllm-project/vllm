# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3 parser for tool calls and reasoning.

Qwen3 XML tool call format::

    <tool_call>
    <function=func_name>
    <parameter=key>value</parameter>
    </function>
    </tool_call>

The argument body consists of `<parameter=NAME>VALUE</parameter>` tags.
The `_qwen3_arg_converter` parses these into a JSON object.
"""

from __future__ import annotations

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

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

THINK_START = "<think>"
THINK_END = "</think>"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
CHATML_TURN_BOUNDARIES = frozenset(("<|im_start|>", "<|im_end|>"))
FUNC_PREFIX = "<function="
FUNC_END = "</function>"
PARAM_START = "<parameter="
PARAM_END = "</parameter>"

_PARAM_TAG_RE = re.compile(r"<\s*parameter\s*=\s*([^>]*)>|<\s*/\s*parameter\s*>")
_PARAM_OPENER_RE = re.compile(r"<\s*parameter\s*=")
_PARAM_CLOSER_RE = re.compile(r"<\s*/\s*parameter\s*>")
_FUNC_OPENER_RE = re.compile(r"<\s*function\s*=")
# Flat parameter matching shared with MiMo's verbatim converter: a value
# ends at the closing tag or the next opener.
_PARAM_RE = re.compile(
    r"<\s*parameter\s*=\s*([^>]*)>"
    r"(.*?)"
    r"(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s*=))",
    re.DOTALL,
)
_PARTIAL_PARAM_RE = re.compile(r"<\s*parameter\s*=\s*([^>]+)>(.*)$", re.DOTALL)
_PARAM_VALUE_END_RE = re.compile(r"<\s*/\s*parameter\s*>|<\s*parameter\s*=")
# Anything that can end or nest inside an open parameter value.
_VALUE_TAG_RE = re.compile(r"<\s*parameter\s*=|<\s*/\s*parameter\s*>|<\s*function\s*=")


def _trim_wrapping_newlines(value: str) -> str:
    """Strip one leading and one trailing newline (the Qwen3 template markup)."""
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


class _ParamNode:
    """One ``<parameter=NAME>...`` span in a raw argument text."""

    __slots__ = (
        "name",
        "open_start",
        "start",
        "end",
        "close_end",
        "closed",
        "children",
    )

    def __init__(self, name: str, open_start: int, start: int) -> None:
        self.name = name
        self.open_start = open_start  # start of the opener tag
        self.start = start  # start of the value text
        self.end = -1  # end of the value text (the closer's start)
        self.close_end = -1  # end of the closer tag itself
        self.closed = False
        self.children: list[_ParamNode] = []


def _parse_param_tree(
    raw_args: str, pos: int = 0, open_root: _ParamNode | None = None
) -> list[_ParamNode]:
    """Parse ``<parameter=...>`` spans as a tree.

    An opener inside an open value nests instead of ending the value, and a
    closer closes the innermost open span. Markup quoted inside a genuine
    value is value text, not markup (issue 58147). *open_root* resumes a
    previously seen parameter that is still open at *pos*.
    """
    roots: list[_ParamNode] = []
    stack: list[_ParamNode] = [open_root] if open_root is not None else []
    for match in _PARAM_TAG_RE.finditer(raw_args, pos):
        if match.group(1) is not None:
            node = _ParamNode(match.group(1), match.start(), match.end())
            if stack:
                stack[-1].children.append(node)
            else:
                roots.append(node)
            stack.append(node)
        elif stack:
            node = stack.pop()
            node.closed = True
            node.end = match.start()
            node.close_end = match.end()
            if not stack:
                roots.append(node)
    for node in stack:
        node.end = len(raw_args)
    if open_root is not None and not open_root.closed:
        roots.append(open_root)
    return roots


def _recover_unclosed(node: _ParamNode, raw_args: str, params: dict[str, str]) -> None:
    """Recovery for a value still open when the call ends (issue 67923).

    The value ends at its first nested ``<parameter=`` or ``<function=``
    opener, and its closed nested parameters promote to the top level, so a
    genuinely missing ``</parameter>`` parses as before. Without such a
    boundary the trailing open value is dropped, also as before.
    """
    inner = raw_args[node.start : node.end]
    cut = len(inner)
    if node.children:
        cut = min(cut, node.children[0].open_start - node.start)
    func = _FUNC_OPENER_RE.search(inner)
    if func is not None:
        cut = min(cut, func.start())
    if cut == len(inner):
        return
    params[node.name] = _trim_wrapping_newlines(inner[:cut])
    for child in node.children:
        if child.closed:
            params[child.name] = _trim_wrapping_newlines(
                raw_args[child.start : child.end]
            )
        else:
            _recover_unclosed(child, raw_args, params)


class _ClosedParams:
    """Parsed parameters of a raw argument text, reused by longer texts.

    Raw arguments only grow while a call streams, so a parse of a text that
    extends `text` resumes where the last parse ended: closed parameters are
    final, and a still-open trailing parameter resumes right after its
    opener tag. Each tag is matched once across the whole stream.
    """

    __slots__ = ("text", "end", "params", "open_node")

    def __init__(self) -> None:
        self.text = ""  # raw text the parsed state depends on
        self.end = 0  # resume position in ``text``; never inside a tag
        self.params: dict[str, str] = {}
        self.open_node: _ParamNode | None = None  # trailing open parameter


def _qwen3_parse_params(
    raw_args: str, partial: bool, closed: _ClosedParams | None = None
) -> tuple[dict[str, str], tuple[str, str] | None]:
    """Return the closed parameters, trimmed, and the open trailing one."""
    params: dict[str, str] = {}
    pos = 0
    open_root = None
    if closed is not None and raw_args.startswith(closed.text):
        params.update(closed.params)
        pos = closed.end
        open_root = closed.open_node

    roots = _parse_param_tree(raw_args, pos, open_root)
    for node in roots:
        if node.closed:
            params[node.name] = _trim_wrapping_newlines(raw_args[node.start : node.end])
        elif not partial:
            _recover_unclosed(node, raw_args, params)

    if closed is not None:
        open_node = roots[-1] if roots and not roots[-1].closed else None
        if open_node is not None:
            # Resume after the open parameter's opener tag; its span must
            # never be re-matched, and anything before it is final.
            end = open_node.start
        elif roots:
            end = roots[-1].close_end
        else:
            end = pos
        closed.text = raw_args[:end]
        closed.end = end
        closed.params = dict(params)
        closed.open_node = open_node

    open_param = None
    if partial and roots and not roots[-1].closed:
        node = roots[-1]
        open_param = (node.name, raw_args[node.start : node.end])
    return params, open_param


def _qwen3_arg_converter(
    raw_args: str, partial: bool, closed: _ClosedParams | None = None
) -> str:
    params, open_param = _qwen3_parse_params(raw_args, partial, closed)
    if open_param is not None and open_param[0]:
        params[open_param[0]] = _trim_wrapping_newlines(open_param[1])

    return json.dumps(params, ensure_ascii=False)


def _param_value_end_window(text: str) -> str:
    """Return the suffix of *text* that may still grow into a value boundary.

    Complete tags are already handled by the caller, so only a partial tail
    is kept.
    """
    start = text.rfind("<")
    if start == -1:
        return ""
    tail = text[start:]
    if _VALUE_TAG_RE.fullmatch(tail):
        return ""
    return tail if _VALUE_TAG_RE.fullmatch(tail, partial=True) else ""


def _scan_value_tags(text: str, depth: int) -> tuple[int, bool]:
    """Update the nesting *depth* over complete in-value tags.

    Returns the new depth and whether the value's own closer was seen.
    """
    for match in _VALUE_TAG_RE.finditer(text):
        tag = match.group(0)
        if _PARAM_CLOSER_RE.fullmatch(tag):
            if depth > 0:
                depth -= 1
            else:
                return depth, True
        elif not _FUNC_OPENER_RE.fullmatch(tag):
            depth += 1
    return depth, False


def _param_value_extender(
    raw_args: str, trim_newlines: bool, closed: _ClosedParams | None = None
) -> ArgValueExtender | None:
    """Extend the open trailing `<parameter=...>` value of *raw_args*.

    The extender streams exactly what the converter would for the same text.
    With *trim_newlines* it follows `_trim_wrapping_newlines`: it drops a
    leading newline of the value and holds back a trailing one.

    From the first complete `<parameter=` or `<function=` inside the open
    value it holds back instead of streaming: the markup may be value text
    of a value a real `</parameter>` still closes (issue 58147), or a
    recovery boundary for a genuinely missing one (issue 67923), and
    streaming cannot tell until the value's own closer or the end of the
    call resolves it.
    """
    closed_params, open_param = _qwen3_parse_params(raw_args, True, closed)
    if open_param is None:
        return None
    name, value = open_param
    # A repeated key keeps its first position, so its value is not last.
    if name in closed_params:
        return None

    window = _param_value_end_window(value)
    started = bool(value)
    held = "\n" if trim_newlines and value.endswith("\n") and value != "\n" else ""
    # Holding starts at the first complete in-value opener and lasts until
    # the value's own closer; once seen, the markup stays unresolved.
    depth, closed_value = _scan_value_tags(value, 0)
    holding = bool(_VALUE_TAG_RE.search(value)) and not closed_value

    def extend(delta: str) -> str | None:
        nonlocal window, started, held, holding, depth
        text = window + delta
        if holding:
            depth, closed_value = _scan_value_tags(text, depth)
            window = _param_value_end_window(text)
            # None lets the conversion stream the resolved text.
            return None if closed_value else ""
        match = _VALUE_TAG_RE.search(text)
        if match is not None:
            tag = match.group(0)
            if _PARAM_CLOSER_RE.fullmatch(tag):
                return None
            holding = True
            depth = 0 if _FUNC_OPENER_RE.fullmatch(tag) else 1
            delta = delta[: max(0, match.start() - len(window))]
        window = _param_value_end_window(window + delta)
        if trim_newlines:
            text = held + delta
            if not started and text:
                started = True
                text = text.removeprefix("\n")
            held = "\n" if text.endswith("\n") else ""
            text = text[: len(text) - len(held)]
        else:
            text = delta
        return json.dumps(text, ensure_ascii=False)[1:-1]

    return extend


def _qwen3_arg_value_extender(
    raw_args: str, closed: _ClosedParams | None = None
) -> ArgValueExtender | None:
    return _param_value_extender(raw_args, trim_newlines=True, closed=closed)


def _legacy_param_value_end_window(text: str) -> str:
    """Return the suffix of *text* that may still grow into a value end."""
    start = text.rfind("<")
    if start == -1:
        return ""
    tail = text[start:]
    return tail if _PARAM_VALUE_END_RE.fullmatch(tail, partial=True) else ""


def _legacy_param_value_extender(
    raw_args: str, trim_newlines: bool
) -> ArgValueExtender | None:
    """Flat value streaming for parsers that keep the next-opener recovery."""
    params: dict[str, str] = {}
    outside: list[str] = []
    pos = 0
    for match in _PARAM_RE.finditer(raw_args):
        outside.append(raw_args[pos : match.start()])
        params[match.group(1)] = match.group(2)
        pos = match.end()
    match = _PARTIAL_PARAM_RE.search("".join(outside) + raw_args[pos:])
    if match is None or not match.group(1):
        return None
    name, value = match.group(1), match.group(2)
    # A repeated key keeps its first position, so its value is not last.
    if name in params:
        return None

    window = _legacy_param_value_end_window(value)
    started = bool(value)
    held = "\n" if trim_newlines and value.endswith("\n") and value != "\n" else ""

    def extend(delta: str) -> str | None:
        nonlocal window, started, held
        text = window + delta
        if _PARAM_VALUE_END_RE.search(text):
            return None
        window = _legacy_param_value_end_window(text)
        if trim_newlines:
            text = held + delta
            if not started and text:
                started = True
                text = text.removeprefix("\n")
            held = "\n" if text.endswith("\n") else ""
            text = text[: len(text) - len(held)]
        else:
            text = delta
        return json.dumps(text, ensure_ascii=False)[1:-1]

    return extend


# With nested parameter values, an open ``<parameter=...>`` value gets its
# own states so that ``<parameter=`` / ``<function=`` markup inside it is
# absorbed as value text (issue 58147) instead of ending the value or the
# call. The malformed-input recoveries of TOOL_ARGS are mirrored on the
# name state, and ``</function>`` / the converter's call-end recovery still
# close a genuinely unclosed value (issue 67923).
#
# Inside a value the engine tracks a bounded nesting depth with states, so
# that the closer of a complete call block quoted inside a value (a file
# that documents the format) is not taken for the real call's end:
#
#   TOOL_PARAM_VALUE         top-level value
#   TOOL_PARAM_VALUE_NESTED  a ``<parameter=`` span inside it
#   TOOL_PARAM_VALUE_FUNC    a ``<function=`` block quoted inside it
#   TOOL_PARAM_VALUE_FUNC_NESTED  a ``<parameter=`` span inside that block
#
# Every tag inside a value is still value text (ARG_VALUE_CHUNK); the
# states only decide which ``</function>`` / ``</parameter>`` is the real
# one. A ``</function>`` at top-level or nested-parameter depth ends the
# call as before, so a genuinely missing ``</parameter>`` (issue 67923)
# still recovers through the converter. Deeper nesting is value text and
# degrades to the previous behaviour. The converter's parameter tree still
# owns the segmentation of the argument text.
_V = ParserState.TOOL_PARAM_VALUE
_VN = ParserState.TOOL_PARAM_VALUE_NESTED
_VF = ParserState.TOOL_PARAM_VALUE_FUNC
_VFN = ParserState.TOOL_PARAM_VALUE_FUNC_NESTED
_TEXT = (EventType.ARG_VALUE_CHUNK,)
_NESTED_PARAM_TRANSITIONS: dict[tuple[ParserState, str], Transition] = {
    # Depth tracking inside a value: all of it is value text.
    (_V, "PARAM_START"): Transition(_VN, _TEXT),
    (_V, "PARAM_END"): Transition(ParserState.TOOL_ARGS, _TEXT),
    (_V, "FUNC_PREFIX"): Transition(_VF, _TEXT),
    (_VN, "PARAM_END"): Transition(_V, _TEXT),
    (_VN, "FUNC_END"): Transition(
        ParserState.TOOL_BETWEEN,
        (EventType.TOOL_CALL_END,),
    ),
    (_VF, "PARAM_START"): Transition(_VFN, _TEXT),
    (_VF, "FUNC_END"): Transition(_V, _TEXT),
    # A bare in-value ``<function=`` mention: the value's own closer.
    (_VF, "PARAM_END"): Transition(ParserState.TOOL_ARGS, _TEXT),
    (_VFN, "PARAM_END"): Transition(_VF, _TEXT),
    (_VFN, "FUNC_END"): Transition(
        ParserState.TOOL_BETWEEN,
        (EventType.TOOL_CALL_END,),
    ),
    (ParserState.TOOL_ARGS, "PARAM_START"): Transition(
        ParserState.TOOL_PARAM_NAME,
        (EventType.ARG_VALUE_CHUNK,),
    ),
    (ParserState.TOOL_PARAM_NAME, "CLOSE_ANGLE"): Transition(
        ParserState.TOOL_PARAM_VALUE,
        (EventType.ARG_VALUE_CHUNK,),
    ),
    (ParserState.TOOL_PARAM_NAME, "PARAM_START"): Transition(
        ParserState.TOOL_PARAM_NAME,
        (EventType.ARG_VALUE_CHUNK,),
    ),
    (ParserState.TOOL_PARAM_NAME, "PARAM_END"): Transition(
        ParserState.TOOL_ARGS,
        (EventType.ARG_VALUE_CHUNK,),
    ),
    (ParserState.TOOL_PARAM_NAME, "FUNC_PREFIX"): Transition(
        ParserState.TOOL_NAME,
        (EventType.TOOL_CALL_END, EventType.TOOL_CALL_START),
    ),
    (ParserState.TOOL_PARAM_NAME, "FUNC_END"): Transition(
        ParserState.TOOL_BETWEEN,
        (EventType.TOOL_CALL_END,),
    ),
    (ParserState.TOOL_PARAM_VALUE, "FUNC_END"): Transition(
        ParserState.TOOL_BETWEEN,
        (EventType.TOOL_CALL_END,),
    ),
}


@functools.cache
def qwen3_config(
    thinking: bool = True,
    *,
    name: str = "qwen3",
    think_start: str = THINK_START,
    think_end: str = THINK_END,
    tool_start: str = TOOL_CALL_START,
    tool_end: str = TOOL_CALL_END,
    turn_boundary_tokens: frozenset[str] = frozenset(),
    nested_param_values: bool = True,
) -> ParserEngineConfig:
    return ParserEngineConfig(
        name=name,
        initial_state=ParserState.REASONING if thinking else ParserState.CONTENT,
        wait_for_reasoning=thinking,
        turn_boundary_tokens=turn_boundary_tokens,
        terminals={
            # Reasoning terminals
            "THINK_START": think_start,
            "THINK_END": think_end,
            # Tool call terminals
            "TOOL_START": tool_start,
            "TOOL_END": tool_end,
            "FUNC_PREFIX": FUNC_PREFIX,
            "FUNC_END": FUNC_END,
            "PARAM_START": PARAM_START,
            "PARAM_END": PARAM_END,
            "CLOSE_ANGLE": ">",
        },
        token_id_terminals={
            "THINK_START": think_start,
            "THINK_END": think_end,
            "TOOL_START": tool_start,
            "TOOL_END": tool_end,
        },
        transitions=(
            {
                # -- Reasoning transitions --
                (ParserState.REASONING, "THINK_START"): Transition(
                    ParserState.REASONING,
                    (),
                ),
                (ParserState.REASONING, "THINK_END"): Transition(
                    ParserState.CONTENT,
                    (EventType.REASONING_END,),
                ),
                # Absorb duplicate </think> — model may emit it after
                # already transitioning to CONTENT; drop it silently.
                (ParserState.CONTENT, "THINK_END"): Transition(
                    ParserState.CONTENT,
                    (),
                ),
                # Tool call directly from reasoning (implicit end)
                (ParserState.REASONING, "TOOL_START"): Transition(
                    ParserState.TOOL_PREAMBLE,
                    (EventType.REASONING_END, EventType.TOOL_CALL_START),
                ),
                # -- Tool call transitions --
                (ParserState.CONTENT, "TOOL_START"): Transition(
                    ParserState.TOOL_PREAMBLE,
                    (EventType.REASONING_END, EventType.TOOL_CALL_START),
                ),
                # Fallback: <function= without a preceding <tool_call>
                (ParserState.CONTENT, "FUNC_PREFIX"): Transition(
                    ParserState.TOOL_NAME,
                    (EventType.TOOL_CALL_START,),
                ),
                (ParserState.TOOL_PREAMBLE, "TOOL_END"): Transition(
                    ParserState.CONTENT,
                    (EventType.TOOL_CALL_END,),
                ),
                (ParserState.TOOL_PREAMBLE, "FUNC_PREFIX"): Transition(
                    ParserState.TOOL_NAME,
                    (),
                ),
                (ParserState.TOOL_NAME, "CLOSE_ANGLE"): Transition(
                    ParserState.TOOL_ARGS,
                    (),
                ),
                # Malformed: </function> while still in TOOL_NAME (no closing >)
                (ParserState.TOOL_NAME, "FUNC_END"): Transition(
                    ParserState.TOOL_BETWEEN,
                    (EventType.TOOL_CALL_END,),
                ),
                # regression issue 58227
                # <function=...</tool_call>
                # function name is not closed, go to content
                (ParserState.TOOL_NAME, "TOOL_END"): Transition(
                    ParserState.CONTENT,
                    (EventType.TOOL_CALL_END,),
                ),
                # regression issue 58227
                # <function=...<tool_call>
                # unclosed tool call followed by another tool call
                (ParserState.TOOL_NAME, "TOOL_START"): Transition(
                    ParserState.TOOL_PREAMBLE,
                    (EventType.TOOL_CALL_END, EventType.TOOL_CALL_START),
                ),
                # regression issue 58227
                # <function=...<function=
                # unclosed function name followed by another function name
                (ParserState.TOOL_NAME, "FUNC_PREFIX"): Transition(
                    ParserState.TOOL_NAME,
                    (EventType.TOOL_CALL_END, EventType.TOOL_CALL_START),
                ),
                (ParserState.TOOL_ARGS, "FUNC_END"): Transition(
                    ParserState.TOOL_BETWEEN,
                    (EventType.TOOL_CALL_END,),
                ),
                # regression issue 58227
                # <function=abc>...<function=
                (ParserState.TOOL_ARGS, "FUNC_PREFIX"): Transition(
                    ParserState.TOOL_NAME,
                    (EventType.TOOL_CALL_END, EventType.TOOL_CALL_START),
                ),
                (ParserState.TOOL_ARGS, "PARAM_START"): Transition(
                    ParserState.TOOL_ARGS,
                    (EventType.ARG_VALUE_CHUNK,),
                ),
                (ParserState.TOOL_ARGS, "PARAM_END"): Transition(
                    ParserState.TOOL_ARGS,
                    (EventType.ARG_VALUE_CHUNK,),
                ),
                (ParserState.TOOL_BETWEEN, "TOOL_END"): Transition(
                    ParserState.CONTENT,
                    (),
                ),
                # Consecutive tool call without closing </tool_call>
                (ParserState.TOOL_BETWEEN, "TOOL_START"): Transition(
                    ParserState.TOOL_PREAMBLE,
                    (EventType.TOOL_CALL_START,),
                ),
                (ParserState.TOOL_BETWEEN, "FUNC_PREFIX"): Transition(
                    ParserState.TOOL_NAME,
                    (EventType.TOOL_CALL_START,),
                ),
            }
            | (_NESTED_PARAM_TRANSITIONS if nested_param_values else {})
        ),
        arg_converter=_qwen3_arg_converter,
        arg_value_extender=_qwen3_arg_value_extender,
        stream_arg_deltas=True,
        strip_trailing_reasoning_whitespace=False,
        tool_args_json=False,
    )


class Qwen3Parser(ParserEngine):
    """Qwen3 parser: `<think>`/`</think>` reasoning +
    `<tool_call>` XML tool calls in a single engine.

    - `<tool_call>` as implicit reasoning end (a grammar transition, so it
      also feeds `is_reasoning_end` and the structured-output gate)

    Subclasses that share the grammar but differ only in the four wrapper
    token strings (reasoning + tool-call) override the class attributes
    below; everything else is inherited unchanged.
    """

    CONFIG_NAME = "qwen3"
    THINK_START = THINK_START
    THINK_END = THINK_END
    TOOL_START = TOOL_CALL_START
    TOOL_END = TOOL_CALL_END
    TURN_BOUNDARIES: frozenset[str] = CHATML_TURN_BOUNDARIES

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        self.thinking_enabled = chat_kwargs.get("enable_thinking", True)
        kwargs.setdefault(
            "parser_engine_config",
            qwen3_config(
                thinking=self.thinking_enabled,
                name=self.CONFIG_NAME,
                think_start=self.THINK_START,
                think_end=self.THINK_END,
                tool_start=self.TOOL_START,
                tool_end=self.TOOL_END,
                turn_boundary_tokens=self.TURN_BOUNDARIES,
            ),
        )
        super().__init__(
            tokenizer,
            tools,
            **kwargs,
        )
        # Each streaming conversion resumes after the last closed parameter.
        if self._arg_converter is _qwen3_arg_converter:
            closed = _ClosedParams()
            self._arg_converter = functools.partial(_qwen3_arg_converter, closed=closed)
            self._arg_value_extender = functools.partial(
                _qwen3_arg_value_extender, closed=closed
            )

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        if not self.thinking_enabled:
            return None, model_output
        return super().extract_reasoning(model_output, request)
