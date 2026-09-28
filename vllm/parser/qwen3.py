# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3 parser for tool calls and reasoning.

Qwen3 XML tool call format::

    <tool_call>
    <function=func_name>
    <parameter=key>value</parameter>
    </function>
    </tool_call>

The argument body consists of ``<parameter=NAME>VALUE</parameter>`` tags.
The ``_qwen3_arg_converter`` parses these into a JSON object.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

import regex as re

from vllm.entrypoints.generate.base.protocol import (
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.parser.engine.events import EventType, SemanticEvent
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
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

_PARAM_RE = re.compile(
    r"<\s*parameter\s*=\s*([^>]*)>"
    r"(.*?)"
    r"(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s*=))",
    re.DOTALL,
)
_PARTIAL_PARAM_RE = re.compile(r"<\s*parameter\s*=\s*([^>]+)>(.*)$", re.DOTALL)
_PARAM_OPEN_RE = re.compile(r"<\s*parameter\s*=\s*([^>]*)>")
_FUNC_OPEN_RE = re.compile(r"<function=([^>]*)>")
_FUNC_BODY_RE = re.compile(r"<function=[^>]*>(.*)</function>", re.DOTALL)


def _trim_wrapping_newlines(value: str) -> str:
    """Strip one leading and one trailing newline (the Qwen3 template markup)."""
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


def _greedy_qwen_args(raw_args: str) -> str:
    """Keep the first parameter through the final closer.

    Used when closers outnumber openers, so a later ``<parameter=`` tag is
    text inside the value rather than another parameter.
    """
    match = _PARAM_OPEN_RE.search(raw_args)
    if match is None:
        return "{}"
    name = match.group(1).strip()
    if not name:
        return "{}"
    end = raw_args.rfind(PARAM_END)
    value_start = match.end()
    value = raw_args[value_start:] if end < value_start else raw_args[value_start:end]
    return json.dumps(
        {name: _trim_wrapping_newlines(value)},
        ensure_ascii=False,
    )


def _qwen3_arg_converter(raw_args: str, partial: bool) -> str:
    params: dict[str, object] = {}

    for match in _PARAM_RE.finditer(raw_args):
        name = match.group(1)
        value = match.group(2)
        params[name] = _trim_wrapping_newlines(value)

    if partial:
        remaining = _PARAM_RE.sub("", raw_args)
        m = _PARTIAL_PARAM_RE.search(remaining)
        if m:
            name = m.group(1)
            value = m.group(2)
            if name:
                params[name] = _trim_wrapping_newlines(value)

    return json.dumps(params, ensure_ascii=False)


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
        transitions={
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
            (ParserState.TOOL_ARGS, "FUNC_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
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
        },
        arg_converter=_qwen3_arg_converter,
        stream_arg_deltas=True,
        strip_trailing_reasoning_whitespace=False,
        tool_args_json=False,
    )


class Qwen3Parser(ParserEngine):
    """Qwen3 parser: ``<think>``/``</think>`` reasoning +
    ``<tool_call>`` XML tool calls in a single engine.

    - ``<tool_call>`` as implicit reasoning end (a grammar transition, so it
      also feeds ``is_reasoning_end`` and the structured-output gate)

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
        self._qwen_source_text = ""
        self._held_later_calls: list[DeltaToolCall] = []
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

    def _reset(self, initial_state: ParserState | None = None) -> None:
        super()._reset(initial_state=initial_state)
        self._qwen_source_text = ""
        self._held_later_calls = []

    def _feed(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
    ) -> list[SemanticEvent]:
        self._qwen_source_text += delta_text
        return super()._feed(delta_text, delta_token_ids)

    def _closers_exceed_openers(self, text: str) -> bool:
        """True when a closer was copied into a parameter value.

        The model's own closer then has no matching opener. Balanced
        parallel calls do not.
        """
        pairs = (
            (PARAM_START, PARAM_END),
            (FUNC_PREFIX, FUNC_END),
            (self.TOOL_START, self.TOOL_END),
        )
        return any(text.count(closer) > text.count(opener) for opener, closer in pairs)

    def _collapsed_call(self, text: str) -> tuple[str, str] | None:
        opened = _FUNC_OPEN_RE.search(text)
        if opened is None:
            return None
        name = opened.group(1).strip()
        if not name or not self._accept_tool_name(name):
            return None
        body_match = _FUNC_BODY_RE.search(text)
        body = body_match.group(1) if body_match is not None else ""
        args_json = self._fix_arg_types(_greedy_qwen_args(body), name)
        return name, args_json

    def _build_extracted_result(
        self,
        *deltas: DeltaMessage | None,
    ) -> ExtractedToolCallInformation:
        result = super()._build_extracted_result(*deltas)
        text = self._qwen_source_text
        if not text or not self._closers_exceed_openers(text):
            return result
        collapsed = self._collapsed_call(text)
        if collapsed is None:
            if len(result.tool_calls) <= 1:
                return result
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=result.tool_calls[:1],
                content=result.content,
            )
        name, args_json = collapsed
        call_id = result.tool_calls[0].id if result.tool_calls else None
        tool_call = ToolCall(
            function=FunctionCall(name=name, arguments=args_json),
            **({"id": call_id} if call_id else {}),
        )
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=[tool_call],
            content=result.content,
        )

    def _park_later_calls(self, delta: DeltaMessage | None) -> DeltaMessage | None:
        if delta is None or not delta.tool_calls:
            return delta
        kept: list[DeltaToolCall] = []
        for call in delta.tool_calls:
            if call.index > 0:
                self._held_later_calls.append(call)
            else:
                kept.append(call)
        delta.tool_calls = kept
        if not delta.tool_calls and not delta.content and not delta.reasoning:
            return None
        return delta

    def _release_held_calls(self, delta: DeltaMessage | None) -> DeltaMessage | None:
        delta = self._park_later_calls(delta)
        later = self._held_later_calls
        self._held_later_calls = []
        if not later or self._closers_exceed_openers(self._qwen_source_text):
            return delta
        if delta is None:
            return DeltaMessage(tool_calls=later)
        delta.tool_calls = list(delta.tool_calls) + later
        return delta

    def _events_to_delta(
        self,
        events: list[SemanticEvent],
        finished: bool = False,
    ) -> DeltaMessage | None:
        delta = super()._events_to_delta(events, finished=finished)
        if finished:
            return self._release_held_calls(delta)
        return self._park_later_calls(delta)

    def finish_streaming(self) -> DeltaMessage | None:
        delta = super().finish_streaming()
        if self._held_later_calls:
            return self._release_held_calls(delta)
        return delta

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        if not self.thinking_enabled:
            return None, model_output
        return super().extract_reasoning(model_output, request)
