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
from collections.abc import Sequence
from typing import TYPE_CHECKING

import regex as re
from openai.types.responses import CustomTool
from pydantic import BaseModel
from xgrammar import StructuralTag
from xgrammar.structural_tag import GrammarFormat, TagFormat

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.exceptions import VLLMValidationError
from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ArgValueExtender,
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.tool_parsers.utils import (
    compile_custom_tool_grammar,
    custom_tools_with_grammar,
    partial_tag_overlap,
    validate_custom_tool_input,
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


def constrain_custom_tool_inputs(
    tag: StructuralTag, tools: dict[str, CustomTool]
) -> None:
    """Constrain raw input inside the Qwen XML envelope, preserving payload newlines."""
    pending = {
        f"{TOOL_CALL_START}\n{FUNC_PREFIX}{name}>\n": tool
        for name, tool in tools.items()
    }

    def visit(node: object) -> None:
        if isinstance(node, TagFormat) and node.begin in pending:
            tool = pending.pop(node.begin)
            assert tool.format is not None and tool.format.type == "grammar"
            grammar = compile_custom_tool_grammar(
                tool.format.syntax, tool.format.definition
            ).grammar
            # The parser removes one wrapping newline on each side of a parameter.
            node.content = TagFormat(
                begin=f"{PARAM_START}input>\n",
                content=GrammarFormat(grammar=str(grammar)),
                end=f"\n{PARAM_END}",
            )
            return
        if isinstance(node, BaseModel):
            for name in type(node).model_fields:
                visit(getattr(node, name))
        elif isinstance(node, (list, tuple)):
            for element in node:
                visit(element)

    visit(tag.format)
    if pending:
        raise VLLMValidationError(
            "Cannot constrain custom tools in this Qwen tool-call format.",
            parameter="tools",
        )


_PARAM_RE = re.compile(
    r"<\s*parameter\s*=\s*([^>]*)>"
    r"(.*?)"
    r"(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s*=))",
    re.DOTALL,
)
_PARTIAL_PARAM_RE = re.compile(r"<\s*parameter\s*=\s*([^>]+)>(.*)$", re.DOTALL)
_PARAM_OPENER_RE = re.compile(r"<\s*parameter\s*=")


def _trim_wrapping_newlines(value: str) -> str:
    """Strip one leading and one trailing newline (the Qwen3 template markup)."""
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


class _ClosedParams:
    """Closed parameters of a raw argument text, reused by longer texts.

    Raw arguments only grow while a call streams, so a parse of a text that
    extends `text` resumes after its last closed parameter.
    """

    __slots__ = ("text", "end", "params", "outside")

    def __init__(self) -> None:
        self.text = ""  # raw text the closed parameters depend on
        self.end = 0  # end of the last closed parameter in ``text``
        self.params: dict[str, str] = {}
        self.outside = ""  # text before ``end`` outside the closed parameters


def _qwen3_parse_params(
    raw_args: str, partial: bool, closed: _ClosedParams | None = None
) -> tuple[dict[str, str], re.Match | None]:
    """Return the closed parameters, trimmed, and the open trailing one."""
    params: dict[str, str] = {}
    outside: list[str] = []
    pos = 0
    if closed is not None and raw_args.startswith(closed.text):
        params.update(closed.params)
        outside.append(closed.outside)
        pos = closed.end
    start = pos

    for match in _PARAM_RE.finditer(raw_args, pos):
        outside.append(raw_args[pos : match.start()])
        params[match.group(1)] = _trim_wrapping_newlines(match.group(2))
        pos = match.end()

    if closed is not None and pos > start:
        # A value closed by the next opener holds only while that opener follows.
        opener = _PARAM_OPENER_RE.match(raw_args, pos)
        closed.text = raw_args[: opener.end() if opener else pos]
        closed.end = pos
        closed.params = dict(params)
        closed.outside = "".join(outside)

    open_param = None
    if partial:
        open_param = _PARTIAL_PARAM_RE.search("".join(outside) + raw_args[pos:])
    return params, open_param


def _qwen3_arg_converter(
    raw_args: str, partial: bool, closed: _ClosedParams | None = None
) -> str:
    params, open_param = _qwen3_parse_params(raw_args, partial, closed)
    if open_param is not None and open_param.group(1):
        params[open_param.group(1)] = _trim_wrapping_newlines(open_param.group(2))

    return json.dumps(params, ensure_ascii=False)


# What ends an open value in ``_PARAM_RE``: a closing tag or the next opener.
_PARAM_VALUE_END_RE = re.compile(r"<\s*/\s*parameter\s*>|<\s*parameter\s*=")


def _param_value_end_window(text: str) -> str:
    """Return the suffix of *text* that may still grow into a value end."""
    start = text.rfind("<")
    if start == -1:
        return ""
    tail = text[start:]
    return tail if _PARAM_VALUE_END_RE.fullmatch(tail, partial=True) else ""


def _param_value_extender(
    raw_args: str, trim_newlines: bool, closed: _ClosedParams | None = None
) -> ArgValueExtender | None:
    """Extend the open trailing `<parameter=...>` value of *raw_args*.

    The extender streams exactly what the converter would for the same text.
    With *trim_newlines* it follows `_trim_wrapping_newlines`: it drops a
    leading newline of the value and holds back a trailing one.
    """
    closed_params, match = _qwen3_parse_params(raw_args, True, closed)
    if match is None:
        return None
    name, value = match.group(1), match.group(2)
    # A repeated key keeps its first position, so its value is not last.
    if name in closed_params:
        return None

    window = _param_value_end_window(value)
    started = bool(value)
    held = "\n" if trim_newlines and value.endswith("\n") and value != "\n" else ""

    def extend(delta: str) -> str | None:
        nonlocal window, started, held
        text = window + delta
        if _PARAM_VALUE_END_RE.search(text):
            return None
        window = _param_value_end_window(text)
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
        },
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
        self._custom_grammar_tools = custom_tools_with_grammar(tools)
        self._grammar_chunks: list[str] = []
        self._grammar_token_ids: list[int] = []
        self._grammar_initial_state: ParserState | None = None
        self._grammar_probe = ""

    def _single_pass_parse(
        self,
        text: str,
        token_ids: Sequence[int],
        initial_state: ParserState | None = None,
    ) -> tuple[str | None, str | None, ExtractedToolCallInformation]:
        if (
            not self._custom_grammar_tools
            or self.skip_tool_parsing
            or self._suppress_tool_calls
        ):
            return super()._single_pass_parse(text, token_ids, initial_state)
        match = re.search(r"<tool_call>\n<function=([^>\n]+)>\n", text)
        if match is not None and match[1] not in self._custom_grammar_tools:
            return super()._single_pass_parse(text, token_ids, initial_state)

        start_id = self.vocab.get(TOOL_CALL_START)
        prefix_ids = (
            token_ids
            if match is None
            else (
                token_ids[: token_ids.index(start_id)]
                if start_id is not None and start_id in token_ids
                else []
            )
        )
        prefix = text if match is None else text[: match.start()]
        self.skip_tool_parsing = True
        try:
            reasoning, content, plain = super()._single_pass_parse(
                prefix, prefix_ids, initial_state
            )
        finally:
            self.skip_tool_parsing = False
        if match is None:
            return reasoning, content, plain
        begin = f"{PARAM_START}input>\n"
        end = f"\n{PARAM_END}\n{FUNC_END}\n{TOOL_CALL_END}"
        body = text[match.end() :]
        calls = []
        if body.startswith(begin) and body.endswith(end):
            # Grammar tools stop after one call. The last delimiter belongs to the
            # envelope; delimiter-looking text inside the payload remains data.
            payload = body[len(begin) : -len(end)]
            validate_custom_tool_input(self._custom_grammar_tools[match[1]], payload)
            calls.append(
                ToolCall(
                    function=FunctionCall(
                        name=match[1],
                        arguments=json.dumps({"input": payload}, ensure_ascii=False),
                    )
                )
            )
        return (
            reasoning,
            content,
            ExtractedToolCallInformation(
                tools_called=bool(calls), tool_calls=calls, content=content
            ),
        )

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> ExtractedToolCallInformation:
        if not self._custom_grammar_tools or request.tool_choice == "none":
            return super().extract_tool_calls(model_output, request)
        self._check_skip_tool_parsing(request)
        return self._single_pass_parse(model_output, [])[2]

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> DeltaMessage | None:
        if not self._custom_grammar_tools or request.tool_choice == "none":
            return super().extract_tool_calls_streaming(
                previous_text,
                current_text,
                delta_text,
                previous_token_ids,
                current_token_ids,
                delta_token_ids,
                request,
            )
        self._grammar_initial_state = ParserState.CONTENT
        if not self._grammar_chunks:
            text = self._grammar_probe + delta_text
            start = text.find(TOOL_CALL_START)
            if start == -1:
                overlap = partial_tag_overlap(text, TOOL_CALL_START)
                self._grammar_probe = text[-overlap:] if overlap else ""
                content = text[:-overlap] if overlap else text
                return DeltaMessage(content=content) if content else None
            self._grammar_probe = ""
            self._grammar_chunks.append(text[start:])
            self._grammar_token_ids.extend(delta_token_ids)
            return DeltaMessage(content=text[:start]) if start else None
        self._grammar_chunks.append(delta_text)
        self._grammar_token_ids.extend(delta_token_ids)
        return None

    def parse_delta(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        request: ChatCompletionRequest | ResponsesRequest,
        prompt_token_ids: list[int] | None = None,
        *,
        finished: bool,
    ) -> DeltaMessage | None:
        if not self._custom_grammar_tools or request.tool_choice == "none":
            return super().parse_delta(
                delta_text,
                delta_token_ids,
                request,
                prompt_token_ids,
                finished=finished,
            )
        self._grammar_chunks.append(delta_text)
        self._grammar_token_ids.extend(delta_token_ids)
        if prompt_token_ids is not None:
            self.set_prompt_token_ids(prompt_token_ids)
        delta = self.finish_streaming() if finished else None
        if delta and not request.include_reasoning:
            delta.reasoning = None
        return delta

    def finish_streaming(self) -> DeltaMessage | None:
        if not self._grammar_chunks:
            if self._grammar_probe:
                remaining_text, self._grammar_probe = self._grammar_probe, ""
                return DeltaMessage(content=remaining_text)
            return super().finish_streaming()
        reasoning, content, result = self._single_pass_parse(
            "".join(self._grammar_chunks),
            self._grammar_token_ids,
            self._grammar_initial_state,
        )
        self._grammar_chunks.clear()
        self._grammar_token_ids.clear()
        return DeltaMessage(
            reasoning=reasoning,
            content=content,
            tool_calls=[
                DeltaToolCall(
                    index=index,
                    id=call.id,
                    type="function",
                    function=DeltaFunctionCall(
                        name=call.function.name, arguments=call.function.arguments
                    ),
                )
                for index, call in enumerate(result.tool_calls)
            ],
        )

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        if not self.thinking_enabled:
            return None, model_output
        return super().extract_reasoning(model_output, request)
