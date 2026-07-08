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

import ast
import functools
import json
from typing import TYPE_CHECKING

import regex as re

from vllm.entrypoints.openai.engine.protocol import (
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.parser.engine.events import EventType
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
FUNC_PREFIX = "<function="
FUNC_PREFIX_IMSTART_QUOTED = '<|im_start|>="'
FUNC_PREFIX_IMSTART_FUNCTION = "<|im_start|>function="
FUNC_PREFIX_BARE_EQ = "="
FUNC_END = "</function>"
FUNC_NAME_QUOTE_END = '"'
FUNC_NAME_NEWLINE_END = "\n"
PARAM_START = "<parameter="
PARAM_END = "</parameter>"

_PARAM_RE = re.compile(
    r"<\s*parameter\s*=\s*([^>]*)>"
    r"(.*?)"
    r"(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s*=))",
    re.DOTALL,
)
_PARTIAL_PARAM_RE = re.compile(r"<\s*parameter\s*=\s*([^>]+)>(.*)$", re.DOTALL)
_DEFAULT_API_CALL_RE = re.compile(
    r"call:\s*default_api:([A-Za-z_][A-Za-z0-9_]*)\s*(\{.*?\})",
    re.DOTALL,
)
_JSON_OBJECT_START_RE = re.compile(r"\{")


def _strip_wrapping_quotes(value: str, *, partial: bool = False) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    if partial and value:
        if value[0] in ("'", '"'):
            value = value[1:]
        if value and value[-1] in ("'", '"'):
            value = value[:-1]
    return value


def _qwen3_arg_converter(raw_args: str, partial: bool) -> str:
    stripped = raw_args.strip()
    if stripped.startswith("{") and (not partial or stripped.endswith("}")):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                parsed = None
        if isinstance(parsed, dict):
            return json.dumps(
                {str(k): v for k, v in parsed.items()},
                ensure_ascii=False,
            )

    params: dict[str, object] = {}

    for match in _PARAM_RE.finditer(raw_args):
        name = _strip_wrapping_quotes(match.group(1))
        value = _strip_wrapping_quotes(match.group(2), partial=partial)
        params[name] = value

    if partial:
        remaining = _PARAM_RE.sub("", raw_args)
        m = _PARTIAL_PARAM_RE.search(remaining)
        if m:
            name = _strip_wrapping_quotes(m.group(1))
            value = _strip_wrapping_quotes(m.group(2), partial=partial)
            if name:
                params[name] = value

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
) -> ParserEngineConfig:
    return ParserEngineConfig(
        name=name,
        initial_state=ParserState.REASONING if thinking else ParserState.CONTENT,
        terminals={
            # Reasoning terminals
            "THINK_START": think_start,
            "THINK_END": think_end,
            # Tool call terminals
            "TOOL_START": tool_start,
            "TOOL_END": tool_end,
            "FUNC_PREFIX": FUNC_PREFIX,
            "FUNC_PREFIX_IMSTART_QUOTED": FUNC_PREFIX_IMSTART_QUOTED,
            "FUNC_PREFIX_IMSTART_FUNCTION": FUNC_PREFIX_IMSTART_FUNCTION,
            "FUNC_PREFIX_BARE_EQ": FUNC_PREFIX_BARE_EQ,
            "FUNC_END": FUNC_END,
            "FUNC_NAME_QUOTE_END": FUNC_NAME_QUOTE_END,
            "FUNC_NAME_NEWLINE_END": FUNC_NAME_NEWLINE_END,
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
        preserve_tokens=frozenset({"<|im_start|>"}),
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
            # Some Qwen3.6 checkpoints occasionally emit malformed auto
            # tool-call function openers while still producing valid parameter
            # XML. Accept the observed variants only after <tool_call>.
            (ParserState.TOOL_PREAMBLE, "FUNC_PREFIX_IMSTART_QUOTED"): Transition(
                ParserState.TOOL_NAME,
                (),
            ),
            (ParserState.TOOL_PREAMBLE, "FUNC_PREFIX_IMSTART_FUNCTION"): Transition(
                ParserState.TOOL_NAME,
                (),
            ),
            (ParserState.TOOL_PREAMBLE, "FUNC_PREFIX_BARE_EQ"): Transition(
                ParserState.TOOL_NAME,
                (),
            ),
            (ParserState.TOOL_NAME, "CLOSE_ANGLE"): Transition(
                ParserState.TOOL_ARGS,
                (),
            ),
            (ParserState.TOOL_NAME, "FUNC_NAME_QUOTE_END"): Transition(
                ParserState.TOOL_ARGS,
                (),
            ),
            (ParserState.TOOL_NAME, "FUNC_NAME_NEWLINE_END"): Transition(
                ParserState.TOOL_ARGS,
                (),
            ),
            (ParserState.TOOL_NAME, "PARAM_START"): Transition(
                ParserState.TOOL_ARGS,
                (EventType.ARG_VALUE_CHUNK,),
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

    - ``<tool_call>`` as implicit reasoning end
    - Unpaired ``<tool_call>`` token ID detection for ``is_reasoning_end``

    Subclasses that share the grammar but differ only in the four wrapper
    token strings (reasoning + tool-call) override the class attributes
    below; everything else is inherited unchanged.
    """

    CONFIG_NAME = "qwen3"
    THINK_START = THINK_START
    THINK_END = THINK_END
    TOOL_START = TOOL_CALL_START
    TOOL_END = TOOL_CALL_END

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
            ),
        )
        super().__init__(
            tokenizer,
            tools,
            **kwargs,
        )
        vocab = self.vocab
        self._tool_call_token_id: int | None = vocab.get(self.TOOL_START)
        self._tool_call_end_token_id: int | None = vocab.get(self.TOOL_END)

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        if not self.thinking_enabled:
            return None, model_output
        return super().extract_reasoning(model_output, request)

    def extract_tool_calls_from_content(
        self,
        content: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        result = super().extract_tool_calls_from_content(content, request)
        if result.tools_called:
            return result

        tool_calls: list[ToolCall] = []
        content_parts: list[str] = []
        pos = 0
        for match in _DEFAULT_API_CALL_RE.finditer(content):
            name = match.group(1)
            args = match.group(2)
            try:
                args_json = _qwen3_arg_converter(args, partial=False)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
            content_parts.append(content[pos : match.start()])
            pos = match.end()
            tool_calls.append(
                ToolCall(
                    type="function",
                    function=FunctionCall(name=name, arguments=args_json),
                )
            )

        if not tool_calls:
            json_fallback = self._extract_json_tool_call_objects(content)
            if json_fallback is not None:
                return json_fallback
            return result

        content_parts.append(content[pos:])
        fallback_content = "".join(content_parts).strip() or None
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=fallback_content,
        )

    def _extract_json_tool_call_objects(
        self,
        content: str,
    ) -> ExtractedToolCallInformation | None:
        decoder = json.JSONDecoder()
        tool_calls: list[ToolCall] = []
        spans: list[tuple[int, int]] = []
        search_pos = 0

        while True:
            match = _JSON_OBJECT_START_RE.search(content, search_pos)
            if match is None:
                break
            start = match.start()
            try:
                obj, end_offset = decoder.raw_decode(content[start:])
            except json.JSONDecodeError:
                search_pos = start + 1
                continue
            end = start + end_offset
            search_pos = end

            if not isinstance(obj, dict):
                continue
            name = obj.get("tool") or obj.get("name")
            args = obj.get("arguments") or obj.get("parameters")
            if not isinstance(name, str) or not isinstance(args, dict):
                continue

            tool_calls.append(
                ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=name,
                        arguments=json.dumps(args, ensure_ascii=False),
                    ),
                )
            )
            spans.append((start, end))

        if not tool_calls:
            return None

        content_parts: list[str] = []
        pos = 0
        for start, end in spans:
            content_parts.append(content[pos:start])
            pos = end
        content_parts.append(content[pos:])
        fallback_content = "".join(content_parts).strip() or None
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=fallback_content,
        )

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        if super().is_reasoning_end(input_ids):
            return True
        tool_call_id = self._tool_call_token_id
        tool_call_end_id = self._tool_call_end_token_id
        reasoning_start_id = self._reasoning_start_token_id
        if tool_call_id is not None:
            for i in range(len(input_ids) - 1, -1, -1):
                if (
                    reasoning_start_id is not None
                    and input_ids[i] == reasoning_start_id
                ):
                    return False
                if input_ids[i] == tool_call_id:
                    if tool_call_end_id is not None and any(
                        input_ids[j] == tool_call_end_id
                        for j in range(i + 1, len(input_ids))
                    ):
                        continue
                    return True
        return False
