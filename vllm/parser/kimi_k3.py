# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 parser for the XTML channel format.

A complete assistant output has the following shape::

    <|open|>think<|sep|>
    reasoning
    <|close|>think<|sep|>
    <|open|>response<|sep|>
    user-visible content
    <|close|>response<|sep|>
    <|open|>tools<|sep|>
    <|open|>call tool="get_weather" index="1"<|sep|>
    <|open|>argument key="city" type="string"<|sep|>
    Hangzhou
    <|close|>argument<|sep|>
    <|open|>argument key="days" type="number"<|sep|>
    3
    <|close|>argument<|sep|>
    <|open|>argument key="include_details" type="boolean"<|sep|>
    true
    <|close|>argument<|sep|>
    <|close|>call<|sep|>
    <|open|>call tool="search" index="2"<|sep|>
    <|open|>argument key="query" type="string"<|sep|>
    vLLM parser
    <|close|>argument<|sep|>
    <|close|>call<|sep|>
    <|close|>tools<|sep|>
    <|close|>message<|sep|>

The response and tools channels are independent: either may be empty, and
multiple call blocks may occur inside the tools channel. Argument ``type`` is
used to restore JSON values such as numbers, booleans, objects, and arrays.
The chat template may consume the opening think or response marker as the
generation prefix, so generated text can begin directly with channel content.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import regex as re
from openai.types.responses import ToolChoiceFunction

from vllm import envs
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaToolCall, FunctionCall
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.exceptions import VLLMValidationError
from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine, ToolCallSlot
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.sampling_params import StructuredOutputsParams
from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

OPEN = "<|open|>"
CLOSE = "<|close|>"
SEP = "<|sep|>"

THINK_OPEN = f"{OPEN}think{SEP}"
THINK_CLOSE = f"{CLOSE}think{SEP}"
RESPONSE_OPEN = f"{OPEN}response{SEP}"
RESPONSE_CLOSE = f"{CLOSE}response{SEP}"
TOOLS_OPEN = f"{OPEN}tools{SEP}"
TOOLS_CLOSE = f"{CLOSE}tools{SEP}"
CALL_PREFIX = f'{OPEN}call tool="'
CALL_NAME_END = '" index="'
CALL_HEADER_END = f'"{SEP}'
CALL_CLOSE = f"{CLOSE}call{SEP}"
MESSAGE_CLOSE = f"{CLOSE}message{SEP}"

_O = re.escape(OPEN)
_C = re.escape(CLOSE)
_S = re.escape(SEP)
_ARG_RE = re.compile(
    rf"{_O}argument\s+(?P<attrs>(?:(?!{_S}).)*?){_S}"
    rf"(?P<value>.*?){_C}argument{_S}",
    re.DOTALL,
)
_PARTIAL_ARG_RE = re.compile(
    rf"{_O}argument\s+(?P<attrs>(?:(?!{_S}).)*?){_S}(?P<value>.*)$",
    re.DOTALL,
)
_ATTR_RE = re.compile(r'(?P<key>\w+)="(?P<value>[^"]*)"')


def _attrs(text: str) -> dict[str, str]:
    return {
        match["key"]: match["value"].replace("&quot;", '"').replace("&amp;", "&")
        for match in _ATTR_RE.finditer(text)
    }


def _decode_argument(attrs_text: str, raw_value: str) -> tuple[str, Any] | None:
    attrs = _attrs(attrs_text)
    key = attrs.get("key")
    if key is None:
        return None
    if attrs.get("type", "string") == "string":
        return key, raw_value
    try:
        return key, json.loads(raw_value)
    except json.JSONDecodeError:
        return key, raw_value


def _kimi_k3_arg_converter(raw_args: str, partial: bool) -> str:
    arguments: dict[str, Any] = {}
    last_end = 0
    for match in _ARG_RE.finditer(raw_args):
        decoded = _decode_argument(match["attrs"], match["value"])
        if decoded is not None:
            arguments[decoded[0]] = decoded[1]
        last_end = match.end()

    if partial:
        match = _PARTIAL_ARG_RE.search(raw_args, last_end)
        if match is not None:
            decoded = _decode_argument(match["attrs"], match["value"])
            if decoded is not None:
                arguments[decoded[0]] = decoded[1]

    return json.dumps(arguments, ensure_ascii=False)


@functools.cache
def kimi_k3_config(thinking: bool = True) -> ParserEngineConfig:
    return ParserEngineConfig(
        name="kimi_k3",
        initial_state=ParserState.REASONING if thinking else ParserState.CONTENT,
        terminals={
            "THINK_OPEN": THINK_OPEN,
            "THINK_CLOSE": THINK_CLOSE,
            "RESPONSE_OPEN": RESPONSE_OPEN,
            "RESPONSE_CLOSE": RESPONSE_CLOSE,
            "TOOLS_OPEN": TOOLS_OPEN,
            "TOOLS_CLOSE": TOOLS_CLOSE,
            "CALL_PREFIX": CALL_PREFIX,
            "CALL_NAME_END": CALL_NAME_END,
            "CALL_HEADER_END": CALL_HEADER_END,
            "CALL_CLOSE": CALL_CLOSE,
            "MESSAGE_CLOSE": MESSAGE_CLOSE,
        },
        transitions={
            (ParserState.REASONING, "THINK_OPEN"): Transition(ParserState.REASONING),
            (ParserState.REASONING, "THINK_CLOSE"): Transition(
                ParserState.CONTENT,
                (EventType.REASONING_END,),
            ),
            (ParserState.CONTENT, "THINK_OPEN"): Transition(
                ParserState.REASONING,
                (EventType.REASONING_START,),
            ),
            (ParserState.CONTENT, "THINK_CLOSE"): Transition(ParserState.CONTENT),
            (ParserState.CONTENT, "RESPONSE_OPEN"): Transition(ParserState.CONTENT),
            (ParserState.CONTENT, "RESPONSE_CLOSE"): Transition(ParserState.CONTENT),
            (ParserState.CONTENT, "TOOLS_OPEN"): Transition(ParserState.TOOL_PREAMBLE),
            (ParserState.TOOL_PREAMBLE, "CALL_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_BETWEEN, "CALL_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_NAME, "CALL_NAME_END"): Transition(
                ParserState.MESSAGE_HEADER
            ),
            (ParserState.MESSAGE_HEADER, "CALL_HEADER_END"): Transition(
                ParserState.TOOL_ARGS
            ),
            (ParserState.TOOL_ARGS, "CALL_CLOSE"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
            ),
            (ParserState.TOOL_PREAMBLE, "TOOLS_CLOSE"): Transition(ParserState.CONTENT),
            (ParserState.TOOL_BETWEEN, "TOOLS_CLOSE"): Transition(ParserState.CONTENT),
            (ParserState.TOOL_ARGS, "TOOLS_CLOSE"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
            (ParserState.CONTENT, "MESSAGE_CLOSE"): Transition(ParserState.CONTENT),
        },
        arg_converter=_kimi_k3_arg_converter,
        arg_structural_chars=frozenset("<>"),
        stream_arg_deltas=False,
        tool_args_json=False,
        preserve_tokens=frozenset((OPEN, CLOSE, SEP)),
        strip_content_whitespace_with_tools=False,
        validate_tool_names=False,
    )


class KimiK3Parser(ParserEngine):
    """Parse Kimi K3 reasoning, response, and tool channels in one engine."""

    supports_required_and_named = False
    structural_tag_model = "kimi_k3"
    engine_based_streaming = True

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        chat_kwargs = kwargs.pop("chat_template_kwargs", None) or {}
        thinking = chat_kwargs.get("thinking")
        if thinking is None:
            thinking = chat_kwargs.get("enable_thinking", True)
        super().__init__(
            tokenizer,
            tools,
            parser_engine_config=kimi_k3_config(bool(thinking)),
            **kwargs,
        )
        self._think_open_ids = tokenizer.encode(THINK_OPEN, add_special_tokens=False)
        self._think_close_ids = tokenizer.encode(THINK_CLOSE, add_special_tokens=False)

    def adjust_request(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> ChatCompletionRequest | ResponsesRequest:
        request.skip_special_tokens = False
        if hasattr(request, "spaces_between_special_tokens"):
            request.spaces_between_special_tokens = False

        if not request.tools or request.tool_choice == "none":
            return request

        named = isinstance(
            request.tool_choice,
            (ChatCompletionNamedToolChoiceParam, ToolChoiceFunction),
        )
        structured_outputs = getattr(request, "structured_outputs", None)
        has_structural_tag = (
            structured_outputs is not None
            and structured_outputs.structural_tag is not None
        )
        if has_structural_tag:
            return request
        if not envs.VLLM_ENFORCE_STRICT_TOOL_CALLING:
            if named:
                raise VLLMValidationError(
                    "Named tool choice for Kimi K3 requires strict tool calling "
                    "(VLLM_ENFORCE_STRICT_TOOL_CALLING) so the XTML structural "
                    "tag can force the call. Otherwise use `tool_choice` set to "
                    '"auto", "required", or "none".',
                    parameter="tool_choice",
                    value=request.tool_choice,
                )
            return request

        structural_tag = get_model_structural_tag(
            model="kimi_k3",
            tools=request.tools,
            tool_choice=request.tool_choice,
            reasoning=False,
        )
        if structural_tag is None:
            return request
        request.structured_outputs = StructuredOutputsParams(
            structural_tag=json.dumps(structural_tag.model_dump())
        )
        if isinstance(request, ResponsesRequest):
            request.text = None
        else:
            request.response_format = None
        return request

    def _ensure_tool_id(self, slot: ToolCallSlot, name: str) -> None:
        if not slot.id:
            slot.id = f"{name}:{self._tool_slots.index(slot)}"

    def _emit_name_delta(
        self,
        idx: int,
        deltas: list[DeltaToolCall],
        name: str | None,
    ) -> None:
        if name:
            name = name.replace("&quot;", '"').replace("&amp;", "&")
        super()._emit_name_delta(idx, deltas, name)

    def parse(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
        enable_auto_tools: bool = False,
        model_output_token_ids: Sequence[int] = (),
    ) -> tuple[str | None, str | None, list[FunctionCall] | None]:
        reasoning, content, tool_calls = super().parse(
            model_output,
            request,
            enable_auto_tools,
            model_output_token_ids,
        )
        if tool_calls is not None:
            tool_calls = tool_calls[: model_output.count(CALL_CLOSE)] or None
        return reasoning, content, tool_calls

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        if self.parser_engine_config.initial_state != ParserState.REASONING:
            return True
        last_open = _subsequence_index(input_ids, self._think_open_ids)
        last_close = _subsequence_index(input_ids, self._think_close_ids)
        if last_open == -1:
            return last_close != -1
        return last_close > last_open

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if self.parser_engine_config.initial_state != ParserState.REASONING:
            return input_ids
        close = _subsequence_index(input_ids, self._think_close_ids)
        if close == -1:
            return []
        return input_ids[close + len(self._think_close_ids) :]


def _subsequence_index(haystack: Sequence[int], needle: Sequence[int]) -> int:
    if not needle:
        return -1
    for index in range(len(haystack) - len(needle), -1, -1):
        if list(haystack[index : index + len(needle)]) == list(needle):
            return index
    return -1
