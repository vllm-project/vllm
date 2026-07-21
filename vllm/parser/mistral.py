# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import functools
import json
from collections.abc import Sequence
from enum import Enum, auto
from random import choices
from string import ascii_letters, digits
from typing import TYPE_CHECKING, Any, Literal

import ijson
import regex as re
from mistral_common.protocol.instruct.tool_calls import (
    NamedToolChoice as MistralNamedToolChoice,
)
from mistral_common.protocol.instruct.tool_calls import (
    Tool as MistralTool,
)
from mistral_common.protocol.instruct.tool_calls import (
    ToolChoice as MistralToolChoice,
)
from mistral_common.protocol.instruct.tool_calls import (
    ToolChoiceEnum as MistralToolChoiceEnum,
)
from mistral_common.tokens.tokenizers.base import SpecialTokens
from pydantic import Field

from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.sampling_params import StructuredOutputsParams
from vllm.utils.mistral import is_mistral_tokenizer

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.parser.engine.events import SemanticEvent
    from vllm.parser.engine.parser_engine import ToolCallSlot
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

logger = init_logger(__name__)

_ALPHANUMERIC = ascii_letters + digits
_DEFAULT_JSON_SCHEMA: dict[str, Any] = {
    "anyOf": [{"type": "object"}, {"type": "array"}]
}

# Special token strings as emitted/decoded by MistralTokenizer.
_TOOL_CALLS = SpecialTokens.tool_calls.value
_ARGS = SpecialTokens.args.value
_THINK_START_SPECIAL = SpecialTokens.begin_think.value
_THINK_END_SPECIAL = SpecialTokens.end_think.value

# Plain-text reasoning markers used by v11 tokenizers.
_THINK_START_TEXT = "<think>"
_THINK_END_TEXT = "</think>"

_OPEN_BRACE = "{"


class StreamingState(Enum):
    """Streaming parsing state for pre-v11 tool call extraction."""

    WAITING_FOR_TOOL_START = auto()
    WAITING_FOR_TOOL_KEY = auto()
    PARSING_NAME = auto()
    PARSING_NAME_COMPLETED = auto()
    WAITING_FOR_ARGUMENTS_START = auto()
    PARSING_ARGUMENTS = auto()
    PARSING_ARGUMENTS_COMPLETED = auto()
    TOOL_COMPLETE = auto()
    ALL_TOOLS_COMPLETE = auto()


class MistralToolCall(ToolCall):
    """ToolCall with a Mistral-compatible random alphanumeric id."""

    id: str = Field(default_factory=lambda: MistralToolCall.generate_random_id())

    @staticmethod
    def generate_random_id() -> str:
        # Mistral Tool Call Ids must be alphanumeric with a length of 9.
        # https://github.com/mistralai/mistral-common/blob/21ee9f6cee3441e9bb1e6ed2d10173f90bd9b94b/src/mistral_common/protocol/instruct/validator.py#L299
        return "".join(choices(_ALPHANUMERIC, k=9))

    @staticmethod
    def is_valid_id(id: str) -> bool:
        return id.isalnum() and len(id) == 9


def _is_pre_v11_tokeniser(model_tokenizer: TokenizerLike) -> bool:
    if is_mistral_tokenizer(model_tokenizer):
        return model_tokenizer.version < 11
    vocab: dict[str, int] = getattr(model_tokenizer, "get_vocab", lambda: {})()
    return _ARGS not in vocab


@functools.cache
def mistral_config(
    *,
    reasoning_encoding: Literal["special_token", "text", "none"],
    name: str = "mistral",
) -> ParserEngineConfig:
    """Return a :class:`ParserEngineConfig` for the Mistral output format.

    Args:
        reasoning_encoding: Reasoning token format.
            ``"special_token"`` – v13+ ``[THINK]``/``[/THINK]`` special
            tokens placed in both ``terminals`` and ``token_id_terminals``.
            ``"text"`` – v11 ``<think>``/``</think>`` plain text placed in
            ``terminals`` only; no ``token_id_terminals`` for think tokens.
            ``"none"`` – no reasoning support.
        name: Name embedded in the returned config (used for debugging).

    Returns:
        A frozen :class:`ParserEngineConfig` with ``initial_state=CONTENT``.
    """
    if reasoning_encoding == "special_token":
        think_start = _THINK_START_SPECIAL
        think_end = _THINK_END_SPECIAL
        reasoning_terminals: dict[str, str] = {
            "THINK_START": think_start,
            "THINK_END": think_end,
        }
        reasoning_token_id_terminals: dict[str, str] = {
            "THINK_START": think_start,
            "THINK_END": think_end,
        }
    elif reasoning_encoding == "text":
        think_start = _THINK_START_TEXT
        think_end = _THINK_END_TEXT
        reasoning_terminals = {
            "THINK_START": think_start,
            "THINK_END": think_end,
        }
        # Text think markers have no token-id terminals.
        reasoning_token_id_terminals = {}
    else:
        reasoning_terminals = {}
        reasoning_token_id_terminals = {}

    if reasoning_encoding != "none":
        reasoning_transitions: dict[tuple[ParserState, str], Transition] = {
            (ParserState.CONTENT, "THINK_START"): Transition(
                ParserState.REASONING,
                (),
            ),
            # Absorb a duplicate/re-emitted THINK_START inside reasoning.
            (ParserState.REASONING, "THINK_START"): Transition(
                ParserState.REASONING,
                (),
            ),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                (EventType.REASONING_END,),
            ),
            # Absorb stray THINK_END that arrives after reasoning ended.
            (ParserState.CONTENT, "THINK_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
            # [TOOL_CALLS] directly from reasoning implicitly ends it.
            (ParserState.REASONING, "TOOL_CALLS"): Transition(
                ParserState.TOOL_NAME,
                (EventType.REASONING_END, EventType.TOOL_CALL_START),
            ),
        }
    else:
        reasoning_transitions = {}

    return ParserEngineConfig(
        name=name,
        initial_state=ParserState.CONTENT,
        terminals={
            **reasoning_terminals,
            "TOOL_CALLS": _TOOL_CALLS,
            "ARGS": _ARGS,
            "OPEN_BRACE": _OPEN_BRACE,
        },
        token_id_terminals={
            **reasoning_token_id_terminals,
            "TOOL_CALLS": _TOOL_CALLS,
            "ARGS": _ARGS,
        },
        transitions={
            **reasoning_transitions,
            # A tool call from content implicitly ends reasoning when enabled.
            (ParserState.CONTENT, "TOOL_CALLS"): Transition(
                ParserState.TOOL_NAME,
                (EventType.REASONING_END, EventType.TOOL_CALL_START)
                if reasoning_encoding != "none"
                else (EventType.TOOL_CALL_START,),
            ),
            # NAME→ARGS via explicit [ARGS] separator (v11+): consumed, no events.
            (ParserState.TOOL_NAME, "ARGS"): Transition(
                ParserState.TOOL_ARGS,
                (),
            ),
            # NAME→ARGS via "{": carried as ARG_VALUE_CHUNK so the opening
            # brace lands in the JSON argument buffer (fallback for name{args}).
            (ParserState.TOOL_NAME, "OPEN_BRACE"): Transition(
                ParserState.TOOL_ARGS,
                (EventType.ARG_VALUE_CHUNK,),
            ),
            # Parallel tool calls: next [TOOL_CALLS] ends current call.
            (ParserState.TOOL_ARGS, "TOOL_CALLS"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_END, EventType.TOOL_CALL_START),
            ),
        },
        stream_arg_deltas=True,
        tool_args_json=True,
        strip_trailing_reasoning_whitespace=True,
        drop_whitespace_only_content_before_tools=True,
    )


class MistralParser(ParserEngine):
    """Mistral parser: engine-based reasoning + ``[TOOL_CALLS]`` tool calls.

    Reasoning encoding is auto-detected from the tokenizer:

    - ``"special_token"`` – ``[THINK]`` present in vocab (v13+).
    - ``"text"`` – tokenizer supports grammar but has no ``[THINK]`` (v11).
    - ``"none"`` – no grammar support; reasoning disabled.

    Tool calls use the ``[TOOL_CALLS]func_name{...}`` format.  The opening
    ``{`` doubles as the NAME→ARGS separator and is included in the argument
    buffer via an ``ARG_VALUE_CHUNK`` event on the transition (mirrors
    kimi_k2, JSON args, `tool_args_json=True`).

    When the tokenizer does not support grammar (``_reasoning_encoding ==
    "none"``), the legacy ``[TOOL_CALLS]``-based extraction path is used
    instead of the declarative engine, handling both pre-v11 JSON-array and
    v11+ ``funcname{args}`` formats.
    """

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        vocab = tokenizer.get_vocab()
        self._reasoning_encoding: Literal["special_token", "text", "none"]
        if _THINK_START_SPECIAL in vocab:
            self._reasoning_encoding = "special_token"
        elif getattr(tokenizer, "supports_grammar", False):
            self._reasoning_encoding = "text"
        else:
            self._reasoning_encoding = "none"

        kwargs.setdefault(
            "parser_engine_config",
            mistral_config(reasoning_encoding=self._reasoning_encoding),
        )
        super().__init__(tokenizer, tools, **kwargs)

        self._tool_calls_token_id: int | None = self.vocab.get(_TOOL_CALLS)

        # EOS stripped in _feed by token id; keep text for the endswith check.
        eos_id = getattr(tokenizer, "eos_token_id", None)
        self._eos_id: int | None = eos_id
        self._eos_text: str | None = None
        if eos_id is not None:
            try:
                self._eos_text = tokenizer.decode([eos_id]) or None
            except Exception:
                self._eos_text = None

        # Tool calls use the legacy parser for all tokenizer versions;
        # reasoning is handled by the engine.
        self.bot_token: str = _TOOL_CALLS
        self.bot_token_id: int | None = self._tool_calls_token_id
        if self.bot_token_id is None:
            raise RuntimeError(
                "Mistral parser could not locate the tool call token in the tokenizer!"
            )

        # Legacy tool-call streaming state.
        self.prev_tool_call_arr: list[dict[str, Any]] = []
        self.current_tool_id: int = -1
        self.streaming_state: StreamingState = StreamingState.WAITING_FOR_TOOL_START
        self.tool_call_started: bool = False
        self.current_tool_name: str | None = None
        self.current_tool_mistral_id: str | None = None
        self.starting_new_tool: bool = False
        self.streamed_args_for_tool: list[str] = []
        self._is_pre_v11: bool = _is_pre_v11_tokeniser(tokenizer)
        self.parse_coro = None
        if self._is_pre_v11:
            self.parse_coro = ijson.parse_coro(
                self.update_stream_state_pre_v11_tokenizer()
            )
        self.tool_call_regex = re.compile(r"\[{.*}\]", re.DOTALL)

    def _feed(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
    ) -> list[SemanticEvent]:
        # EOS is a stop token, not output; strip it before the engine emits it.
        if self._eos_id is not None and self._eos_id in delta_token_ids:
            new_ids = [tid for tid in delta_token_ids if tid != self._eos_id]
            if self._eos_text and delta_text.endswith(self._eos_text):
                delta_text = delta_text[: -len(self._eos_text)]
            return super()._feed(delta_text, new_ids)
        return super()._feed(delta_text, delta_token_ids)

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        if not isinstance(request, ResponsesRequest) and request._grammar_from_parser:
            return request
        so_non_supported_attributes = [
            "regex",
            "choice",
            "grammar",
            # whitespace_pattern is not a constraint type but an option;
            # Mistral grammar factory does not support it.
            "whitespace_pattern",
            "structural_tag",
        ]
        any_so_non_supported_active = request.structured_outputs is not None and any(
            getattr(request.structured_outputs, attribute) is not None
            for attribute in so_non_supported_attributes
        )
        response_format_non_supported_active = (
            isinstance(request, ResponsesRequest)
            or request.response_format is not None
            and request.response_format.type == "structural_tag"
        )

        if (
            not is_mistral_tokenizer(self.model_tokenizer)
            or isinstance(request, ResponsesRequest)
            or not self.model_tokenizer.supports_grammar
            or any_so_non_supported_active
            or response_format_non_supported_active
        ):
            request = super().adjust_request(request)
            if request.tools and request.tool_choice != "none":
                # Keep special tokens so the [TOOL_CALLS] marker
                # survives for tool detection.
                request.skip_special_tokens = False
            return request

        json_schema: dict[str, Any] | None = None
        if request.structured_outputs is not None:
            if request.structured_outputs.json_object is not None:
                json_schema = _DEFAULT_JSON_SCHEMA
            elif request.structured_outputs.json is not None:
                if isinstance(request.structured_outputs.json, str):
                    json_schema = json.loads(request.structured_outputs.json)
                else:
                    json_schema = request.structured_outputs.json
            else:
                raise ValueError(
                    "Unsupported request.structured_outputs for MistralParser. "
                    "Only `json` and `json_object` are supported."
                )
        elif (
            request.response_format is not None
            and request.response_format.type != "text"
        ):
            if request.response_format.type == "json_object":
                json_schema = _DEFAULT_JSON_SCHEMA
            elif request.response_format.type == "json_schema":
                if request.response_format.json_schema is not None:
                    json_schema = request.response_format.json_schema.json_schema
                else:
                    json_schema = _DEFAULT_JSON_SCHEMA
            else:
                raise ValueError(
                    "MistralParser only accepts `text`, `json_object` or "
                    f"`json_schema`, got {request.response_format=}"
                )
            request.response_format = None

        grammar_factory = self.model_tokenizer.grammar_factory

        # Rendering grammar is cached in mistral-common given tools, template and mode.
        template = grammar_factory.select_jinja_template()

        mistral_tools = (
            [MistralTool.from_openai(tool.model_dump()) for tool in request.tools]
            if request.tools is not None
            else None
        )

        tool_choice: MistralToolChoice
        match request.tool_choice:
            case "none" | "auto" | "required":
                tool_choice = MistralToolChoiceEnum(request.tool_choice)
            case None:
                tool_choice = MistralToolChoiceEnum.auto
            # _ == Named tool choice
            case _:
                tool_choice = MistralNamedToolChoice.model_validate(
                    {
                        "type": "function",
                        "function": {"name": request.tool_choice.function.name},
                    }
                )

        match tool_choice, json_schema is not None:
            case MistralToolChoiceEnum.none, True:
                lark_grammar = grammar_factory.get_lark_for_json_schema(
                    template=template, json_schema=json_schema
                )
            case _, _:
                lark_grammar = grammar_factory.get_lark_from_jinja(
                    template=template,
                    mode=tool_choice,
                    tools=mistral_tools,
                    json_schema=json_schema,
                    parallel_tool_calls=request.parallel_tool_calls,
                    json_only=False,
                )

        request.structured_outputs = StructuredOutputsParams(grammar=lark_grammar)
        request._grammar_from_parser = True
        return request

    def _ensure_tool_id(self, slot: ToolCallSlot, name: str) -> None:
        """Assign a Mistral-compatible 9-char alphanumeric id to `slot`."""
        if not slot.id:
            slot.id = MistralToolCall.generate_random_id()

    def extract_tool_calls_from_content(
        self,
        content: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if self._is_pre_v11:
            return self._legacy_extract_tool_calls(content, request)
        return super().extract_tool_calls_from_content(content, request)

    def _legacy_extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Pre-v11 non-streaming extraction.

        Handles the ``[TOOL_CALLS][{...}]`` JSON-array format.
        """
        if self.bot_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        content_and_raw_tool_calls = model_output.split(self.bot_token)
        content = content_and_raw_tool_calls[0]
        raw_tool_calls = content_and_raw_tool_calls[1:]

        # pre-v11: content[BOT] [{tool_call1},{tool_call2}]
        if len(raw_tool_calls) != 1:
            raise ValueError(
                "Only one BOT token should have been outputted, "
                f"but got {model_output}."
            )
        stringified_tool_calls = raw_tool_calls[0].strip()
        try:
            # Use raw_decode to parse the first valid JSON value,
            # ignoring trailing tokens the model may emit after
            # the tool call array.
            tool_calls, _ = json.JSONDecoder().raw_decode(stringified_tool_calls)
        except json.JSONDecodeError:
            try:
                raw_tool_call = self.tool_call_regex.findall(stringified_tool_calls)[0]
                tool_calls = json.loads(raw_tool_call)
                tool_calls = [
                    {
                        "name": tool_call["name"],
                        "arguments": json.dumps(
                            tool_call.get("arguments", {}),
                            ensure_ascii=False,
                        ),
                    }
                    for tool_call in tool_calls
                ]
            except (IndexError, json.JSONDecodeError):
                logger.exception("Error in extracting tool call from response.")
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=stringified_tool_calls,
                )
        else:
            tool_calls = [
                {
                    "name": tool_call["name"],
                    "arguments": json.dumps(
                        tool_call.get("arguments", {}),
                        ensure_ascii=False,
                    ),
                }
                for tool_call in tool_calls
            ]

        mistral_tool_calls: list[MistralToolCall] = [
            MistralToolCall(
                type="function",
                function=FunctionCall(
                    name=tool_call["name"],
                    arguments=tool_call.get("arguments", "{}"),
                ),
            )
            for tool_call in tool_calls
        ]

        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=mistral_tool_calls,
            content=content if content.strip() else None,
        )

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
        if not self._is_pre_v11:
            return super().extract_tool_calls_streaming(
                previous_text,
                current_text,
                delta_text,
                previous_token_ids,
                current_token_ids,
                delta_token_ids,
                request,
            )
        # Pre-v11: latch on [TOOL_CALLS] then delegate to legacy state machine.
        if self.bot_token_id in delta_token_ids or self.bot_token in delta_text:
            self.tool_call_started = True
        if not self.tool_call_started:
            return DeltaMessage(content=delta_text)
        try:
            return self._extract_tool_calls_streaming_pre_v11_tokenizer(
                delta_text=delta_text,
                delta_token_ids=delta_token_ids,
            )
        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            return None

    @ijson.coroutine
    def update_stream_state_pre_v11_tokenizer(self):
        while True:
            (prefix, event, value) = yield

            if prefix == "item" and event == "start_map":
                self.streaming_state = StreamingState.WAITING_FOR_TOOL_KEY
                self.starting_new_tool = True
            if prefix == "item" and event == "map_key" and value == "name":
                self.streaming_state = StreamingState.PARSING_NAME
            if prefix == "item.name" and event == "string":
                self.current_tool_name = value
                self.streaming_state = StreamingState.PARSING_NAME_COMPLETED
            if prefix == "item" and event == "map_key" and value == "arguments":
                self.streaming_state = StreamingState.WAITING_FOR_ARGUMENTS_START
            if prefix == "item.arguments" and event == "start_map":
                self.streaming_state = StreamingState.PARSING_ARGUMENTS
            if prefix == "item.arguments" and event == "end_map":
                self.streaming_state = StreamingState.PARSING_ARGUMENTS_COMPLETED
            if prefix == "item" and event == "end_map":
                self.streaming_state = StreamingState.TOOL_COMPLETE
            if prefix == "" and event == "end_array":
                self.streaming_state = StreamingState.ALL_TOOLS_COMPLETE

    def _extract_tool_calls_streaming_pre_v11_tokenizer(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """Extract tool calls for pre-v11 Mistral models.

        Handles ``[TOOL_CALLS][{"name": "add", "arguments":{"a": 3.5}}]``.
        """
        assert self.parse_coro is not None
        content = None
        delta_tool_calls: list[DeltaToolCall] = []
        current_tool_call: DeltaToolCall = DeltaToolCall(
            index=self.current_tool_id, type="function"
        )
        current_tool_call_modified = False
        if self.bot_token_id in delta_token_ids or self.bot_token in delta_text:
            # this is the first tool call
            if not delta_text.startswith(self.bot_token):
                content = delta_text.split(self.bot_token)[0]
            delta_text = "".join(delta_text.split(self.bot_token)[1:])

        # ijson gives no text index per event, so split the delta manually
        # to know where each event is emitted from.
        while len(delta_text) > 0:
            streaming_state_before_parse = self.streaming_state

            if self.streaming_state == StreamingState.WAITING_FOR_TOOL_START:
                delta_to_be_parsed, delta_text = self._split_delta(
                    delta_text=delta_text,
                    stop_after_opening_curly_braces=1,
                )
            elif self.streaming_state == StreamingState.WAITING_FOR_TOOL_KEY:
                delta_to_be_parsed, delta_text = self._split_delta(
                    delta_text=delta_text,
                    stop_after_colon=1,
                    stop_after_opening_curly_braces=1,
                )
            elif self.streaming_state == StreamingState.PARSING_NAME:
                delta_to_be_parsed, delta_text = self._split_delta(
                    delta_text=delta_text,
                    stop_after_comma=1,
                    stop_after_closing_brackets=1,
                )
            elif self.streaming_state == StreamingState.WAITING_FOR_ARGUMENTS_START:
                delta_to_be_parsed, delta_text = self._split_delta(
                    delta_text=delta_text,
                    stop_after_opening_curly_braces=1,
                )
            elif self.streaming_state == StreamingState.PARSING_ARGUMENTS:
                delta_to_be_parsed, delta_text = self._split_delta(
                    delta_text=delta_text,
                    stop_after_closing_curly_braces=1,
                )
            elif self.streaming_state in [
                StreamingState.PARSING_ARGUMENTS_COMPLETED,
                StreamingState.PARSING_NAME_COMPLETED,
            ]:
                delta_to_be_parsed, delta_text = self._split_delta(
                    delta_text=delta_text,
                    stop_after_closing_curly_braces=1,
                    stop_after_closing_brackets=1,
                )
            elif self.streaming_state == StreamingState.TOOL_COMPLETE:
                delta_to_be_parsed, delta_text = self._split_delta(
                    delta_text=delta_text,
                    stop_after_opening_curly_braces=1,
                    stop_after_closing_brackets=1,
                )
            elif self.streaming_state == StreamingState.ALL_TOOLS_COMPLETE:
                content = delta_text
                delta_text = ""
            else:
                delta_to_be_parsed = delta_text
                delta_text = ""

            if self.streaming_state != StreamingState.ALL_TOOLS_COMPLETE:
                self.parse_coro.send(delta_to_be_parsed.encode("utf-8"))

            # start_map is the authoritative new-tool signal and survives
            # batched deltas, unlike comparing pre/post streaming states.
            if self.starting_new_tool:
                self.starting_new_tool = False
                if current_tool_call_modified:
                    if self.current_tool_mistral_id is not None:
                        current_tool_call.id = self.current_tool_mistral_id
                        self.current_tool_mistral_id = None
                    self._track_streamed_args_pre_v11(current_tool_call)
                    delta_tool_calls.append(current_tool_call)
                current_tool_call_modified = False
                self.current_tool_id += 1
                self.streamed_args_for_tool.append("")
                self.prev_tool_call_arr.append({})
                self.current_tool_mistral_id = MistralToolCall.generate_random_id()
                current_tool_call = DeltaToolCall(
                    index=self.current_tool_id,
                    type="function",
                )
            if current_tool_call.function is None:
                current_tool_call.function = DeltaFunctionCall()

            if self.current_tool_name is not None:
                current_tool_call_modified = True
                current_tool_call.function.name = self.current_tool_name
                self.prev_tool_call_arr[self.current_tool_id]["name"] = (
                    self.current_tool_name
                )
                self.current_tool_name = None
            if self.streaming_state == StreamingState.PARSING_NAME_COMPLETED:
                self.streaming_state = StreamingState.WAITING_FOR_TOOL_KEY
            if self.streaming_state in [
                StreamingState.PARSING_ARGUMENTS,
                StreamingState.PARSING_ARGUMENTS_COMPLETED,
            ]:
                if self.streaming_state == StreamingState.PARSING_ARGUMENTS_COMPLETED:
                    self.streaming_state = StreamingState.WAITING_FOR_TOOL_KEY
                current_tool_call_modified = True
                if current_tool_call.function.arguments is None:
                    current_tool_call.function.arguments = delta_to_be_parsed
                else:
                    current_tool_call.function.arguments += delta_to_be_parsed
                if streaming_state_before_parse != StreamingState.PARSING_ARGUMENTS:
                    # It's the first chunk of arg. let's lstrip it
                    current_tool_call.function.arguments = (
                        current_tool_call.function.arguments.lstrip()
                    )

        if current_tool_call_modified:
            if self.current_tool_mistral_id is not None:
                current_tool_call.id = self.current_tool_mistral_id
                self.current_tool_mistral_id = None
            self._track_streamed_args_pre_v11(current_tool_call)
            delta_tool_calls.append(current_tool_call)

        if content or len(delta_tool_calls) > 0:
            delta_message = DeltaMessage()
            if content:
                delta_message.content = content
            if len(delta_tool_calls) > 0:
                delta_message.tool_calls = delta_tool_calls
            return delta_message
        else:
            if self.streaming_state == StreamingState.ALL_TOOLS_COMPLETE:
                return DeltaMessage()
            else:
                return None

    def _track_streamed_args_pre_v11(self, tool_call: DeltaToolCall) -> None:
        r"""Accumulate `tool_call` arguments into the streaming state."""
        if tool_call.function is not None and tool_call.function.arguments is not None:
            self.streamed_args_for_tool[self.current_tool_id] += (
                tool_call.function.arguments
            )
            self.prev_tool_call_arr[self.current_tool_id]["arguments"] = (
                self.streamed_args_for_tool[self.current_tool_id]
            )

    def _split_delta(
        self,
        delta_text: str,
        stop_after_quotes: int = -1,
        stop_after_opening_curly_braces: int = -1,
        stop_after_closing_curly_braces: int = -1,
        stop_after_closing_brackets: int = -1,
        stop_after_colon: int = -1,
        stop_after_comma: int = -1,
    ) -> tuple[str, str]:
        delta_to_be_parsed = ""
        for i, c in enumerate(delta_text):
            if c in ['"', "'"]:
                delta_to_be_parsed += c
                stop_after_quotes -= 1
                if stop_after_quotes == 0:
                    return (delta_to_be_parsed, delta_text[i + 1 :])
            elif c == "{":
                delta_to_be_parsed += c
                stop_after_opening_curly_braces -= 1
                if stop_after_opening_curly_braces == 0:
                    return (delta_to_be_parsed, delta_text[i + 1 :])
            elif c == "}":
                delta_to_be_parsed += c
                stop_after_closing_curly_braces -= 1
                if stop_after_closing_curly_braces == 0:
                    return (delta_to_be_parsed, delta_text[i + 1 :])
            elif c == "]":
                delta_to_be_parsed += c
                stop_after_closing_brackets -= 1
                if stop_after_closing_brackets == 0:
                    return (delta_to_be_parsed, delta_text[i + 1 :])
            elif c == ":":
                delta_to_be_parsed += c
                stop_after_colon -= 1
                if stop_after_colon == 0:
                    return (delta_to_be_parsed, delta_text[i + 1 :])
            elif c == ",":
                delta_to_be_parsed += c
                stop_after_comma -= 1
                if stop_after_comma == 0:
                    return (delta_to_be_parsed, delta_text[i + 1 :])
            else:
                delta_to_be_parsed += c

        return (delta_to_be_parsed, "")

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        if self._reasoning_encoding == "none":
            return True
        if super().is_reasoning_end(input_ids):
            return True
        # [TOOL_CALLS] acts as an implicit reasoning-end marker
        if self.bot_token_id is not None:
            reasoning_start_id = self._reasoning_start_token_id
            for i in range(len(input_ids) - 1, -1, -1):
                if (
                    reasoning_start_id is not None
                    and input_ids[i] == reasoning_start_id
                ):
                    return False
                if input_ids[i] == self.bot_token_id:
                    return True
        return False

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        if self._reasoning_encoding == "none":
            return None, model_output
        return super().extract_reasoning(model_output, request)

    def _handle_arg_chunk(
        self,
        event: SemanticEvent,
        deltas: list[DeltaToolCall],
    ) -> None:
        """Emit the opening ``{`` as an arg delta when the name is first sent.

        When the TOOL_NAME→TOOL_ARGS transition fires, ``{`` arrives as an
        ARG_VALUE_CHUNK before ``name_sent`` is True.  The parent emits the
        name delta but not the ``{`` arg delta.  This override re-emits the
        current chunk so streaming clients receive a valid JSON prefix.
        """
        idx = event.tool_index
        name_sent_before = (
            0 <= idx < len(self._tool_slots) and self._tool_slots[idx].name_sent
        )
        super()._handle_arg_chunk(event, deltas)
        if (
            event.value
            and not name_sent_before
            and 0 <= idx < len(self._tool_slots)
            and self._tool_slots[idx].name_sent
        ):
            deltas.append(
                DeltaToolCall(
                    index=idx,
                    function=DeltaFunctionCall(arguments=event.value),
                )
            )

    def _extract_args_json(self, raw_args: str, func_name: str) -> str:
        return raw_args.strip() or "{}"
