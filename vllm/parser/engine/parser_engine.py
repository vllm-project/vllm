# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parser engine base that handles both reasoning and tool call
extraction with a single :class:`StreamingParserEngine`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from functools import cached_property
from typing import TYPE_CHECKING

import regex as re

from vllm.entrypoints.chat_utils import get_tool_call_id_type, make_tool_call_id
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.parser.abstract_parser import Parser, StreamState
from vllm.parser.engine.events import EventType, SemanticEvent
from vllm.parser.engine.parser_engine_config import ParserEngineConfig, ParserState
from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine
from vllm.tool_parsers.utils import (
    coerce_to_schema_type,
    extract_types_from_schema,
    find_tool_name,
    find_tool_properties,
)

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

logger = init_logger(__name__)


class ToolCallSlot:
    __slots__ = (
        "id",
        "name",
        "_args_parts",
        "_args_joined",
        "name_sent",
        "string_keys",
        "streamed_json",
    )

    def __init__(self) -> None:
        self.id: str = ""
        self.name: str = ""
        self._args_parts: list[str] = []
        self._args_joined: str | None = ""
        self.name_sent: bool = False
        self.string_keys: set[str] | None = None
        self.streamed_json: str = ""

    @property
    def args(self) -> str:
        if self._args_joined is None:
            self._args_joined = "".join(self._args_parts)
        return self._args_joined

    def append_args(self, value: str) -> None:
        self._args_parts.append(value)
        self._args_joined = None


class ParserEngine(Parser):
    """A :class:`Parser` backed by a single declarative engine config.

    Subclasses set the ``ParserEngineConfig`` in ``__init__`` to define the
    complete output format for a model (reasoning + tool calls).
    """

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        *,
        parser_engine_config: ParserEngineConfig,
        model_config=None,
        **kwargs,
    ) -> None:
        self.model_tokenizer = tokenizer
        self._tools = tools
        self._stream_state = StreamState(
            tool_call_id_type=(
                get_tool_call_id_type(model_config)
                if model_config is not None
                else "random"
            ),
        )
        self._reasoning_parser = None
        self._tool_parser = None
        self.parser_engine_config = parser_engine_config
        self._engine = StreamingParserEngine(
            parser_engine_config, tokenizer, vocab=self.vocab
        )

        self._reasoning_ended: bool = False
        self._streaming_initialized: bool = False
        self._prompt_streaming_prepared: bool = False

        self._tool_slots: list[ToolCallSlot] = []
        self._deferred_content: str = ""
        self._deferred_reasoning: str = ""
        self._content_has_nonws: bool = False

        self._arg_converter = parser_engine_config.arg_converter
        self._arg_structural_chars = parser_engine_config.arg_structural_chars
        self._stream_arg_deltas = parser_engine_config.stream_arg_deltas
        self._strip_trailing_reasoning_ws = (
            parser_engine_config.strip_trailing_reasoning_whitespace
        )
        self._drop_ws_only_content_before_tools = (
            parser_engine_config.drop_whitespace_only_content_before_tools
        )
        self._strip_content_ws_with_tools = (
            parser_engine_config.strip_content_whitespace_with_tools
        )

        vocab = self.vocab
        self._reasoning_start_token_id: int | None = None
        self._reasoning_end_token_id: int | None = None

        start_text = parser_engine_config.token_id_terminals.get("THINK_START")
        end_text = parser_engine_config.token_id_terminals.get("THINK_END")
        if start_text:
            self._reasoning_start_token_id = vocab.get(start_text)
        if end_text:
            self._reasoning_end_token_id = vocab.get(end_text)

    @property
    def reasoning_start_str(self) -> str | None:
        return self.parser_engine_config.terminals.get("THINK_START")

    @property
    def reasoning_end_str(self) -> str | None:
        return self.parser_engine_config.terminals.get("THINK_END")

    @cached_property
    def vocab(self) -> dict[str, int]:
        return self.model_tokenizer.get_vocab()

    # ── Engine lifecycle ──────────────────────────────────────────────

    @property
    def skip_tool_parsing(self) -> bool:
        return self._engine.skip_tool_parsing

    @skip_tool_parsing.setter
    def skip_tool_parsing(self, value: bool) -> None:
        self._engine.skip_tool_parsing = value

    @property
    def reasoning_ended(self) -> bool:
        return self._reasoning_ended

    def initialize_streaming(
        self,
        initial_state: ParserState | None = None,
    ) -> None:
        if not self._streaming_initialized:
            self._streaming_initialized = True
            self._reset(initial_state=initial_state)

    def adjust_initial_state_from_prompt(self, prompt_token_ids: Sequence[int]) -> None:
        """See :meth:`ReasoningParser.adjust_initial_state_from_prompt`."""
        return

    def finish_streaming(self) -> DeltaMessage | None:
        events = self._engine.finish()
        if events or self._deferred_content:
            return self._events_to_delta(events, finished=True)
        return None

    def _reset(self, initial_state: ParserState | None = None) -> None:
        self._engine.reset(initial_state=initial_state)
        self._reasoning_ended = False
        self._tool_slots.clear()
        self._deferred_content = ""
        self._deferred_reasoning = ""
        self._content_has_nonws = False
        self._prompt_streaming_prepared = False

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request.skip_special_tokens = False
        return request

    def _preprocess_feed(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
    ) -> tuple[str, Sequence[int]]:
        return delta_text, delta_token_ids

    def _feed(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
    ) -> list[SemanticEvent]:
        delta_text, delta_token_ids = self._preprocess_feed(delta_text, delta_token_ids)
        return self._engine.feed(delta_text, delta_token_ids)

    # ── Schema-aware type correction ─────────────────────────────────

    @staticmethod
    def _coerce_value(value: object, schema: dict) -> tuple[object, bool]:
        """Coerce a single value according to its schema.

        Returns ``(coerced_value, changed)``.
        """
        if isinstance(value, str):
            types = extract_types_from_schema(schema)
            coerced = coerce_to_schema_type(value, types)
            if coerced is not value:
                return coerced, True
            return value, False

        if isinstance(value, dict):
            nested_props = schema.get("properties")
            if isinstance(nested_props, dict):
                _, changed = ParserEngine._coerce_dict(value, nested_props)
                return value, changed
            return value, False

        if isinstance(value, list):
            items_schema = schema.get("items")
            if isinstance(items_schema, dict):
                changed = False
                for i, item in enumerate(value):
                    coerced, item_changed = ParserEngine._coerce_value(
                        item, items_schema
                    )
                    if item_changed:
                        value[i] = coerced
                        changed = True
                return value, changed
            return value, False

        types = extract_types_from_schema(schema)
        as_str = json.dumps(value, ensure_ascii=False)
        coerced = coerce_to_schema_type(as_str, types)
        if coerced != value:
            return coerced, True
        return value, False

    @staticmethod
    def _coerce_dict(args: dict, properties: dict) -> tuple[dict, bool]:
        """Coerce all values in *args* using *properties* schemas."""
        changed = False
        for key, value in args.items():
            prop = properties.get(key)
            if not isinstance(prop, dict):
                continue
            coerced, val_changed = ParserEngine._coerce_value(value, prop)
            if val_changed:
                args[key] = coerced
                changed = True
        return args, changed

    @staticmethod
    def _safe_arg_prefix(json_str: str, string_keys: set[str] | None = None) -> str:
        """Return the prefix of *json_str* up to the last top-level value.

        Middle values (followed by a comma) are stable across streaming
        ticks and included.  The trailing value is excluded for non-string
        values because type coercion may change its serialised form between
        ticks, which would violate the ``startswith(prev)`` prefix invariant.
        String values for keys in ``string_keys`` are prefix-stable, so stream
        their unterminated content instead of buffering long arguments until
        the closing tag arrives.
        """
        last_colon = -1
        last_key: str | None = None
        pending_key: str | None = None
        in_string = False
        escape = False
        string_start = -1
        depth = 0
        for i, c in enumerate(json_str):
            if escape:
                escape = False
                continue
            if in_string:
                if c == "\\":
                    escape = True
                elif c == '"':
                    in_string = False
                    if depth == 1 and string_start >= 0:
                        pending_key = json_str[string_start + 1 : i]
                continue
            if c == '"':
                in_string = True
                string_start = i
            elif c in ("{", "["):
                depth += 1
            elif c in ("}", "]"):
                depth -= 1
            elif c == ":" and depth == 1:
                last_colon = i
                last_key = pending_key
                pending_key = None
        if last_colon < 0:
            return ""
        end = last_colon + 1
        while end < len(json_str) and json_str[end] in (" ", "\t", "\n", "\r"):
            end += 1
        if end >= len(json_str) or json_str[end] != '"':
            return json_str[:end]
        if string_keys is not None and last_key not in string_keys:
            return json_str[:end]

        escape = False
        for i in range(end + 1, len(json_str)):
            c = json_str[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                return json_str[:i]
        return json_str

    @staticmethod
    def _streamable_string_keys(properties: dict) -> set[str] | None:
        """Return keys whose trailing string values can safely stream.

        ``None`` means there is no schema, so all string values keep their
        JSON representation as strings.  With a schema, only fields that can
        remain strings are safe to emit before the value is closed; fields
        coerced to bool/number/null/object/array may serialize differently.
        """
        if not properties:
            return None

        streamable: set[str] = set()
        for key, schema in properties.items():
            if set(extract_types_from_schema(schema)) == {"string"}:
                streamable.add(key)
        return streamable

    def _fix_arg_types(self, args_json: str, func_name: str) -> str:
        """Correct parameter types using the tool schema.

        String values are coerced via :func:`coerce_to_schema_type`.
        Nested objects and arrays are recursed into when the schema
        defines ``properties`` or ``items``.  Without a schema, values
        stay as strings.
        """
        if not self._tools or not func_name:
            return args_json
        try:
            args = json.loads(args_json)
        except (json.JSONDecodeError, ValueError):
            return args_json
        if not isinstance(args, dict):
            return args_json

        properties = find_tool_properties(self._tools, func_name)
        if not properties:
            return args_json

        _, changed = self._coerce_dict(args, properties)

        if changed:
            return json.dumps(args, ensure_ascii=False)
        return args_json

    def _is_valid_tool_name(self, name: str) -> bool:
        if not self.parser_engine_config.validate_tool_names:
            return True
        if not self._tools:
            return True
        return find_tool_name(self._tools, name)

    # ── Private helpers ─────────────────────────────────────────────

    def _check_skip_tool_parsing(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> None:
        tools = getattr(request, "tools", None)
        if tools:
            self._tools = tools
        if not self.skip_tool_parsing:
            tool_choice = getattr(request, "tool_choice", None)
            if tool_choice == "none" and tools:
                self.skip_tool_parsing = True

    def _strip_content_whitespace(
        self,
        content: str,
        tools_called: bool,
    ) -> str | None:
        if tools_called:
            if self._strip_content_ws_with_tools:
                content = content.strip()
            elif self._drop_ws_only_content_before_tools and not content.strip():
                content = ""
        return content or None

    # ── Streaming: parse_delta ────────────────────────────────────────

    def parse_delta(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        request: ChatCompletionRequest | ResponsesRequest,
        prompt_token_ids: list[int] | None = None,
        *,
        finished: bool,
    ) -> DeltaMessage | None:
        self._initialize_history_tool_call_cnt(request)
        if not self._prompt_streaming_prepared and prompt_token_ids is not None:
            # NOTE: call the hook BEFORE setting the flag, because the hook
            # may invoke ``_reset`` (e.g. via ``initialize_streaming``) which
            # clears ``_prompt_streaming_prepared``.
            self.adjust_initial_state_from_prompt(prompt_token_ids)
            self._prompt_streaming_prepared = True
        self._check_skip_tool_parsing(request)
        events = self._feed(delta_text, delta_token_ids)
        if finished:
            events.extend(self._engine.finish())
        result = self._events_to_delta(events, finished=finished)
        return self._strip_trailing_reasoning(result)

    def _strip_trailing_reasoning(
        self,
        delta: DeltaMessage | None,
    ) -> DeltaMessage | None:
        """Strip trailing whitespace from reasoning, deferring it until we
        know whether more reasoning follows or reasoning has ended.

        Runs in ``parse_delta`` *after* ``_events_to_delta`` (and any
        subclass overrides) so that overrides see the raw reasoning text.

        Gated by ``strip_trailing_reasoning_whitespace``; when disabled,
        passes through unchanged.
        """
        if not self._strip_trailing_reasoning_ws:
            return delta
        if delta is not None and delta.reasoning is not None:
            combined = self._deferred_reasoning + delta.reasoning
            trimmed = combined.rstrip()
            self._deferred_reasoning = combined[len(trimmed) :]
            delta.reasoning = trimmed or None
            if (
                delta.reasoning is None
                and delta.content is None
                and not delta.tool_calls
            ):
                return None
        elif self._deferred_reasoning and self._reasoning_ended:
            self._deferred_reasoning = ""
        return delta

    # ── Non-streaming: extract_reasoning ──────────────────────────────

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        self._reset()
        events = self._feed(model_output, [])
        events.extend(self._engine.finish())

        reasoning_parts: list[str] = []
        content_parts: list[str] = []

        for event in events:
            if event.type == EventType.REASONING_CHUNK:
                reasoning_parts.append(event.value)
            elif event.type == EventType.TEXT_CHUNK:
                content_parts.append(event.value)
            elif event.type == EventType.REASONING_END:
                self._reasoning_ended = True

        raw_reasoning = "".join(reasoning_parts)
        if self._strip_trailing_reasoning_ws:
            raw_reasoning = raw_reasoning.rstrip()
        reasoning = raw_reasoning or None
        content = "".join(content_parts) or None
        return reasoning, content

    # ── Non-streaming: extract_reasoning_streaming ────────────────────

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        self.initialize_streaming()
        events = self._feed(delta_text, delta_token_ids)
        return self._strip_trailing_reasoning(self._events_to_delta(events))

    # ── Non-streaming: extract_tool_calls ─────────────────────────────

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> ExtractedToolCallInformation:
        self._reset()
        self._streaming_initialized = True
        result = self.extract_tool_calls_streaming(
            previous_text="",
            current_text=model_output,
            delta_text=model_output,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )
        finish_delta = self.finish_streaming()
        return self._build_extracted_result(result, finish_delta)

    def extract_tool_calls_from_content(
        self,
        content: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from reasoning-stripped content.

        Unlike :meth:`extract_tool_calls` which re-parses the full model
        output, this method starts the parser engine in ``CONTENT`` state
        so it can parse content that has already had reasoning stripped.
        """
        self._check_skip_tool_parsing(request)
        _, parsed_content, tool_call_info = self._single_pass_parse(
            content,
            [],
            initial_state=ParserState.CONTENT,
        )
        if parsed_content is not None and tool_call_info.content is None:
            tool_call_info = ExtractedToolCallInformation(
                tools_called=tool_call_info.tools_called,
                tool_calls=tool_call_info.tool_calls,
                content=parsed_content,
            )
        return tool_call_info

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
        self.initialize_streaming()
        self._check_skip_tool_parsing(request)
        events = self._feed(delta_text, delta_token_ids)
        return self._strip_trailing_reasoning(self._events_to_delta(events))

    # ── Reasoning state queries ───────────────────────────────────────

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        end_id = self._reasoning_end_token_id
        start_id = self._reasoning_start_token_id
        if end_id is not None:
            if not input_ids:
                return self.parser_engine_config.initial_state != ParserState.REASONING
            for i in range(len(input_ids) - 1, -1, -1):
                if input_ids[i] == end_id:
                    return True
                if start_id is not None and input_ids[i] == start_id:
                    return False
            return False
        return self._reasoning_ended

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        end_id = self._reasoning_end_token_id
        if end_id is not None:
            for i in range(len(input_ids) - 1, -1, -1):
                if input_ids[i] == end_id:
                    return input_ids[i + 1 :]
        return input_ids

    def get_streaming_fallback_content(
        self,
        text: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> str | None:
        return None

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        start_id = self._reasoning_start_token_id
        end_id = self._reasoning_end_token_id
        if start_id is None or end_id is None:
            return 0
        count = 0
        depth = 0
        for token_id in token_ids:
            if token_id == start_id:
                depth += 1
                continue
            if token_id == end_id:
                if depth > 0:
                    depth -= 1
                continue
            if depth > 0:
                count += 1
        return count

    # ── Single-pass parse helper ────────────────────────────────────────

    def _single_pass_parse(
        self,
        text: str,
        token_ids: Sequence[int],
        initial_state: ParserState | None = None,
    ) -> tuple[str | None, str | None, ExtractedToolCallInformation]:
        """Reset, feed, finish, and extract results in one pass.

        Must be called as a unit — ``_events_to_delta`` populates tool
        state that ``_build_extracted_result`` reads.
        """
        self._reset(initial_state=initial_state)
        events = self._feed(text, token_ids)
        events.extend(self._engine.finish())

        delta = self._events_to_delta(events)
        tool_call_info = self._build_extracted_result()

        reasoning = delta.reasoning if delta else None
        if reasoning and self._strip_trailing_reasoning_ws:
            reasoning = reasoning.rstrip() or None

        content = delta.content if delta else None
        if content:
            content = self._strip_content_whitespace(
                content, tool_call_info.tools_called
            )

        return reasoning, content, tool_call_info

    # ── Non-streaming: parse ───────────────────────────────────────────

    def parse(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
        enable_auto_tools: bool = False,
        model_output_token_ids: Sequence[int] = (),
    ) -> tuple[str | None, str | None, list[FunctionCall] | None]:
        self._initialize_history_tool_call_cnt(request)
        self._check_skip_tool_parsing(request)
        reasoning, content, tool_call_info = self._single_pass_parse(
            model_output,
            model_output_token_ids,
        )

        tool_calls: list[FunctionCall] | None = None
        if tool_call_info.tools_called:
            tool_calls = [
                FunctionCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                )
                for tc in tool_call_info.tool_calls
            ]

        return reasoning, content, tool_calls

    # ── Event-to-delta conversion ─────────────────────────────────────

    def _events_to_delta(
        self,
        events: list[SemanticEvent],
        finished: bool = False,
    ) -> DeltaMessage | None:
        if not events and not self._deferred_content:
            return None

        tool_call_deltas: list[DeltaToolCall] = []
        content_parts: list[str] = []
        reasoning_parts: list[str] = []

        seen_tool_event = False
        for event in events:
            match event.type:
                case EventType.TEXT_CHUNK:
                    if seen_tool_event:
                        self._deferred_content += event.value
                    else:
                        content_parts.append(event.value)
                case EventType.REASONING_CHUNK:
                    reasoning_parts.append(event.value)
                case EventType.REASONING_END:
                    self._reasoning_ended = True
                case EventType.TOOL_CALL_START:
                    seen_tool_event = True
                    self._ensure_slot(event.tool_index)
                case EventType.TOOL_NAME:
                    seen_tool_event = True
                    self._handle_tool_name(event)
                case EventType.ARG_VALUE_CHUNK:
                    seen_tool_event = True
                    self._handle_arg_chunk(event, tool_call_deltas)
                case EventType.TOOL_CALL_END:
                    seen_tool_event = True
                    self._handle_tool_end(event, tool_call_deltas)
                case EventType.REASONING_START:
                    pass  # no delta-level effect

        if len(tool_call_deltas) > 1:
            tool_call_deltas = self._coalesce_tool_call_deltas(tool_call_deltas)

        if self._deferred_content and (not seen_tool_event or not tool_call_deltas):
            content_parts.insert(0, self._deferred_content)
            self._deferred_content = ""

        content_str = "".join(content_parts)

        if self._content_has_nonws:
            pass
        elif content_str:
            stripped = content_str.strip()
            if stripped:
                self._content_has_nonws = True
            elif self._tool_slots:
                if self._drop_ws_only_content_before_tools:
                    content_str = ""
            elif not finished:
                self._deferred_content = content_str
                content_str = ""

        content = content_str or None
        reasoning = "".join(reasoning_parts) or None

        if content or tool_call_deltas or reasoning:
            kwargs: dict[str, object] = {}
            if content is not None:
                kwargs["content"] = content
            if reasoning is not None:
                kwargs["reasoning"] = reasoning
            if tool_call_deltas:
                kwargs["tool_calls"] = tool_call_deltas
            return DeltaMessage(**kwargs)
        return None

    def _ensure_slot(self, idx: int) -> None:
        while len(self._tool_slots) <= idx:
            self._tool_slots.append(ToolCallSlot())

    def _ensure_tool_id(self, slot: ToolCallSlot, name: str) -> None:
        if not slot.id:
            state = self._stream_state
            slot.id = make_tool_call_id(
                id_type=state.tool_call_id_type,
                func_name=name,
                idx=state.history_tool_call_cnt,
            )
            state.history_tool_call_cnt += 1

    def _handle_tool_name(self, event: SemanticEvent) -> None:
        idx = event.tool_index
        self._tool_slots[idx].name += event.value

    def _emit_name_delta(
        self,
        idx: int,
        deltas: list[DeltaToolCall],
        name: str | None,
    ) -> None:
        if not name or not self._is_valid_tool_name(name):
            return
        slot = self._tool_slots[idx]
        slot.name = name
        slot.name_sent = True
        slot.string_keys = self._streamable_string_keys(
            find_tool_properties(self._tools, name)
        )
        self._ensure_tool_id(slot, name)
        deltas.append(
            DeltaToolCall(
                index=idx,
                id=slot.id,
                type="function",
                function=DeltaFunctionCall(name=name),
            )
        )

    def _handle_arg_chunk(
        self,
        event: SemanticEvent,
        deltas: list[DeltaToolCall],
    ) -> None:
        idx = event.tool_index
        slot = self._tool_slots[idx]
        if event.value:
            slot.append_args(event.value)

        if not slot.name_sent:
            if slot.name:
                self._emit_name_delta(idx, deltas, slot.name)
            elif event.value:
                # Name not yet known — try to extract from accumulated args
                name = self._try_extract_name(idx)
                self._emit_name_delta(idx, deltas, name)
        elif event.value:
            # Name already sent — emit arg delta
            arg_delta = self._compute_arg_delta(idx, event.value)
            if arg_delta:
                deltas.append(
                    DeltaToolCall(
                        index=idx,
                        function=DeltaFunctionCall(arguments=arg_delta),
                    )
                )

    def _handle_tool_end(
        self,
        event: SemanticEvent,
        deltas: list[DeltaToolCall],
    ) -> None:
        idx = event.tool_index
        if idx >= len(self._tool_slots):
            return

        remaining = self._flush_arg_converter(idx)
        slot = self._tool_slots[idx]

        if not slot.name_sent:
            name = slot.name or self._try_extract_name(idx)
            if name and self._is_valid_tool_name(name):
                slot.name = name
                slot.name_sent = True
                slot.string_keys = self._streamable_string_keys(
                    find_tool_properties(self._tools, name)
                )
                self._ensure_tool_id(slot, name)
                deltas.append(
                    DeltaToolCall(
                        index=idx,
                        id=slot.id,
                        type="function",
                        function=DeltaFunctionCall(
                            name=name,
                            arguments=remaining or "",
                        ),
                    )
                )
                remaining = None

        if remaining and slot.name_sent:
            deltas.append(
                DeltaToolCall(
                    index=idx,
                    function=DeltaFunctionCall(arguments=remaining),
                )
            )

    # ── Tool-call delta coalescing ──────────────────────────────────────

    @staticmethod
    def _coalesce_tool_call_deltas(
        deltas: list[DeltaToolCall],
    ) -> list[DeltaToolCall]:
        """Merge entries that share the same index into one per index."""
        merged: dict[int, DeltaToolCall] = {}
        for tc in deltas:
            existing = merged.get(tc.index)
            if existing is None:
                merged[tc.index] = tc
                continue
            if tc.id is not None and existing.id is None:
                existing.id = tc.id
            if tc.type is not None and existing.type is None:
                existing.type = tc.type
            if tc.function is not None:
                if existing.function is None:
                    existing.function = tc.function
                else:
                    if tc.function.name is not None and existing.function.name is None:
                        existing.function.name = tc.function.name
                    if tc.function.arguments is not None:
                        if existing.function.arguments is None:
                            existing.function.arguments = tc.function.arguments
                        else:
                            existing.function.arguments += tc.function.arguments
        if len(merged) == len(deltas):
            return deltas
        return list(merged.values())

    # ── Arg conversion helpers ─────────────────────────────────────────

    def _compute_arg_delta(self, idx: int, raw_delta: str) -> str | None:
        converter = self._arg_converter
        if converter is None:
            return raw_delta

        if not self._stream_arg_deltas:
            return None

        structural = self._arg_structural_chars
        if structural is not None and structural.isdisjoint(raw_delta):
            return None

        slot = self._tool_slots[idx]
        try:
            current_json = converter(slot.args, True)
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.debug("arg converter failed (streaming): %s", slot.args[:80])
            return None

        if not current_json:
            return None

        if slot.name:
            current_json = self._fix_arg_types(current_json, slot.name)

        prev = slot.streamed_json
        safe_json = self._safe_arg_prefix(current_json, slot.string_keys)

        if not safe_json or safe_json == prev:
            return None

        if prev:
            if not safe_json.startswith(prev):
                return None
            diff = safe_json[len(prev) :]
        else:
            diff = safe_json

        if diff:
            slot.streamed_json = safe_json
            return diff
        return None

    def _flush_arg_converter(self, idx: int) -> str | None:
        converter = self._arg_converter
        if converter is None:
            return None

        slot = self._tool_slots[idx]
        try:
            final_json = converter(slot.args, False)
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.debug("arg converter failed (flush): %s", slot.args[:80])
            return None

        if final_json:
            final_json = self._fix_arg_types(final_json, slot.name)

        prev = slot.streamed_json
        if final_json and len(final_json) > len(prev):
            if prev and not final_json.startswith(prev):
                return None
            diff = final_json[len(prev) :]
            slot.streamed_json = final_json
            return diff
        return None

    _NAME_RE = re.compile(r'"name"\s*:\s*"([^"]*)"')

    def _try_extract_name(self, idx: int) -> str | None:
        m = self._NAME_RE.search(self._tool_slots[idx].args)
        if m:
            name = m.group(1)
            if name:
                return name
        return None

    # ── Build ExtractedToolCallInformation ─────────────────────────────

    def _build_extracted_result(
        self,
        *deltas: DeltaMessage | None,
    ) -> ExtractedToolCallInformation:
        content_parts: list[str] = []
        for delta in deltas:
            if delta is not None and delta.content:
                content_parts.append(delta.content)

        tool_calls: list[ToolCall] = []
        for idx, slot in enumerate(self._tool_slots):
            if not slot.name and not slot.args:
                continue

            name = slot.name.strip()
            raw_body = slot.args

            if not name and raw_body.strip():
                name, args_json = self._extract_name_and_args(raw_body)
            elif raw_body.strip():
                converter = self._arg_converter
                if converter is not None:
                    try:
                        args_json = converter(raw_body, False)
                    except (json.JSONDecodeError, ValueError, TypeError):
                        logger.debug(
                            "arg converter failed (extract): %s", raw_body[:80]
                        )
                        args_json = self._extract_args_json(raw_body, name)
                else:
                    args_json = self._extract_args_json(raw_body, name)
            else:
                args_json = "{}"

            if name and self._is_valid_tool_name(name):
                self._ensure_tool_id(slot, name)
                args_json = self._fix_arg_types(args_json, name)
                tool_calls.append(
                    ToolCall(
                        id=slot.id,
                        function=FunctionCall(name=name, arguments=args_json),
                    )
                )

        content_str = "".join(content_parts)
        content = self._strip_content_whitespace(content_str, len(tool_calls) > 0)

        return ExtractedToolCallInformation(
            tools_called=len(tool_calls) > 0,
            tool_calls=tool_calls,
            content=content,
        )

    @staticmethod
    def _extract_args_value(parsed: dict) -> str | None:
        for key in ("arguments", "parameters"):
            if key in parsed:
                val = parsed[key]
                if isinstance(val, str):
                    return val
                return json.dumps(val, ensure_ascii=False)
        return None

    def _extract_name_and_args(
        self,
        raw_body: str,
    ) -> tuple[str, str]:
        raw_body = raw_body.strip()
        try:
            parsed = json.loads(raw_body)
        except json.JSONDecodeError:
            return "", raw_body

        if not isinstance(parsed, dict):
            return "", raw_body

        name = parsed.get("name", "")
        args = self._extract_args_value(parsed)
        if args is not None:
            return name, args

        without_name = {k: v for k, v in parsed.items() if k != "name"}
        return name, json.dumps(without_name, ensure_ascii=False)

    def _extract_args_json(self, raw_args: str, func_name: str) -> str:
        if not raw_args.strip():
            return "{}"
        _, args = self._extract_name_and_args(raw_args)
        return args
