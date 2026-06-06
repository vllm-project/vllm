# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parser engine base that handles both reasoning and tool call
extraction with a single :class:`StreamingParserEngine`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import regex as re

from vllm.entrypoints.chat_utils import make_tool_call_id
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
from vllm.tool_parsers.utils import find_tool_properties

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

logger = init_logger(__name__)


@dataclass
class ToolCallSlot:
    id: str = ""
    name: str = ""
    args: str = ""
    name_sent: bool = False
    streamed_json: str = ""


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
        **kwargs,
    ) -> None:
        self.model_tokenizer = tokenizer
        self._tools = tools
        self._stream_state = StreamState()
        self._reasoning_parser = None
        self._tool_parser = None
        self.parser_engine_config = parser_engine_config
        self._engine = StreamingParserEngine(
            parser_engine_config, tokenizer, vocab=self.vocab
        )

        self._reasoning_ended: bool = False
        self._streaming_initialized: bool = False

        self._tool_slots: list[ToolCallSlot] = []
        self._deferred_content: str = ""
        self._deferred_reasoning: str = ""
        self._content_has_nonws: bool = False

        self._arg_converter = parser_engine_config.arg_converter
        self._arg_structural_chars = parser_engine_config.arg_structural_chars
        self._strip_trailing_quotes = parser_engine_config.strip_trailing_quotes
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

    def _reset(self, initial_state: ParserState | None = None) -> None:
        self._engine.reset(initial_state=initial_state)
        self._reasoning_ended = False
        self._tool_slots.clear()
        self._deferred_content = ""
        self._deferred_reasoning = ""
        self._content_has_nonws = False

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request.skip_special_tokens = False
        return request

    # ── Schema-aware type correction ─────────────────────────────────

    def _fix_arg_types(self, args_json: str, func_name: str) -> str:
        """Correct parameter types wrongly coerced by the arg_converter.

        The arg_converter may blindly coerce e.g. ``"1"`` to ``1``.  If the
        tool schema declares the parameter as ``"string"``, revert to string.
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

        changed = False
        for key, value in args.items():
            if isinstance(value, str):
                continue
            prop = properties.get(key)
            if not isinstance(prop, dict) or prop.get("type") != "string":
                continue
            if isinstance(value, bool):
                args[key] = "true" if value else "false"
            elif value is None:
                args[key] = "null"
            else:
                args[key] = str(value)
            changed = True

        if changed:
            return json.dumps(args, ensure_ascii=False)
        return args_json

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
        if not self._engine.skip_tool_parsing:
            tool_choice = getattr(request, "tool_choice", None)
            tools = getattr(request, "tools", None)
            if tool_choice == "none" and tools:
                self._engine.skip_tool_parsing = True
        events = self._engine.feed(delta_text, delta_token_ids)
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
        events = self._engine.feed(model_output, [])
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
        if not self._streaming_initialized:
            self._streaming_initialized = True
            self._reset()
        events = self._engine.feed(delta_text, delta_token_ids)
        return self._strip_trailing_reasoning(self._events_to_delta(events))

    # ── Non-streaming: extract_tool_calls ─────────────────────────────

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
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
        finish_events = self._engine.finish()
        finish_delta = self._events_to_delta(finish_events) if finish_events else None
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
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        if not self._streaming_initialized:
            self._streaming_initialized = True
            self._reset()
        events = self._engine.feed(delta_text, delta_token_ids)
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
        if self._reasoning_ended:
            return []
        return input_ids

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
        events = self._engine.feed(text, token_ids)
        events.extend(self._engine.finish())

        delta = self._events_to_delta(events)
        tool_call_info = self._build_extracted_result()

        reasoning = delta.reasoning if delta else None
        if reasoning and self._strip_trailing_reasoning_ws:
            reasoning = reasoning.rstrip() or None

        content = delta.content if delta else None
        if tool_call_info.tools_called and content:
            if self._strip_content_ws_with_tools:
                content = content.strip() or None
            elif self._drop_ws_only_content_before_tools and not content.strip():
                content = None

        return reasoning, content, tool_call_info

    # ── Non-streaming: parse ───────────────────────────────────────────

    def parse(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
        enable_auto_tools: bool = False,
    ) -> tuple[str | None, str | None, list[FunctionCall] | None]:
        reasoning, content, tool_call_info = self._single_pass_parse(
            model_output,
            [],
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

    # ── Response outputs (Responses API) ──────────────────────────────

    def extract_response_outputs(
        self,
        *,
        model_output: str,
        model_output_token_ids: Sequence[int],
        request: ResponsesRequest,
        enable_auto_tools: bool = False,
        tool_call_id_type: str = "random",
        logprobs=None,
    ) -> list:
        from openai.types.responses import (
            ResponseFunctionToolCall,
            ResponseOutputMessage,
            ResponseOutputText,
        )
        from openai.types.responses.response_output_item import ResponseOutputItem
        from openai.types.responses.response_reasoning_item import (
            Content as ResponseReasoningTextContent,
        )
        from openai.types.responses.response_reasoning_item import (
            ResponseReasoningItem,
        )

        from vllm.utils import random_uuid

        # Token IDs let the engine distinguish real special tokens
        # from text that happens to look like them.
        reasoning, content, tool_call_info = self._single_pass_parse(
            model_output,
            model_output_token_ids,
        )

        outputs: list[ResponseOutputItem] = []

        if reasoning:
            outputs.append(
                ResponseReasoningItem(
                    id=f"rs_{random_uuid()}",
                    summary=[],
                    type="reasoning",
                    content=[
                        ResponseReasoningTextContent(
                            text=reasoning, type="reasoning_text"
                        )
                    ],
                    status=None,
                )
            )

        if content:
            outputs.append(
                ResponseOutputMessage(
                    id=f"msg_{random_uuid()}",
                    content=[
                        ResponseOutputText(
                            text=content,
                            annotations=[],
                            type="output_text",
                            logprobs=logprobs,
                        )
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )

        if tool_call_info.tools_called:
            for i, tc in enumerate(tool_call_info.tool_calls):
                outputs.append(
                    ResponseFunctionToolCall(
                        id=f"fc_{random_uuid()}",
                        call_id=tc.id
                        if tc.id
                        else make_tool_call_id(
                            id_type=tool_call_id_type,
                            func_name=tc.function.name,
                            idx=i,
                        ),
                        type="function_call",
                        status="completed",
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    )
                )

        return outputs

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

        if self._deferred_content:
            content_parts.append(self._deferred_content)
            self._deferred_content = ""

        for event in events:
            match event.type:
                case EventType.TEXT_CHUNK:
                    content_parts.append(event.value)
                case EventType.REASONING_CHUNK:
                    reasoning_parts.append(event.value)
                case EventType.REASONING_END:
                    self._reasoning_ended = True
                case EventType.TOOL_CALL_START:
                    self._init_tool_slot(event)
                case EventType.TOOL_NAME:
                    self._handle_tool_name(event)
                case EventType.ARG_VALUE_CHUNK:
                    self._handle_arg_chunk(event, tool_call_deltas)
                case EventType.TOOL_CALL_END:
                    self._handle_tool_end(event, tool_call_deltas)
                case EventType.REASONING_START:
                    pass  # no delta-level effect

        content_str = "".join(content_parts)
        stripped = content_str.strip() if content_str else ""

        if self._tool_slots:
            if (
                self._drop_ws_only_content_before_tools
                and not self._content_has_nonws
                and not stripped
            ):
                content_str = ""
        elif (
            content_str
            and not stripped
            and not self._content_has_nonws
            and not finished
        ):
            self._deferred_content = content_str
            content_str = ""

        if stripped:
            self._content_has_nonws = True

        content = content_str or None
        reasoning = "".join(reasoning_parts) or None

        if content or tool_call_deltas or reasoning:
            return DeltaMessage(
                content=content,
                reasoning=reasoning,
                tool_calls=tool_call_deltas,
            )
        return None

    def _ensure_slot(self, idx: int) -> None:
        while len(self._tool_slots) <= idx:
            self._tool_slots.append(ToolCallSlot())

    def _init_tool_slot(self, event: SemanticEvent) -> None:
        idx = event.tool_index
        self._ensure_slot(idx)
        state = self._stream_state
        self._tool_slots[idx].id = make_tool_call_id(
            id_type=state.tool_call_id_type,
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
        if not name:
            return
        slot = self._tool_slots[idx]
        slot.name = name
        slot.name_sent = True
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
            slot.args += event.value

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
            if name:
                slot.name = name
                slot.name_sent = True
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

    # ── Arg conversion helpers ─────────────────────────────────────────

    def _compute_arg_delta(self, idx: int, raw_delta: str) -> str | None:
        converter = self._arg_converter
        if converter is None:
            return raw_delta

        if not self._strip_trailing_quotes:
            return None

        structural = self._arg_structural_chars
        if structural is not None and structural.isdisjoint(raw_delta):
            return None

        slot = self._tool_slots[idx]
        try:
            current_json = converter(slot.args, True)
        except Exception:
            return None

        if not current_json:
            return None

        prev = slot.streamed_json
        safe_json = current_json
        while safe_json and safe_json[-1] in ("}", '"', "]"):
            safe_json = safe_json[:-1]

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
        except Exception:
            return None

        if final_json:
            final_json = self._fix_arg_types(final_json, slot.name)

        prev = slot.streamed_json
        if final_json and len(final_json) > len(prev):
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
            if not slot.id:
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
                    except Exception:
                        args_json = self._extract_args_json(raw_body, name)
                else:
                    args_json = self._extract_args_json(raw_body, name)
            else:
                args_json = "{}"

            if name:
                args_json = self._fix_arg_types(args_json, name)
                tool_calls.append(
                    ToolCall(
                        id=slot.id,
                        function=FunctionCall(name=name, arguments=args_json),
                    )
                )

        content_str = "".join(content_parts)
        if tool_calls:
            if self._strip_content_ws_with_tools:
                content_str = content_str.strip()
            elif self._drop_ws_only_content_before_tools and not content_str.strip():
                content_str = ""
        content = content_str or None

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
        raw_args = raw_args.strip()
        if not raw_args:
            return "{}"

        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            if self.parser_engine_config.value_postprocessor:
                return self.parser_engine_config.value_postprocessor(raw_args)
            return raw_args

        if isinstance(parsed, dict):
            args = self._extract_args_value(parsed)
            if args is not None:
                return args
            if "name" in parsed:
                without_name = {k: v for k, v in parsed.items() if k != "name"}
                return json.dumps(without_name, ensure_ascii=False)

        return raw_args
