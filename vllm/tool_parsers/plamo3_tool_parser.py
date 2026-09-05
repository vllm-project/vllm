# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.reasoning.plamo3_reasoning_parser import (
    BEGIN_TOOL_ARGS_TAG,
    BEGIN_TOOL_NAME_TAG,
    BEGIN_TOOL_REQUEST_TAG,
    BEGIN_TOOL_REQUESTS_TAG,
    END_TOOL_ARGS_TAG,
    END_TOOL_NAME_TAG,
    END_TOOL_REQUEST_TAG,
    END_TOOL_REQUESTS_TAG,
    EOT_TAG,
    compute_safe_until,
    strip_at_eot,
    strip_trailing_partial_marker,
)
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import ToolParser

logger = init_logger(__name__)


def parse_model_output(model_output: str) -> tuple[str, list[ToolCall]]:
    model_output = strip_at_eot(model_output)

    if (pos_begin_requests := model_output.find(BEGIN_TOOL_REQUESTS_TAG)) == -1:
        return model_output, []

    content = model_output[:pos_begin_requests]
    index = pos_begin_requests + len(BEGIN_TOOL_REQUESTS_TAG)
    tool_calls: list[ToolCall] = []

    while True:
        if not model_output.startswith(BEGIN_TOOL_REQUEST_TAG, index):
            if not tool_calls:
                return content, []
            break

        index += len(BEGIN_TOOL_REQUEST_TAG)

        if not model_output.startswith(BEGIN_TOOL_NAME_TAG, index):
            return content, []
        name_start = index + len(BEGIN_TOOL_NAME_TAG)
        if (name_end := model_output.find(END_TOOL_NAME_TAG, name_start)) == -1:
            return content, []
        tool_name = model_output[name_start:name_end]
        index = name_end + len(END_TOOL_NAME_TAG)

        if not model_output.startswith(BEGIN_TOOL_ARGS_TAG, index):
            return content, []
        args_start = index + len(BEGIN_TOOL_ARGS_TAG)
        if (args_end := model_output.find(END_TOOL_ARGS_TAG, args_start)) == -1:
            return content, []
        tool_arguments = model_output[args_start:args_end]
        index = args_end + len(END_TOOL_ARGS_TAG)

        if not model_output.startswith(END_TOOL_REQUEST_TAG, index):
            return content, []
        index += len(END_TOOL_REQUEST_TAG)

        tool_calls.append(
            ToolCall(function=FunctionCall(name=tool_name, arguments=tool_arguments))
        )

    if not model_output.startswith(END_TOOL_REQUESTS_TAG, index):
        return content, []

    return content, tool_calls


class ToolParserStreamPhase(Enum):
    CONTENT = "content"
    BEGIN_TOOL_REQUEST = "begin_tool_request"
    BEGIN_TOOL_NAME = "begin_tool_name"
    BEGIN_TOOL_ARGUMENTS = "begin_tool_arguments"
    IN_TOOL_ARGUMENTS = "in_tool_arguments"
    END_TOOL_REQUEST = "end_tool_request"
    MAYBE_NEXT_TOOL_OR_END = "maybe_next_tool_or_end"
    AFTER_TOOL_REQUESTS = "after_tool_requests"
    DONE = "done"


@dataclass
class _ToolParserStreamState:
    phase: ToolParserStreamPhase = ToolParserStreamPhase.CONTENT
    parse_pos: int = 0
    content_emit_pos: int = 0
    argument_emit_pos: int | None = None
    tool_index: int = -1
    tool_id: str | None = None


@dataclass
class _ToolCallDelta:
    index: int
    tool_id: str | None = None
    name: str | None = None
    argument_parts: list[str] = field(default_factory=list)

    def build(self) -> DeltaToolCall:
        arguments = "".join(self.argument_parts) or None
        return DeltaToolCall(
            index=self.index,
            type="function",
            id=self.tool_id,
            function=DeltaFunctionCall(
                name=self.name,
                arguments=arguments,
            ).model_dump(exclude_none=True),
        )


@dataclass
class _StreamDeltaBuilder:
    content_parts: list[str] = field(default_factory=list)
    tool_calls: dict[int, _ToolCallDelta] = field(default_factory=dict)

    def add_content(self, content: str) -> None:
        if content:
            self.content_parts.append(content)

    def add_tool_name(self, index: int, tool_id: str, name: str) -> None:
        self.tool_calls[index] = _ToolCallDelta(
            index=index,
            tool_id=tool_id,
            name=name,
        )

    def add_tool_arguments(self, index: int, arguments: str) -> None:
        if not arguments:
            return
        delta = self.tool_calls.setdefault(index, _ToolCallDelta(index=index))
        delta.argument_parts.append(arguments)

    def build(self) -> DeltaMessage | None:
        content = "".join(self.content_parts)
        tool_calls = [delta.build() for delta in self.tool_calls.values()]

        # Exclude unset fields to ensure clean SSE chunks. Since vLLM streams via
        # `model_dump_json(exclude_unset=True)`, this prevents mixed payloads
        # like empty `content` or `tool_calls`.
        kwargs: dict[str, Any] = {}
        if content:
            kwargs["content"] = content
        if tool_calls:
            kwargs["tool_calls"] = tool_calls

        return DeltaMessage(**kwargs) if kwargs else None


class Plamo3ToolParser(ToolParser):
    """Tool parser for PLaMo-3 explicit tool-call tags.

    Parses structures such as:

        <|plamo:begin_tool_requests:plamo|>
          <|plamo:begin_tool_request:plamo|>
            <|plamo:begin_tool_name:plamo|>get_weather<|plamo:end_tool_name:plamo|>
            <|plamo:begin_tool_arguments:plamo|>...<|plamo:end_tool_arguments:plamo|>
          <|plamo:end_tool_request:plamo|>
        <|plamo:end_tool_requests:plamo|>

    The tags are not added special tokens, so they may be split across
    streaming chunks; the streaming parser works on the reconstructed text.
    The arguments block contains JSON, such as ``{"city": "Tokyo"}``.
    """

    def __init__(self, tokenizer: TokenizerLike, tools=None):
        super().__init__(tokenizer, tools)
        self._stream = _ToolParserStreamState()

    def _start_tool_call(self) -> None:
        state = self._stream
        state.tool_id = make_tool_call_id()
        state.argument_emit_pos = None
        state.tool_index += 1

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        content, tool_calls = parse_model_output(model_output)
        content = strip_trailing_partial_marker(content)
        return ExtractedToolCallInformation(
            tools_called=(len(tool_calls) != 0),
            tool_calls=tool_calls,
            content=content,
        )

    def _emit_content_delta(
        self,
        buf: str,
        until: int,
        output: _StreamDeltaBuilder,
    ) -> None:
        state = self._stream
        if until <= state.content_emit_pos:
            return
        output.add_content(buf[state.content_emit_pos : until])
        state.content_emit_pos = until

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
        state = self._stream
        output = _StreamDeltaBuilder()

        while True:
            if state.phase == ToolParserStreamPhase.CONTENT:
                next_begin = current_text.find(BEGIN_TOOL_REQUESTS_TAG, state.parse_pos)
                next_eos = current_text.find(EOT_TAG, state.parse_pos)
                candidates = [p for p in [next_begin, next_eos] if p != -1]
                if candidates:
                    next_pos = min(candidates)
                    self._emit_content_delta(current_text, next_pos, output)
                else:
                    # EOT_TAG ("<|plamo:tag|>") is a single token and therefore
                    # never appears partially in the buffer, so it needs no
                    # hold-back and is omitted here.
                    if state.content_emit_pos < len(current_text):
                        safe_until = compute_safe_until(
                            current_text,
                            state.content_emit_pos,
                            (BEGIN_TOOL_REQUESTS_TAG,),
                        )
                        self._emit_content_delta(current_text, safe_until, output)
                    break

                if next_eos != -1 and (next_begin == -1 or next_eos < next_begin):
                    state.parse_pos = next_eos + len(EOT_TAG)
                    state.phase = ToolParserStreamPhase.DONE
                else:
                    state.parse_pos = next_begin + len(BEGIN_TOOL_REQUESTS_TAG)
                    state.phase = ToolParserStreamPhase.BEGIN_TOOL_REQUEST
                continue

            if state.phase == ToolParserStreamPhase.BEGIN_TOOL_REQUEST:
                if current_text.startswith(BEGIN_TOOL_REQUEST_TAG, state.parse_pos):
                    state.parse_pos += len(BEGIN_TOOL_REQUEST_TAG)
                    self._start_tool_call()
                    state.phase = ToolParserStreamPhase.BEGIN_TOOL_NAME
                    continue
                break

            if state.phase == ToolParserStreamPhase.BEGIN_TOOL_NAME:
                if current_text.startswith(BEGIN_TOOL_NAME_TAG, state.parse_pos):
                    name_start = state.parse_pos + len(BEGIN_TOOL_NAME_TAG)
                    if (
                        name_end := current_text.find(END_TOOL_NAME_TAG, name_start)
                    ) == -1:
                        break
                    tool_name = current_text[name_start:name_end]
                    state.parse_pos = name_end + len(END_TOOL_NAME_TAG)
                    state.phase = ToolParserStreamPhase.BEGIN_TOOL_ARGUMENTS
                    assert state.tool_id is not None
                    output.add_tool_name(
                        state.tool_index,
                        state.tool_id,
                        tool_name,
                    )
                    continue
                break

            if state.phase == ToolParserStreamPhase.BEGIN_TOOL_ARGUMENTS:
                if current_text.startswith(BEGIN_TOOL_ARGS_TAG, state.parse_pos):
                    state.parse_pos += len(BEGIN_TOOL_ARGS_TAG)
                    state.argument_emit_pos = state.parse_pos
                    state.phase = ToolParserStreamPhase.IN_TOOL_ARGUMENTS
                    continue
                break

            if state.phase == ToolParserStreamPhase.IN_TOOL_ARGUMENTS:
                if (
                    args_end_start := current_text.find(
                        END_TOOL_ARGS_TAG, state.parse_pos
                    )
                ) != -1:
                    assert state.argument_emit_pos is not None
                    output.add_tool_arguments(
                        state.tool_index,
                        current_text[state.argument_emit_pos : args_end_start],
                    )
                    state.argument_emit_pos = args_end_start
                    state.parse_pos = args_end_start + len(END_TOOL_ARGS_TAG)
                    state.phase = ToolParserStreamPhase.END_TOOL_REQUEST
                    continue

                if state.argument_emit_pos is not None and (
                    state.argument_emit_pos < len(current_text)
                ):
                    safe_until = compute_safe_until(
                        current_text,
                        state.argument_emit_pos,
                        (END_TOOL_ARGS_TAG,),
                    )
                    if safe_until > state.argument_emit_pos:
                        output.add_tool_arguments(
                            state.tool_index,
                            current_text[state.argument_emit_pos : safe_until],
                        )
                        state.argument_emit_pos = safe_until
                break

            if state.phase == ToolParserStreamPhase.END_TOOL_REQUEST:
                if current_text.startswith(END_TOOL_REQUEST_TAG, state.parse_pos):
                    state.parse_pos += len(END_TOOL_REQUEST_TAG)
                    state.phase = ToolParserStreamPhase.MAYBE_NEXT_TOOL_OR_END
                    continue
                break

            if state.phase == ToolParserStreamPhase.MAYBE_NEXT_TOOL_OR_END:
                if current_text.startswith(BEGIN_TOOL_REQUEST_TAG, state.parse_pos):
                    self._start_tool_call()
                    state.parse_pos += len(BEGIN_TOOL_REQUEST_TAG)
                    state.phase = ToolParserStreamPhase.BEGIN_TOOL_NAME
                    continue
                if current_text.startswith(END_TOOL_REQUESTS_TAG, state.parse_pos):
                    state.parse_pos += len(END_TOOL_REQUESTS_TAG)
                    state.phase = ToolParserStreamPhase.AFTER_TOOL_REQUESTS
                    continue
                break

            if state.phase == ToolParserStreamPhase.AFTER_TOOL_REQUESTS:
                if current_text.startswith(EOT_TAG, state.parse_pos):
                    state.parse_pos += len(EOT_TAG)
                    state.phase = ToolParserStreamPhase.DONE
                    continue
                break

            if state.phase == ToolParserStreamPhase.DONE:
                break

        return output.build()
