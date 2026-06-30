# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import contextlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, NamedTuple

from openai_harmony import HarmonyError

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    FunctionCall,
)
from vllm.entrypoints.openai.parser.harmony_utils import (
    extract_function_from_recipient,
    get_streamable_parser_for_assistant,
    is_function_recipient,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.abstract_parser import DelegatingParser
from vllm.reasoning.gptoss_reasoning_parser import GptOssReasoningParser
from vllm.tool_parsers.gptoss_tool_parser import GptOssToolParser

if TYPE_CHECKING:
    from openai_harmony import Message, StreamableParser


class _SegmentType(Enum):
    TOOL = auto()
    REASONING = auto()
    CONTENT = auto()
    IGNORE = auto()

    @staticmethod
    def from_channel_and_recipient(
        channel: str | None, recipient: str | None
    ) -> _SegmentType:
        if recipient and is_function_recipient(recipient):
            return _SegmentType.TOOL
        if channel == "analysis":
            return _SegmentType.REASONING
        if channel == "final" or (channel == "commentary" and recipient is None):
            return _SegmentType.CONTENT
        return _SegmentType.IGNORE


class Segment(NamedTuple):
    channel: str | None
    recipient: str | None
    delta: str
    completed_message: Message | None = None


@dataclass
class ChunkResult:
    segments: list[Segment]
    reasoning_token_count: int


class HarmonyParser(DelegatingParser):
    def __init__(self, tokenizer, tools=None, *args, **kwargs):
        super().__init__(tokenizer, tools, *args, **kwargs)

        if self.reasoning_parser and not isinstance(
            self.reasoning_parser, GptOssReasoningParser
        ):
            raise ValueError(
                "Harmony requires GptOssReasoningParser, "
                f"got {self.reasoning_parser.__class__.__name__}."
            )

        if self.tool_parser and not isinstance(self.tool_parser, GptOssToolParser):
            raise ValueError(
                "Harmony requires GptOssToolParser, "
                f"got {self.tool_parser.__class__.__name__}."
            )

        self._parser: StreamableParser | None = None
        self._next_tool_call_index = 0
        self._num_processed_messages = 0

    @property
    def _harmony_parser(self) -> StreamableParser:
        """Lazily initializes the Harmony parser."""
        if self._parser is None:
            self._parser = get_streamable_parser_for_assistant()
        return self._parser

    def _poll_completed_message(self) -> Message | None:
        messages = self._harmony_parser.messages
        if len(messages) <= self._num_processed_messages:
            return None
        msg = messages[self._num_processed_messages]
        self._num_processed_messages += 1
        return msg

    def flush(self) -> Segment | None:
        msg = None
        with contextlib.suppress(HarmonyError):
            self._harmony_parser.process_eos()
            # TODO: Consider reraising

        msg = self._poll_completed_message()

        # Reset to the initial assistant-parser state for the next turn.
        self._parser = None
        self._num_processed_messages = 0

        if msg is None:
            return None

        return Segment(
            channel=msg.channel,
            recipient=msg.recipient,
            delta="",
            completed_message=msg,
        )

    def parse(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
        enable_auto_tools: bool = False,
        model_output_token_ids: Sequence[int] = (),
    ) -> tuple[str | None, str | None, list[FunctionCall] | None]:
        """Parse Harmony output from token IDs.

        Tool calls are always extracted regardless of ``enable_auto_tools``.
        Callers must decide whether to surface them.
        """
        result = self.process_chunk(model_output_token_ids)
        flushed_segment = self.flush()
        if flushed_segment is not None:
            result.segments.append(flushed_segment)

        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        tool_calls: list[FunctionCall] = []

        for segment in result.segments:
            msg = segment.completed_message
            if msg is None:
                continue
            if msg.author.role != "assistant" or not msg.content:
                continue
            text = msg.content[0].text
            segment_type = _SegmentType.from_channel_and_recipient(
                msg.channel, msg.recipient
            )
            match segment_type:
                case _SegmentType.REASONING if self.reasoning_parser and text:
                    reasoning_parts.append(text)
                case _SegmentType.CONTENT if text:
                    content_parts.append(text)
                case _SegmentType.TOOL if self.tool_parser:
                    recipient = msg.recipient
                    content_type = msg.content_type
                    assert recipient is not None
                    if content_type is not None and "json" not in content_type:
                        arguments = text
                    else:
                        try:
                            arguments = json.dumps(json.loads(text))
                        except json.JSONDecodeError:
                            arguments = text
                    tool_calls.append(
                        FunctionCall(
                            name=extract_function_from_recipient(recipient),
                            arguments=arguments,
                        )
                    )

        reasoning = "\n".join(reasoning_parts) or None
        content = "\n".join(content_parts) or None
        return reasoning, content, tool_calls or None

    def parse_delta(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        request: ChatCompletionRequest | ResponsesRequest,
        prompt_token_ids: list[int] | None = None,
        *,
        finished: bool,
    ) -> DeltaMessage | None:
        prev_recipient = self._harmony_parser.current_recipient
        result = self.process_chunk(delta_token_ids)
        if finished:
            flushed_segment = self.flush()
            if flushed_segment is not None:
                result.segments.append(flushed_segment)
        combined_content = ""
        combined_reasoning = ""
        tool_messages: list[DeltaToolCall] = []

        for segment in result.segments:
            if segment.completed_message is not None:
                prev_recipient = None
                continue

            segment_type = _SegmentType.from_channel_and_recipient(
                segment.channel, segment.recipient
            )
            match segment_type:
                case _SegmentType.REASONING if self.reasoning_parser:
                    combined_reasoning += segment.delta
                case _SegmentType.CONTENT:
                    combined_content += segment.delta
                case _SegmentType.TOOL if self.tool_parser:
                    assert segment.recipient is not None
                    if prev_recipient != segment.recipient:
                        tool_name = extract_function_from_recipient(segment.recipient)
                        tool_messages.append(
                            DeltaToolCall(
                                # HarmonyParser does not use _stream_state;
                                # "random" tool_call_id_type is always used
                                id=make_tool_call_id(),
                                type="function",
                                function=DeltaFunctionCall(
                                    name=tool_name,
                                    arguments=segment.delta,
                                ),
                                index=self._next_tool_call_index,
                            )
                        )
                        self._next_tool_call_index += 1
                        prev_recipient = segment.recipient
                    elif segment.delta:
                        idx = self._next_tool_call_index - 1
                        if tool_messages:
                            tool_msg = tool_messages[-1]
                            assert tool_msg.index == idx
                            fn = tool_msg.function
                            assert fn is not None and fn.arguments is not None
                            fn.arguments += segment.delta
                        else:
                            tool_messages.append(
                                DeltaToolCall(
                                    index=idx,
                                    function=DeltaFunctionCall(arguments=segment.delta),
                                )
                            )

        if finished:
            self._next_tool_call_index = 0

        if not combined_content and not combined_reasoning and not tool_messages:
            return None

        delta_message = DeltaMessage()
        if combined_content:
            delta_message.content = combined_content
        if combined_reasoning:
            delta_message.reasoning = combined_reasoning
        if tool_messages:
            delta_message.tool_calls = tool_messages

        # Suppress reasoning deltas if not requested
        if delta_message and not request.include_reasoning:
            delta_message.reasoning = None

            # If only reasoning was in the message (no content, no tool_calls)
            # skip emitting entirely
            if not delta_message.content and not delta_message.tool_calls:
                return None

        return delta_message

    def process_chunk(self, token_ids: Sequence[int]) -> ChunkResult:
        if not token_ids:
            return ChunkResult(segments=[], reasoning_token_count=0)

        segments: list[Segment] = []
        reasoning_token_count = 0
        for token_id in token_ids:
            self._harmony_parser.process(token_id)
            channel = self._harmony_parser.current_channel
            recipient = self._harmony_parser.current_recipient
            delta = self._harmony_parser.last_content_delta or ""
            completed_message = self._poll_completed_message()

            if channel == "analysis" or (
                channel == "commentary" and recipient is not None
            ):
                reasoning_token_count += 1

            segments.append(
                Segment(
                    channel=channel,
                    recipient=recipient,
                    delta=delta,
                    completed_message=completed_message,
                )
            )

            # TODO: Optionally merge and suppress empty Segments

        return ChunkResult(
            segments=segments,
            reasoning_token_count=reasoning_token_count,
        )
