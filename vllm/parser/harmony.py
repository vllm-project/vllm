# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, NamedTuple

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
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
    from openai_harmony import Message, Role
    from openai_harmony import StreamState as HarmonyStreamState


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
    is_boundary: bool = False
    completed_message: Message | None = None


@dataclass
class ChunkResult:
    segments: list[Segment]
    reasoning_token_count: int


class HarmonyParser(DelegatingParser):
    def __init__(self, tokenizer, tools=None, *args, **kwargs):
        super().__init__(tokenizer, tools, *args, **kwargs)

        if self._reasoning_parser and not isinstance(
            self._reasoning_parser, GptOssReasoningParser
        ):
            raise ValueError(
                "Harmony requires GptOssReasoningParser, "
                f"got {self._reasoning_parser.__class__.__name__}."
            )

        if self._tool_parser and not isinstance(self._tool_parser, GptOssToolParser):
            raise ValueError(
                "Harmony requires GptOssToolParser, "
                f"got {self._tool_parser.__class__.__name__}."
            )

        self._harmony_parser = get_streamable_parser_for_assistant()

    @property
    def messages(self) -> list[Message]:
        return self._harmony_parser.messages

    @property
    def state(self) -> HarmonyStreamState:
        return self._harmony_parser.state

    @property
    def current_role(self) -> Role | None:
        return self._harmony_parser.current_role

    @property
    def current_channel(self) -> str | None:
        return self._harmony_parser.current_channel

    @property
    def current_recipient(self) -> str | None:
        return self._harmony_parser.current_recipient

    @property
    def current_content(self) -> str:
        return self._harmony_parser.current_content

    @property
    def current_content_type(self) -> str | None:
        return self._harmony_parser.current_content_type

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

        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        tool_calls: list[FunctionCall] = []

        def _append_parsed_message(
            channel: str | None,
            recipient: str | None,
            text: str,
            content_type: str | None = None,
        ) -> None:
            segment_type = _SegmentType.from_channel_and_recipient(channel, recipient)
            match segment_type:
                case _SegmentType.REASONING if self.reasoning_parser and text:
                    reasoning_parts.append(text)
                case _SegmentType.CONTENT if text:
                    content_parts.append(text)
                case _SegmentType.TOOL if self.tool_parser:
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

        for segment in result.segments:
            msg = segment.completed_message
            if msg is None:
                continue
            if msg.author.role != "assistant" or not msg.content:
                continue
            _append_parsed_message(
                channel=msg.channel,
                recipient=msg.recipient,
                text=msg.content[0].text,
                content_type=msg.content_type,
            )

        if (
            self.current_channel is not None
            or self.current_recipient is not None
            or self.current_content
        ):
            _append_parsed_message(
                channel=self.current_channel,
                recipient=self.current_recipient,
                text=self.current_content,
                content_type=self.current_content_type,
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
        raise NotImplementedError(
            "HarmonyParser streaming parsing is deferred. "
            "Use the existing harmony streaming path."
        )

    def process_chunk(self, token_ids: Sequence[int]) -> ChunkResult:
        if not token_ids:
            return ChunkResult(segments=[], reasoning_token_count=0)

        from openai_harmony import StreamState

        segments: list[Segment] = []
        reasoning_token_count = 0
        for token_id in token_ids:
            self._harmony_parser.process(token_id)
            channel = self.current_channel
            recipient = self.current_recipient
            delta = self._harmony_parser.last_content_delta or ""
            completed_message = None
            is_boundary = self.state == StreamState.EXPECT_START
            if is_boundary and self.messages:
                completed_message = self.messages[-1]

            if channel == "analysis" or (
                channel == "commentary" and recipient is not None
            ):
                reasoning_token_count += 1

            segments.append(
                Segment(
                    channel=channel,
                    recipient=recipient,
                    delta=delta,
                    is_boundary=is_boundary,
                    completed_message=completed_message,
                )
            )

            # TODO: Optionally merge and suppress empty Segments

        return ChunkResult(
            segments=segments,
            reasoning_token_count=reasoning_token_count,
        )
