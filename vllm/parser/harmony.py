# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import TYPE_CHECKING, NamedTuple

from openai_harmony import HarmonyError, Message, Role
from xgrammar import StructuralTag
from xgrammar.openai_tool_call_schema import BuiltinToolParam, FunctionToolParam
from xgrammar.structural_tag import (
    AnyTextFormat,
    ConstStringFormat,
    Format,
    JSONSchemaFormat,
    OptionalFormat,
    OrFormat,
    SequenceFormat,
    TagFormat,
)

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    FunctionCall,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.parser.harmony_utils import (
    extract_function_from_recipient,
    get_encoding,
    get_streamable_parser_for_assistant,
    is_function_recipient,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.parser.abstract_parser import DelegatingParser, structured_outputs_to_format
from vllm.reasoning.gptoss_reasoning_parser import GptOssReasoningParser
from vllm.renderers.chat_utils import make_tool_call_id
from vllm.tool_parsers.gptoss_tool_parser import GptOssToolParser
from vllm.tool_parsers.structural_tag_registry import (
    SimplifiedToolChoice,
    get_function_parameters,
    register_vllm_structural_tag,
)

if TYPE_CHECKING:
    from openai_harmony import Message, StreamableParser

    from vllm.logprobs import LogprobsOnePosition


logger = init_logger(__name__)


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
    # Index into the token_ids passed to process_chunk of the token that
    # produced this segment. None for segments created by flush().
    token_index: int | None = None
    # (token_id, logprobs) of an output text token. Set by HarmonyContext
    # only when logprobs are requested.
    token_logprobs: tuple[int, LogprobsOnePosition] | None = None


@dataclass(slots=True)
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

        self._num_counted_tokens = 0
        self._num_reasoning_tokens = 0

        # For error recovery
        self._current_message_tokens: list[int] = []
        self._current_header_tokens: list[int] = []
        self._active_header: tuple[str | None, str | None] | None = None

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
        if self._active_header is not None:
            msg.channel, msg.recipient = self._active_header
        else:
            msg.channel, msg.recipient = self._normalize_header(
                msg.channel,
                msg.recipient,
            )
        self._num_processed_messages += 1
        self._active_header = None
        self._current_header_tokens.clear()
        return msg

    def flush(self) -> list[Segment]:
        segments: list[Segment] = []
        try:
            self._harmony_parser.process_eos()
            msg = self._poll_completed_message()
        except HarmonyError:
            logger.warning(
                "Harmony parser ended in a non-terminal state; returning the "
                "recovered raw output."
            )

            final_channel = "final"
            text = self.model_tokenizer.decode(self._current_message_tokens)
            segments.append(
                Segment(
                    channel=final_channel,
                    recipient=None,
                    delta=text,
                    completed_message=None,
                )
            )
            msg = Message.from_role_and_content(Role.ASSISTANT, text).with_channel(
                final_channel
            )

        # Reset to the initial assistant-parser state for the next turn.
        self._parser = None
        self._num_processed_messages = 0
        self._current_message_tokens.clear()
        self._current_header_tokens.clear()
        self._active_header = None

        if msg is None:
            return segments

        segments.append(
            Segment(
                channel=msg.channel,
                recipient=msg.recipient,
                delta="",
                completed_message=msg,
            )
        )
        return segments

    def parse(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
        enable_auto_tools: bool = False,
        model_output_token_ids: Sequence[int] = (),
    ) -> tuple[str | None, str | None, list[FunctionCall] | None]:
        """Parse Harmony output from token IDs.

        Tool calls are always extracted regardless of `enable_auto_tools`.
        Callers must decide whether to surface them.
        """
        result = self.process_chunk(model_output_token_ids)
        flushed_segments = self.flush()
        if flushed_segments:
            result.segments.extend(flushed_segments)

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
        _, prev_recipient = self._normalize_header(
            self._harmony_parser.current_channel,
            self._harmony_parser.current_recipient,
        )
        result = self.process_chunk(delta_token_ids)
        if finished:
            flushed_segments = self.flush()
            if flushed_segments:
                result.segments.extend(flushed_segments)
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
        for token_index, token_id in enumerate(token_ids):
            previous_state = self._harmony_parser.state_data["state"]
            try:
                self._harmony_parser.process(token_id)
            except HarmonyError:
                continue
            current_state = self._harmony_parser.state_data["state"]
            # Capture raw header tokens, including the token that starts the header.
            if current_state == "Header":
                if previous_state != "Header":
                    self._current_header_tokens.clear()
                    self._active_header = None
                self._current_header_tokens.append(token_id)
            # Resolve the header when the parser enters message content.
            elif previous_state == "Header" and current_state == "Content":
                self._active_header = self._resolve_header(
                    self._current_header_tokens,
                    self._harmony_parser.current_channel,
                    self._harmony_parser.current_recipient,
                )
            if self._active_header is not None:
                channel, recipient = self._active_header
            else:
                channel, recipient = self._normalize_header(
                    self._harmony_parser.current_channel,
                    self._harmony_parser.current_recipient,
                )
            delta = self._harmony_parser.last_content_delta or ""
            completed_message = self._poll_completed_message()
            if completed_message is not None:
                channel, recipient = (
                    completed_message.channel,
                    completed_message.recipient,
                )
                self._current_message_tokens.clear()
            else:
                self._current_message_tokens.append(token_id)
            if self._is_reasoning_token(token_id, channel, recipient):
                reasoning_token_count += 1
            segments.append(
                Segment(
                    channel=channel,
                    recipient=recipient,
                    delta=delta,
                    completed_message=completed_message,
                    token_index=token_index,
                )
            )

        self._num_counted_tokens += len(token_ids)
        self._num_reasoning_tokens += reasoning_token_count
        return ChunkResult(
            segments=segments,
            reasoning_token_count=reasoning_token_count,
        )

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        if len(token_ids) == self._num_counted_tokens:
            return self._num_reasoning_tokens

        parser = get_streamable_parser_for_assistant()
        header_tokens: list[int] = []
        active_header: tuple[str | None, str | None] | None = None
        count = 0
        for token_id in token_ids:
            previous_state = parser.state_data["state"]
            try:
                parser.process(token_id)
            except HarmonyError:
                continue
            current_state = parser.state_data["state"]
            if current_state == "Header":
                if previous_state != "Header":
                    header_tokens.clear()
                active_header = None
                header_tokens.append(token_id)
            elif previous_state == "Header" and current_state == "Content":
                active_header = self._resolve_header(
                    header_tokens,
                    parser.current_channel,
                    parser.current_recipient,
                )
            if active_header is not None:
                channel, recipient = active_header
            else:
                channel, recipient = self._normalize_header(
                    parser.current_channel,
                    parser.current_recipient,
                )
            if self._is_reasoning_token(token_id, channel, recipient):
                count += 1
        return count

    @staticmethod
    def _is_reasoning_token(
        token_id: int, channel: str | None, recipient: str | None
    ) -> bool:
        is_reasoning_channel = channel == "analysis" or (
            channel == "commentary" and recipient is not None
        )
        return is_reasoning_channel and not get_encoding().is_special_token(token_id)

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = _adjust_output_format(request)
        return super().adjust_request(request)

    @staticmethod
    def _normalize_header(
        channel: str | None, recipient: str | None
    ) -> tuple[str | None, str | None]:
        """Repair header tokens that Harmony misparses into the recipient.

        Older Harmony leaks `<|constrain|>` into the recipient. gpt-oss can
        also restart a header, e.g. `<|channel|>commentary to=assistant
        <|channel|>analysis`, in which case the last channel applies.
        """
        if recipient is None:
            return channel, None

        recipient = recipient.split("<|constrain|>", 1)[0]
        recipient, restarted, restarted_channel = recipient.partition("<|channel|>")
        recipient = recipient.strip()
        if restarted:
            channel = restarted_channel.strip() or channel
            # A restarted header addressed to the assistant itself is stray.
            if recipient == "assistant":
                recipient = ""
        return channel, recipient or None

    @classmethod
    def _resolve_header(
        cls,
        header_tokens: Sequence[int],
        channel: str | None,
        recipient: str | None,
    ) -> tuple[str | None, str | None]:
        channel, recipient = cls._normalize_header(channel, recipient)
        if not header_tokens:
            return channel, recipient
        raw_header = get_encoding().decode(list(header_tokens))
        # A restarted channel header is indicated by multiple channel markers.
        if raw_header.count("<|channel|>") <= 1:
            return channel, recipient
        # The last channel declaration takes precedence.
        restarted_channel = raw_header.rsplit("<|channel|>", 1)[1]
        restarted_channel = restarted_channel.split(maxsplit=1)[0]
        restarted_channel = restarted_channel.split("<|", 1)[0].strip()
        if not restarted_channel:
            return channel, recipient
        channel = restarted_channel
        # "assistant" is not a tool recipient. In this malformed/restarted
        # header it represents the stray recipient from the earlier declaration.
        if recipient == "assistant":
            recipient = None
        return channel, recipient


_JSON_CONSTRAINS = [" json", " <|constrain|>json"]
# Harmomy's stop tokens are <|return|>, <|call|>, <|endoftext|>
# They are represented as "" since xgrammar disallows stop tokens while
# under constraints, leading to bad or infinite generations.
_END_TAG = ["<|end|>", ""]
_FINAL_BEGIN = "<|channel|>final{constrain}<|message|>"
_TOOL_CALL_CHANNELS = [
    "<|channel|>commentary",
    "<|channel|>analysis",
    "<|channel|>final",
]
_FUNCTION_CALL_BEGINS = [
    " to=functions.{name}{channel}{constrain}<|message|>",
    "{channel} to=functions.{name}{constrain}<|message|>",
]
_ANY_CONTENT = AnyTextFormat()


def _assemble_tag(
    allow_analysis: bool, allow_commentary: bool, content: Format
) -> StructuralTag:
    tags = []
    if allow_analysis:
        analysis_tag = OptionalFormat(
            content=SequenceFormat(
                elements=[
                    TagFormat(
                        begin="<|channel|>analysis<|message|>",
                        content=_ANY_CONTENT,
                        end="<|end|>",
                    ),
                    ConstStringFormat(value="<|start|>assistant"),
                ]
            )
        )
        tags.append(analysis_tag)

    if allow_commentary:
        commentary_tag = OptionalFormat(
            content=SequenceFormat(
                elements=[
                    TagFormat(
                        begin="<|channel|>commentary<|message|>",
                        content=_ANY_CONTENT,
                        end="<|end|>",
                    ),
                    ConstStringFormat(value="<|start|>assistant"),
                ]
            )
        )
        tags.append(commentary_tag)

    tags.append(content)

    return StructuralTag(format=SequenceFormat(elements=tags))


@register_vllm_structural_tag("harmony", builtin_tools=True)
def get_harmony_structural_tag(
    tools: list[FunctionToolParam],
    builtin_tools: list[BuiltinToolParam],
    tool_choice: SimplifiedToolChoice,
    reasoning: bool,
    token_suffix: str = "",
) -> StructuralTag:
    # reasoning always enabled for Harmony
    del reasoning, token_suffix

    if builtin_tools:
        # Fallback for built-in tools
        tags = [
            TagFormat(
                begin="to=",
                content=AnyTextFormat(excludes=["<|start|>"]),
                end=_END_TAG,
            )
        ]
        tags.extend(
            TagFormat(
                begin=channel + " to=",
                content=AnyTextFormat(excludes=["<|start|>", "<|channel|>"]),
                end=_END_TAG,
            )
            for channel in _TOOL_CALL_CHANNELS
        )
    else:
        tags = [
            TagFormat(
                begin=pattern.format(
                    name=tool.function.name, channel=channel, constrain=constrain
                ),
                content=JSONSchemaFormat(
                    json_schema=get_function_parameters(tool.function)
                ),
                end=_END_TAG,
            )
            for tool in tools
            for pattern in _FUNCTION_CALL_BEGINS
            for constrain in _JSON_CONSTRAINS
            for channel in _TOOL_CALL_CHANNELS
        ]

    if tool_choice == "auto":
        tags.extend(
            TagFormat(
                begin=_FINAL_BEGIN.format(constrain=constrain),
                content=_ANY_CONTENT,
                end=_END_TAG,
            )
            for constrain in _JSON_CONSTRAINS + [""]
        )

    return _assemble_tag(
        allow_analysis=True, allow_commentary=True, content=OrFormat(elements=tags)
    )


def _adjust_output_format(
    request: ChatCompletionRequest | ResponsesRequest,
) -> ChatCompletionRequest | ResponsesRequest:
    """Canonicalize request constraints into a reasoning-aware StructuralTag."""
    params = request.extract_structured_outputs()
    if params is None:
        return request

    final_content = structured_outputs_to_format(params)
    if final_content is None:
        return request

    if isinstance(final_content, JSONSchemaFormat):
        content = OrFormat(
            elements=[
                TagFormat(
                    begin=_FINAL_BEGIN.format(constrain=constrain),
                    content=final_content,
                    end=_END_TAG,
                )
                for constrain in _JSON_CONSTRAINS
            ]
        )
    else:
        content = TagFormat(
            begin=_FINAL_BEGIN.format(constrain=""), content=final_content, end=_END_TAG
        )
    structural_tag = _assemble_tag(
        allow_analysis=True, allow_commentary=False, content=content
    )

    request.structured_outputs = replace(
        params,
        json=None,
        regex=None,
        choice=None,
        grammar=None,
        json_object=None,
        structural_tag=json.dumps(structural_tag.model_dump()),
    )
    if isinstance(request, ResponsesRequest):
        request.text = None
    else:
        request.response_format = None
    return request
