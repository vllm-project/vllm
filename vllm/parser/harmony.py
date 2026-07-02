# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, NamedTuple

from openai_harmony import HarmonyError
from xgrammar import normalize_tool_choice
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

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
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
from vllm.logger import init_logger
from vllm.parser.abstract_parser import DelegatingParser
from vllm.reasoning.gptoss_reasoning_parser import GptOssReasoningParser
from vllm.sampling_params import StructuredOutputsParams
from vllm.tool_parsers.gptoss_tool_parser import GptOssToolParser
from vllm.tool_parsers.structural_tag_registry import (
    any_tool_strict,
    dump_tool_choice_for_xgrammar,
    dump_tool_for_xgrammar,
    get_function_parameters,
)

if TYPE_CHECKING:
    from openai_harmony import Message, StreamableParser


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
        msg.recipient = self._normalize_recipient(msg.recipient)
        self._num_processed_messages += 1
        return msg

    def flush(self) -> Segment | None:
        try:
            self._harmony_parser.process_eos()
            msg = self._poll_completed_message()
        except HarmonyError:
            logger.warning(
                "Harmony parser ended in a non-terminal state; returning the "
                "raw unparsed output. This usually indicates a malformed "
                "assistant turn, e.g. a 'final' channel missing the "
                "<|message|> delimiter."
            )
            raise
        finally:
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
        try:
            flushed_segment = self.flush()
        except HarmonyError:
            return None, model_output, None
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
        prev_recipient = self._normalize_recipient(
            self._harmony_parser.current_recipient
        )
        result = self.process_chunk(delta_token_ids)
        if finished:
            try:
                flushed_segment = self.flush()
            except HarmonyError:
                self._next_tool_call_index = 0
                return DeltaMessage(content=delta_text)
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
        return delta_message

    def process_chunk(self, token_ids: Sequence[int]) -> ChunkResult:
        if not token_ids:
            return ChunkResult(segments=[], reasoning_token_count=0)

        segments: list[Segment] = []
        reasoning_token_count = 0
        for token_id in token_ids:
            self._harmony_parser.process(token_id)
            channel = self._harmony_parser.current_channel
            recipient = self._normalize_recipient(
                self._harmony_parser.current_recipient
            )
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

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = _adjust_structural_tag(request)
        request = super().adjust_request(request)
        return request

    @staticmethod
    def _normalize_recipient(recipient: str | None) -> str | None:
        """Remove constrained formats misparsed into recipients by older Harmony."""
        if recipient is None:
            return None

        constrain_index = recipient.find("<|constrain|>")
        if constrain_index == -1:
            return recipient
        return recipient[:constrain_index].rstrip() or None


# Harmomy can parse either <|end|>, <|call|>, <|endoftext|>, or <|return|>
# <|return|> is represented as `""` since it's an xgrammar stop token
_END_TAG = ["<|end|>", "<|call|>", "<|endoftext|>", ""]
_FINAL_CONSTRAIN_BEGINS = [
    # "<|channel|>final json<|message|>", # disabled to trigger _normalize_recipient
    "<|channel|>final <|constrain|>json<|message|>",
]
_TOOL_CALL_CHANNELS = [
    "<|channel|>commentary",
    "<|channel|>analysis",
    "<|channel|>final",
]
_FUNCTION_CALL_BEGINS = [
    "to=functions.{name} {channel} json<|message|>",
    "to=functions.{name} {channel} <|constrain|>json<|message|>",
    "{channel} to=functions.{name} json<|message|>",
    "{channel} to=functions.{name} <|constrain|>json<|message|>",
]
_JSON_CONTENT = JSONSchemaFormat(json_schema={"type": "object"})
_ANY_CONTENT = AnyTextFormat()


def _assemble_tag(
    allow_analysis: bool, allow_commentary: bool, content: Format
) -> Format:
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

    return SequenceFormat(elements=tags)


def _get_tool_structural_tag(
    request: ChatCompletionRequest | ResponsesRequest,
) -> Format | None:
    """Create structural tag for strict tool calling."""
    tools = request.tools
    tool_choice = request.tool_choice

    if not tools or tool_choice == "none":
        return None

    if tool_choice == "auto" and not any_tool_strict(tools):
        return None

    dumped_tools = [dump_tool_for_xgrammar(tool) for tool in tools]
    dumped_tool_choice = dump_tool_choice_for_xgrammar(tool_choice)
    function_tools, builtin_tools, simplified_tool_choice = normalize_tool_choice(
        dumped_tools,
        dumped_tool_choice,
    )

    tags = []
    if builtin_tools:
        # Fallback for builtin tools
        tags.append(
            TagFormat(
                begin="to=",
                content=AnyTextFormat(excludes=["<|start|>"]),
                end=_END_TAG,
            )
        )
        for channel in _TOOL_CALL_CHANNELS:
            tags.append(
                TagFormat(
                    begin=channel + " to=",
                    content=AnyTextFormat(excludes=["<|start|>", "<|channel|>"]),
                    end=_END_TAG,
                )
            )
    else:
        for tool in function_tools:
            name = tool.function.name
            tool_schema = JSONSchemaFormat(
                json_schema=get_function_parameters(tool.function)
            )
            tool_tags = [
                TagFormat(
                    begin=pattern.format(name=name, channel=channel),
                    content=tool_schema,
                    end=_END_TAG,
                )
                for pattern in _FUNCTION_CALL_BEGINS
                for channel in _TOOL_CALL_CHANNELS
            ]
            tags.extend(tool_tags)

    if simplified_tool_choice == "auto":
        # Allow final channel
        for pattern in _FINAL_CONSTRAIN_BEGINS:
            tags.append(TagFormat(begin=pattern, content=_JSON_CONTENT, end=_END_TAG))
        tags.append(
            TagFormat(
                begin="<|channel|>final<|message|>",
                content=_ANY_CONTENT,
                end=_END_TAG,
            )
        )

    return _assemble_tag(
        allow_analysis=True, allow_commentary=True, content=OrFormat(elements=tags)
    )


def _get_output_format_structural_tag(
    request: ChatCompletionRequest | ResponsesRequest,
) -> Format | None:
    if isinstance(request, ResponsesRequest) and request.text is not None:
        response_format = request.text.format
    elif isinstance(request, ChatCompletionRequest):
        response_format = request.response_format
    else:
        return None

    if response_format is None or response_format.type in (
        "text",
        "structural_tag",
    ):
        return None

    if response_format.type == "json_object":
        final_content = _JSON_CONTENT
    elif response_format.type == "json_schema":
        # Chat Completions nests the schema; Responses exposes `schema_`.
        schema_wrapper = getattr(response_format, "json_schema", None)
        if schema_wrapper is not None:
            schema = getattr(schema_wrapper, "json_schema", None)
        else:
            schema = getattr(response_format, "schema_", None)
        if schema is None:
            return None
        final_content = JSONSchemaFormat(json_schema=schema)
    else:
        return None

    return _assemble_tag(
        allow_analysis=True,
        allow_commentary=False,
        content=OrFormat(
            elements=[
                TagFormat(begin=begin, content=final_content, end=_END_TAG)
                for begin in _FINAL_CONSTRAIN_BEGINS
            ]
        ),
    )


def _adjust_structural_tag(
    request: ChatCompletionRequest | ResponsesRequest,
) -> ChatCompletionRequest | ResponsesRequest:
    # Favor tool structural tag when specified to match non-harmony behavior
    structural_tag = _get_tool_structural_tag(
        request
    ) or _get_output_format_structural_tag(request)
    if structural_tag is None:
        return request

    request.structured_outputs = StructuredOutputsParams(
        structural_tag=json.dumps(
            {
                "type": "structural_tag",
                "format": structural_tag.model_dump(by_alias=True),
            }
        ),
    )
    if isinstance(request, ResponsesRequest):
        request.text = None
    else:
        request.response_format = None
    return request
