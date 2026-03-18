# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import json
import uuid
from abc import abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from functools import cached_property

from openai.types.responses import (
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallItem,
    ResponseOutputItem,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ToolChoiceFunction,
)
from openai.types.responses.response_output_text import Logprob
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningTextContent,
)
from pydantic import TypeAdapter, ValidationError

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
    FunctionCall,
    FunctionDefinition,
)
from vllm.entrypoints.openai.responses.context import ConversationContext, SimpleContext
from vllm.entrypoints.openai.responses.protocol import (
    ResponseReasoningPartAddedEvent,
    ResponseReasoningPartDoneEvent,
    ResponsesRequest,
    StreamingResponsesResponse,
)
from vllm.logger import init_logger
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import ToolParser
from vllm.utils import random_uuid
from vllm.utils.collection_utils import as_list

logger = init_logger(__name__)


@dataclass
class StreamingParseState:
    """Per-request state for streaming parse operations.

    Parser instances are shared across requests, so this holds the
    mutable state that varies per streaming session.
    """

    reasoning_ended: bool = False
    tool_call_text_started: bool = False
    previous_text: str = ""
    previous_token_ids: list[int] = field(default_factory=list)
    prompt_is_reasoning_end: bool | None = None


class Parser:
    """
    Abstract Parser class that unifies ReasoningParser and ToolParser into
    a single interface for parsing model output.

    This class provides a unified way to handle both reasoning extraction
    (e.g., chain-of-thought content in <think> tags) and tool call extraction
    (e.g., function calls in XML/JSON format) from model outputs.

    Subclasses can either:
    1. Override the abstract methods directly for custom parsing logic
    2. Set `reasoning_parser` and `tool_parser` properties to delegate to
       existing parser implementations

    Class Attributes:
        reasoning_parser_cls: The ReasoningParser class to use (for compatibility
            with code that needs the class, not instance).
        tool_parser_cls: The ToolParser class to use (for compatibility with
            code that needs the class, not instance).
    """

    # Class-level parser classes for compatibility with existing patterns
    # Subclasses should override these if they use specific parser classes
    reasoning_parser_cls: type[ReasoningParser] | None = None
    tool_parser_cls: type[ToolParser] | None = None

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        """
        Initialize the Parser.

        Args:
            tokenizer: The tokenizer used by the model. This is required for
                token-based parsing operations.
        """
        self.model_tokenizer = tokenizer
        self._reasoning_parser: ReasoningParser | None = None
        self._tool_parser: ToolParser | None = None

    @cached_property
    def vocab(self) -> dict[str, int]:
        """Get the vocabulary mapping from tokens to IDs."""
        return self.model_tokenizer.get_vocab()

    @property
    def reasoning_parser(self) -> ReasoningParser | None:
        """The underlying reasoning parser, if any."""
        return self._reasoning_parser

    @reasoning_parser.setter
    def reasoning_parser(self, parser: ReasoningParser | None) -> None:
        self._reasoning_parser = parser

    @property
    def tool_parser(self) -> ToolParser | None:
        """The underlying tool parser, if any."""
        return self._tool_parser

    @tool_parser.setter
    def tool_parser(self, parser: ToolParser | None) -> None:
        self._tool_parser = parser

    # ========== Reasoning Parser Methods ==========

    @abstractmethod
    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        """
        Check if the reasoning content ends in the input_ids.

        Used by structured engines like `xgrammar` to check if the
        reasoning content ends in the model output.

        Args:
            input_ids: The token IDs of the model output.

        Returns:
            True if the reasoning content ends in the input_ids.
        """

    def is_reasoning_end_streaming(
        self, input_ids: list[int], delta_ids: list[int]
    ) -> bool:
        """
        Check if the reasoning content ends during a decode step.

        Args:
            input_ids: The entire model output token IDs.
            delta_ids: The last few computed tokens at the current decode step.

        Returns:
            True if the reasoning content ends in the delta_ids.
        """
        return self.is_reasoning_end(input_ids)

    @abstractmethod
    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract content token IDs from the input_ids.

        This extracts the non-reasoning content (e.g., everything after
        the </think> tag).

        Args:
            input_ids: The token IDs of the model output.

        Returns:
            The extracted content token IDs.
        """

    @abstractmethod
    def extract_response_outputs(
        self,
        *,
        model_output: str,
        model_output_token_ids: Sequence[int],
        request: ResponsesRequest,
        enable_auto_tools: bool = False,
        tool_call_id_type: str = "random",
        logprobs: list[Logprob] | None = None,
    ) -> list[ResponseOutputItem]:
        """
        Extract reasoning, content, and tool calls from a complete
        model-generated string and return as ResponseOutputItem objects.

        Used for non-streaming responses where we have the entire model
        response available before sending to the client.

        Args:
            model_output: The complete model-generated string.
            model_output_token_ids: The token IDs of the model output.
            request: The request object used to generate the output.
            enable_auto_tools: Whether to enable automatic tool call parsing.
            tool_call_id_type: Type of tool call ID generation ("random", etc).
            logprobs: Pre-computed logprobs for the output text, if any.

        Returns:
            A list of ResponseOutputItem objects.
        """

    @abstractmethod
    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from a complete model-generated string.

        Used for non-streaming responses where we have the entire model
        response available before sending to the client.

        Args:
            model_output: The complete model-generated string.
            request: The request object used to generate the output.

        Returns:
            A tuple of (reasoning_content, response_content).
        """

    @abstractmethod
    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """
        Extract reasoning content from a streaming delta message.

        Args:
            previous_text: Text from all previous tokens.
            current_text: Text including the current delta.
            delta_text: The new text in this delta.
            previous_token_ids: Token IDs from previous generation.
            current_token_ids: All token IDs including current.
            delta_token_ids: The new token IDs in this delta.

        Returns:
            A DeltaMessage with reasoning and/or content fields, or None.
        """

    # ========== Unified Streaming Method ==========

    @abstractmethod
    def extract_streaming_delta(
        self,
        state: StreamingParseState,
        delta_text: str,
        delta_token_ids: list[int],
        prompt_token_ids: list[int] | None,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> DeltaMessage | None:
        """
        Extract a streaming delta message, handling reasoning-to-tool
        transitions and dispatching to the appropriate parser.

        This method consolidates the 4-branch dispatch logic for streaming:
        - Both reasoning + tool parsers: reasoning first, then tool parsing
        - Reasoning only: delegate to reasoning parser
        - Tool only: delegate to tool parser
        - Neither: return content directly

        Args:
            state: Mutable per-request streaming state.
            delta_text: The new text in this delta.
            delta_token_ids: The new token IDs in this delta.
            prompt_token_ids: The prompt token IDs (used on first call to
                check if reasoning already ended in prompt). None after first
                call.
            request: The request object.

        Returns:
            A DeltaMessage, or None if no output for this delta.
        """

    # ========== Tool Parser Methods ==========

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """
        Adjust the request parameters for tool calling.

        Can be overridden by subclasses to modify request parameters
        (e.g., setting structured output schemas for tool calling).

        Args:
            request: The original request.

        Returns:
            The adjusted request.
        """
        return request

    @abstractmethod
    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete model-generated string.

        Used for non-streaming responses.

        Args:
            model_output: The complete model-generated string.
            request: The request object used to generate the output.

        Returns:
            ExtractedToolCallInformation containing the tool calls.
        """

    @abstractmethod
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
        """
        Extract tool calls from a streaming delta message.

        Args:
            previous_text: Text from all previous tokens.
            current_text: Text including the current delta.
            delta_text: The new text in this delta.
            previous_token_ids: Token IDs from previous generation.
            current_token_ids: All token IDs including current.
            delta_token_ids: The new token IDs in this delta.
            request: The request object.

        Returns:
            A DeltaMessage with tool_calls field, or None.
        """


class DelegatingParser(Parser):
    """
    A Parser implementation that delegates to separate ReasoningParser and
    ToolParser instances.

    This is the recommended base class for creating model-specific parsers
    that combine existing reasoning and tool parser implementations.
    Subclasses should set `self._reasoning_parser` and `self._tool_parser`
    in their `__init__` method.

    If either parser is None, the corresponding methods will return default
    values (no reasoning extraction, no tool calls).
    """

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        if self._reasoning_parser is None:
            return True  # No reasoning parser = reasoning is always "ended"
        return self._reasoning_parser.is_reasoning_end(input_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if self._reasoning_parser is None:
            return input_ids
        return self._reasoning_parser.extract_content_ids(input_ids)

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        if self._reasoning_parser is None:
            return None, model_output
        return self._reasoning_parser.extract_reasoning(model_output, request)

    def extract_response_outputs(
        self,
        *,
        model_output: str,
        model_output_token_ids: Sequence[int],
        request: ResponsesRequest,
        enable_auto_tools: bool = False,
        tool_call_id_type: str = "random",
        logprobs: list[Logprob] | None = None,
    ) -> list[ResponseOutputItem]:
        # First extract reasoning
        reasoning, content = self.extract_reasoning(model_output, request)

        # Then parse tool calls from the content
        tool_calls, content = self._parse_tool_calls(
            request=request,
            content=content,
            enable_auto_tools=enable_auto_tools,
        )

        # Build output items
        outputs: list[ResponseOutputItem] = []

        # Add reasoning item if present
        if reasoning:
            reasoning_item = ResponseReasoningItem(
                id=f"rs_{random_uuid()}",
                summary=[],
                type="reasoning",
                content=[
                    ResponseReasoningTextContent(text=reasoning, type="reasoning_text")
                ],
                status=None,  # NOTE: Only the last output item has status.
            )
            outputs.append(reasoning_item)

        # Add message item if there's content
        if content:
            res_text_part = ResponseOutputText(
                text=content,
                annotations=[],
                type="output_text",
                logprobs=logprobs,
            )
            message_item = ResponseOutputMessage(
                id=f"msg_{random_uuid()}",
                content=[res_text_part],
                role="assistant",
                status="completed",
                type="message",
            )
            outputs.append(message_item)

        if tool_calls:
            # We use a simple counter for history_tool_call_count because
            # we don't track the history of tool calls in the Responses API yet.
            # This means that the tool call index will start from 0 for each
            # request.
            for history_tool_call_cnt, tool_call in enumerate(tool_calls):
                tool_call_item = ResponseFunctionToolCall(
                    id=f"fc_{random_uuid()}",
                    call_id=tool_call.id
                    if tool_call.id
                    else make_tool_call_id(
                        id_type=tool_call_id_type,
                        func_name=tool_call.name,
                        idx=history_tool_call_cnt,
                    ),
                    type="function_call",
                    status="completed",
                    name=tool_call.name,
                    arguments=tool_call.arguments,
                )
                outputs.append(tool_call_item)

        return outputs

    def _parse_tool_calls(
        self,
        request: ResponsesRequest,
        content: str | None,
        enable_auto_tools: bool,
    ) -> tuple[list[FunctionCall], str | None]:
        """
        TODO(qandrew): merge _parse_tool_calls_from_content
        for ChatCompletions into this function
        Parse tool calls from content based on request tool_choice settings.

        Returns:
            A tuple of (function_calls, remaining_content) if tool calls
            were parsed
        """
        function_calls: list[FunctionCall] = []

        if request.tool_choice and isinstance(request.tool_choice, ToolChoiceFunction):
            # Forced Function Call (Responses API style)
            assert content is not None
            function_calls.append(
                FunctionCall(name=request.tool_choice.name, arguments=content)
            )
            return function_calls, None  # Clear content since tool is called.

        if request.tool_choice and isinstance(
            request.tool_choice, ChatCompletionNamedToolChoiceParam
        ):
            # Forced Function Call (Chat Completion API style)
            assert content is not None
            function_calls.append(
                FunctionCall(name=request.tool_choice.function.name, arguments=content)
            )
            return function_calls, None  # Clear content since tool is called.

        if request.tool_choice == "required":
            # Required tool calls - parse JSON
            tool_calls = []
            with contextlib.suppress(ValidationError):
                content = content or ""
                tool_calls = TypeAdapter(list[FunctionDefinition]).validate_json(
                    content
                )
            for tool_call in tool_calls:
                function_calls.append(
                    FunctionCall(
                        name=tool_call.name,
                        arguments=json.dumps(tool_call.parameters, ensure_ascii=False),
                    )
                )
            return function_calls, None  # Clear content since tool is called.

        if (
            self._tool_parser is not None
            and enable_auto_tools
            and (request.tool_choice == "auto" or request.tool_choice is None)
        ):
            # Automatic Tool Call Parsing
            tool_call_info = self._tool_parser.extract_tool_calls(
                content if content is not None else "",
                request=request,  # type: ignore
            )
            if tool_call_info is not None and tool_call_info.tools_called:
                function_calls.extend(
                    FunctionCall(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                    )
                    for tool_call in tool_call_info.tool_calls
                )
                remaining_content = tool_call_info.content
                if remaining_content and remaining_content.strip() == "":
                    remaining_content = None
                return function_calls, remaining_content

        # No tool calls
        return [], content

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if self._reasoning_parser is None:
            return DeltaMessage(content=delta_text)
        return self._reasoning_parser.extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if self._tool_parser is None:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        return self._tool_parser.extract_tool_calls(model_output, request)

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
        if self._tool_parser is None:
            return None
        return self._tool_parser.extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request,
        )

    def extract_streaming_delta(
        self,
        state: StreamingParseState,
        delta_text: str,
        delta_token_ids: list[int],
        prompt_token_ids: list[int] | None,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> DeltaMessage | None:
        # Initialize prompt_is_reasoning_end on first call
        if (
            self._reasoning_parser is not None
            and state.prompt_is_reasoning_end is None
            and prompt_token_ids is not None
        ):
            state.prompt_is_reasoning_end = self._reasoning_parser.is_reasoning_end(
                prompt_token_ids
            )

        current_text = state.previous_text + delta_text
        current_token_ids = state.previous_token_ids + delta_token_ids

        delta_message: DeltaMessage | None = None

        if self._reasoning_parser is not None and self._tool_parser is not None:
            # Both parsers: reasoning first, then transition to tool parsing
            if state.prompt_is_reasoning_end:
                state.reasoning_ended = True

            if not state.reasoning_ended:
                delta_message = self._reasoning_parser.extract_reasoning_streaming(
                    previous_text=state.previous_text,
                    current_text=current_text,
                    delta_text=delta_text,
                    previous_token_ids=state.previous_token_ids,
                    current_token_ids=current_token_ids,
                    delta_token_ids=delta_token_ids,
                )
                if self._reasoning_parser.is_reasoning_end(delta_token_ids):
                    state.reasoning_ended = True
                    current_token_ids = self._reasoning_parser.extract_content_ids(
                        delta_token_ids
                    )
                    if delta_message and delta_message.content:
                        current_text = delta_message.content
                        delta_message.content = None
                    else:
                        current_text = ""

            if state.reasoning_ended:
                if not state.tool_call_text_started:
                    state.tool_call_text_started = True
                    state.previous_text = ""
                    state.previous_token_ids = []
                    delta_text = current_text
                    delta_token_ids = current_token_ids

                delta_message = self._tool_parser.extract_tool_calls_streaming(
                    previous_text=state.previous_text,
                    current_text=current_text,
                    delta_text=delta_text,
                    previous_token_ids=state.previous_token_ids,
                    current_token_ids=current_token_ids,
                    delta_token_ids=delta_token_ids,
                    request=request,  # type: ignore[arg-type]
                )

        elif self._reasoning_parser is not None:
            delta_message = self._reasoning_parser.extract_reasoning_streaming(
                previous_text=state.previous_text,
                current_text=current_text,
                delta_text=delta_text,
                previous_token_ids=state.previous_token_ids,
                current_token_ids=current_token_ids,
                delta_token_ids=delta_token_ids,
            )

        elif self._tool_parser is not None:
            delta_message = self._tool_parser.extract_tool_calls_streaming(
                previous_text=state.previous_text,
                current_text=current_text,
                delta_text=delta_text,
                previous_token_ids=state.previous_token_ids,
                current_token_ids=current_token_ids,
                delta_token_ids=delta_token_ids,
                request=request,  # type: ignore[arg-type]
            )

        else:
            delta_message = DeltaMessage(content=delta_text)

        state.previous_text = current_text
        state.previous_token_ids = current_token_ids

        return delta_message

    # ========== Streaming Event Generation ==========

    async def process_streaming_events(
        self,
        request: ResponsesRequest,
        result_generator: AsyncIterator[ConversationContext | None],
        request_id: str,
        raise_if_error: Callable[[str | None, str], None],
        create_logprobs: Callable[..., list] | None,
        top_logprobs: int | None,
        increment_sequence_number: Callable[
            [StreamingResponsesResponse], StreamingResponsesResponse
        ],
    ) -> AsyncGenerator[StreamingResponsesResponse, None]:
        """Generate streaming Responses API events from a result generator.

        This is the default implementation that handles reasoning, content,
        and tool call streaming transitions. Subclasses can override for
        custom behavior (e.g., attaching metadata, custom event emission).

        Args:
            request: The Responses API request.
            result_generator: Async iterator yielding SimpleContext objects.
            request_id: The request ID for error reporting.
            raise_if_error: Callback to raise on error finish reasons.
            create_logprobs: Callback to create logprob objects from
                (token_ids, logprobs, tokenizer, top_logprobs). None when
                logprobs not requested.
            top_logprobs: Number of top logprobs to include.
            increment_sequence_number: Callback to assign sequence numbers
                to streaming events.
        """
        current_content_index = 0
        current_output_index = 0
        current_item_id = ""
        current_tool_call_id = ""
        current_tool_call_name = ""
        streaming_state = StreamingParseState()
        first_delta_sent = False
        previous_delta_messages: list[DeltaMessage] = []
        async for ctx in result_generator:
            assert isinstance(ctx, SimpleContext)
            if ctx.last_output is None:
                continue
            if ctx.last_output.outputs:
                output = ctx.last_output.outputs[0]
                # finish_reason='error' indicates a retryable error
                raise_if_error(output.finish_reason, request_id)
                delta_text = output.text
                delta_token_ids = as_list(output.token_ids)

                # Pass prompt_token_ids on first call only
                prompt_token_ids = (
                    ctx.last_output.prompt_token_ids
                    if streaming_state.prompt_is_reasoning_end is None
                    else None
                )
                delta_message = self.extract_streaming_delta(
                    state=streaming_state,
                    delta_text=delta_text,
                    delta_token_ids=delta_token_ids,
                    prompt_token_ids=prompt_token_ids,
                    request=request,
                )
                if not delta_message:
                    continue
                if not first_delta_sent:
                    current_item_id = random_uuid()
                    if delta_message.tool_calls:
                        current_tool_call_id = f"call_{random_uuid()}"
                        assert len(delta_message.tool_calls) == 1, (
                            "Multiple tool calls in one delta is not supported"
                        )
                        assert delta_message.tool_calls[0].function is not None, (
                            "Tool call without function is not supported"
                        )
                        assert delta_message.tool_calls[0].function.name is not None, (
                            "Tool call without function name is not supported"
                        )
                        current_tool_call_name = delta_message.tool_calls[
                            0
                        ].function.name
                        yield increment_sequence_number(
                            ResponseOutputItemAddedEvent(
                                type="response.output_item.added",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item=ResponseFunctionToolCallItem(
                                    type="function_call",
                                    id=current_item_id,
                                    call_id=current_tool_call_id,
                                    name=current_tool_call_name,
                                    arguments=delta_message.tool_calls[
                                        0
                                    ].function.arguments,
                                    status="in_progress",
                                ),
                            )
                        )
                    elif delta_message.reasoning:
                        yield increment_sequence_number(
                            ResponseOutputItemAddedEvent(
                                type="response.output_item.added",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item=ResponseReasoningItem(
                                    type="reasoning",
                                    id=current_item_id,
                                    summary=[],
                                    status="in_progress",
                                ),
                            )
                        )
                        yield increment_sequence_number(
                            ResponseReasoningPartAddedEvent(
                                type="response.reasoning_part.added",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item_id=current_item_id,
                                content_index=current_content_index,
                                part=ResponseReasoningTextContent(
                                    text="",
                                    type="reasoning_text",
                                ),
                            )
                        )
                    elif not delta_message.tool_calls:
                        yield increment_sequence_number(
                            ResponseOutputItemAddedEvent(
                                type="response.output_item.added",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item=ResponseOutputMessage(
                                    id=current_item_id,
                                    type="message",
                                    role="assistant",
                                    content=[],
                                    status="in_progress",
                                ),
                            )
                        )
                        yield increment_sequence_number(
                            ResponseContentPartAddedEvent(
                                type="response.content_part.added",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item_id=current_item_id,
                                content_index=current_content_index,
                                part=ResponseOutputText(
                                    type="output_text",
                                    text="",
                                    annotations=[],
                                    logprobs=[],
                                ),
                            )
                        )
                    first_delta_sent = True

                # check delta message and previous delta message are
                # same as content or reasoning content
                if (
                    previous_delta_messages
                    and previous_delta_messages[-1].reasoning is not None
                    and delta_message.content is not None
                ):
                    # from reasoning to normal content, send done
                    # event for reasoning
                    reason_content = "".join(
                        pm.reasoning
                        for pm in previous_delta_messages
                        if pm.reasoning is not None
                    )

                    # delta message could have both reasoning and
                    # content. Include current delta's reasoning in the
                    # finalization since it may carry the tail end of
                    # reasoning text (e.g. when reasoning end and
                    # content start arrive in the same delta).
                    if delta_message.reasoning is not None:
                        yield increment_sequence_number(
                            ResponseReasoningTextDeltaEvent(
                                type="response.reasoning_text.delta",
                                sequence_number=-1,
                                content_index=current_content_index,
                                output_index=current_output_index,
                                item_id=current_item_id,
                                delta=delta_message.reasoning,
                            )
                        )
                        reason_content += delta_message.reasoning
                        delta_message = DeltaMessage(content=delta_message.content)

                    yield increment_sequence_number(
                        ResponseReasoningTextDoneEvent(
                            type="response.reasoning_text.done",
                            item_id=current_item_id,
                            sequence_number=-1,
                            output_index=current_output_index,
                            content_index=current_content_index,
                            text=reason_content,
                        )
                    )
                    yield increment_sequence_number(
                        ResponseReasoningPartDoneEvent(
                            type="response.reasoning_part.done",
                            sequence_number=-1,
                            item_id=current_item_id,
                            output_index=current_output_index,
                            content_index=current_content_index,
                            part=ResponseReasoningTextContent(
                                text=reason_content,
                                type="reasoning_text",
                            ),
                        )
                    )
                    current_content_index = 0
                    reasoning_item = ResponseReasoningItem(
                        type="reasoning",
                        content=[
                            ResponseReasoningTextContent(
                                text=reason_content,
                                type="reasoning_text",
                            ),
                        ],
                        status="completed",
                        id=current_item_id,
                        summary=[],
                    )
                    yield increment_sequence_number(
                        ResponseOutputItemDoneEvent(
                            type="response.output_item.done",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item=reasoning_item,
                        )
                    )
                    current_output_index += 1
                    current_item_id = str(uuid.uuid4())
                    yield increment_sequence_number(
                        ResponseOutputItemAddedEvent(
                            type="response.output_item.added",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item=ResponseOutputMessage(
                                id=current_item_id,
                                type="message",
                                role="assistant",
                                content=[],
                                status="in_progress",
                            ),
                        )
                    )
                    yield increment_sequence_number(
                        ResponseContentPartAddedEvent(
                            type="response.content_part.added",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                            content_index=current_content_index,
                            part=ResponseOutputText(
                                type="output_text",
                                text="",
                                annotations=[],
                                logprobs=[],
                            ),
                        )
                    )
                    # reset previous delta messages
                    previous_delta_messages = []
                if delta_message.tool_calls and delta_message.tool_calls[0].function:
                    if delta_message.tool_calls[0].function.arguments:
                        yield increment_sequence_number(
                            ResponseFunctionCallArgumentsDeltaEvent(
                                type=("response.function_call_arguments.delta"),
                                sequence_number=-1,
                                output_index=current_output_index,
                                item_id=current_item_id,
                                delta=delta_message.tool_calls[0].function.arguments,
                            )
                        )
                    # tool call initiated with no arguments
                    elif delta_message.tool_calls[0].function.name:
                        # send done with current content part
                        # and add new function call item
                        yield increment_sequence_number(
                            ResponseTextDoneEvent(
                                type="response.output_text.done",
                                sequence_number=-1,
                                output_index=current_output_index,
                                content_index=current_content_index,
                                text="",
                                logprobs=[],
                                item_id=current_item_id,
                            )
                        )
                        yield increment_sequence_number(
                            ResponseContentPartDoneEvent(
                                type="response.content_part.done",
                                sequence_number=-1,
                                item_id=current_item_id,
                                output_index=current_output_index,
                                content_index=current_content_index,
                                part=ResponseOutputText(
                                    type="output_text",
                                    text="",
                                    annotations=[],
                                    logprobs=[],
                                ),
                            )
                        )
                        yield increment_sequence_number(
                            ResponseOutputItemDoneEvent(
                                type="response.output_item.done",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item=ResponseOutputMessage(
                                    id=current_item_id,
                                    type="message",
                                    role="assistant",
                                    content=[],
                                    status="completed",
                                ),
                            )
                        )
                        current_output_index += 1
                        current_item_id = random_uuid()
                        assert delta_message.tool_calls[0].function is not None
                        current_tool_call_name = delta_message.tool_calls[
                            0
                        ].function.name
                        current_tool_call_id = f"call_{random_uuid()}"
                        yield increment_sequence_number(
                            ResponseOutputItemAddedEvent(
                                type="response.output_item.added",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item=ResponseFunctionToolCallItem(
                                    type="function_call",
                                    id=current_item_id,
                                    call_id=current_tool_call_id,
                                    name=current_tool_call_name,
                                    arguments="",
                                    status="in_progress",
                                ),
                            )
                        )
                        # skip content part for tool call
                        current_content_index = 1
                        continue
                elif delta_message.reasoning is not None:
                    yield increment_sequence_number(
                        ResponseReasoningTextDeltaEvent(
                            type="response.reasoning_text.delta",
                            sequence_number=-1,
                            content_index=current_content_index,
                            output_index=current_output_index,
                            item_id=current_item_id,
                            delta=delta_message.reasoning,
                        )
                    )
                elif delta_message.content:
                    yield increment_sequence_number(
                        ResponseTextDeltaEvent(
                            type="response.output_text.delta",
                            sequence_number=-1,
                            content_index=current_content_index,
                            output_index=current_output_index,
                            item_id=current_item_id,
                            delta=delta_message.content,
                            logprobs=(
                                create_logprobs(
                                    token_ids=output.token_ids,
                                    logprobs=output.logprobs,
                                    tokenizer=self.model_tokenizer,
                                    top_logprobs=top_logprobs,
                                )
                                if create_logprobs
                                else []
                            ),
                        )
                    )

                previous_delta_messages.append(delta_message)

        if previous_delta_messages:
            parts = []
            for pm in previous_delta_messages:
                if pm.tool_calls:
                    assert len(pm.tool_calls) == 1, (
                        "Multiple tool calls in one delta is not supported"
                    )
                    assert pm.tool_calls[0].function is not None, (
                        "Tool call without function is not supported"
                    )
                    parts.append(pm.tool_calls[0].function.arguments or "")

            tool_call_arguments = "".join(parts)
            if tool_call_arguments:
                yield increment_sequence_number(
                    ResponseFunctionCallArgumentsDoneEvent(
                        type="response.function_call_arguments.done",
                        sequence_number=-1,
                        output_index=current_output_index,
                        item_id=current_item_id,
                        arguments=tool_call_arguments,
                        name=current_tool_call_name,
                    )
                )
                current_content_index = 0
                function_call_item = ResponseFunctionToolCall(
                    type="function_call",
                    name=current_tool_call_name,
                    arguments=tool_call_arguments,
                    status="completed",
                    id=current_item_id,
                    call_id=current_tool_call_id,
                )
                yield increment_sequence_number(
                    ResponseOutputItemDoneEvent(
                        type="response.output_item.done",
                        sequence_number=-1,
                        output_index=current_output_index,
                        item=function_call_item,
                    )
                )

            elif previous_delta_messages[-1].reasoning is not None:
                reason_content = "".join(
                    pm.reasoning
                    for pm in previous_delta_messages
                    if pm.reasoning is not None
                )
                yield increment_sequence_number(
                    ResponseReasoningTextDoneEvent(
                        type="response.reasoning_text.done",
                        item_id=current_item_id,
                        sequence_number=-1,
                        output_index=current_output_index,
                        content_index=current_content_index,
                        text=reason_content,
                    )
                )
                yield increment_sequence_number(
                    ResponseReasoningPartDoneEvent(
                        type="response.reasoning_part.done",
                        sequence_number=-1,
                        item_id=current_item_id,
                        output_index=current_output_index,
                        content_index=current_content_index,
                        part=ResponseReasoningTextContent(
                            text=reason_content,
                            type="reasoning_text",
                        ),
                    )
                )
                reasoning_item = ResponseReasoningItem(
                    type="reasoning",
                    content=[
                        ResponseReasoningTextContent(
                            text=reason_content,
                            type="reasoning_text",
                        ),
                    ],
                    status="completed",
                    id=current_item_id,
                    summary=[],
                )
                yield increment_sequence_number(
                    ResponseOutputItemDoneEvent(
                        type="response.output_item.done",
                        sequence_number=-1,
                        output_index=current_output_index,
                        item=reasoning_item,
                    )
                )
            elif previous_delta_messages[-1].content:
                final_content = "".join(
                    pm.content for pm in previous_delta_messages if pm.content
                )
                yield increment_sequence_number(
                    ResponseTextDoneEvent(
                        type="response.output_text.done",
                        sequence_number=-1,
                        output_index=current_output_index,
                        content_index=current_content_index,
                        text=final_content,
                        logprobs=[],
                        item_id=current_item_id,
                    )
                )
                part = ResponseOutputText(
                    text=final_content,
                    type="output_text",
                    annotations=[],
                )
                yield increment_sequence_number(
                    ResponseContentPartDoneEvent(
                        type="response.content_part.done",
                        sequence_number=-1,
                        item_id=current_item_id,
                        output_index=current_output_index,
                        content_index=current_content_index,
                        part=part,
                    )
                )
                item = ResponseOutputMessage(
                    type="message",
                    role="assistant",
                    content=[
                        part,
                    ],
                    status="completed",
                    id=current_item_id,
                    summary=[],
                )
                yield increment_sequence_number(
                    ResponseOutputItemDoneEvent(
                        type="response.output_item.done",
                        sequence_number=-1,
                        output_index=current_output_index,
                        item=item,
                    )
                )


class _WrappedParser(DelegatingParser):
    """
    A DelegatingParser subclass that instantiates parsers from class attributes.

    This class is used to dynamically create a parser that wraps individual
    ReasoningParser and ToolParser classes. The class attributes
    `reasoning_parser_cls` and `tool_parser_cls` should be set before
    instantiation.

    Usage:
        _WrappedParser.reasoning_parser_cls = MyReasoningParser
        _WrappedParser.tool_parser_cls = MyToolParser
        parser = _WrappedParser(tokenizer)
    """

    reasoning_parser_cls: type[ReasoningParser] | None = None
    tool_parser_cls: type[ToolParser] | None = None

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)
        # Instantiate the underlying parsers from class attributes
        if self.__class__.reasoning_parser_cls is not None:
            self._reasoning_parser = self.__class__.reasoning_parser_cls(tokenizer)
        if self.__class__.tool_parser_cls is not None:
            self._tool_parser = self.__class__.tool_parser_cls(tokenizer)
