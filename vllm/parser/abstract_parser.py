# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import json
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property

from openai.types.responses import ToolChoiceFunction
from pydantic import TypeAdapter, ValidationError

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
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.parser.metrics import record_tool_parser_invocation
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser
from vllm.tool_parsers.streaming import (
    extract_named_tool_call_streaming,
    extract_required_tool_call_streaming,
)

logger = init_logger(__name__)


@dataclass
class StreamState:
    """Mutable state for ``Parser.parse_delta()``. One per stream."""

    reasoning_ended: bool = False
    tool_call_text_started: bool = False
    prompt_reasoning_checked: bool = False
    previous_text: str = ""
    previous_token_ids: list[int] = field(default_factory=list)
    history_tool_call_cnt: int = 0
    tool_call_id_type: str = "random"
    # only used for "required" and "named tool" choices,
    # tracks whether function name has been fully returned in the stream yet
    function_name_returned: bool = False


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

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        *args,
        **kwargs,
    ):
        self.model_tokenizer = tokenizer
        self._reasoning_parser: ReasoningParser | None = None
        self._tool_parser: ToolParser | None = None
        self._stream_state = StreamState()

        if self.__class__.reasoning_parser_cls is not None:
            self._reasoning_parser = self.__class__.reasoning_parser_cls(
                tokenizer, *args, **kwargs
            )
        if self.__class__.tool_parser_cls is not None:
            self._tool_parser = self.__class__.tool_parser_cls(tokenizer, tools)

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
            A tuple of (reasoning, response_content).
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

    # ========== Tool Parser Methods ==========

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
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
        request: ChatCompletionRequest | ResponsesRequest,
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
        request: ChatCompletionRequest | ResponsesRequest,
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

    @abstractmethod
    def parse(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
        enable_auto_tools: bool = False,
    ) -> tuple[str | None, str | None, list[FunctionCall] | None]:
        """Parse a complete model output, extracting reasoning and tool calls.

        Args:
            model_output: The complete model-generated string.
            request: The request object used to generate the output.
            enable_auto_tools: Whether to enable automatic tool call parsing.

        Returns:
            A tuple of (reasoning, content, tool_calls).
        """

    @abstractmethod
    def parse_delta(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        request: ChatCompletionRequest | ResponsesRequest,
        prompt_token_ids: list[int] | None = None,
        *,
        finished: bool,
    ) -> DeltaMessage | None:
        """Parse a single streaming delta, orchestrating reasoning then
        tool call extraction via internal stream state.
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

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        if self._reasoning_parser is None:
            return None, model_output
        return self._reasoning_parser.extract_reasoning(model_output, request)

    def _get_function_name(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> str:
        if request.tool_choice and isinstance(request.tool_choice, ToolChoiceFunction):
            return request.tool_choice.name
        if request.tool_choice and isinstance(
            request.tool_choice, ChatCompletionNamedToolChoiceParam
        ):
            return request.tool_choice.function.name
        raise ValueError("Invalid tool_choice for function name extraction.")

    def _extract_tool_calls(
        self,
        content: str | None,
        request: ChatCompletionRequest | ResponsesRequest,
        enable_auto_tools: bool = False,
    ) -> tuple[list[FunctionCall] | None, str | None]:
        tool_parser = self._tool_parser
        if tool_parser is None:
            return [], content

        supports_required_and_named = tool_parser.supports_required_and_named
        is_named_tool_choice = request.tool_choice and isinstance(
            request.tool_choice,
            (ToolChoiceFunction, ChatCompletionNamedToolChoiceParam),
        )
        is_required_tool_choice = request.tool_choice == "required"
        is_auto_tool_choice = enable_auto_tools and (
            request.tool_choice == "auto"
            or request.tool_choice is None
            or (
                not supports_required_and_named
                and (is_named_tool_choice or is_required_tool_choice)
            )
        )

        tool_calls = list[FunctionCall]()
        if is_named_tool_choice and supports_required_and_named:
            if content is None:
                return [], None
            tool_calls.append(
                FunctionCall(
                    name=self._get_function_name(request),
                    arguments=content,
                )
            )
            content = None
        elif is_required_tool_choice and supports_required_and_named:
            # "required" with standard JSON-based parsing
            parsed_calls = []
            with contextlib.suppress(ValidationError):
                content = content or ""
                parsed_calls = TypeAdapter(list[FunctionDefinition]).validate_json(
                    content
                )
            for tc in parsed_calls:
                tool_calls.append(
                    FunctionCall(
                        name=tc.name,
                        arguments=json.dumps(tc.parameters, ensure_ascii=False),
                    )
                )
            content = None
        elif is_auto_tool_choice:
            # Automatic Tool Call Parsing (also used as fallback for
            # required/named when supports_required_and_named=False)
            tool_call_info = self.extract_tool_calls(
                content if content is not None else "",
                request=request,
            )
            if tool_call_info is not None and tool_call_info.tools_called:
                tool_calls.extend(
                    FunctionCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    )
                    for tc in tool_call_info.tool_calls
                )
                content = tool_call_info.content
                if content and content.strip() == "":
                    content = None
            else:
                # No tool calls.
                return None, content

        return tool_calls, content

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        if self._reasoning_parser is not None:
            request = self._reasoning_parser.adjust_request(request)
        if self._tool_parser is not None:
            request = self._tool_parser.adjust_request(request)
        return request

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
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> ExtractedToolCallInformation:
        if self._tool_parser is None:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        result = None
        is_tool_called: bool | Exception = False
        try:
            result = self._tool_parser.extract_tool_calls(
                model_output,
                request=request,  # type: ignore[arg-type]
            )
            is_tool_called = bool(result.tools_called)
        except Exception as e:
            is_tool_called = e
            raise
        finally:
            record_tool_parser_invocation(
                is_tool_called=is_tool_called,
                is_streaming=False,
                request=request,
            )
        return result

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
        if self._tool_parser is None:
            return None
        result = None
        is_tool_called: bool | Exception = False
        try:
            result = self._tool_parser.extract_tool_calls_streaming(
                previous_text,
                current_text,
                delta_text,
                previous_token_ids,
                current_token_ids,
                delta_token_ids,
                request,  # type: ignore[arg-type]
            )
            is_tool_called = bool(result and result.tool_calls)
        except Exception as e:
            is_tool_called = e
            raise
        finally:
            record_tool_parser_invocation(
                is_tool_called=is_tool_called,
                is_streaming=True,
                request=request,
            )
        return result

    def _extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest | ResponsesRequest,
        # The following parameters are used for "required" tool choice parsing and are
        # tracked in StreamState for streaming parsing.
        tool_call_idx: int | None = None,
        tool_call_id_type: str = "random",
        function_name_returned: bool = False,
    ) -> tuple[DeltaMessage | None, bool]:
        if request.tool_choice == "none":
            return (DeltaMessage(content=delta_text) if delta_text else None), False

        assert self._tool_parser is not None
        supports_required_and_named = self._tool_parser.supports_required_and_named
        if (
            supports_required_and_named
            and request.tool_choice
            and isinstance(
                request.tool_choice,
                (ToolChoiceFunction, ChatCompletionNamedToolChoiceParam),
            )
        ):
            delta_message, function_name_returned = extract_named_tool_call_streaming(
                delta_text=delta_text,
                function_name=self._get_function_name(request),
                function_name_returned=function_name_returned,
                tool_call_idx=tool_call_idx,
                tool_call_id_type=tool_call_id_type,
                tokenizer=self.model_tokenizer,
            )
            return delta_message, function_name_returned

        if supports_required_and_named and request.tool_choice == "required":
            delta_message, function_name_returned = (
                extract_required_tool_call_streaming(
                    previous_text=previous_text,
                    current_text=current_text,
                    delta_text=delta_text,
                    function_name_returned=function_name_returned,
                    tool_call_idx=tool_call_idx,
                    tool_call_id_type=tool_call_id_type,
                )
            )
            return delta_message, function_name_returned
        return self.extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request,
        ), False

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        if self._reasoning_parser is None:
            return False
        return self._reasoning_parser.is_reasoning_end(input_ids)

    def is_reasoning_end_streaming(
        self, input_ids: list[int], delta_ids: list[int]
    ) -> bool:
        if self._reasoning_parser is None:
            return False
        return self._reasoning_parser.is_reasoning_end_streaming(input_ids, delta_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if self._reasoning_parser is None:
            return input_ids
        return self._reasoning_parser.extract_content_ids(input_ids)

    def _in_reasoning_phase(self, state: StreamState) -> bool:
        if self._reasoning_parser is None:
            return False
        return not state.reasoning_ended

    def _in_tool_call_phase(self, state: StreamState) -> bool:
        if self._tool_parser is None:
            return False
        return state.reasoning_ended

    def _append_unstreamed_tool_args(
        self,
        delta_message: DeltaMessage | None,
    ) -> None:
        """Append parsed-but-unstreamed tool-call arguments to *delta_message*."""
        if (
            self._tool_parser is not None
            and delta_message
            and delta_message.tool_calls
            and (last_tc := delta_message.tool_calls[-1]).function
        ):
            last_tc.function.arguments = (
                last_tc.function.arguments or ""
            ) + self._tool_parser.get_remaining_unstreamed_args()

    def finalize_generation(
        self,
        delta_message: DeltaMessage | None,
        request: ChatCompletionRequest | ResponsesRequest,
        state: StreamState,
    ) -> DeltaMessage | None:
        """Finalize generation for cases where generation was incomplete.
        For example, if streaming terminated before reasoning ended
        """
        fallback_fn = getattr(
            self._reasoning_parser, "get_streaming_fallback_content", None
        )
        if fallback_fn is not None and not state.reasoning_ended:
            promoted = fallback_fn(state.previous_text, request)
            if promoted:
                if delta_message is None:
                    delta_message = DeltaMessage()
                delta_message.content = (delta_message.content or "") + promoted

        self._append_unstreamed_tool_args(delta_message)
        return delta_message

    def parse(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
        enable_auto_tools: bool = False,
    ) -> tuple[str | None, str | None, list[FunctionCall] | None]:
        reasoning, content = self.extract_reasoning(model_output, request)
        tool_calls, content = self._extract_tool_calls(
            content=content,
            request=request,
            enable_auto_tools=enable_auto_tools,
        )
        return reasoning, content, tool_calls

    def parse_delta(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        request: ChatCompletionRequest | ResponsesRequest,
        prompt_token_ids: list[int] | None = None,
        *,
        finished: bool,
    ) -> DeltaMessage | None:
        state = self._stream_state

        if not state.prompt_reasoning_checked and prompt_token_ids is not None:
            state.prompt_reasoning_checked = True
            if self._reasoning_parser is None or self.is_reasoning_end(
                prompt_token_ids
            ):
                state.reasoning_ended = True

        current_text = state.previous_text + delta_text
        current_token_ids = state.previous_token_ids + delta_token_ids
        delta_message: DeltaMessage | None = None

        # Reasoning extraction
        if self._in_reasoning_phase(state):
            delta_message = self.extract_reasoning_streaming(
                previous_text=state.previous_text,
                current_text=current_text,
                delta_text=delta_text,
                previous_token_ids=state.previous_token_ids,
                current_token_ids=current_token_ids,
                delta_token_ids=delta_token_ids,
            )
            if self.is_reasoning_end_streaming(current_token_ids, delta_token_ids):
                state.reasoning_ended = True
                current_token_ids = self.extract_content_ids(delta_token_ids)
                current_text = (
                    delta_message.content
                    if delta_message and delta_message.content
                    else ""
                )
                delta_text = current_text
                delta_token_ids = current_token_ids

        # Tool call extraction
        if self._in_tool_call_phase(state):
            if not state.tool_call_text_started:
                state.tool_call_text_started = True
                state.previous_text = ""
                state.previous_token_ids = []
                delta_text = current_text
                delta_token_ids = current_token_ids

            # A boundary delta may carry both reasoning and tool call,
            # save it before the tool parser overwrites delta_message.
            reasoning = delta_message.reasoning if delta_message else None
            delta_message, state.function_name_returned = (
                self._extract_tool_calls_streaming(
                    previous_text=state.previous_text,
                    current_text=current_text,
                    delta_text=delta_text,
                    previous_token_ids=state.previous_token_ids,
                    current_token_ids=current_token_ids,
                    delta_token_ids=delta_token_ids,
                    request=request,  # type: ignore[arg-type]
                    tool_call_idx=state.history_tool_call_cnt,
                    tool_call_id_type=state.tool_call_id_type,
                    function_name_returned=state.function_name_returned,
                )
            )
            if reasoning:
                if not delta_message:
                    delta_message = DeltaMessage()
                delta_message.reasoning = reasoning

            if (
                delta_message
                and delta_message.tool_calls
                and delta_message.tool_calls[0].id is not None
            ):
                state.history_tool_call_cnt += 1

        # No phase active: pass through as content
        if (
            delta_message is None
            and not self._in_reasoning_phase(state)
            and not self._in_tool_call_phase(state)
        ):
            delta_message = DeltaMessage(content=delta_text)

        state.previous_text = current_text
        state.previous_token_ids = current_token_ids

        if finished:
            delta_message = self.finalize_generation(delta_message, request, state)

        return delta_message
