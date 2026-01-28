# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import logging
from abc import abstractmethod
from collections.abc import Sequence
from functools import cached_property

from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ToolChoiceFunction,
)
from openai.types.responses.response_output_text import Logprob
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningTextContent,
)
from pydantic import TypeAdapter

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
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
)
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import ToolParser
from vllm.utils import random_uuid

logger = logging.getLogger(__name__)


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
        model_output: str,
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
        model_output: str,
        request: ResponsesRequest,
        enable_auto_tools: bool = False,
        tool_call_id_type: str = "random",
        logprobs: list[Logprob] | None = None,
    ) -> list[ResponseOutputItem]:
        # First extract reasoning
        reasoning, content = self.extract_reasoning(model_output, request)

        # Then parse tool calls from the content
        tool_calls_result = self._parse_tool_calls(
            request=request,
            content=content,
            enable_auto_tools=enable_auto_tools,
        )

        # Update content if tool calls were extracted
        if tool_calls_result is not None:
            tool_calls, content = tool_calls_result
        else:
            tool_calls = None

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

        # Add tool call items
        if tool_calls:
            for idx, tool_call in enumerate(tool_calls):
                tool_call_item = ResponseFunctionToolCall(
                    id=f"fc_{random_uuid()}",
                    call_id=tool_call.id
                    if tool_call.id
                    else make_tool_call_id(
                        id_type=tool_call_id_type,
                        func_name=tool_call.name,
                        idx=idx,
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
    ) -> tuple[list[FunctionCall], str | None] | None:
        """
        Parse tool calls from content based on request tool_choice settings.

        Returns:
            A tuple of (function_calls, remaining_content) if tool calls
            were parsed, or None if no tool calls.
        """
        function_calls: list[FunctionCall] = []

        if request.tool_choice and isinstance(request.tool_choice, ToolChoiceFunction):
            # Forced Function Call (Responses API style)
            if content is None:
                return None
            function_calls.append(
                FunctionCall(name=request.tool_choice.name, arguments=content)
            )
            return function_calls, None

        if request.tool_choice and isinstance(
            request.tool_choice, ChatCompletionNamedToolChoiceParam
        ):
            # Forced Function Call (Chat Completion API style)
            if content is None:
                return None
            function_calls.append(
                FunctionCall(name=request.tool_choice.function.name, arguments=content)
            )
            return function_calls, None

        if request.tool_choice == "required":
            # Required tool calls - parse JSON
            if content is None:
                return None
            try:
                tool_calls_data = TypeAdapter(list[FunctionDefinition]).validate_json(
                    content
                )
                function_calls.extend(
                    FunctionCall(
                        name=tool_call.name,
                        arguments=json.dumps(tool_call.parameters, ensure_ascii=False),
                    )
                    for tool_call in tool_calls_data
                )
                return function_calls, None
            except Exception:
                logger.warning("Failed to parse required tool calls from content")
                return None

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
        return None

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
