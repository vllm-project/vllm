# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
from collections.abc import Callable
from typing import Literal

from openai.types.responses import ResponseFunctionToolCall, ResponseOutputItem
from openai.types.responses.response_function_tool_call_output_item import (
    ResponseFunctionToolCallOutputItem,
)
from openai.types.responses.response_output_item import McpCall
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.response_reasoning_item import (
    Content,
    ResponseReasoningItem,
)

from vllm.entrypoints.constants import MCP_PREFIX
from vllm.entrypoints.openai.protocol import (
    DeltaMessage,
    ResponseInputOutputItem,
    ResponsesRequest,
)
from vllm.outputs import CompletionOutput
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import ToolParser
from vllm.utils import random_uuid

logger = logging.getLogger(__name__)


class ResponsesParser:
    """Incremental parser over completion tokens with reasoning support."""

    def __init__(
        self,
        *,
        tokenizer: TokenizerLike,
        reasoning_parser_cls: Callable[[TokenizerLike], ReasoningParser],
        response_messages: list[ResponseInputOutputItem],
        request: ResponsesRequest,
        tool_parser_cls: Callable[[TokenizerLike], ToolParser] | None,
    ):
        self.response_messages: list[ResponseInputOutputItem] = (
            # TODO: initial messages may not be properly typed
            response_messages
        )
        self.num_init_messages = len(response_messages)
        self.tokenizer = tokenizer
        self.request = request

        self.reasoning_parser_instance = reasoning_parser_cls(tokenizer)
        self.tool_parser_instance = None
        if tool_parser_cls is not None:
            self.tool_parser_instance = tool_parser_cls(tokenizer)

    def process(self, output: CompletionOutput) -> "ResponsesParser":
        reasoning_content, content = self.reasoning_parser_instance.extract_reasoning(
            output.text, request=self.request
        )
        if reasoning_content:
            self.response_messages.append(
                ResponseReasoningItem(
                    type="reasoning",
                    id=f"rs_{random_uuid()}",
                    summary=[],
                    content=[
                        Content(
                            type="reasoning_text",
                            text=reasoning_content,
                        )
                    ],
                )
            )

        function_calls: list[ResponseFunctionToolCall] = []
        if self.tool_parser_instance is not None:
            tool_call_info = self.tool_parser_instance.extract_tool_calls(
                content if content is not None else "",
                request=self.request,  # type: ignore
            )
            if tool_call_info is not None and tool_call_info.tools_called:
                # extract_tool_calls() returns a list of tool calls.
                function_calls.extend(
                    ResponseFunctionToolCall(
                        id=f"fc_{random_uuid()}",
                        call_id=f"call_{random_uuid()}",
                        type="function_call",
                        status="completed",
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                    )
                    for tool_call in tool_call_info.tool_calls
                )
                content = tool_call_info.content
                if content and content.strip() == "":
                    content = None

        if content:
            self.response_messages.append(
                ResponseOutputMessage(
                    type="message",
                    id=f"msg_{random_uuid()}",
                    status="completed",
                    role="assistant",
                    content=[
                        ResponseOutputText(
                            annotations=[],  # TODO
                            type="output_text",
                            text=content,
                            logprobs=None,  # TODO
                        )
                    ],
                )
            )
        if len(function_calls) > 0:
            self.response_messages.extend(function_calls)

        return self

    def make_response_output_items_from_parsable_context(
        self,
    ) -> list[ResponseOutputItem]:
        """Given a list of sentences, construct ResponseOutput Items."""
        response_messages = self.response_messages[self.num_init_messages :]
        output_messages: list[ResponseOutputItem] = []
        for message in response_messages:
            if not isinstance(message, ResponseFunctionToolCallOutputItem):
                output_messages.append(message)
            else:
                if len(output_messages) == 0:
                    raise ValueError(
                        "Cannot have a FunctionToolCallOutput before FunctionToolCall."
                    )
                if isinstance(output_messages[-1], ResponseFunctionToolCall):
                    mcp_message = McpCall(
                        id=f"{MCP_PREFIX}{random_uuid()}",
                        arguments=output_messages[-1].arguments,
                        name=output_messages[-1].name,
                        server_label=output_messages[
                            -1
                        ].name,  # TODO: store the server label
                        type="mcp_call",
                        status="completed",
                        output=message.output,
                        # TODO: support error output
                    )
                    output_messages[-1] = mcp_message

        return output_messages


class StreamableResponsesParser:
    """Streaming parser that processes output tokens one by one."""

    def __init__(
        self,
        *,
        tokenizer: TokenizerLike,
        response_messages: list[ResponseInputOutputItem],
        reasoning_parser_cls: Callable[[TokenizerLike], ReasoningParser],
        request: ResponsesRequest,
        tool_parser_cls: Callable[[TokenizerLike], ToolParser] | None,
    ):
        # Store initial messages (can be appended to for tool outputs)
        self.response_messages: list[ResponseInputOutputItem] = [
            *response_messages,
        ]
        self.num_init_messages = len(response_messages)
        self.response_delta_messages: list[DeltaMessage] = []
        self.tokenizer = tokenizer
        self.request = request

        self.reasoning_parser_cls = reasoning_parser_cls
        self.reasoning_parser_instance = reasoning_parser_cls(tokenizer)
        self.tool_parser_instance = None
        self.tool_parser_cls = tool_parser_cls
        if tool_parser_cls is not None:
            self.tool_parser_instance = tool_parser_cls(tokenizer)

        # Accumulated state for current turn
        self._previous_output_token_ids: list[int] = []
        self._previous_output_text: str = ""

        # Track whether reasoning has ended
        self._reasoning_ended: bool = False

        # Accumulated content after reasoning ended (for tool parsing)
        self._previous_content_text: str = ""
        self._previous_content_token_ids: list[int] = []

        # if token cannot be decoded into valid text, cache it first
        self._cached_uncompleted_tokens: list[int] = []

        # like openharmony
        # analysis: current token is reasoning token
        # commentary: current token is tool call token
        # final: current token is to be output to user
        self._current_channel: Literal["final", "analysis", "commentary"] = "final"

    @property
    def current_channel(self) -> Literal["final", "analysis", "commentary"]:
        return self._current_channel

    def process(self, token_id: int) -> "StreamableResponsesParser":
        """Process output token one by one and return delta message.

        Args:
            token_id: The token id to process
            is_final: Whether this is the final token of the output

        Returns:
            DeltaMessage containing the delta for this token, or None if no output
        """
        # ignore special tokens
        if token_id in self.tokenizer.all_special_ids:
            self._current_channel = "final"
            return self
        # Decode the token to get delta_text
        delta_token_ids = self._cached_uncompleted_tokens + [token_id]
        delta_text = self.tokenizer.decode(delta_token_ids, skip_special_tokens=True)
        # if text ends with replacement character, cache the token id and return
        if delta_text and delta_text[-1] == "\ufffd":
            self._cached_uncompleted_tokens.append(token_id)
            # Reasoning text must be the first token in the output,
            # or the following code maybe wrong
            if not self._reasoning_ended:
                self.response_delta_messages.append(DeltaMessage(reasoning=""))
            else:
                self.response_delta_messages.append(DeltaMessage(content=""))
            return self
        # clear cached tokens
        self._cached_uncompleted_tokens = []

        # Update current state
        current_token_ids = self._previous_output_token_ids + delta_token_ids
        current_text = self._previous_output_text + delta_text

        delta_message: DeltaMessage | None = None
        content_delta: str | None = None

        # If reasoning already ended, skip reasoning parser
        # and treat as content directly
        if self._reasoning_ended:
            content_delta = delta_text
        else:
            # Parse reasoning using ReasoningParser
            reasoning_delta = (
                self.reasoning_parser_instance.extract_reasoning_streaming(
                    previous_text=self._previous_output_text,
                    current_text=current_text,
                    delta_text=delta_text,
                    previous_token_ids=self._previous_output_token_ids,
                    current_token_ids=current_token_ids,
                    delta_token_ids=delta_token_ids,
                )
            )

            # Check if reasoning has ended with this token
            is_end = self.reasoning_parser_instance.is_reasoning_end_streaming(
                current_token_ids, delta_token_ids
            )
            if is_end:
                self._reasoning_ended = True

            # Process reasoning parser output
            if reasoning_delta is not None:
                if reasoning_delta.reasoning is not None:
                    # Still in reasoning phase
                    delta_message = DeltaMessage(reasoning=reasoning_delta.reasoning)
                    self._current_channel = "analysis"
                elif reasoning_delta.content is not None:
                    # Content from reasoning parser (after reasoning or no reasoning)
                    content_delta = reasoning_delta.content
                    # Assume reasoning is done if content is produced
                    self._reasoning_ended = True
            else:
                # DeltaMessage with empty string to indicate
                # this token is reasoning token
                delta_message = DeltaMessage(reasoning="")
                self._current_channel = "analysis"

        # Process content (either from reasoning parser or direct)
        if content_delta is not None:
            # Update content tracking for tool parsing
            current_content_text = self._previous_content_text + content_delta
            current_content_token_ids = (
                self._previous_content_token_ids + delta_token_ids
            )

            # Try to parse tool calls from content
            if self.tool_parser_instance is not None:
                tool_delta = self.tool_parser_instance.extract_tool_calls_streaming(
                    previous_text=self._previous_content_text,
                    current_text=current_content_text,
                    delta_text=content_delta,
                    previous_token_ids=self._previous_content_token_ids,
                    current_token_ids=current_content_token_ids,
                    delta_token_ids=delta_token_ids,
                    # TODO： currently, only some tool parser need to access
                    # request.tools in extract_tool_calls_streaming, and
                    # ResponsesRequest have this attribute too. Maybe we should
                    # update ToolParser params to allow ResponsesRequest along
                    # with ChatCompletionRequest.
                    request=self.request,  # type: ignore
                )

                if tool_delta is not None:
                    # Use tool parser's output (may contain tool_calls and/or content)
                    delta_message = DeltaMessage(
                        content=tool_delta.content,
                        tool_calls=tool_delta.tool_calls,
                    )
                    self._current_channel = "commentary"
                else:
                    # tool_delta is None means tool parser is analyzing the text,
                    # and content is cached
                    delta_message = DeltaMessage(content="")
                    self._current_channel = "final"
            else:
                delta_message = DeltaMessage(content=content_delta)
                self._current_channel = "final"

            # Update content state
            self._previous_content_text = current_content_text
            self._previous_content_token_ids = current_content_token_ids

        # Update previous state for next iteration
        self._previous_output_token_ids = current_token_ids
        self._previous_output_text = current_text

        # Store delta message, which must not be none
        assert delta_message is not None
        self.response_delta_messages.append(delta_message)

        return self

    def reset(self) -> None:
        """Reset parser state for a new turn."""
        self._previous_output_token_ids = []
        self._previous_output_text = ""
        self._reasoning_ended = False
        self._previous_content_text = ""
        self._previous_content_token_ids = []
        self.response_delta_messages = []
        self._cached_uncompleted_tokens = []
        if self.tool_parser_cls is not None:
            self.tool_parser_instance = self.tool_parser_cls(self.tokenizer)
        if self.reasoning_parser_cls is not None:
            self.reasoning_parser_instance = self.reasoning_parser_cls(self.tokenizer)

    @property
    def final_output(self) -> list[ResponseInputOutputItem]:
        """Merge all delta messages into final output items.

        This converts streaming delta messages into the same format as
        ResponsesParser.process() produces for non-streaming responses.
        Can be used for both non-streaming final output and streaming
        response.completed event.

        Returns:
            List of ResponseInputOutputItem containing:
            - ResponseReasoningItem (if any reasoning content)
            - ResponseOutputMessage (if any text content)
            - ResponseFunctionToolCall (for each tool call)
        """
        logger.debug("total output text: %s", self._previous_output_text)
        output_items: list[ResponseInputOutputItem] = []

        # 1. Merge all reasoning deltas into a single ResponseReasoningItem
        reasoning_text = self.accumulated_reasoning
        if reasoning_text:
            output_items.append(
                ResponseReasoningItem(
                    type="reasoning",
                    id=f"rs_{random_uuid()}",
                    summary=[],
                    content=[
                        Content(
                            type="reasoning_text",
                            text=reasoning_text,
                        )
                    ],
                )
            )

        # 2. Merge all tool calls from deltas by index
        # Tool calls are accumulated incrementally by their index
        tool_calls_by_index: dict[int, dict] = {}
        for dm in self.response_delta_messages:
            if dm.tool_calls:
                for tc in dm.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_by_index:
                        tool_calls_by_index[idx] = {
                            "id": None,
                            "name": "",
                            "arguments": "",
                        }
                    # Update id if provided
                    if tc.id is not None:
                        tool_calls_by_index[idx]["id"] = tc.id
                    # Accumulate function name and arguments
                    if tc.function is not None:
                        if tc.function.name is not None:
                            tool_calls_by_index[idx]["name"] += tc.function.name
                        if tc.function.arguments is not None:
                            tool_calls_by_index[idx]["arguments"] += (
                                tc.function.arguments
                            )

        # 3. Get content text
        content_text = self.accumulated_content.strip()

        # 4. Create ResponseOutputMessage if there's content
        if content_text:
            output_items.append(
                ResponseOutputMessage(
                    type="message",
                    id=f"msg_{random_uuid()}",
                    status="completed",
                    role="assistant",
                    content=[
                        ResponseOutputText(
                            annotations=[],
                            type="output_text",
                            text=content_text,
                            logprobs=None,
                        )
                    ],
                )
            )

        # 5. Create ResponseFunctionToolCall for each accumulated tool call
        for idx in sorted(tool_calls_by_index.keys()):
            tc_data = tool_calls_by_index[idx]
            output_items.append(
                ResponseFunctionToolCall(
                    id=tc_data["id"] or f"fc_{random_uuid()}",
                    call_id=f"call_{random_uuid()}",
                    type="function_call",
                    status="completed",
                    name=tc_data["name"],
                    arguments=tc_data["arguments"],
                )
            )

        return output_items

    @property
    def accumulated_reasoning(self) -> str:
        """Return all accumulated reasoning text."""
        return "".join(
            dm.reasoning
            for dm in self.response_delta_messages
            if dm.reasoning is not None
        )

    @property
    def accumulated_content(self) -> str:
        """Return all accumulated content text."""
        return "".join(
            dm.content for dm in self.response_delta_messages if dm.content is not None
        )

    def make_response_output_items_from_parsable_context(
        self,
    ) -> list[ResponseOutputItem]:
        """Given a list of sentences, construct ResponseOutput Items."""
        response_messages = self.response_messages[self.num_init_messages :]
        output_messages: list[ResponseOutputItem] = []
        for message in response_messages:
            if not isinstance(message, ResponseFunctionToolCallOutputItem):
                output_messages.append(message)
            else:
                if len(output_messages) == 0:
                    raise ValueError(
                        "Cannot have a FunctionToolCallOutput before FunctionToolCall."
                    )
                if isinstance(output_messages[-1], ResponseFunctionToolCall):
                    mcp_message = McpCall(
                        id=f"{MCP_PREFIX}{random_uuid()}",
                        arguments=output_messages[-1].arguments,
                        name=output_messages[-1].name,
                        server_label=output_messages[
                            -1
                        ].name,  # TODO: store the server label
                        type="mcp_call",
                        status="completed",
                        output=message.output,
                        # TODO: support error output
                    )
                    output_messages[-1] = mcp_message

        return output_messages


def get_responses_parser_for_simple_context(
    *,
    tokenizer: TokenizerLike,
    reasoning_parser_cls: Callable[[TokenizerLike], ReasoningParser],
    response_messages: list[ResponseInputOutputItem],
    request: ResponsesRequest,
    tool_parser_cls,
) -> ResponsesParser:
    """Factory function to create a ResponsesParser with
    optional reasoning parser.

    Returns:
        ResponsesParser instance configured with the provided parser
    """
    return ResponsesParser(
        tokenizer=tokenizer,
        reasoning_parser_cls=reasoning_parser_cls,
        response_messages=response_messages,
        request=request,
        tool_parser_cls=tool_parser_cls,
    )


def get_streamable_responses_parser(
    *,
    tokenizer: TokenizerLike,
    reasoning_parser_cls: Callable[[TokenizerLike], ReasoningParser],
    response_messages: list[ResponseInputOutputItem],
    request: ResponsesRequest,
    tool_parser_cls: Callable[[TokenizerLike], ToolParser] | None,
) -> StreamableResponsesParser:
    """Factory function to create a StreamableResponsesParser.

    Returns:
        StreamableResponsesParser instance configured with the provided parsers
    """
    return StreamableResponsesParser(
        tokenizer=tokenizer,
        response_messages=response_messages,
        reasoning_parser_cls=reasoning_parser_cls,
        request=request,
        tool_parser_cls=tool_parser_cls,
    )
