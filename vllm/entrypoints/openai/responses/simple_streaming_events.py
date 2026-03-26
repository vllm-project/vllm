# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field

from openai.types.responses import (
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallItem,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)
from openai.types.responses.response_content_part_added_event import (
    PartReasoningText as AddedPartReasoningText,
)
from openai.types.responses.response_content_part_done_event import (
    PartReasoningText as DonePartReasoningText,
)
from openai.types.responses.response_reasoning_item import (
    Content as ReasoningContent,
)

from vllm.entrypoints.openai.engine.protocol import DeltaMessage, GenerationError
from vllm.entrypoints.openai.responses.context import SimpleContext
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
    StreamingResponsesResponse,
)
from vllm.logger import init_logger
from vllm.logprobs import SampleLogprobs
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser
from vllm.utils import random_uuid
from vllm.utils.collection_utils import as_list

logger = init_logger(__name__)


@dataclass
class SimpleStreamingState:
    """Mutable state for simple streaming event processing."""

    current_content_index: int = 0
    current_output_index: int = 0
    current_item_id: str = ""
    current_tool_call_name: str = ""
    current_tool_call_id: str = ""
    prompt_is_reasoning_end: bool | None = None
    reasoning_ended: bool = False
    tool_call_text_started: bool = False
    first_tool_call_started: bool = False
    event_stream_started: bool = False
    maybe_important_output_text_spaces: str = ""
    previous_delta_messages: list[DeltaMessage] = field(default_factory=list)
    previous_text: str = ""
    previous_token_ids: list[int] = field(default_factory=list)
    started_tool_call_indices: set[int] = field(default_factory=set)


def _parse_delta_message(
    request: "ResponsesRequest | None",
    output,
    reasoning_parser: "ReasoningParser | None",
    tool_parser: "ToolParser | None",
    state: SimpleStreamingState,
) -> tuple[DeltaMessage | None, str, list[int]]:
    """Parse a generation output into a DeltaMessage.

    Extracts reasoning, tool calls, and content from the raw output using
    the appropriate parsers.
    """
    delta_text = output.text
    delta_token_ids = as_list(output.token_ids)
    current_text = state.previous_text + delta_text
    current_token_ids = state.previous_token_ids + delta_token_ids

    delta_message: DeltaMessage | None = None

    if reasoning_parser and tool_parser:
        # reasoning parser didn't see the reasoning end marker, because
        # it was already in the prompt.
        if state.prompt_is_reasoning_end:
            state.reasoning_ended = True
        # Track content extracted when reasoning ends, so it can be restored
        # if the tool_parser doesn't find a tool call
        extracted_content: str | None = None
        if not state.reasoning_ended:
            delta_message = reasoning_parser.extract_reasoning_streaming(
                previous_text=state.previous_text,
                current_text=current_text,
                delta_text=delta_text,
                previous_token_ids=state.previous_token_ids,
                current_token_ids=current_token_ids,
                delta_token_ids=delta_token_ids,
            )
            if reasoning_parser.is_reasoning_end(delta_token_ids):
                state.reasoning_ended = True
                current_token_ids = reasoning_parser.extract_content_ids(
                    delta_token_ids
                )
                # Save the content before stripping it, so it can be restored
                # if the tool_parser doesn't find a tool call
                extracted_content = delta_message.content if delta_message else None
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
            potential_reasoning_delta = delta_message
            delta_message = tool_parser.extract_tool_calls_streaming(
                previous_text=state.previous_text,
                current_text=current_text,
                delta_text=delta_text,
                previous_token_ids=state.previous_token_ids,
                current_token_ids=current_token_ids,
                delta_token_ids=delta_token_ids,
                request=request,  # type: ignore[arg-type]
            )
            if delta_message is None:
                delta_message = potential_reasoning_delta
            # If the tool_parser returned None and we had extracted content,
            # restore it to the delta_message so it can be emitted as text
            if (
                delta_message
                and delta_message is potential_reasoning_delta
                and extracted_content
            ):
                delta_message.content = extracted_content

    elif reasoning_parser:
        delta_message = reasoning_parser.extract_reasoning_streaming(
            previous_text=state.previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=state.previous_token_ids,
            current_token_ids=current_token_ids,
            delta_token_ids=delta_token_ids,
        )
    elif tool_parser:
        delta_message = tool_parser.extract_tool_calls_streaming(
            previous_text=state.previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=state.previous_token_ids,
            current_token_ids=current_token_ids,
            delta_token_ids=delta_token_ids,
            request=request,  # type: ignore[arg-type]
        )
    else:
        delta_message = DeltaMessage(
            content=output.text,
        )

    if delta_message is None:
        return None, current_text, current_token_ids

    # parser/model quirk: handle tool call related quirks
    # (applies when tool_parser is configured, regardless of reasoning_parser)
    if tool_parser and delta_message:
        # remove any spurious output text after the first tool call has
        # started streaming
        if state.first_tool_call_started:
            delta_message.content = None
        # skip whitespaces only output text before tool calls (otherwise we stream
        # phantom messages with, for example, just 2 newlines).
        # TODO: might be tool_parser responsibility?
        if (
            delta_message.content
            and (
                not state.previous_delta_messages
                or (
                    state.previous_delta_messages
                    and not state.previous_delta_messages[-1].content
                )
            )
            and delta_message.content.isspace()
        ):
            state.maybe_important_output_text_spaces += delta_message.content
            delta_message.content = None
        # restore accumulated whitespace on first non-whitespace content
        elif (
            delta_message.content
            and state.maybe_important_output_text_spaces
            and (
                not state.previous_delta_messages
                or not state.previous_delta_messages[-1].content
            )
        ):
            delta_message.content = (
                state.maybe_important_output_text_spaces + delta_message.content
            )
            state.maybe_important_output_text_spaces = ""

    if delta_message.tool_calls:
        assert len(delta_message.tool_calls) == 1, (
            "Multiple tool calls in one delta is not supported"
        )
        assert not (delta_message.content and delta_message.reasoning), (
            "Tool call with output_text and reasoning in same delta is not supported"
        )
        assert delta_message.tool_calls[0].function is not None, (
            "Tool call without function is not supported"
        )

    # filter empty text output/reasoning
    if delta_message and delta_message.content and delta_message.content == "":
        delta_message.content = None

    if delta_message and delta_message.reasoning and delta_message.reasoning == "":
        delta_message.reasoning = None
    # filter empty delta
    if (
        delta_message
        and delta_message.reasoning is None
        and delta_message.content is None
        and not delta_message.tool_calls
    ):
        delta_message = None

    return delta_message, current_text, current_token_ids


def _events_add_content_reasoning_item(
    state: SimpleStreamingState,
) -> list[StreamingResponsesResponse]:
    return [
        ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            sequence_number=-1,
            output_index=state.current_output_index,
            item=ResponseReasoningItem(
                type="reasoning",
                id=state.current_item_id,
                summary=[],
                status="in_progress",
            ),
        ),
        ResponseContentPartAddedEvent(
            type="response.content_part.added",
            sequence_number=-1,
            output_index=state.current_output_index,
            item_id=state.current_item_id,
            content_index=state.current_content_index,
            part=AddedPartReasoningText(
                type="reasoning_text",
                text="",
            ),
        ),
    ]


def _event_reasoning_delta(
    state: SimpleStreamingState, delta_message: DeltaMessage
) -> StreamingResponsesResponse:
    return ResponseReasoningTextDeltaEvent(
        type="response.reasoning_text.delta",
        sequence_number=-1,
        content_index=state.current_content_index,
        output_index=state.current_output_index,
        item_id=state.current_item_id,
        delta=delta_message.reasoning,
    )


def _events_reasoning_done(
    state: SimpleStreamingState,
    reason_content: str,
) -> list[StreamingResponsesResponse]:
    """Events when reasoning completes."""

    return [
        ResponseReasoningTextDoneEvent(
            type="response.reasoning_text.done",
            item_id=state.current_item_id,
            sequence_number=-1,
            output_index=state.current_output_index,
            content_index=state.current_content_index,
            text=reason_content,
        ),
        ResponseContentPartDoneEvent(
            type="response.content_part.done",
            sequence_number=-1,
            item_id=state.current_item_id,
            output_index=state.current_output_index,
            content_index=state.current_content_index,
            part=DonePartReasoningText(
                text=reason_content,
                type="reasoning_text",
            ),
        ),
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            sequence_number=-1,
            output_index=state.current_output_index,
            item=ResponseReasoningItem(
                type="reasoning",
                content=[
                    ReasoningContent(
                        text=reason_content,
                        type="reasoning_text",
                    )
                ],
                status="completed",
                id=state.current_item_id,
                summary=[],
            ),
        ),
    ]


def _events_add_content_output_text_item(
    state: SimpleStreamingState,
    logprobs: list | None = None,
) -> list[StreamingResponsesResponse]:
    return [
        ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            sequence_number=-1,
            output_index=state.current_output_index,
            item=ResponseOutputMessage(
                id=state.current_item_id,
                type="message",
                role="assistant",
                content=[],
                status="in_progress",
            ),
        ),
        ResponseContentPartAddedEvent(
            type="response.content_part.added",
            sequence_number=-1,
            output_index=state.current_output_index,
            item_id=state.current_item_id,
            content_index=state.current_content_index,
            part=ResponseOutputText(
                type="output_text",
                text="",
                annotations=[],
                logprobs=logprobs if logprobs is not None else [],
            ),
        ),
    ]


def _event_output_text_delta(
    state: SimpleStreamingState,
    delta_message: DeltaMessage,
    logprobs: list | None = None,
) -> StreamingResponsesResponse:
    return ResponseTextDeltaEvent(
        type="response.output_text.delta",
        sequence_number=-1,
        content_index=state.current_content_index,
        output_index=state.current_output_index,
        item_id=state.current_item_id,
        delta=delta_message.content,
        logprobs=logprobs if logprobs is not None else [],
    )


def _events_output_text_done(
    state: SimpleStreamingState,
    text_content: str,
    logprobs: list | None = None,
) -> list[StreamingResponsesResponse]:
    """Events when output text completes."""
    events: list[StreamingResponsesResponse] = []

    events.append(
        ResponseTextDoneEvent(
            type="response.output_text.done",
            item_id=state.current_item_id,
            sequence_number=-1,
            output_index=state.current_output_index,
            content_index=state.current_content_index,
            text=text_content,
            logprobs=logprobs if logprobs is not None else [],
        )
    )
    events.append(
        ResponseContentPartDoneEvent(
            type="response.content_part.done",
            sequence_number=-1,
            item_id=state.current_item_id,
            output_index=state.current_output_index,
            content_index=state.current_content_index,
            part=ResponseOutputText(
                text=text_content,
                type="output_text",
                annotations=[],
            ),
        )
    )
    events.append(
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            sequence_number=-1,
            output_index=state.current_output_index,
            item=ResponseOutputMessage(
                type="message",
                role="assistant",
                content=[
                    ResponseOutputText(
                        text=text_content,
                        type="output_text",
                        annotations=[],
                    )
                ],
                status="completed",
                id=state.current_item_id,
                annotations=[],
                logprobs=logprobs if logprobs is not None else [],
            ),
        )
    )

    return events


def _event_add_tc_output_item(
    state: SimpleStreamingState, delta_message: DeltaMessage
) -> StreamingResponsesResponse:
    function = delta_message.tool_calls[0].function
    arguments = (
        function.arguments if function and function.arguments is not None else ""
    )
    return ResponseOutputItemAddedEvent(
        type="response.output_item.added",
        sequence_number=-1,
        output_index=state.current_output_index,
        item=ResponseFunctionToolCallItem(
            type="function_call",
            id=state.current_item_id,
            call_id=state.current_tool_call_id,
            name=state.current_tool_call_name,
            arguments=arguments,
            status="in_progress",
        ),
    )


def _event_tc_args_delta(
    state: SimpleStreamingState,
    delta_message: DeltaMessage,
) -> StreamingResponsesResponse:
    function = delta_message.tool_calls[0].function
    delta = function.arguments if function and function.arguments is not None else ""
    return ResponseFunctionCallArgumentsDeltaEvent(
        type="response.function_call_arguments.delta",
        sequence_number=-1,
        output_index=state.current_output_index,
        item_id=state.current_item_id,
        delta=delta,
    )


def _events_tc_done(
    state: SimpleStreamingState,
    arguments: str,
) -> list[StreamingResponsesResponse]:
    """Events when a tool call completes."""

    return [
        ResponseFunctionCallArgumentsDoneEvent(
            type="response.function_call_arguments.done",
            sequence_number=-1,
            output_index=state.current_output_index,
            item_id=state.current_item_id,
            arguments=arguments,
            name=state.current_tool_call_name,
        ),
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            sequence_number=-1,
            output_index=state.current_output_index,
            item=ResponseFunctionToolCall(
                type="function_call",
                name=state.current_tool_call_name,
                arguments=arguments,
                status="completed",
                id=state.current_item_id,
                call_id=state.current_tool_call_id,
            ),
        ),
    ]


async def process_simple_stream(
    result_generator: AsyncIterator,
    reasoning_parser: "ReasoningParser | None",
    tool_parser: "ToolParser | None",
    request: "ResponsesRequest | None",
    tokenizer: "TokenizerLike | None",
    _emit_event: Callable[[StreamingResponsesResponse], StreamingResponsesResponse],
    get_logprobs: Callable[
        [Sequence[int], SampleLogprobs | None, TokenizerLike, int | None], list
    ]
    | None = None,
    raise_if_error: Callable[[str | None, str], None] | None = None,
) -> AsyncGenerator[StreamingResponsesResponse, None]:
    """Process simple (non-Harmony) streaming deltas and emit events.

    This is the main generator that processes delta messages and yields
    appropriate streaming events.

    In a regex-like syntax the following sequence pattern is supported:
    (?:reasoning)?(?:output text)?(?:tool call)*
    Note that this implementation currently does not support multiple reasoning
    and/or output text blocks, nor tool calls within reasoning.
    """
    state = SimpleStreamingState()

    async for ctx in result_generator:
        assert isinstance(ctx, SimpleContext)
        if ctx.last_output is None:
            continue

        output = ctx.last_output
        # Check for error finish reason before processing
        if output.outputs and output.outputs[0].finish_reason == "error":
            if raise_if_error:
                raise_if_error(
                    output.outputs[0].finish_reason,
                    request.request_id if request else "unknown",
                )
            else:
                logger.error(
                    "Request %s failed with an internal error during generation",
                    request.request_id if request else "unknown",
                )
                raise GenerationError("Internal server error")

        # Initialize prompt_is_reasoning_end on first iteration
        if reasoning_parser and state.prompt_is_reasoning_end is None:
            state.prompt_is_reasoning_end = reasoning_parser.is_reasoning_end(
                output.prompt_token_ids
            )

        # Parse the delta message
        if not output.outputs:
            continue
        output = output.outputs[0]

        # Pre-compute logprobs for this iteration if needed
        current_token_ids = as_list(output.token_ids)
        logprobs_data: list | None = None
        if get_logprobs and request and request.is_include_output_logprobs():
            logprobs_data = get_logprobs(
                current_token_ids,
                output.logprobs,
                tokenizer,  # type: ignore[arg-type]
                request.top_logprobs,
            )
        result = _parse_delta_message(
            request=request,
            output=output,
            reasoning_parser=reasoning_parser,
            tool_parser=tool_parser,
            state=state,
        )
        delta_message, current_text, current_token_ids = result

        # Update state
        state.previous_text = current_text
        state.previous_token_ids = current_token_ids

        if not delta_message:
            continue

        if not state.event_stream_started:
            if delta_message.tool_calls:
                function = delta_message.tool_calls[0].function
                if not function or function.name is None:
                    continue
                state.first_tool_call_started = True
                state.current_item_id = random_uuid()
                state.started_tool_call_indices.add(0)
                state.current_tool_call_id = f"call_{random_uuid()}"
                state.current_tool_call_name = function.name
                yield _emit_event(_event_add_tc_output_item(state, delta_message))
            elif delta_message.reasoning:
                state.current_item_id = random_uuid()
                for event in _events_add_content_reasoning_item(state):
                    yield _emit_event(event)
            elif delta_message.content:
                state.current_item_id = random_uuid()
                for event in _events_add_content_output_text_item(state, logprobs_data):
                    yield _emit_event(event)

            state.event_stream_started = True

        # Handle reasoning to output text transition.
        if (
            state.previous_delta_messages
            and state.previous_delta_messages[-1].reasoning is not None
            and delta_message.content is not None
        ):
            assert not delta_message.tool_calls, (
                "Reasoning, output text, and tool call not supported in a single delta."
            )

            final_reasoning = "".join(
                pm.reasoning for pm in state.previous_delta_messages if pm.reasoning
            )
            if delta_message.reasoning is not None:
                yield _emit_event(_event_reasoning_delta(state, delta_message))
                final_reasoning += delta_message.reasoning
                delta_message.reasoning = None

            for event in _events_reasoning_done(state, final_reasoning):
                yield _emit_event(event)
            state.current_output_index += 1
            state.reasoning_ended = True
            state.current_content_index = 0
            state.previous_delta_messages = []

            state.current_item_id = random_uuid()
            for event in _events_add_content_output_text_item(state, logprobs_data):
                yield _emit_event(event)
            event = _event_output_text_delta(state, delta_message, logprobs_data)
            yield _emit_event(event)
        # handle tool calls
        elif delta_message.tool_calls and delta_message.tool_calls[0].function:
            # handle transition from reasoning or output text to tool call
            assert not (delta_message.reasoning and delta_message.content), (
                "Reasoning, output text, and tool call not supported "
                "to be in a single delta."
            )
            if state.previous_delta_messages and (
                state.previous_delta_messages[-1].reasoning is not None
                or state.previous_delta_messages[-1].content is not None
            ):
                # Finalize any potentially accumulated reasoning or output text
                final_reasoning = "".join(
                    pm.reasoning for pm in state.previous_delta_messages if pm.reasoning
                )
                if delta_message.reasoning is not None:
                    yield _emit_event(_event_reasoning_delta(state, delta_message))
                    final_reasoning += delta_message.reasoning
                    delta_message.reasoning = None

                if state.previous_delta_messages[-1].reasoning:
                    for event in _events_reasoning_done(state, final_reasoning):
                        yield _emit_event(event)
                    state.current_output_index += 1
                    state.reasoning_ended = True

                final_content = "".join(
                    pm.content for pm in state.previous_delta_messages if pm.content
                )
                if delta_message.content is not None:
                    yield _emit_event(
                        _event_output_text_delta(state, delta_message, logprobs_data)
                    )
                    final_content += delta_message.content
                    delta_message.content = None

                if state.previous_delta_messages[-1].content:
                    for event in _events_output_text_done(
                        state, final_content, logprobs_data
                    ):
                        yield _emit_event(event)
                    state.current_output_index += 1

                state.previous_delta_messages = []
                state.current_content_index = 0

            # handle transition from tool call to next tool call
            if (
                state.previous_delta_messages
                and state.previous_delta_messages[-1].tool_calls
                and state.previous_delta_messages[-1].tool_calls[0].index
                != delta_message.tool_calls[0].index
            ):
                arguments = "".join(
                    (pm.tool_calls[0].function.arguments or "")
                    for pm in state.previous_delta_messages
                    if pm.tool_calls and pm.tool_calls[0].function
                )
                for event in _events_tc_done(state, arguments):
                    yield _emit_event(event)
                state.previous_delta_messages = []
                state.current_output_index += 1

            tool_call = delta_message.tool_calls[0]
            tool_call_index = tool_call.index if tool_call.index is not None else 0

            # Emit tool call start events for new tool call indices
            if (
                delta_message.tool_calls[0].function.name
                and tool_call_index not in state.started_tool_call_indices
            ):
                function = delta_message.tool_calls[0].function
                state.first_tool_call_started = True
                state.current_item_id = random_uuid()
                state.current_tool_call_id = f"call_{random_uuid()}"
                state.current_tool_call_name = function.name or ""
                yield _emit_event(_event_add_tc_output_item(state, delta_message))
                state.started_tool_call_indices.add(tool_call_index)

            if (
                delta_message.tool_calls[0].function.arguments
                and delta_message.tool_calls[0].function.arguments != ""
            ):
                yield _emit_event(_event_tc_args_delta(state, delta_message))
        elif delta_message.reasoning is not None:
            yield _emit_event(_event_reasoning_delta(state, delta_message))
        elif delta_message.content is not None:
            yield _emit_event(
                _event_output_text_delta(state, delta_message, logprobs_data)
            )
        state.previous_delta_messages.append(delta_message)

    # result_generator is finished, finalize any open items
    if state.previous_delta_messages:
        # tool call
        if state.previous_delta_messages[-1].tool_calls:
            arguments = "".join(
                pm.tool_calls[0].function.arguments or ""
                for pm in state.previous_delta_messages
                if pm.tool_calls and pm.tool_calls[0].function
            )
            for event in _events_tc_done(state, arguments):
                yield _emit_event(event)
        # reasoning
        elif state.previous_delta_messages[-1].reasoning is not None:
            final_reasoning = "".join(
                pm.reasoning or ""
                for pm in state.previous_delta_messages
                if pm.reasoning
            )
            for event in _events_reasoning_done(state, final_reasoning):
                yield _emit_event(event)
            state.reasoning_ended = True
        # output text
        elif state.previous_delta_messages[-1].content is not None:
            final_content = "".join(
                pm.content or "" for pm in state.previous_delta_messages if pm.content
            )
            for event in _events_output_text_done(state, final_content, logprobs_data):
                yield _emit_event(event)
