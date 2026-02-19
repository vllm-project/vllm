# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Streaming SSE event builders for the Responses API.

Pure functions that translate streaming state + delta data into
OpenAI Response API SSE events. Used by the streaming event
processors in serving.py.
"""

import json
from dataclasses import dataclass
from typing import Final

from openai.types.responses import (
    ResponseCodeInterpreterCallCodeDeltaEvent,
    ResponseCodeInterpreterCallCodeDoneEvent,
    ResponseCodeInterpreterCallCompletedEvent,
    ResponseCodeInterpreterCallInProgressEvent,
    ResponseCodeInterpreterCallInterpretingEvent,
    ResponseCodeInterpreterToolCallParam,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseMcpCallArgumentsDeltaEvent,
    ResponseMcpCallArgumentsDoneEvent,
    ResponseMcpCallCompletedEvent,
    ResponseMcpCallInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseWebSearchCallCompletedEvent,
    ResponseWebSearchCallInProgressEvent,
    ResponseWebSearchCallSearchingEvent,
    response_function_web_search,
)
from openai.types.responses.response_output_item import McpCall
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningTextContent,
)

from vllm.entrypoints.mcp.tool_server import ToolServer
from vllm.entrypoints.openai.responses.context import StreamingHarmonyContext
from vllm.entrypoints.openai.responses.protocol import (
    ResponseReasoningPartAddedEvent,
    ResponseReasoningPartDoneEvent,
    StreamingResponsesResponse,
)
from vllm.utils import random_uuid

TOOL_NAME_TO_MCP_SERVER_LABEL: Final[dict[str, str]] = {
    "python": "code_interpreter",
    "container": "container",
    "browser": "web_search_preview",
}


@dataclass
class HarmonyStreamingState:
    """Mutable state for harmony streaming event processing."""

    current_content_index: int = -1
    current_output_index: int = 0
    current_item_id: str = ""
    sent_output_item_added: bool = False
    is_first_function_call_delta: bool = False

    def reset_for_new_item(self) -> None:
        """Reset state when expecting a new output item."""
        self.current_output_index += 1
        self.sent_output_item_added = False
        self.is_first_function_call_delta = False


def is_mcp_tool_by_namespace(recipient: str | None) -> bool:
    """
    Determine if a tool call is an MCP tool based on recipient prefix.

    - Tools starting with "functions." are function calls
    - Everything else is an MCP tool
    """
    if recipient is None:
        return False

    # Function calls have "functions." prefix
    # Everything else is an MCP tool
    return not recipient.startswith("functions.")


def emit_function_call_done_events(
    previous_item,
    state: HarmonyStreamingState,
) -> list[StreamingResponsesResponse]:
    """Emit events when a function call completes."""
    function_name = previous_item.recipient[len("functions.") :]
    events: list[StreamingResponsesResponse] = []
    events.append(
        ResponseFunctionCallArgumentsDoneEvent(
            type="response.function_call_arguments.done",
            arguments=previous_item.content[0].text,
            name=function_name,
            item_id=state.current_item_id,
            output_index=state.current_output_index,
            sequence_number=-1,
        )
    )
    function_call_item = ResponseFunctionToolCall(
        type="function_call",
        arguments=previous_item.content[0].text,
        name=function_name,
        item_id=state.current_item_id,
        output_index=state.current_output_index,
        sequence_number=-1,
        call_id=f"fc_{random_uuid()}",
        status="completed",
    )
    events.append(
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            sequence_number=-1,
            output_index=state.current_output_index,
            item=function_call_item,
        )
    )
    return events


def emit_mcp_call_done_events(
    previous_item,
    state: HarmonyStreamingState,
) -> list[StreamingResponsesResponse]:
    """Emit events when an MCP tool call completes."""
    server_label = TOOL_NAME_TO_MCP_SERVER_LABEL.get(
        previous_item.recipient, previous_item.recipient
    )
    events: list[StreamingResponsesResponse] = []
    events.append(
        ResponseMcpCallArgumentsDoneEvent(
            type="response.mcp_call_arguments.done",
            arguments=previous_item.content[0].text,
            name=previous_item.recipient,
            item_id=state.current_item_id,
            output_index=state.current_output_index,
            sequence_number=-1,
        )
    )
    events.append(
        ResponseMcpCallCompletedEvent(
            type="response.mcp_call.completed",
            sequence_number=-1,
            output_index=state.current_output_index,
            item_id=state.current_item_id,
        )
    )
    events.append(
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            sequence_number=-1,
            output_index=state.current_output_index,
            item=McpCall(
                type="mcp_call",
                arguments=previous_item.content[0].text,
                name=previous_item.recipient,
                id=state.current_item_id,
                server_label=server_label,
                status="completed",
            ),
        )
    )
    return events


def emit_reasoning_done_events(
    previous_item,
    state: HarmonyStreamingState,
) -> list[StreamingResponsesResponse]:
    """Emit events when a reasoning (analysis) item completes."""
    content = ResponseReasoningTextContent(
        text=previous_item.content[0].text,
        type="reasoning_text",
    )
    reasoning_item = ResponseReasoningItem(
        type="reasoning",
        content=[content],
        status="completed",
        id=state.current_item_id,
        summary=[],
    )
    events: list[StreamingResponsesResponse] = []
    events.append(
        ResponseReasoningTextDoneEvent(
            type="response.reasoning_text.done",
            item_id=state.current_item_id,
            sequence_number=-1,
            output_index=state.current_output_index,
            content_index=state.current_content_index,
            text=previous_item.content[0].text,
        )
    )
    events.append(
        ResponseReasoningPartDoneEvent(
            type="response.reasoning_part.done",
            sequence_number=-1,
            item_id=state.current_item_id,
            output_index=state.current_output_index,
            content_index=state.current_content_index,
            part=content,
        )
    )
    events.append(
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            sequence_number=-1,
            output_index=state.current_output_index,
            item=reasoning_item,
        )
    )
    return events


def emit_text_output_done_events(
    previous_item,
    state: HarmonyStreamingState,
) -> list[StreamingResponsesResponse]:
    """Emit events when a final text output item completes."""
    text_content = ResponseOutputText(
        type="output_text",
        text=previous_item.content[0].text,
        annotations=[],
    )
    events: list[StreamingResponsesResponse] = []
    events.append(
        ResponseTextDoneEvent(
            type="response.output_text.done",
            sequence_number=-1,
            output_index=state.current_output_index,
            content_index=state.current_content_index,
            text=previous_item.content[0].text,
            logprobs=[],
            item_id=state.current_item_id,
        )
    )
    events.append(
        ResponseContentPartDoneEvent(
            type="response.content_part.done",
            sequence_number=-1,
            item_id=state.current_item_id,
            output_index=state.current_output_index,
            content_index=state.current_content_index,
            part=text_content,
        )
    )
    events.append(
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            sequence_number=-1,
            output_index=state.current_output_index,
            item=ResponseOutputMessage(
                id=state.current_item_id,
                type="message",
                role="assistant",
                content=[text_content],
                status="completed",
            ),
        )
    )
    return events


def emit_previous_item_done_events(
    previous_item,
    state: HarmonyStreamingState,
) -> list[StreamingResponsesResponse]:
    """Emit done events for the previous item when expecting a new start."""
    if previous_item.recipient is not None:
        # Deal with tool call
        if previous_item.recipient.startswith("functions."):
            return emit_function_call_done_events(previous_item, state)
        elif (
            is_mcp_tool_by_namespace(previous_item.recipient)
            and state.current_item_id is not None
            and state.current_item_id.startswith("mcp_")
        ):
            return emit_mcp_call_done_events(previous_item, state)
    elif previous_item.channel == "analysis":
        return emit_reasoning_done_events(previous_item, state)
    elif previous_item.channel == "final":
        return emit_text_output_done_events(previous_item, state)
    return []


def emit_final_channel_delta_events(
    ctx: StreamingHarmonyContext,
    state: HarmonyStreamingState,
) -> list[StreamingResponsesResponse]:
    """Emit events for final channel text delta streaming."""
    events: list[StreamingResponsesResponse] = []
    if not state.sent_output_item_added:
        state.sent_output_item_added = True
        state.current_item_id = f"msg_{random_uuid()}"
        events.append(
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
            )
        )
        state.current_content_index += 1
        events.append(
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
                    logprobs=[],
                ),
            )
        )
    events.append(
        ResponseTextDeltaEvent(
            type="response.output_text.delta",
            sequence_number=-1,
            content_index=state.current_content_index,
            output_index=state.current_output_index,
            item_id=state.current_item_id,
            delta=ctx.last_content_delta,
            # TODO, use logprobs from ctx.last_request_output
            logprobs=[],
        )
    )
    return events


def emit_analysis_channel_delta_events(
    ctx: StreamingHarmonyContext,
    state: HarmonyStreamingState,
) -> list[StreamingResponsesResponse]:
    """Emit events for analysis channel reasoning delta streaming."""
    events: list[StreamingResponsesResponse] = []
    if not state.sent_output_item_added:
        state.sent_output_item_added = True
        state.current_item_id = f"msg_{random_uuid()}"
        events.append(
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
            )
        )
        state.current_content_index += 1
        events.append(
            ResponseReasoningPartAddedEvent(
                type="response.reasoning_part.added",
                sequence_number=-1,
                output_index=state.current_output_index,
                item_id=state.current_item_id,
                content_index=state.current_content_index,
                part=ResponseReasoningTextContent(
                    text="",
                    type="reasoning_text",
                ),
            )
        )
    events.append(
        ResponseReasoningTextDeltaEvent(
            type="response.reasoning_text.delta",
            item_id=state.current_item_id,
            output_index=state.current_output_index,
            content_index=state.current_content_index,
            delta=ctx.last_content_delta,
            sequence_number=-1,
        )
    )
    return events


def emit_mcp_tool_delta_events(
    ctx: StreamingHarmonyContext,
    state: HarmonyStreamingState,
    recipient: str,
) -> list[StreamingResponsesResponse]:
    """Emit events for MCP tool delta streaming."""
    server_label = TOOL_NAME_TO_MCP_SERVER_LABEL.get(recipient, recipient)
    events: list[StreamingResponsesResponse] = []
    if not state.sent_output_item_added:
        state.sent_output_item_added = True
        state.current_item_id = f"mcp_{random_uuid()}"
        events.append(
            ResponseOutputItemAddedEvent(
                type="response.output_item.added",
                sequence_number=-1,
                output_index=state.current_output_index,
                item=McpCall(
                    type="mcp_call",
                    id=state.current_item_id,
                    name=recipient,
                    arguments="",
                    server_label=server_label,
                    status="in_progress",
                ),
            )
        )
        events.append(
            ResponseMcpCallInProgressEvent(
                type="response.mcp_call.in_progress",
                sequence_number=-1,
                output_index=state.current_output_index,
                item_id=state.current_item_id,
            )
        )
    events.append(
        ResponseMcpCallArgumentsDeltaEvent(
            type="response.mcp_call_arguments.delta",
            sequence_number=-1,
            output_index=state.current_output_index,
            item_id=state.current_item_id,
            delta=ctx.last_content_delta,
        )
    )
    return events


def emit_code_interpreter_delta_events(
    ctx: StreamingHarmonyContext,
    state: HarmonyStreamingState,
) -> list[StreamingResponsesResponse]:
    """Emit events for code interpreter delta streaming."""
    events: list[StreamingResponsesResponse] = []
    if not state.sent_output_item_added:
        state.sent_output_item_added = True
        state.current_item_id = f"tool_{random_uuid()}"
        events.append(
            ResponseOutputItemAddedEvent(
                type="response.output_item.added",
                sequence_number=-1,
                output_index=state.current_output_index,
                item=ResponseCodeInterpreterToolCallParam(
                    type="code_interpreter_call",
                    id=state.current_item_id,
                    code=None,
                    container_id="auto",
                    outputs=None,
                    status="in_progress",
                ),
            )
        )
        events.append(
            ResponseCodeInterpreterCallInProgressEvent(
                type="response.code_interpreter_call.in_progress",
                sequence_number=-1,
                output_index=state.current_output_index,
                item_id=state.current_item_id,
            )
        )
    events.append(
        ResponseCodeInterpreterCallCodeDeltaEvent(
            type="response.code_interpreter_call_code.delta",
            sequence_number=-1,
            output_index=state.current_output_index,
            item_id=state.current_item_id,
            delta=ctx.last_content_delta,
        )
    )
    return events


def emit_mcp_prefix_delta_events(
    ctx: StreamingHarmonyContext,
    state: HarmonyStreamingState,
) -> list[StreamingResponsesResponse]:
    """Emit events for MCP prefix (mcp.*) delta streaming."""
    events: list[StreamingResponsesResponse] = []
    if not state.sent_output_item_added:
        state.sent_output_item_added = True
        state.current_item_id = f"mcp_{random_uuid()}"
        mcp_name = ctx.parser.current_recipient[len("mcp.") :]

        events.append(
            ResponseOutputItemAddedEvent(
                type="response.output_item.added",
                sequence_number=-1,
                output_index=state.current_output_index,
                item=McpCall(
                    type="mcp_call",
                    id=state.current_item_id,
                    name=mcp_name,
                    arguments="",
                    server_label=mcp_name,
                    status="in_progress",
                ),
            )
        )
        events.append(
            ResponseMcpCallInProgressEvent(
                type="response.mcp_call.in_progress",
                sequence_number=-1,
                output_index=state.current_output_index,
                item_id=state.current_item_id,
            )
        )

    events.append(
        ResponseMcpCallArgumentsDeltaEvent(
            type="response.mcp_call_arguments.delta",
            sequence_number=-1,
            output_index=state.current_output_index,
            item_id=state.current_item_id,
            delta=ctx.last_content_delta,
        )
    )
    return events


def emit_function_call_delta_events(
    ctx: StreamingHarmonyContext,
    state: HarmonyStreamingState,
) -> list[StreamingResponsesResponse]:
    """Emit events for developer function calls on commentary channel."""
    if not (
        ctx.parser.current_channel == "commentary"
        and ctx.parser.current_recipient
        and ctx.parser.current_recipient.startswith("functions.")
    ):
        return []

    events: list[StreamingResponsesResponse] = []
    if state.is_first_function_call_delta is False:
        state.is_first_function_call_delta = True
        fc_name = ctx.parser.current_recipient[len("functions.") :]
        state.current_item_id = f"fc_{random_uuid()}"
        tool_call_item = ResponseFunctionToolCall(
            name=fc_name,
            type="function_call",
            id=state.current_item_id,
            call_id=f"call_{random_uuid()}",
            arguments="",
            status="in_progress",
        )
        events.append(
            ResponseOutputItemAddedEvent(
                type="response.output_item.added",
                sequence_number=-1,
                output_index=state.current_output_index,
                item=tool_call_item,
            )
        )
    # Always emit the delta (including on first call)
    events.append(
        ResponseFunctionCallArgumentsDeltaEvent(
            item_id=state.current_item_id,
            delta=ctx.last_content_delta,
            output_index=state.current_output_index,
            sequence_number=-1,
            type="response.function_call_arguments.delta",
        )
    )
    return events


def emit_content_delta_events(
    ctx: StreamingHarmonyContext,
    state: HarmonyStreamingState,
) -> list[StreamingResponsesResponse]:
    """Emit events for content delta streaming based on channel type."""
    if not ctx.last_content_delta:
        return []

    if ctx.parser.current_channel == "final" and ctx.parser.current_recipient is None:
        return emit_final_channel_delta_events(ctx, state)
    elif (
        ctx.parser.current_channel == "analysis"
        and ctx.parser.current_recipient is None
    ):
        return emit_analysis_channel_delta_events(ctx, state)
    # built-in tools will be triggered on the analysis channel
    # However, occasionally built-in tools will
    # still be output to commentary.
    elif (
        ctx.parser.current_channel == "commentary"
        or ctx.parser.current_channel == "analysis"
    ) and ctx.parser.current_recipient is not None:
        recipient = ctx.parser.current_recipient
        # Check for function calls first - they have their own event handling
        if recipient.startswith("functions."):
            return emit_function_call_delta_events(ctx, state)
        if is_mcp_tool_by_namespace(recipient):
            return emit_mcp_tool_delta_events(ctx, state, recipient)
        else:
            return emit_code_interpreter_delta_events(ctx, state)
    elif (
        (
            ctx.parser.current_channel == "commentary"
            or ctx.parser.current_channel == "analysis"
        )
        and ctx.parser.current_recipient is not None
        and ctx.parser.current_recipient.startswith("mcp.")
    ):
        return emit_mcp_prefix_delta_events(ctx, state)

    return []


def emit_browser_tool_events(
    previous_item,
    state: HarmonyStreamingState,
) -> list[StreamingResponsesResponse]:
    """Emit events for browser tool calls (web search)."""
    function_name = previous_item.recipient[len("browser.") :]
    parsed_args = json.loads(previous_item.content[0].text)
    action = None

    if function_name == "search":
        action = response_function_web_search.ActionSearch(
            type="search",
            query=parsed_args["query"],
        )
    elif function_name == "open":
        action = response_function_web_search.ActionOpenPage(
            type="open_page",
            # TODO: translate to url
            url=f"cursor:{parsed_args.get('cursor', '')}",
        )
    elif function_name == "find":
        action = response_function_web_search.ActionFind(
            type="find",
            pattern=parsed_args["pattern"],
            # TODO: translate to url
            url=f"cursor:{parsed_args.get('cursor', '')}",
        )
    else:
        raise ValueError(f"Unknown function name: {function_name}")

    state.current_item_id = f"tool_{random_uuid()}"
    events: list[StreamingResponsesResponse] = []
    events.append(
        ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            sequence_number=-1,
            output_index=state.current_output_index,
            item=response_function_web_search.ResponseFunctionWebSearch(
                # TODO: generate a unique id for web search call
                type="web_search_call",
                id=state.current_item_id,
                action=action,
                status="in_progress",
            ),
        )
    )
    events.append(
        ResponseWebSearchCallInProgressEvent(
            type="response.web_search_call.in_progress",
            sequence_number=-1,
            output_index=state.current_output_index,
            item_id=state.current_item_id,
        )
    )
    events.append(
        ResponseWebSearchCallSearchingEvent(
            type="response.web_search_call.searching",
            sequence_number=-1,
            output_index=state.current_output_index,
            item_id=state.current_item_id,
        )
    )
    # enqueue
    events.append(
        ResponseWebSearchCallCompletedEvent(
            type="response.web_search_call.completed",
            sequence_number=-1,
            output_index=state.current_output_index,
            item_id=state.current_item_id,
        )
    )
    events.append(
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            sequence_number=-1,
            output_index=state.current_output_index,
            item=ResponseFunctionWebSearch(
                type="web_search_call",
                id=state.current_item_id,
                action=action,
                status="completed",
            ),
        )
    )
    return events


def emit_mcp_tool_completion_events(
    previous_item,
    state: HarmonyStreamingState,
) -> list[StreamingResponsesResponse]:
    """Emit events when an MCP tool completes during assistant action turn."""
    recipient = previous_item.recipient
    server_label = TOOL_NAME_TO_MCP_SERVER_LABEL.get(recipient, recipient)
    events: list[StreamingResponsesResponse] = []
    events.append(
        ResponseMcpCallArgumentsDoneEvent(
            type="response.mcp_call_arguments.done",
            sequence_number=-1,
            output_index=state.current_output_index,
            item_id=state.current_item_id,
            arguments=previous_item.content[0].text,
            name=recipient,
        )
    )
    events.append(
        ResponseMcpCallCompletedEvent(
            type="response.mcp_call.completed",
            sequence_number=-1,
            output_index=state.current_output_index,
            item_id=state.current_item_id,
        )
    )
    events.append(
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            sequence_number=-1,
            output_index=state.current_output_index,
            item=McpCall(
                type="mcp_call",
                id=state.current_item_id,
                name=recipient,
                arguments=previous_item.content[0].text,
                server_label=server_label,
                status="completed",
            ),
        )
    )
    return events


def emit_code_interpreter_completion_events(
    previous_item,
    state: HarmonyStreamingState,
) -> list[StreamingResponsesResponse]:
    """Emit events when code interpreter completes."""
    events: list[StreamingResponsesResponse] = []
    events.append(
        ResponseCodeInterpreterCallCodeDoneEvent(
            type="response.code_interpreter_call_code.done",
            sequence_number=-1,
            output_index=state.current_output_index,
            item_id=state.current_item_id,
            code=previous_item.content[0].text,
        )
    )
    events.append(
        ResponseCodeInterpreterCallInterpretingEvent(
            type="response.code_interpreter_call.interpreting",
            sequence_number=-1,
            output_index=state.current_output_index,
            item_id=state.current_item_id,
        )
    )
    events.append(
        ResponseCodeInterpreterCallCompletedEvent(
            type="response.code_interpreter_call.completed",
            sequence_number=-1,
            output_index=state.current_output_index,
            item_id=state.current_item_id,
        )
    )
    events.append(
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            sequence_number=-1,
            output_index=state.current_output_index,
            item=ResponseCodeInterpreterToolCallParam(
                type="code_interpreter_call",
                id=state.current_item_id,
                code=previous_item.content[0].text,
                container_id="auto",
                outputs=[],
                status="completed",
            ),
        )
    )
    return events


def emit_mcp_prefix_completion_events(
    previous_item,
    state: HarmonyStreamingState,
) -> list[StreamingResponsesResponse]:
    """Emit events when an MCP prefix tool (mcp.*) completes."""
    mcp_name = previous_item.recipient[len("mcp.") :]
    events: list[StreamingResponsesResponse] = []
    events.append(
        ResponseMcpCallArgumentsDoneEvent(
            type="response.mcp_call_arguments.done",
            sequence_number=-1,
            output_index=state.current_output_index,
            item_id=state.current_item_id,
            arguments=previous_item.content[0].text,
            name=mcp_name,
        )
    )
    events.append(
        ResponseMcpCallCompletedEvent(
            type="response.mcp_call.completed",
            sequence_number=-1,
            output_index=state.current_output_index,
            item_id=state.current_item_id,
        )
    )
    events.append(
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            sequence_number=-1,
            output_index=state.current_output_index,
            item=McpCall(
                type="mcp_call",
                id=state.current_item_id,
                name=mcp_name,
                arguments=previous_item.content[0].text,
                server_label=mcp_name,
                status="completed",
            ),
        )
    )
    return events


def emit_tool_action_events(
    ctx: StreamingHarmonyContext,
    state: HarmonyStreamingState,
    tool_server: ToolServer | None,
) -> list[StreamingResponsesResponse]:
    """Emit events for tool action turn."""
    if not ctx.is_assistant_action_turn() or len(ctx.parser.messages) == 0:
        return []

    events: list[StreamingResponsesResponse] = []
    previous_item = ctx.parser.messages[-1]

    # Handle browser tool
    if (
        tool_server is not None
        and tool_server.has_tool("browser")
        and previous_item.recipient is not None
        and previous_item.recipient.startswith("browser.")
    ):
        events.extend(emit_browser_tool_events(previous_item, state))

    # Handle tool completion
    if (
        tool_server is not None
        and previous_item.recipient is not None
        and state.current_item_id is not None
        and state.sent_output_item_added
    ):
        recipient = previous_item.recipient
        # Handle MCP prefix tool completion first
        if recipient.startswith("mcp."):
            events.extend(emit_mcp_prefix_completion_events(previous_item, state))
        else:
            # Handle other MCP tool and code interpreter completion
            is_mcp_tool = is_mcp_tool_by_namespace(
                recipient
            ) and state.current_item_id.startswith("mcp_")
            if is_mcp_tool:
                events.extend(emit_mcp_tool_completion_events(previous_item, state))
            else:
                events.extend(
                    emit_code_interpreter_completion_events(previous_item, state)
                )

    return events
