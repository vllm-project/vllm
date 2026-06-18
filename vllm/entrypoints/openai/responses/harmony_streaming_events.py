# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Harmony-specific streaming SSE event builders for the Responses API.

Translates Harmony protocol segments into OpenAI Response API SSE events.
Delegates to the shared leaf helpers in streaming_events.py for text,
reasoning, and function call events.

The file is organized as:
  1. Leaf helpers — open events (MCP, code interpreter)
  2. Leaf helpers — delta events (MCP, code interpreter)
  3. Leaf helpers — done events (MCP, code interpreter, browser)
  4. Harmony segment dispatchers (entry point + open/delta/done routing)
"""

from __future__ import annotations

from dataclasses import dataclass

from openai.types.responses import (
    ResponseCodeInterpreterCallCodeDeltaEvent,
    ResponseCodeInterpreterCallCodeDoneEvent,
    ResponseCodeInterpreterCallCompletedEvent,
    ResponseCodeInterpreterCallInProgressEvent,
    ResponseCodeInterpreterCallInterpretingEvent,
    ResponseMcpCallArgumentsDeltaEvent,
    ResponseMcpCallArgumentsDoneEvent,
    ResponseMcpCallCompletedEvent,
    ResponseMcpCallFailedEvent,
    ResponseMcpCallInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseWebSearchCallCompletedEvent,
    ResponseWebSearchCallInProgressEvent,
    ResponseWebSearchCallSearchingEvent,
)
from openai_harmony import Message as HarmonyMessage

from vllm.entrypoints.openai.responses.harmony import (
    ResponseItemKind,
    ResponseItemType,
    build_code_interpreter_call,
    build_mcp_or_container_call,
    build_web_search_call,
    message_text_content,
    resolve_response_item_type,
)
from vllm.entrypoints.openai.responses.protocol import (
    StreamingResponsesResponse,
)
from vllm.entrypoints.openai.responses.streaming_events import (
    emit_function_call_delta_events,
    emit_function_call_done_events,
    emit_function_call_open_events,
    emit_reasoning_delta_events,
    emit_reasoning_done_events,
    emit_reasoning_open_events,
    emit_text_delta_events,
    emit_text_done_events,
    emit_text_open_events,
)
from vllm.parser.harmony import Segment
from vllm.utils import random_uuid

# =====================================================================
# Leaf helpers — open events
# =====================================================================


def emit_mcp_open_events(
    item_type: ResponseItemType,
    state: HarmonyStreamingState,
) -> list[StreamingResponsesResponse]:
    return [
        ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            sequence_number=-1,
            output_index=state.output_index,
            item=build_mcp_or_container_call(
                item_type, "", state.current_item_id, status="in_progress"
            ),
        ),
        ResponseMcpCallInProgressEvent(
            type="response.mcp_call.in_progress",
            sequence_number=-1,
            output_index=state.output_index,
            item_id=state.current_item_id,
        ),
    ]


def emit_code_interpreter_open_events(
    item_type: ResponseItemType,
    state: HarmonyStreamingState,
) -> list[StreamingResponsesResponse]:
    return [
        ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            sequence_number=-1,
            output_index=state.output_index,
            item=build_code_interpreter_call(
                item_type, "", state.current_item_id, status="in_progress"
            ),
        ),
        ResponseCodeInterpreterCallInProgressEvent(
            type="response.code_interpreter_call.in_progress",
            sequence_number=-1,
            output_index=state.output_index,
            item_id=state.current_item_id,
        ),
    ]


# =====================================================================
# Leaf helpers — delta events
# =====================================================================


def emit_mcp_delta_events(
    delta: str,
    state: HarmonyStreamingState,
) -> list[StreamingResponsesResponse]:
    return [
        ResponseMcpCallArgumentsDeltaEvent(
            type="response.mcp_call_arguments.delta",
            sequence_number=-1,
            output_index=state.output_index,
            item_id=state.current_item_id,
            delta=delta,
        )
    ]


def emit_code_interpreter_delta_events(
    delta: str,
    state: HarmonyStreamingState,
) -> list[StreamingResponsesResponse]:
    return [
        ResponseCodeInterpreterCallCodeDeltaEvent(
            type="response.code_interpreter_call_code.delta",
            sequence_number=-1,
            output_index=state.output_index,
            item_id=state.current_item_id,
            delta=delta,
        )
    ]


# =====================================================================
# Leaf helpers — done events
# =====================================================================


def emit_mcp_or_container_done_events(
    item_type: ResponseItemType,
    arguments: str,
    state: HarmonyStreamingState,
) -> list[StreamingResponsesResponse]:
    item = build_mcp_or_container_call(item_type, arguments, state.current_item_id)
    return [
        ResponseMcpCallArgumentsDoneEvent(
            type="response.mcp_call_arguments.done",
            arguments=arguments,
            name=item.name,
            item_id=state.current_item_id,
            output_index=state.output_index,
            sequence_number=-1,
        ),
        (
            ResponseMcpCallCompletedEvent(
                type="response.mcp_call.completed",
                sequence_number=-1,
                output_index=state.output_index,
                item_id=state.current_item_id,
            )
            if item.error is None
            else ResponseMcpCallFailedEvent(
                type="response.mcp_call.failed",
                sequence_number=-1,
                output_index=state.output_index,
                item_id=state.current_item_id,
            )
        ),
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            sequence_number=-1,
            output_index=state.output_index,
            item=item,
        ),
    ]


def emit_code_interpreter_done_events(
    item_type: ResponseItemType,
    code: str,
    state: HarmonyStreamingState,
) -> list[StreamingResponsesResponse]:
    return [
        ResponseCodeInterpreterCallCodeDoneEvent(
            type="response.code_interpreter_call_code.done",
            sequence_number=-1,
            output_index=state.output_index,
            item_id=state.current_item_id,
            code=code,
        ),
        ResponseCodeInterpreterCallInterpretingEvent(
            type="response.code_interpreter_call.interpreting",
            sequence_number=-1,
            output_index=state.output_index,
            item_id=state.current_item_id,
        ),
        ResponseCodeInterpreterCallCompletedEvent(
            type="response.code_interpreter_call.completed",
            sequence_number=-1,
            output_index=state.output_index,
            item_id=state.current_item_id,
        ),
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            sequence_number=-1,
            output_index=state.output_index,
            item=build_code_interpreter_call(item_type, code, state.current_item_id),
        ),
    ]


def emit_browser_done_events(
    item_type: ResponseItemType,
    text: str,
    state: HarmonyStreamingState,
) -> list[StreamingResponsesResponse]:
    item = build_web_search_call(item_type, text, state.current_item_id)
    events: list[StreamingResponsesResponse] = [
        ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            sequence_number=-1,
            output_index=state.output_index,
            item=item,
        ),
        ResponseWebSearchCallInProgressEvent(
            type="response.web_search_call.in_progress",
            sequence_number=-1,
            output_index=state.output_index,
            item_id=state.current_item_id,
        ),
        ResponseWebSearchCallSearchingEvent(
            type="response.web_search_call.searching",
            sequence_number=-1,
            output_index=state.output_index,
            item_id=state.current_item_id,
        ),
        ResponseWebSearchCallCompletedEvent(
            type="response.web_search_call.completed",
            sequence_number=-1,
            output_index=state.output_index,
            item_id=state.current_item_id,
        ),
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            sequence_number=-1,
            output_index=state.output_index,
            item=item,
        ),
    ]

    return events


# =====================================================================
# Harmony segment dispatchers
# =====================================================================


@dataclass
class HarmonyStreamingState:
    """Mutable state for Harmony streaming event processing."""

    output_index: int = 0
    content_index: int = -1
    current_item_id: str = ""
    tool_call_id: str = ""
    current_kind: ResponseItemKind | None = None

    def reset_for_new_item(self) -> None:
        """Reset state when expecting a new output item."""
        self.output_index += 1
        self.content_index = -1
        self.current_item_id = ""
        self.tool_call_id = ""
        self.current_kind = None


def emit_harmony_open(
    item_type: ResponseItemType,
    state: HarmonyStreamingState,
) -> list[StreamingResponsesResponse]:
    assert state.current_kind is None, (
        "Attempted to open a new Harmony item before closing"
    )

    state.current_kind = item_type.kind
    match item_type.kind:
        case ResponseItemKind.REASONING:
            state.current_item_id = f"rs_{random_uuid()}"
            state.content_index = 0
            events = emit_reasoning_open_events(state)
        case ResponseItemKind.TEXT:
            state.current_item_id = f"msg_{random_uuid()}"
            state.content_index = 0
            events = emit_text_open_events(state)
        case ResponseItemKind.FUNCTION:
            assert item_type.action is not None
            rid = random_uuid()
            state.current_item_id = f"fc_{rid}"
            state.tool_call_id = f"call_{rid}"
            events = emit_function_call_open_events(
                item_type.action,
                state,
            )
        case ResponseItemKind.MCP | ResponseItemKind.CONTAINER:
            assert item_type.action is not None
            state.current_item_id = f"mcp_{random_uuid()}"
            events = emit_mcp_open_events(item_type, state)
        case ResponseItemKind.CODE_INTERPRETER:
            state.current_item_id = f"tool_{random_uuid()}"
            events = emit_code_interpreter_open_events(item_type, state)
        case ResponseItemKind.WEB_SEARCH:
            state.current_item_id = f"ws_{random_uuid()}"
            events = []
        case _:
            events = []
    return events


def emit_harmony_delta(
    item_type: ResponseItemType,
    delta: str,
    state: HarmonyStreamingState,
) -> list[StreamingResponsesResponse]:
    assert state.current_kind is item_type.kind
    assert state.current_item_id

    match item_type.kind:
        case _ if not delta:
            events = []
        case ResponseItemKind.TEXT:
            events = emit_text_delta_events(delta, state, logprobs=[])
        case ResponseItemKind.REASONING:
            events = emit_reasoning_delta_events(delta, state)
        case ResponseItemKind.FUNCTION:
            events = emit_function_call_delta_events(delta, state)
        case ResponseItemKind.MCP | ResponseItemKind.CONTAINER:
            events = emit_mcp_delta_events(delta, state)
        case ResponseItemKind.CODE_INTERPRETER:
            events = emit_code_interpreter_delta_events(delta, state)
        case _:
            events = []

    return events


def emit_harmony_done(
    item_type: ResponseItemType,
    completed_message: HarmonyMessage,
    state: HarmonyStreamingState,
) -> list[StreamingResponsesResponse]:
    assert state.current_item_id, "Harmony item closed before it was opened"
    assert state.current_kind is item_type.kind

    text = message_text_content(completed_message)[0]
    match item_type.kind:
        case ResponseItemKind.REASONING:
            events = emit_reasoning_done_events(text, state)
        case ResponseItemKind.TEXT:
            events = emit_text_done_events(text, state)
        case ResponseItemKind.FUNCTION:
            assert item_type.action is not None
            events = emit_function_call_done_events(item_type.action, text, state)
        case ResponseItemKind.MCP | ResponseItemKind.CONTAINER:
            events = emit_mcp_or_container_done_events(item_type, text, state)
        case ResponseItemKind.CODE_INTERPRETER:
            events = emit_code_interpreter_done_events(item_type, text, state)
        case ResponseItemKind.WEB_SEARCH:
            events = emit_browser_done_events(item_type, text, state)
        case _:
            events = []

    state.reset_for_new_item()
    return events


def emit_harmony_segment_events(
    segment: Segment,
    state: HarmonyStreamingState,
    function_tool_names: frozenset[str] | None = None,
) -> list[StreamingResponsesResponse]:
    completed_message = segment.completed_message
    if completed_message is not None:
        assert not segment.delta
        channel = completed_message.channel
        recipient = completed_message.recipient
    else:
        channel = segment.channel
        recipient = segment.recipient

    item_type = resolve_response_item_type(channel, recipient, function_tool_names)
    if item_type.kind is ResponseItemKind.IGNORE:
        return []

    if completed_message is not None:
        return emit_harmony_done(item_type, completed_message, state)

    events = []
    if state.current_kind is None:
        events.extend(emit_harmony_open(item_type, state))
    events.extend(emit_harmony_delta(item_type, segment.delta, state))
    return events
