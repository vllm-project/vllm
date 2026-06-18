# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Streaming SSE event builders for the Responses API.

Contains shared leaf helpers for building SSE events (text, reasoning,
function call) and the SimpleStreamingEventProcessor for the non-Harmony
streaming path.

Harmony-specific streaming logic lives in harmony_streaming_events.py.

The file is organized as:
  1. Shared leaf helpers — open events (text, reasoning, function call)
  2. Shared leaf helpers — delta events (text, reasoning, function call)
  3. Shared leaf helpers — done events (text, reasoning, function call)
  4. Simple streaming state machine (SimpleStreamingState,
     SimpleStreamingEventProcessor, split_delta)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

from openai.types.responses import (
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    response_text_delta_event,
)
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningTextContent,
)

from vllm.entrypoints.openai.engine.protocol import DeltaMessage, DeltaToolCall
from vllm.entrypoints.openai.responses.protocol import (
    ResponseReasoningPartAddedEvent,
    ResponseReasoningPartDoneEvent,
    StreamingResponsesResponse,
)
from vllm.outputs import CompletionOutput
from vllm.utils import random_uuid

if TYPE_CHECKING:
    from vllm.entrypoints.openai.responses.harmony_streaming_events import (
        HarmonyStreamingState,
    )


# =====================================================================
# Shared leaf helpers — open events
# =====================================================================


def emit_text_open_events(
    state: SimpleStreamingState | HarmonyStreamingState,
) -> list[StreamingResponsesResponse]:
    return [
        ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            sequence_number=-1,
            output_index=state.output_index,
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
            output_index=state.output_index,
            item_id=state.current_item_id,
            content_index=state.content_index,
            part=ResponseOutputText(
                type="output_text",
                text="",
                annotations=[],
                logprobs=[],
            ),
        ),
    ]


def emit_reasoning_open_events(
    state: SimpleStreamingState | HarmonyStreamingState,
) -> list[StreamingResponsesResponse]:
    return [
        ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            sequence_number=-1,
            output_index=state.output_index,
            item=ResponseReasoningItem(
                type="reasoning",
                id=state.current_item_id,
                summary=[],
                status="in_progress",
            ),
        ),
        ResponseReasoningPartAddedEvent(
            type="response.reasoning_part.added",
            sequence_number=-1,
            output_index=state.output_index,
            item_id=state.current_item_id,
            content_index=state.content_index,
            part=ResponseReasoningTextContent(
                text="",
                type="reasoning_text",
            ),
        ),
    ]


def emit_function_call_open_events(
    function_name: str,
    state: SimpleStreamingState | HarmonyStreamingState,
) -> list[StreamingResponsesResponse]:
    return [
        ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            sequence_number=-1,
            output_index=state.output_index,
            item=ResponseFunctionToolCall(
                name=function_name,
                type="function_call",
                id=state.current_item_id,
                call_id=state.tool_call_id,
                arguments="",
                status="in_progress",
            ),
        )
    ]


# =====================================================================
# Shared leaf helpers — delta events
# =====================================================================


def emit_text_delta_events(
    delta: str,
    state: SimpleStreamingState | HarmonyStreamingState,
    logprobs: list[response_text_delta_event.Logprob] | None = None,
) -> list[StreamingResponsesResponse]:
    """Emit events for text content delta streaming."""
    return [
        ResponseTextDeltaEvent(
            type="response.output_text.delta",
            sequence_number=-1,
            content_index=state.content_index,
            output_index=state.output_index,
            item_id=state.current_item_id,
            delta=delta,
            logprobs=logprobs or [],
        )
    ]


def emit_reasoning_delta_events(
    delta: str,
    state: SimpleStreamingState | HarmonyStreamingState,
) -> list[StreamingResponsesResponse]:
    """Emit events for reasoning text delta streaming."""
    return [
        ResponseReasoningTextDeltaEvent(
            type="response.reasoning_text.delta",
            item_id=state.current_item_id,
            output_index=state.output_index,
            content_index=state.content_index,
            delta=delta,
            sequence_number=-1,
        )
    ]


def emit_function_call_delta_events(
    delta: str,
    state: SimpleStreamingState | HarmonyStreamingState,
) -> list[StreamingResponsesResponse]:
    """Emit events for function call argument deltas."""
    return [
        ResponseFunctionCallArgumentsDeltaEvent(
            item_id=state.current_item_id,
            delta=delta,
            output_index=state.output_index,
            sequence_number=-1,
            type="response.function_call_arguments.delta",
        )
    ]


# =====================================================================
# Shared leaf helpers — done events
# =====================================================================


def emit_text_done_events(
    text: str,
    state: SimpleStreamingState | HarmonyStreamingState,
) -> list[StreamingResponsesResponse]:
    """Emit events when a final text output item completes."""
    text_content = ResponseOutputText(
        type="output_text",
        text=text,
        annotations=[],
    )
    events: list[StreamingResponsesResponse] = []
    events.append(
        ResponseTextDoneEvent(
            type="response.output_text.done",
            sequence_number=-1,
            output_index=state.output_index,
            content_index=state.content_index,
            text=text,
            logprobs=[],
            item_id=state.current_item_id,
        )
    )
    events.append(
        ResponseContentPartDoneEvent(
            type="response.content_part.done",
            sequence_number=-1,
            item_id=state.current_item_id,
            output_index=state.output_index,
            content_index=state.content_index,
            part=text_content,
        )
    )
    events.append(
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            sequence_number=-1,
            output_index=state.output_index,
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


def emit_reasoning_done_events(
    text: str,
    state: SimpleStreamingState | HarmonyStreamingState,
) -> list[StreamingResponsesResponse]:
    """Emit events when a reasoning (analysis) item completes."""
    content = ResponseReasoningTextContent(
        text=text,
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
            output_index=state.output_index,
            content_index=state.content_index,
            text=text,
        )
    )
    events.append(
        ResponseReasoningPartDoneEvent(
            type="response.reasoning_part.done",
            sequence_number=-1,
            item_id=state.current_item_id,
            output_index=state.output_index,
            content_index=state.content_index,
            part=content,
        )
    )
    events.append(
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            sequence_number=-1,
            output_index=state.output_index,
            item=reasoning_item,
        )
    )
    return events


def emit_function_call_done_events(
    function_name: str,
    arguments: str,
    state: SimpleStreamingState | HarmonyStreamingState,
) -> list[StreamingResponsesResponse]:
    """Emit events when a function call completes."""
    events: list[StreamingResponsesResponse] = []
    events.append(
        ResponseFunctionCallArgumentsDoneEvent(
            type="response.function_call_arguments.done",
            arguments=arguments,
            name=function_name,
            item_id=state.current_item_id,
            output_index=state.output_index,
            sequence_number=-1,
        )
    )
    events.append(
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            sequence_number=-1,
            output_index=state.output_index,
            item=ResponseFunctionToolCall(
                type="function_call",
                arguments=arguments,
                name=function_name,
                id=state.current_item_id,
                call_id=state.tool_call_id,
                status="completed",
            ),
        )
    )
    return events


# =====================================================================
# Simple streaming helpers
# =====================================================================


class _StateType(Enum):
    NONE = auto()
    CONTENT = auto()
    REASONING = auto()
    TOOL_CALL = auto()


@dataclass
class SimpleStreamingState:
    output_index: int = 0
    current_item_id: str = ""
    content_index: int = 0
    accumulated_text: str = ""
    tool_call_id: str = ""
    tool_call_name: str = ""
    tool_call_index: int | None = None
    current_state: _StateType = field(default_factory=lambda: _StateType.NONE)


def emit_simple_content_open(
    state: SimpleStreamingState,
) -> list[StreamingResponsesResponse]:
    state.current_state = _StateType.CONTENT
    state.current_item_id = random_uuid()
    state.content_index = 0
    state.accumulated_text = ""
    return emit_text_open_events(state)


def emit_simple_content_delta(
    state: SimpleStreamingState,
    delta: str,
    logprobs: list[response_text_delta_event.Logprob] | None = None,
) -> list[StreamingResponsesResponse]:
    state.accumulated_text += delta
    return emit_text_delta_events(delta, state, logprobs=logprobs)


def emit_simple_content_done(
    state: SimpleStreamingState,
) -> list[StreamingResponsesResponse]:
    events = emit_text_done_events(state.accumulated_text, state)
    state.output_index += 1
    state.current_state = _StateType.NONE
    return events


def emit_simple_reasoning_open(
    state: SimpleStreamingState,
) -> list[StreamingResponsesResponse]:
    state.current_state = _StateType.REASONING
    state.current_item_id = random_uuid()
    state.content_index = 0
    state.accumulated_text = ""
    return emit_reasoning_open_events(state)


def emit_simple_reasoning_delta(
    state: SimpleStreamingState,
    delta: str,
) -> list[StreamingResponsesResponse]:
    state.accumulated_text += delta
    return emit_reasoning_delta_events(delta, state)


def emit_simple_reasoning_done(
    state: SimpleStreamingState,
) -> list[StreamingResponsesResponse]:
    events = emit_reasoning_done_events(state.accumulated_text, state)
    state.output_index += 1
    state.current_state = _StateType.NONE
    return events


def emit_simple_tool_call_open(
    state: SimpleStreamingState,
    name: str,
    index: int | None,
) -> list[StreamingResponsesResponse]:
    state.current_state = _StateType.TOOL_CALL
    state.current_item_id = random_uuid()
    state.tool_call_id = f"call_{random_uuid()}"
    state.tool_call_name = name
    state.tool_call_index = index
    state.accumulated_text = ""
    return emit_function_call_open_events(name, state)


def emit_simple_tool_call_delta(
    state: SimpleStreamingState,
    delta: str,
) -> list[StreamingResponsesResponse]:
    state.accumulated_text += delta
    return emit_function_call_delta_events(delta, state)


def emit_simple_tool_call_done(
    state: SimpleStreamingState,
) -> list[StreamingResponsesResponse]:
    events = emit_function_call_done_events(
        state.tool_call_name, state.accumulated_text, state
    )
    state.output_index += 1
    state.current_state = _StateType.NONE
    return events


class _StateHandlers(NamedTuple):
    """Tuple for each state: open(start), delta(chunk), done(finish)."""

    open_fn: Callable[..., list[StreamingResponsesResponse]]
    delta_fn: Callable[..., list[StreamingResponsesResponse]]
    done_fn: Callable[..., list[StreamingResponsesResponse]]


def split_delta(delta: DeltaMessage) -> list[DeltaMessage]:
    """Decompose a DeltaMessage with multiple fields into atomic deltas.

    The Responses API emits typed SSE events (one type per event), so a
    compound DeltaMessage must be split before entering the state machine.
    Order: reasoning -> content -> tool_calls (grouped by index).
    """
    has_reasoning = delta.reasoning is not None
    has_content = delta.content is not None
    has_tools = bool(delta.tool_calls)
    parts = int(has_reasoning) + int(has_content) + int(has_tools)

    if parts <= 1 and (
        not has_tools
        or len({tc.index for tc in delta.tool_calls if tc.index is not None}) <= 1
    ):
        return [delta]

    deltas: list[DeltaMessage] = []
    if has_reasoning:
        deltas.append(DeltaMessage(reasoning=delta.reasoning))
    if has_content:
        deltas.append(DeltaMessage(content=delta.content))
    if has_tools:
        groups: dict[int | None, list[DeltaToolCall]] = {}
        for tc in delta.tool_calls:
            groups.setdefault(tc.index, []).append(tc)
        for tcs in groups.values():
            deltas.append(DeltaMessage(tool_calls=tcs))
    return deltas or [delta]


class SimpleStreamingEventProcessor:
    """
    State-machine processor for the simple (non-Harmony) streaming path.

    Core flow:
      1. Resolve the target state from the delta_message
         (CONTENT / REASONING / TOOL_CALL).
      2. If the target state differs from the current one,
         close_current() then open() the new state.
      3. emit_delta() produces the incremental events for the state.

    State lifecycle:
      open()  ->  repeated emit_delta()  ->  close_current()
    """

    _STATE_HANDLERS: ClassVar[dict[_StateType, _StateHandlers]] = {
        _StateType.CONTENT: _StateHandlers(
            emit_simple_content_open,
            emit_simple_content_delta,
            emit_simple_content_done,
        ),
        _StateType.REASONING: _StateHandlers(
            emit_simple_reasoning_open,
            emit_simple_reasoning_delta,
            emit_simple_reasoning_done,
        ),
        _StateType.TOOL_CALL: _StateHandlers(
            emit_simple_tool_call_open,
            emit_simple_tool_call_delta,
            emit_simple_tool_call_done,
        ),
    }

    def __init__(self, state: SimpleStreamingState | None = None) -> None:
        self.state = state or SimpleStreamingState()

    def resolve_target_state(
        self, delta_message: DeltaMessage
    ) -> tuple[_StateType, Any]:
        """
        Decide which state the next delta belongs to.

        Priority: TOOL_CALL > REASONING > CONTENT, fallback to NONE.
        For TOOL_CALL the first tool_call object is also returned so
        callers can detect a switch between consecutive tools.
        """
        if (
            delta_message.tool_calls
            and delta_message.tool_calls[0].function is not None
        ):
            return _StateType.TOOL_CALL, delta_message.tool_calls[0]
        if delta_message.reasoning is not None:
            return _StateType.REASONING, None
        if delta_message.content:
            return _StateType.CONTENT, None
        return _StateType.NONE, None

    def needs_transition(self, target_state: _StateType, tool_call: Any) -> bool:
        """
        Return True when we must close the current state and open a new one.

        Two cases trigger a transition:
          1. The target state differs from the current state
             (e.g. CONTENT -> TOOL_CALL).
          2. We are already in TOOL_CALL but the next tool_call has a
             different index (multiple consecutive tool calls).
        """
        if self.state.current_state != target_state:
            return True
        return (
            target_state == _StateType.TOOL_CALL
            and tool_call is not None
            and self.state.tool_call_index is not None
            and tool_call.index is not None
            and self.state.tool_call_index != tool_call.index
        )

    def close_current(self) -> list[StreamingResponsesResponse]:
        """Close the current state and emit its 'done' event sequence."""
        handlers = self._STATE_HANDLERS.get(self.state.current_state)
        if handlers is None:
            return []
        return handlers.done_fn(self.state)

    def open(
        self, target_state: _StateType, tool_call: Any = None
    ) -> list[StreamingResponsesResponse]:
        """Open a new state and emit its 'added' / 'open' event sequence."""
        handlers = self._STATE_HANDLERS[target_state]
        if target_state == _StateType.TOOL_CALL:
            assert tool_call is not None
            return handlers.open_fn(
                self.state, tool_call.function.name, tool_call.index
            )
        return handlers.open_fn(self.state)

    def emit_delta(
        self,
        delta_message: DeltaMessage,
        output: CompletionOutput,
        get_logprobs: Callable[
            [CompletionOutput], list[response_text_delta_event.Logprob]
        ]
        | None = None,
    ) -> list[StreamingResponsesResponse]:
        """Emit incremental events for the current state from the delta."""
        handlers = self._STATE_HANDLERS[self.state.current_state]

        if self.state.current_state == _StateType.TOOL_CALL:
            assert delta_message.tool_calls is not None
            combined_args = ""
            for tc in delta_message.tool_calls:
                if tc.function is not None and tc.function.arguments:
                    combined_args += tc.function.arguments
            if combined_args:
                return handlers.delta_fn(self.state, combined_args)
            return []
        elif self.state.current_state == _StateType.REASONING:
            assert delta_message.reasoning is not None
            return handlers.delta_fn(self.state, delta_message.reasoning)
        elif self.state.current_state == _StateType.CONTENT:
            assert delta_message.content is not None
            logprobs = get_logprobs(output) if get_logprobs else []
            return handlers.delta_fn(self.state, delta_message.content, logprobs)
        return []
