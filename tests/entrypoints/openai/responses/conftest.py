# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

import pytest

logger = logging.getLogger(__name__)

BASE_TEST_ENV = {
    # The day vLLM said "hello world" on arxiv 🚀
    "VLLM_SYSTEM_START_DATE": "2023-09-12",
}
DEFAULT_MAX_RETRIES = 3


@pytest.fixture
def pairs_of_event_types() -> dict[str, str]:
    """Links the 'done' event type with the corresponding 'start' event type.

    This mapping should link all done <-> start events; if tests mean to
    restrict the allowed events, they should filter this fixture to avoid
    copy + paste errors in the mappings or unexpected KeyErrors due to missing
    events.
    """
    # fmt: off
    event_pairs = {
        "response.completed": "response.created",
        "response.output_item.done": "response.output_item.added",
        "response.content_part.done": "response.content_part.added",
        "response.output_text.done": "response.output_text.delta",
        "response.reasoning_text.done": "response.reasoning_text.delta",
        "response.reasoning_part.done": "response.reasoning_part.added",
        "response.mcp_call_arguments.done": "response.mcp_call_arguments.delta",
        "response.mcp_call.completed": "response.mcp_call.in_progress",
        "response.function_call_arguments.done": "response.function_call_arguments.delta", # noqa: E501
        "response.code_interpreter_call_code.done": "response.code_interpreter_call_code.delta", # noqa: E501
        "response.code_interpreter_call.completed": "response.code_interpreter_call.in_progress", # noqa: E501
        "response.web_search_call.completed": "response.web_search_call.in_progress",
    }
    # fmt: on
    return event_pairs


async def retry_for_tool_call(
    client,
    *,
    model: str,
    expected_tool_type: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    **create_kwargs: Any,
):
    """Call ``client.responses.create`` up to *max_retries* times, returning
    the first response that contains an output item of *expected_tool_type*.

    Returns the **last** response if none match so the caller's assertions
    fire with a clear diagnostic.
    """
    last_response = None
    for attempt in range(max_retries):
        response = await client.responses.create(model=model, **create_kwargs)
        last_response = response
        if any(
            getattr(item, "type", None) == expected_tool_type
            for item in response.output
        ):
            return response
    assert last_response is not None
    return last_response


async def retry_streaming_for(
    client,
    *,
    model: str,
    validate_events: Callable[[list], bool],
    max_retries: int = DEFAULT_MAX_RETRIES,
    **create_kwargs: Any,
) -> list:
    """Call ``client.responses.create(stream=True)`` up to *max_retries*
    times, returning the first event list where *validate_events* returns
    ``True``.
    """
    last_events: list = []
    for attempt in range(max_retries):
        stream = await client.responses.create(
            model=model, stream=True, **create_kwargs
        )
        events: list = []
        async for event in stream:
            events.append(event)
        last_events = events
        if validate_events(events):
            return events
    return last_events


def has_output_type(response, type_name: str) -> bool:
    """Return True if *response* has at least one output item of *type_name*."""
    return any(getattr(item, "type", None) == type_name for item in response.output)


def events_contain_type(events: list, type_substring: str) -> bool:
    """Return True if any event's type contains *type_substring*."""
    return any(type_substring in getattr(e, "type", "") for e in events)


def _validate_event_pairing(events: list, pairs_of_event_types: dict[str, str]) -> None:
    """Validate that streaming events are properly nested/paired.

    Derives push/pop sets from *pairs_of_event_types* so that every
    start/end pair in the dict is handled automatically.
    """
    start_events = set(pairs_of_event_types.values())
    end_events = set(pairs_of_event_types.keys())

    stack: list[str] = []
    for event in events:
        etype = event.type
        if etype in end_events:
            expected_start = pairs_of_event_types[etype]
            assert stack and stack[-1] == expected_start, (
                f"Stack mismatch for {etype}: "
                f"expected {expected_start}, "
                f"got {stack[-1] if stack else '<empty>'}"
            )
            stack.pop()
        elif etype in start_events:
            # Consecutive deltas of the same type share a single stack slot.
            if etype.endswith("delta") and stack and stack[-1] == etype:
                continue
            stack.append(etype)
        # else: passthrough event (e.g. response.in_progress,
        # web_search_call.searching, code_interpreter_call.interpreting)
    assert len(stack) == 0, f"Unclosed events on stack: {stack}"


def _validate_event_ordering(events: list) -> None:
    """Validate that envelope events appear in the correct positions."""
    assert len(events) >= 2, f"Expected at least 2 events, got {len(events)}"

    # First event must be response.created
    assert events[0].type == "response.created", (
        f"First event must be response.created, got {events[0].type}"
    )
    # Last event must be response.completed
    assert events[-1].type == "response.completed", (
        f"Last event must be response.completed, got {events[-1].type}"
    )

    # response.in_progress, if present, must be the second event
    in_progress_indices = [
        i for i, e in enumerate(events) if e.type == "response.in_progress"
    ]
    if in_progress_indices:
        assert in_progress_indices == [1], (
            f"response.in_progress must be the second event, "
            f"found at indices {in_progress_indices}"
        )

    # Exactly one created and one completed
    created_count = sum(1 for e in events if e.type == "response.created")
    completed_count = sum(1 for e in events if e.type == "response.completed")
    assert created_count == 1, (
        f"Expected exactly 1 response.created, got {created_count}"
    )
    assert completed_count == 1, (
        f"Expected exactly 1 response.completed, got {completed_count}"
    )


def _validate_field_consistency(events: list) -> None:
    """Validate item_id, output_index, and content_index consistency.

    Tracks the active output item established by ``output_item.added``
    and verifies that all subsequent events for that item carry matching
    identifiers until ``output_item.done`` closes it.
    """
    _SESSION_EVENTS = {
        "response.created",
        "response.in_progress",
        "response.completed",
    }

    active_item_id: str | None = None
    active_output_index: int | None = None
    last_output_index: int = -1
    active_content_index: int | None = None

    for event in events:
        etype = event.type

        if etype in _SESSION_EVENTS:
            continue

        # --- output_item.added: opens a new item ------------------
        if etype == "response.output_item.added":
            item = getattr(event, "item", None)
            output_index = getattr(event, "output_index", None)

            assert item is not None, "output_item.added must have an item"
            item_id = getattr(item, "id", None)
            assert item_id, "output_item.added item must have an id"

            # output_index must be non-decreasing across items
            if output_index is not None:
                assert output_index >= last_output_index, (
                    f"output_index went backwards: {output_index} < {last_output_index}"
                )
                last_output_index = output_index

            active_item_id = item_id
            active_output_index = output_index
            active_content_index = None
            continue

        # --- output_item.done: closes the active item -------------
        if etype == "response.output_item.done":
            item = getattr(event, "item", None)
            output_index = getattr(event, "output_index", None)

            assert item is not None, "output_item.done must have an item"
            done_item_id = getattr(item, "id", None)

            if active_item_id is not None and done_item_id:
                assert done_item_id == active_item_id, (
                    f"output_item.done item.id mismatch: "
                    f"expected {active_item_id}, got {done_item_id}"
                )
            if active_output_index is not None and output_index is not None:
                assert output_index == active_output_index, (
                    f"output_item.done output_index mismatch: "
                    f"expected {active_output_index}, got {output_index}"
                )

            active_item_id = None
            active_output_index = None
            active_content_index = None
            continue

        # --- content_part / reasoning_part added: sets content_index
        if etype in (
            "response.content_part.added",
            "response.reasoning_part.added",
        ):
            _assert_item_fields(event, etype, active_item_id, active_output_index)
            active_content_index = getattr(event, "content_index", None)
            continue

        # --- all other item-level events --------------------------
        _assert_item_fields(event, etype, active_item_id, active_output_index)

        # content_index (only meaningful on events that carry it)
        content_index = getattr(event, "content_index", None)
        if content_index is not None and active_content_index is not None:
            assert content_index == active_content_index, (
                f"{etype} content_index mismatch: "
                f"expected {active_content_index}, got {content_index}"
            )


def _assert_item_fields(
    event,
    etype: str,
    active_item_id: str | None,
    active_output_index: int | None,
) -> None:
    """Check that *event*'s item_id and output_index match the active item."""
    event_item_id = getattr(event, "item_id", None)
    output_index = getattr(event, "output_index", None)

    if active_item_id is not None and event_item_id is not None:
        assert event_item_id == active_item_id, (
            f"{etype} item_id mismatch: expected {active_item_id}, got {event_item_id}"
        )
    if active_output_index is not None and output_index is not None:
        assert output_index == active_output_index, (
            f"{etype} output_index mismatch: "
            f"expected {active_output_index}, got {output_index}"
        )


def validate_streaming_event_stack(
    events: list, pairs_of_event_types: dict[str, str]
) -> None:
    """Validate streaming events: pairing, ordering, and field consistency.

    Checks three aspects:
    1. **Event pairing** — start/end events are properly nested
       (stack-based matching derived from *pairs_of_event_types*).
    2. **Event ordering** — envelope events (``created``,
       ``in_progress``, ``completed``) appear at the correct positions.
    3. **Field consistency** — ``item_id``, ``output_index``, and
       ``content_index`` are consistent across related events within
       each output item's lifecycle.
    """
    _validate_event_pairing(events, pairs_of_event_types)
    _validate_event_ordering(events)
    _validate_field_consistency(events)


def log_response_diagnostics(
    response,
    *,
    label: str = "Response Diagnostics",
) -> dict[str, Any]:
    """Extract and log diagnostic info from a Responses API response.

    Logs reasoning, tool-call attempts, MCP items, and output types so
    that CI output (``pytest -s`` or ``--log-cli-level=INFO``) gives
    full visibility into model behaviour even on passing runs.

    Returns the extracted data so callers can make additional assertions
    if needed.
    """
    reasoning_texts = [
        text
        for item in response.output
        if getattr(item, "type", None) == "reasoning"
        for content in getattr(item, "content", [])
        if (text := getattr(content, "text", None))
    ]

    tool_call_attempts = [
        {
            "recipient": msg.get("recipient"),
            "channel": msg.get("channel"),
        }
        for msg in response.output_messages
        if (msg.get("recipient") or "").startswith("python")
    ]

    mcp_items = [
        {
            "name": getattr(item, "name", None),
            "status": getattr(item, "status", None),
        }
        for item in response.output
        if getattr(item, "type", None) == "mcp_call"
    ]

    output_types = [getattr(o, "type", None) for o in response.output]

    diagnostics = {
        "model_attempted_tool_calls": bool(tool_call_attempts),
        "tool_call_attempts": tool_call_attempts,
        "mcp_items": mcp_items,
        "reasoning": reasoning_texts,
        "output_text": response.output_text,
        "output_types": output_types,
    }

    logger.info(
        "\n====== %s ======\n%s\n==============================",
        label,
        json.dumps(diagnostics, indent=2, default=str),
    )

    return diagnostics
