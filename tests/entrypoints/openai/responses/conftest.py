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
    "VLLM_GPT_OSS_SYSTEM_START_DATE": "2023-09-12",
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
        "response.web_search_call.completed": "response.web_search_call.in_progress",
    }
    # fmt: on
    return event_pairs


async def retry_for_tool_call(
    client,
    *,
    model: str,
    expected_tool_type: str = "function_call",
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


def validate_streaming_event_stack(
    events: list, pairs_of_event_types: dict[str, str]
) -> None:
    """Validate that streaming events are properly nested/paired."""
    stack: list[str] = []
    for event in events:
        etype = event.type
        if etype == "response.created":
            stack.append(etype)
        elif etype == "response.completed":
            assert stack and stack[-1] == pairs_of_event_types[etype], (
                f"Unexpected stack top for {etype}: "
                f"got {stack[-1] if stack else '<empty>'}"
            )
            stack.pop()
        elif etype.endswith("added") or etype == "response.mcp_call.in_progress":
            stack.append(etype)
        elif etype.endswith("delta"):
            if stack and stack[-1] == etype:
                continue
            stack.append(etype)
        elif etype.endswith("done") or etype == "response.mcp_call.completed":
            assert etype in pairs_of_event_types, f"Unknown done event: {etype}"
            expected_start = pairs_of_event_types[etype]
            assert stack and stack[-1] == expected_start, (
                f"Stack mismatch for {etype}: "
                f"expected {expected_start}, "
                f"got {stack[-1] if stack else '<empty>'}"
            )
            stack.pop()
    assert len(stack) == 0, f"Unclosed events on stack: {stack}"


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
