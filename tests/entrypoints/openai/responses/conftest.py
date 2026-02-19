# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest


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
