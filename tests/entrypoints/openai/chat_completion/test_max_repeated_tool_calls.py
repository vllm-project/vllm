# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the opt-in `max_repeated_tool_calls` loop guard (RFC #66858)."""

import json

import pytest
from pydantic import ValidationError

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    count_trailing_identical_tool_calls,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
            },
        },
    }
]


def _call(name: str = "search", arguments: dict | str | None = None, idx: int = 0):
    if arguments is None:
        arguments = {"q": "vllm"}
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return {
        "id": f"call_{idx}",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _step(*calls, idx: int = 0) -> list[dict]:
    """One agent step: an assistant turn with tool calls and their results."""
    return [
        {"role": "assistant", "content": None, "tool_calls": list(calls)},
        *(
            {"role": "tool", "tool_call_id": f"call_{idx}", "content": "result"}
            for _ in calls
        ),
    ]


def _history(*steps: list[dict]) -> list[dict]:
    messages: list[dict] = [{"role": "user", "content": "find it"}]
    for step in steps:
        messages.extend(step)
    return messages


def _request(messages, **kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model", messages=messages, tools=TOOLS, **kwargs
    )


def test_off_by_default():
    request = _request(_history(*[_step(_call()) for _ in range(5)]))
    assert request.max_repeated_tool_calls is None
    assert request.tool_choice == "auto"


@pytest.mark.parametrize("repeats, expected", [(2, "auto"), (3, "none")])
def test_cap_reached_disables_tool_calls(repeats, expected):
    messages = _history(*[_step(_call()) for _ in range(repeats)])
    request = _request(messages, max_repeated_tool_calls=3)
    assert request.tool_choice == expected


def test_run_counts_only_identical_trailing_calls():
    a = _call(arguments={"q": "a"})
    b = _call(arguments={"q": "b"})
    # Different arguments or tool name end the run.
    assert count_trailing_identical_tool_calls(_history(_step(a), _step(b))) == 1
    assert (
        count_trailing_identical_tool_calls(
            _history(_step(_call()), _step(_call(name="lookup")))
        )
        == 1
    )
    # Key order and whitespace don't matter; parallel calls compare as a set.
    reordered = _call(arguments='{"n":1,"q":"x"}')
    assert (
        count_trailing_identical_tool_calls(
            _history(_step(_call(arguments='{"q": "x", "n": 1}')), _step(reordered))
        )
        == 2
    )
    assert count_trailing_identical_tool_calls(_history(_step(a, b), _step(b, a))) == 2
    # A user turn or a plain answer starts the count again.
    messages = _history(_step(a), _step(a))
    messages.append({"role": "user", "content": "try again"})
    messages.extend(_step(a))
    assert count_trailing_identical_tool_calls(messages) == 1
    messages.append({"role": "assistant", "content": "done"})
    assert count_trailing_identical_tool_calls(messages) == 0


@pytest.mark.parametrize(
    "tool_choice", ["required", {"type": "function", "function": {"name": "search"}}]
)
def test_explicit_tool_choice_is_left_alone(tool_choice):
    messages = _history(*[_step(_call()) for _ in range(3)])
    request = _request(messages, tool_choice=tool_choice, max_repeated_tool_calls=1)
    assert request.tool_choice != "none"


def test_cap_must_be_positive():
    with pytest.raises(ValidationError):
        _request(_history(), max_repeated_tool_calls=0)
