# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest


def test_chat_completion_request_with_no_tools():
    # tools key is not present
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "facebook/opt-125m",
        }
    )
    assert request.tool_choice == "none"

    # tools key is None
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "facebook/opt-125m",
            "tools": None,
        }
    )
    assert request.tool_choice == "none"

    # tools key present but empty -- should be rejected
    with pytest.raises(ValueError, match="must not be an empty array"):
        ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "facebook/opt-125m",
                "tools": [],
            }
        )


@pytest.mark.parametrize("tool_choice", ["auto", "required"])
def test_chat_completion_request_with_tool_choice_but_no_tools(tool_choice):
    with pytest.raises(
        ValueError, match="When using `tool_choice`, `tools` must be set."
    ):
        ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "facebook/opt-125m",
                "tool_choice": tool_choice,
            }
        )

    with pytest.raises(
        ValueError, match="When using `tool_choice`, `tools` must be set."
    ):
        ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "facebook/opt-125m",
                "tool_choice": tool_choice,
                "tools": None,
            }
        )


def test_retention_directives_monotonic_priorities_valid():
    """Non-increasing priorities across token positions are accepted."""
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "facebook/opt-125m",
            "retention_directives": [
                {"start": 0, "end": 512, "priority": 90},
                {"start": 512, "end": 1024, "priority": 70},
                {"start": 1024, "end": None, "priority": 50},
            ],
        }
    )
    assert request.retention_directives is not None
    assert len(request.retention_directives) == 3


def test_retention_directives_monotonic_priorities_equal_is_valid():
    """Equal priorities at different starts are accepted (non-increasing)."""
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "facebook/opt-125m",
            "retention_directives": [
                {"start": 0, "end": 512, "priority": 80},
                {"start": 512, "end": 1024, "priority": 80},
            ],
        }
    )
    assert request.retention_directives is not None


def test_retention_directives_empty_or_none_is_valid():
    """Empty and None are trivially valid."""
    values: list[list | None] = [None, []]
    for value in values:
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "facebook/opt-125m",
                "retention_directives": value,
            }
        )
        assert (request.retention_directives or None) is None


def test_retention_directives_single_directive_is_valid():
    """A single directive has no ordering constraint to violate."""
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "facebook/opt-125m",
            "retention_directives": [
                {"start": 0, "end": None, "priority": 42},
            ],
        }
    )
    assert len(request.retention_directives) == 1


def test_retention_directives_increasing_priority_rejected():
    """A later directive with strictly higher priority is rejected."""
    with pytest.raises(ValueError, match="non-increasing"):
        ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "facebook/opt-125m",
                "retention_directives": [
                    {"start": 0, "end": 512, "priority": 50},
                    {"start": 512, "end": 1024, "priority": 90},
                ],
            }
        )


def test_retention_directives_unsorted_input_still_validated():
    """Directives passed out of order are still checked after sort-by-start."""
    with pytest.raises(ValueError, match="non-increasing"):
        ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "facebook/opt-125m",
                "retention_directives": [
                    # out of order; after sort by start: 30 -> 90 is an increase
                    {"start": 512, "end": 1024, "priority": 90},
                    {"start": 0, "end": 512, "priority": 30},
                ],
            }
        )
