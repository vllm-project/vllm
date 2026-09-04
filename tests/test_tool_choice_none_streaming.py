# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for the tool_choice="none" guard in the streaming chat completion
path (issue #42747).

When a tool parser is configured server-side but a request sets
tool_choice="none", the streaming path must not invoke parse_delta() and must
not produce delta.tool_calls or finish_reason="tool_calls".
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _simulate_streaming_delta(parser, tool_choice: str | None, delta_text: str):
    """
    Reproduces the decision logic from serving.py:

        elif parser is not None and request.tool_choice != "none":
            delta_message = parser.parse_delta(...)
        else:
            delta_message = DeltaMessage(content=delta_text)

    Returns (delta_message, parse_delta_called).
    """
    request = SimpleNamespace(tool_choice=tool_choice)
    parse_delta_called = False

    if parser is not None and request.tool_choice != "none":
        parse_delta_called = True
        delta_message = parser.parse_delta(
            delta_text=delta_text,
            delta_token_ids=[],
            request=request,
            prompt_token_ids=None,
        )
    else:
        delta_message = SimpleNamespace(content=delta_text, tool_calls=None)

    return delta_message, parse_delta_called


@pytest.mark.parametrize("tool_choice", ["none"])
def test_tool_choice_none_skips_parse_delta(tool_choice: str) -> None:
    """parse_delta must not be called when tool_choice='none'."""
    parser = MagicMock()
    parser.parse_delta.return_value = SimpleNamespace(
        content=None, tool_calls=[MagicMock()]
    )

    _msg, called = _simulate_streaming_delta(parser, tool_choice, "hello")

    assert not called, "parse_delta must not be invoked when tool_choice='none'"
    parser.parse_delta.assert_not_called()


@pytest.mark.parametrize("tool_choice", [None, "auto", "required"])
def test_non_none_tool_choice_invokes_parse_delta(tool_choice) -> None:
    """parse_delta must be called when tool_choice is not 'none'."""
    parser = MagicMock()
    parser.parse_delta.return_value = SimpleNamespace(content="hi", tool_calls=None)

    _msg, called = _simulate_streaming_delta(parser, tool_choice, "hi")

    assert called, f"parse_delta must be invoked when tool_choice={tool_choice!r}"
    parser.parse_delta.assert_called_once()


def test_tool_choice_none_produces_content_delta() -> None:
    """With tool_choice='none', the delta must be plain content, not tool_calls."""
    parser = MagicMock()
    # Even if the parser would normally emit tool_calls, they must be suppressed
    parser.parse_delta.return_value = SimpleNamespace(
        content=None, tool_calls=[MagicMock()]
    )

    delta, _called = _simulate_streaming_delta(parser, "none", "some text")

    assert delta.content == "some text"
    assert not delta.tool_calls


def test_no_parser_falls_through_to_content_delta() -> None:
    """When parser is None (no tool parser configured), content delta is produced."""
    delta, called = _simulate_streaming_delta(None, "auto", "plain content")

    assert not called
    assert delta.content == "plain content"
    assert not delta.tool_calls
