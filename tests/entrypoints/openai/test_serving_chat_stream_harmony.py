# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for harmony streaming delta extraction.
"""

from dataclasses import dataclass, field
from unittest.mock import patch

import pytest

from vllm.entrypoints.openai.serving_chat_stream_harmony import (
    extract_harmony_streaming_delta,
)


@dataclass
class MockMessage:
    """Mock message object for testing."""

    channel: str | None = None
    recipient: str | None = None


@dataclass
class MockStreamableParser:
    """Mock StreamableParser for testing without openai_harmony dependency."""

    messages: list[MockMessage] = field(default_factory=list)


class TestExtractHarmonyStreamingDelta:
    """Tests for extract_harmony_streaming_delta function."""

    @pytest.mark.parametrize(
        "delta_text,expected_content",
        [
            ("Hello, world!", "Hello, world!"),
            ("", ""),
        ],
    )
    def test_final_channel_returns_content_delta(self, delta_text, expected_content):
        """Test that final channel returns a DeltaMessage with content."""
        parser = MockStreamableParser()
        delta_message, tools_streamed = extract_harmony_streaming_delta(
            harmony_parser=parser,
            cur_channel="final",
            cur_recipient=None,
            prev_recipient=None,
            delta_text=delta_text,
            include_reasoning=False,
        )

        assert delta_message is not None
        assert delta_message.content == expected_content
        assert tools_streamed is False

    @pytest.mark.parametrize(
        "include_reasoning,expected_has_message",
        [
            (True, True),
            (False, False),
        ],
    )
    def test_analysis_channel_reasoning(self, include_reasoning, expected_has_message):
        """Test analysis channel respects include_reasoning flag."""
        parser = MockStreamableParser()
        delta_message, tools_streamed = extract_harmony_streaming_delta(
            harmony_parser=parser,
            cur_channel="analysis",
            cur_recipient=None,
            prev_recipient=None,
            delta_text="Let me think...",
            include_reasoning=include_reasoning,
        )

        if expected_has_message:
            assert delta_message is not None
            assert delta_message.reasoning == "Let me think..."
        else:
            assert delta_message is None
        assert tools_streamed is False

    @pytest.mark.parametrize("channel", ["commentary", "analysis"])
    @patch("vllm.entrypoints.openai.serving_chat_stream_harmony.make_tool_call_id")
    def test_new_tool_call(self, mock_make_tool_call_id, channel):
        """Test new tool call creation when recipient changes."""
        mock_make_tool_call_id.return_value = "call_test123"
        parser = MockStreamableParser()

        delta_message, tools_streamed = extract_harmony_streaming_delta(
            harmony_parser=parser,
            cur_channel=channel,
            cur_recipient="functions.get_weather",
            prev_recipient=None,
            delta_text="",
            include_reasoning=False,
        )

        assert delta_message is not None
        assert len(delta_message.tool_calls) == 1
        tool_call = delta_message.tool_calls[0]
        assert tool_call.id == "call_test123"
        assert tool_call.type == "function"
        assert tool_call.function.name == "get_weather"
        assert tool_call.function.arguments == ""
        assert tool_call.index == 0
        assert tools_streamed is True

    @pytest.mark.parametrize("channel", ["commentary", "analysis"])
    def test_tool_call_argument_streaming(self, channel):
        """Test streaming tool call arguments (same recipient)."""
        parser = MockStreamableParser()

        delta_message, tools_streamed = extract_harmony_streaming_delta(
            harmony_parser=parser,
            cur_channel=channel,
            cur_recipient="functions.get_weather",
            prev_recipient="functions.get_weather",
            delta_text='{"location": "Paris"}',
            include_reasoning=False,
        )

        assert delta_message is not None
        tool_call = delta_message.tool_calls[0]
        assert tool_call.id is None
        assert tool_call.function.arguments == '{"location": "Paris"}'
        assert tool_call.index == 0
        assert tools_streamed is True

    @pytest.mark.parametrize("channel", ["commentary", "analysis"])
    def test_tool_call_empty_arguments_returns_none(self, channel):
        """Test empty delta_text with same recipient returns None."""
        parser = MockStreamableParser()

        delta_message, tools_streamed = extract_harmony_streaming_delta(
            harmony_parser=parser,
            cur_channel=channel,
            cur_recipient="functions.get_weather",
            prev_recipient="functions.get_weather",
            delta_text="",
            include_reasoning=False,
        )

        assert delta_message is None
        assert tools_streamed is False

    def test_tool_call_index_from_previous_messages(self):
        """Test tool call index accounts for previous function messages."""
        messages = [
            MockMessage(channel="analysis", recipient=None),  # Not counted
            MockMessage(channel="commentary", recipient="functions.tool1"),  # Counted
            MockMessage(channel="final", recipient=None),  # Not counted
        ]
        parser = MockStreamableParser(messages=messages)

        delta_message, _ = extract_harmony_streaming_delta(
            harmony_parser=parser,
            cur_channel="commentary",
            cur_recipient="functions.tool2",
            prev_recipient="functions.tool2",
            delta_text="args",
            include_reasoning=False,
        )

        assert delta_message.tool_calls[0].index == 1

    @pytest.mark.parametrize(
        "channel,recipient",
        [
            ("commentary", None),
            ("commentary", "browser.search"),
        ],
    )
    def test_returns_tool_call_preambles(self, channel, recipient):
        """Test that invalid channel/recipient combinations return None."""
        parser = MockStreamableParser()
        delta_text = "some text"
        delta_message, tools_streamed = extract_harmony_streaming_delta(
            harmony_parser=parser,
            cur_channel=channel,
            cur_recipient=recipient,
            prev_recipient=None,
            delta_text=delta_text,
            include_reasoning=True,
        )

        assert delta_message.content == delta_text
        assert tools_streamed is False

    @pytest.mark.parametrize(
        "channel,recipient",
        [
            (None, None),
            ("unknown_channel", None),
        ],
    )
    def test_returns_none_for_invalid_inputs(self, channel, recipient):
        """Test that invalid channel/recipient combinations return None."""
        parser = MockStreamableParser()

        delta_message, tools_streamed = extract_harmony_streaming_delta(
            harmony_parser=parser,
            cur_channel=channel,
            cur_recipient=recipient,
            prev_recipient=None,
            delta_text="some text",
            include_reasoning=True,
        )

        assert delta_message is None
        assert tools_streamed is False
