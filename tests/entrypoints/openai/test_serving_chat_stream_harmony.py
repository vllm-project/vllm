# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for harmony streaming delta extraction.
"""

from dataclasses import dataclass, field
from unittest.mock import patch

import pytest

from vllm.entrypoints.openai.chat_completion.stream_harmony import (
    TokenState,
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

        # Updated to use TokenState list
        token_states = [TokenState(channel="final", recipient=None, text=delta_text)]

        delta_message, tools_streamed = extract_harmony_streaming_delta(
            harmony_parser=parser,
            token_states=token_states,
            prev_recipient=None,
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
        text = "Let me think..."
        token_states = [TokenState(channel="analysis", recipient=None, text=text)]

        delta_message, tools_streamed = extract_harmony_streaming_delta(
            harmony_parser=parser,
            token_states=token_states,
            prev_recipient=None,
            include_reasoning=include_reasoning,
        )

        if expected_has_message:
            assert delta_message is not None
            assert delta_message.reasoning == text
        else:
            assert delta_message is None
        assert tools_streamed is False

    @pytest.mark.parametrize("channel", ["commentary", "analysis"])
    @patch("vllm.entrypoints.openai.chat_completion.stream_harmony.make_tool_call_id")
    def test_new_tool_call(self, mock_make_tool_call_id, channel):
        """Test new tool call creation when recipient changes."""
        mock_make_tool_call_id.return_value = "call_test123"
        parser = MockStreamableParser()

        token_states = [
            TokenState(channel=channel, recipient="functions.get_weather", text="")
        ]

        delta_message, tools_streamed = extract_harmony_streaming_delta(
            harmony_parser=parser,
            token_states=token_states,
            prev_recipient=None,
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
        args_text = '{"location": "Paris"}'

        token_states = [
            TokenState(
                channel=channel, recipient="functions.get_weather", text=args_text
            )
        ]

        delta_message, tools_streamed = extract_harmony_streaming_delta(
            harmony_parser=parser,
            token_states=token_states,
            prev_recipient="functions.get_weather",
            include_reasoning=False,
        )

        assert delta_message is not None
        tool_call = delta_message.tool_calls[0]
        assert tool_call.id is None
        assert tool_call.function.arguments == args_text
        assert tool_call.index == 0
        assert tools_streamed is True

    @pytest.mark.parametrize("channel", ["commentary", "analysis"])
    def test_tool_call_empty_arguments_returns_none(self, channel):
        """Test empty delta_text with same recipient returns None."""
        parser = MockStreamableParser()

        token_states = [
            TokenState(channel=channel, recipient="functions.get_weather", text="")
        ]

        delta_message, tools_streamed = extract_harmony_streaming_delta(
            harmony_parser=parser,
            token_states=token_states,
            prev_recipient="functions.get_weather",
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

        token_states = [
            TokenState(channel="commentary", recipient="functions.tool2", text="args")
        ]

        delta_message, _ = extract_harmony_streaming_delta(
            harmony_parser=parser,
            token_states=token_states,
            prev_recipient="functions.tool2",
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
        """Test that invalid tool recipient on commentary is treated as content."""
        parser = MockStreamableParser()
        delta_text = "some text"

        token_states = [
            TokenState(channel=channel, recipient=recipient, text=delta_text)
        ]

        delta_message, tools_streamed = extract_harmony_streaming_delta(
            harmony_parser=parser,
            token_states=token_states,
            prev_recipient=None,
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

        token_states = [
            TokenState(channel=channel, recipient=recipient, text="some text")
        ]

        delta_message, tools_streamed = extract_harmony_streaming_delta(
            harmony_parser=parser,
            token_states=token_states,
            prev_recipient=None,
            include_reasoning=True,
        )

        assert delta_message is None
        assert tools_streamed is False

    def test_consecutive_token_grouping(self):
        """
        Test that consecutive tokens with the same channel/recipient
        are merged into a single processing group.
        """
        parser = MockStreamableParser()
        token_states = [
            TokenState("final", None, "H"),
            TokenState("final", None, "el"),
            TokenState("final", None, "lo"),
            TokenState("final", None, ","),
            TokenState("final", None, " World"),
        ]

        delta_message, _ = extract_harmony_streaming_delta(
            harmony_parser=parser,
            token_states=token_states,
            prev_recipient=None,
            include_reasoning=False,
        )

        assert delta_message is not None
        assert delta_message.content == "Hello, World"

    @patch("vllm.entrypoints.openai.chat_completion.stream_harmony.make_tool_call_id")
    def test_complex_batch_permutation(self, mock_make_id):
        """
        Test a complex permutation: Reasoning -> Tool Call -> Content.
        This verifies that multiple distinct actions in one batch
        are all captured in the single DeltaMessage.
        """
        mock_make_id.return_value = "call_batch_test"
        parser = MockStreamableParser()

        token_states = [
            # 1. Reasoning
            TokenState("analysis", None, "Reasoning about query..."),
            # 2. Tool Calling
            TokenState("commentary", "functions.search", '{"query":'),
            TokenState("commentary", "functions.search", ' "vllm"}'),
            # 3. Final Content
            TokenState("final", None, "."),
        ]

        delta_message, tools_streamed = extract_harmony_streaming_delta(
            harmony_parser=parser,
            token_states=token_states,
            prev_recipient=None,
            include_reasoning=True,
        )

        assert delta_message is not None

        assert delta_message.reasoning == "Reasoning about query..."

        # We expect 2 objects for 1 logical tool call:
        # 1. The definition (id, name, type)
        # 2. The arguments payload
        assert len(delta_message.tool_calls) == 2

        header = delta_message.tool_calls[0]
        payload = delta_message.tool_calls[1]

        assert header.function.name == "search"
        assert header.id == "call_batch_test"
        assert header.index == 0

        assert payload.index == 0
        assert payload.function.arguments == '{"query": "vllm"}'

        assert delta_message.content == "."
        assert tools_streamed is True

    @patch("vllm.entrypoints.openai.chat_completion.stream_harmony.make_tool_call_id")
    def test_tool_call_index_consistency_with_ongoing_call(self, mock_make_id):
        """
        Test that an ongoing tool call continuation and subsequent new calls
        maintain correct indexing when interleaved with content.
        """
        mock_make_id.side_effect = ["id_b", "id_c"]

        messages = [
            MockMessage(channel="commentary", recipient="functions.previous_tool")
        ]
        parser = MockStreamableParser(messages=messages)

        token_states = [
            TokenState("commentary", "functions.tool_a", '{"key_a": "val_a"}'),
            TokenState("final", None, "Thinking..."),
            TokenState("commentary", "functions.tool_b", '{"key_b": "val_b"}'),
            TokenState("final", None, " Thinking again..."),
            TokenState("commentary", "functions.tool_c", '{"key_c": "val_c"}'),
        ]

        delta_message, _ = extract_harmony_streaming_delta(
            harmony_parser=parser,
            token_states=token_states,
            prev_recipient="functions.tool_a",
            include_reasoning=False,
        )

        assert delta_message is not None

        tool_a_deltas = [t for t in delta_message.tool_calls if t.index == 1]
        assert len(tool_a_deltas) > 0
        assert tool_a_deltas[0].id is None
        assert tool_a_deltas[0].function.arguments == '{"key_a": "val_a"}'

        tool_b_header = next(t for t in delta_message.tool_calls if t.id == "id_b")
        assert tool_b_header.index == 2
        tool_b_args = next(
            t for t in delta_message.tool_calls if t.index == 2 and t.id is None
        )
        assert tool_b_args.function.arguments == '{"key_b": "val_b"}'

        tool_c_start = next(t for t in delta_message.tool_calls if t.id == "id_c")
        assert tool_c_start.index == 3
        tool_c_args = next(
            t for t in delta_message.tool_calls if t.index == 3 and t.id is None
        )
        assert tool_c_args.function.arguments == '{"key_c": "val_c"}'

        assert delta_message.content == "Thinking... Thinking again..."
