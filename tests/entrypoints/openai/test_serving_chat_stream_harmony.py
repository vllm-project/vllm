# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for harmony streaming delta extraction.
"""

from dataclasses import dataclass, field
from unittest.mock import patch

import pytest

from vllm.entrypoints.openai.chat_completion.stream_harmony import (
    HarmonyStreamingState,
    extract_harmony_streaming_delta,
)


@dataclass
class MockMessage:
    """Mock message object for testing."""

    channel: str | None = None
    recipient: str | None = None
    content: str = ""


@dataclass
class MockStreamableParser:
    """Mock StreamableParser for testing without openai_harmony dependency."""

    messages: list[MockMessage] = field(default_factory=list)
    current_channel: str | None = None
    current_recipient: str | None = None
    current_content: str | None = None


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
        parser.current_channel = "final"
        parser.current_content = delta_text
        stream_state = HarmonyStreamingState()

        delta_message, tools_streamed = extract_harmony_streaming_delta(
            harmony_parser=parser,
            stream_state=stream_state,
            include_reasoning=False,
        )

        if expected_content:
            assert delta_message is not None
            assert delta_message.content == expected_content
        else:
            assert delta_message is None
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
        parser.current_channel = "analysis"
        parser.current_content = text
        stream_state = HarmonyStreamingState()

        delta_message, tools_streamed = extract_harmony_streaming_delta(
            harmony_parser=parser,
            stream_state=stream_state,
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
        parser.current_channel = channel
        parser.current_recipient = "functions.get_weather"
        parser.current_content = ""
        stream_state = HarmonyStreamingState()

        delta_message, tools_streamed = extract_harmony_streaming_delta(
            harmony_parser=parser,
            stream_state=stream_state,
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
        parser.current_channel = channel
        parser.current_recipient = "functions.get_weather"
        parser.current_content = args_text
        stream_state = HarmonyStreamingState(
            prev_current_signature=(channel, "functions.get_weather"),
            prev_current_emitted_len=0,
            prev_current_tool_index=0,
            prev_current_tool_header_emitted=True,
        )

        delta_message, tools_streamed = extract_harmony_streaming_delta(
            harmony_parser=parser,
            stream_state=stream_state,
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
        parser.current_channel = channel
        parser.current_recipient = "functions.get_weather"
        parser.current_content = ""
        stream_state = HarmonyStreamingState(
            prev_current_signature=(channel, "functions.get_weather"),
            prev_current_emitted_len=0,
            prev_current_tool_index=0,
            prev_current_tool_header_emitted=True,
        )

        delta_message, tools_streamed = extract_harmony_streaming_delta(
            harmony_parser=parser,
            stream_state=stream_state,
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
        parser.current_channel = "commentary"
        parser.current_recipient = "functions.tool2"
        parser.current_content = "args"
        stream_state = HarmonyStreamingState(
            emitted_message_count=3,
            next_tool_call_index=1,
        )

        delta_message, _ = extract_harmony_streaming_delta(
            harmony_parser=parser,
            stream_state=stream_state,
            include_reasoning=False,
        )

        assert delta_message.tool_calls[0].index == 1

    def test_returns_preambles_as_content(self):
        """Test that commentary with no recipient (preamble) is user content."""
        parser = MockStreamableParser()
        delta_text = "some text"
        parser.current_channel = "commentary"
        parser.current_content = delta_text
        stream_state = HarmonyStreamingState()

        delta_message, tools_streamed = extract_harmony_streaming_delta(
            harmony_parser=parser,
            stream_state=stream_state,
            include_reasoning=True,
        )

        assert delta_message.content == delta_text
        assert tools_streamed is False

    @pytest.mark.parametrize(
        "channel,recipient",
        [
            (None, None),
            ("unknown_channel", None),
            ("commentary", "browser.search"),
        ],
    )
    def test_returns_none_for_invalid_inputs(self, channel, recipient):
        """Test that invalid channel/recipient combinations return None."""
        parser = MockStreamableParser()
        parser.current_channel = channel
        parser.current_recipient = recipient
        parser.current_content = "some text"
        stream_state = HarmonyStreamingState()

        delta_message, tools_streamed = extract_harmony_streaming_delta(
            harmony_parser=parser,
            stream_state=stream_state,
            include_reasoning=True,
        )

        if channel == "commentary" and recipient is not None:
            assert delta_message is not None
            assert delta_message.content == "some text"
        else:
            assert delta_message is None
        assert tools_streamed is False

    def test_consecutive_token_grouping(self):
        """
        Test that consecutive tokens with the same channel/recipient
        are merged into a single processing group.
        """
        parser = MockStreamableParser()
        parser.current_channel = "final"
        parser.current_content = "Hello, World"
        stream_state = HarmonyStreamingState()

        delta_message, _ = extract_harmony_streaming_delta(
            harmony_parser=parser,
            stream_state=stream_state,
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
        parser.messages = [
            MockMessage(channel="analysis", content="Reasoning about query..."),
            MockMessage(
                channel="commentary",
                recipient="functions.search",
                content='{"query": "vllm"}',
            ),
        ]
        parser.current_channel = "final"
        parser.current_content = "."
        stream_state = HarmonyStreamingState()

        delta_message, tools_streamed = extract_harmony_streaming_delta(
            harmony_parser=parser,
            stream_state=stream_state,
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
        mock_make_id.side_effect = ["id_prev", "id_a", "id_b", "id_c"]

        messages = [
            MockMessage(channel="commentary", recipient="functions.previous_tool"),
            MockMessage(
                channel="commentary",
                recipient="functions.tool_a",
                content='{"key_a": "val_a"}',
            ),
            MockMessage(channel="final", content="Thinking..."),
            MockMessage(
                channel="commentary",
                recipient="functions.tool_b",
                content='{"key_b": "val_b"}',
            ),
            MockMessage(channel="final", content=" Thinking again..."),
            MockMessage(
                channel="commentary",
                recipient="functions.tool_c",
                content='{"key_c": "val_c"}',
            ),
        ]
        parser = MockStreamableParser(messages=messages)
        stream_state = HarmonyStreamingState()

        delta_message, _ = extract_harmony_streaming_delta(
            harmony_parser=parser,
            stream_state=stream_state,
            include_reasoning=False,
        )

        assert delta_message is not None

        tool_a_deltas = [t for t in delta_message.tool_calls if t.index == 1]
        assert len(tool_a_deltas) > 0
        tool_a_args = next(t for t in tool_a_deltas if t.id is None)
        assert tool_a_args.function.arguments == '{"key_a": "val_a"}'

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

    def test_carryover_avoids_reemitting_streamed_content(self):
        """
        Simulate two consecutive calls. Call 1 streams partial content
        from the in-progress message. Call 2 sees that message completed
        and should only emit the new portion.
        """
        parser = MockStreamableParser()
        parser.current_channel = "final"
        parser.current_content = "Hello"
        stream_state = HarmonyStreamingState()

        delta_message, _ = extract_harmony_streaming_delta(
            harmony_parser=parser,
            stream_state=stream_state,
            include_reasoning=False,
        )

        assert delta_message is not None
        assert delta_message.content == "Hello"

        parser.messages = [
            MockMessage(channel="final", content="Hello, World"),
        ]
        parser.current_channel = None
        parser.current_content = None

        delta_message, _ = extract_harmony_streaming_delta(
            harmony_parser=parser,
            stream_state=stream_state,
            include_reasoning=False,
        )

        assert delta_message is not None
        assert delta_message.content == ", World"

    @patch("vllm.entrypoints.openai.chat_completion.stream_harmony.make_tool_call_id")
    def test_carryover_tool_call_avoids_reemitting_header(self, mock_make_id):
        """
        Simulate two consecutive calls for a tool call. Call 1 streams
        the header + partial args. Call 2 sees the message completed and
        should only emit the remaining args without a duplicate header.
        """
        mock_make_id.return_value = "call_carry"
        parser = MockStreamableParser()
        parser.current_channel = "commentary"
        parser.current_recipient = "functions.search"
        parser.current_content = '{"q":'
        stream_state = HarmonyStreamingState()

        delta_message, tools_streamed = extract_harmony_streaming_delta(
            harmony_parser=parser,
            stream_state=stream_state,
            include_reasoning=False,
        )

        assert tools_streamed is True
        assert len(delta_message.tool_calls) == 2
        assert delta_message.tool_calls[0].id == "call_carry"
        assert delta_message.tool_calls[1].function.arguments == '{"q":'

        parser.messages = [
            MockMessage(
                channel="commentary",
                recipient="functions.search",
                content='{"q": "vllm"}',
            ),
        ]
        parser.current_channel = None
        parser.current_content = None

        delta_message, tools_streamed = extract_harmony_streaming_delta(
            harmony_parser=parser,
            stream_state=stream_state,
            include_reasoning=False,
        )

        assert tools_streamed is True
        assert len(delta_message.tool_calls) == 1
        assert delta_message.tool_calls[0].id is None
        assert delta_message.tool_calls[0].function.arguments == ' "vllm"}'
