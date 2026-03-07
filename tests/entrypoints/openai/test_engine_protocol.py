# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for vllm.entrypoints.openai.engine.protocol module."""

from vllm.entrypoints.openai.engine.protocol import DeltaMessage


class TestDeltaMessageReasoningCompat:
    """Test backward compatibility for reasoning_content field in DeltaMessage.

    The reasoning_content field was deprecated in favor of reasoning.
    These tests ensure both fields remain in sync for backward compatibility.
    See: https://github.com/vllm-project/vllm/issues/27755
    """

    def test_reasoning_syncs_to_reasoning_content(self):
        """When reasoning is set, reasoning_content should be auto-populated."""
        msg = DeltaMessage(reasoning="thinking steps")
        assert msg.reasoning == "thinking steps"
        assert msg.reasoning_content == "thinking steps"

    def test_reasoning_content_syncs_to_reasoning(self):
        """When reasoning_content is set, reasoning should be auto-populated."""
        msg = DeltaMessage(reasoning_content="thinking steps")
        assert msg.reasoning_content == "thinking steps"
        assert msg.reasoning == "thinking steps"

    def test_both_fields_set_reasoning_takes_precedence(self):
        """When both fields are set, reasoning takes precedence as canonical."""
        msg = DeltaMessage(reasoning="a", reasoning_content="b")
        assert msg.reasoning == "a"
        assert msg.reasoning_content == "a"  # Synced from reasoning

    def test_neither_field_set(self):
        """When neither field is set, both should remain None."""
        msg = DeltaMessage(content="hello")
        assert msg.reasoning is None
        assert msg.reasoning_content is None

    def test_empty_string_reasoning(self):
        """Empty string should sync like any other value."""
        msg = DeltaMessage(reasoning="")
        assert msg.reasoning == ""
        assert msg.reasoning_content == ""

    def test_serialization_includes_both_fields(self):
        """Both fields should appear in serialized output when set."""
        msg = DeltaMessage(reasoning="thinking")
        data = msg.model_dump(exclude_none=True)
        assert data["reasoning"] == "thinking"
        assert data["reasoning_content"] == "thinking"

    def test_streaming_use_case(self):
        """Simulate streaming chunks with reasoning content."""
        chunks = [
            DeltaMessage(role="assistant"),
            DeltaMessage(reasoning="Let me think..."),
            DeltaMessage(reasoning=" about this"),
            DeltaMessage(content="The answer is 42"),
        ]

        # All chunks should have synced fields
        assert chunks[0].reasoning is None
        assert chunks[0].reasoning_content is None

        assert chunks[1].reasoning == "Let me think..."
        assert chunks[1].reasoning_content == "Let me think..."

        assert chunks[2].reasoning == " about this"
        assert chunks[2].reasoning_content == " about this"

        assert chunks[3].reasoning is None
        assert chunks[3].reasoning_content is None
        assert chunks[3].content == "The answer is 42"
