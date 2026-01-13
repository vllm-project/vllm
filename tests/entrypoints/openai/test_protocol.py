# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from openai_harmony import (
    Message,
)

from vllm.entrypoints.openai.protocol import (
    ResponsesRequest,
    serialize_message,
    serialize_messages,
)


def test_serialize_message() -> None:
    dict_value = {"a": 1, "b": "2"}
    assert serialize_message(dict_value) == dict_value

    msg_value = {
        "role": "assistant",
        "name": None,
        "content": [{"type": "text", "text": "Test 1"}],
        "channel": "analysis",
    }
    msg = Message.from_dict(msg_value)
    assert serialize_message(msg) == msg_value


def test_serialize_messages() -> None:
    assert serialize_messages(None) is None
    assert serialize_messages([]) is None

    dict_value = {"a": 3, "b": "4"}
    msg_value = {
        "role": "assistant",
        "name": None,
        "content": [{"type": "text", "text": "Test 2"}],
        "channel": "analysis",
    }
    msg = Message.from_dict(msg_value)
    assert serialize_messages([msg, dict_value]) == [msg_value, dict_value]


class TestOutputTextToInputTextConversion:
    """Tests for output_text to input_text conversion in message content."""

    def test_converts_output_text_to_input_text(self):
        """Test that output_text content type is converted to input_text."""
        input_data = {
            "model": "test-model",
            "input": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Previous response"}],
                }
            ],
        }

        validated = ResponsesRequest.preprocess_input(input_data)
        processed_item = validated["input"][0]

        assert processed_item["content"][0]["type"] == "input_text"
        assert processed_item["content"][0]["text"] == "Previous response"


class TestReasoningContentNormalization:
    """Tests for reasoning content type normalization."""

    def test_converts_output_text_to_reasoning_text(self):
        """Test that output_text in reasoning content is converted to reasoning_text."""
        input_data = {
            "model": "test-model",
            "input": [
                {
                    "type": "reasoning",
                    "id": "rs_123",
                    "summary": [],
                    "content": [{"type": "output_text", "text": "Thinking..."}],
                }
            ],
        }

        validated = ResponsesRequest.preprocess_input(input_data)
        processed_item = validated["input"][0]

        assert processed_item["content"][0]["type"] == "reasoning_text"

    def test_converts_input_text_to_reasoning_text(self):
        """Test that input_text in reasoning content is converted to reasoning_text."""
        input_data = {
            "model": "test-model",
            "input": [
                {
                    "type": "reasoning",
                    "id": "rs_456",
                    "summary": [],
                    "content": [{"type": "input_text", "text": "Reasoning..."}],
                }
            ],
        }

        validated = ResponsesRequest.preprocess_input(input_data)
        processed_item = validated["input"][0]

        assert processed_item["content"][0]["type"] == "reasoning_text"

    def test_preserves_reasoning_text_type(self):
        """Test that reasoning_text content type is preserved unchanged."""
        input_data = {
            "model": "test-model",
            "input": [
                {
                    "type": "reasoning",
                    "id": "rs_123",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "Thinking..."}],
                }
            ],
        }

        validated = ResponsesRequest.preprocess_input(input_data)
        processed_item = validated["input"][0]

        assert processed_item["content"][0]["type"] == "reasoning_text"
