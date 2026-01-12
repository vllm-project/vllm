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


class TestResponsesRequestInputNormalization:
    """Tests for ResponsesRequest input normalization.

    These tests verify the function_call_parsing validator correctly
    normalizes input items from clients with different serialization styles.
    """

    def test_strips_none_values_from_input_items(self):
        """Test that None values are stripped from input items.

        Clients may include None for optional fields with different
        serialization configs. These should be stripped to avoid validation
        errors and ensure consistent processing.
        """
        input_data = {
            "model": "test-model",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": "Hello",
                    "name": None,  # Should be stripped
                    "status": None,  # Should be stripped
                }
            ],
        }

        validated = ResponsesRequest.function_call_parsing(input_data)
        processed_item = validated["input"][0]

        assert "name" not in processed_item
        assert "status" not in processed_item
        assert processed_item["content"] == "Hello"

    def test_handles_string_content_unchanged(self):
        """Test that string content is preserved without modification.

        Messages with simple string content (not array) should pass through
        unchanged.
        """
        input_data = {
            "model": "test-model",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": "Simple string message",
                }
            ],
        }

        validated = ResponsesRequest.function_call_parsing(input_data)
        processed_item = validated["input"][0]

        assert processed_item["content"] == "Simple string message"
