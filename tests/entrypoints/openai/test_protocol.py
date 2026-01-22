# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
from unittest import mock

from openai_harmony import (
    Message,
)

from vllm.entrypoints.openai.responses.protocol import (
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


class TestReasoningIdExtraction:
    """Tests for extracting id from encrypted_content in reasoning items."""

    def test_extracts_id_from_encrypted_content(self):
        """Test that id is extracted from encrypted_content if missing."""
        encrypted = json.dumps({"id": "rs_abc123", "content": []})
        input_data = {
            "model": "test-model",
            "input": [
                {
                    "type": "reasoning",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "Reasoning..."}],
                    "encrypted_content": encrypted,
                }
            ],
        }

        validated = ResponsesRequest.function_call_parsing(input_data)
        processed_item = validated["input"][0]

        assert processed_item["id"] == "rs_abc123"

    def test_preserves_existing_id(self):
        """Test that existing id is not overwritten by encrypted_content."""
        encrypted = json.dumps({"id": "encrypted_id", "content": []})
        input_data = {
            "model": "test-model",
            "input": [
                {
                    "type": "reasoning",
                    "id": "original_id",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "Reasoning..."}],
                    "encrypted_content": encrypted,
                }
            ],
        }

        validated = ResponsesRequest.function_call_parsing(input_data)
        processed_item = validated["input"][0]

        assert processed_item["id"] == "original_id"

    def test_handles_invalid_encrypted_content_gracefully(self):
        """Test that invalid encrypted_content doesn't crash processing."""
        input_data = {
            "model": "test-model",
            "input": [
                {
                    "type": "reasoning",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "Reasoning..."}],
                    "encrypted_content": "not_valid_json_or_encrypted",
                }
            ],
        }

        validated = ResponsesRequest.function_call_parsing(input_data)
        processed_item = validated["input"][0]

        # Should not have extracted an id since content is invalid
        assert "id" not in processed_item

    def test_handles_missing_id_in_encrypted_content(self):
        """Test handling when encrypted_content doesn't have an id field."""
        encrypted = json.dumps({"content": [], "other_field": "value"})
        input_data = {
            "model": "test-model",
            "input": [
                {
                    "type": "reasoning",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "Reasoning..."}],
                    "encrypted_content": encrypted,
                }
            ],
        }

        validated = ResponsesRequest.function_call_parsing(input_data)
        processed_item = validated["input"][0]

        # Should not have extracted an id since it's not in the content
        assert "id" not in processed_item


def _has_cryptography() -> bool:
    """Check if cryptography package is available."""
    try:
        import cryptography  # noqa: F401

        return True
    except ImportError:
        return False


class TestReasoningIdExtractionWithEncryption:
    """Tests for extracting id from truly encrypted content."""

    def setup_method(self):
        """Reset encryption state before each test."""
        from vllm.entrypoints.openai.reasoning_encryption import (
            _reset_encryption_state,
        )

        _reset_encryption_state()

    def teardown_method(self):
        """Reset encryption state after each test."""
        from vllm.entrypoints.openai.reasoning_encryption import (
            _reset_encryption_state,
        )

        _reset_encryption_state()
        # Clean up env var
        os.environ.pop("VLLM_ENCRYPT_REASONING_CONTENT", None)

    def test_extracts_id_from_fernet_encrypted_content(self):
        """Test that id is extracted from Fernet-encrypted content."""
        if not _has_cryptography():
            return  # Skip if cryptography not available

        with mock.patch.dict(
            os.environ, {"VLLM_ENCRYPT_REASONING_CONTENT": "1"}, clear=False
        ):
            from vllm.entrypoints.openai.reasoning_encryption import (
                _reset_encryption_state,
                encrypt_reasoning_content,
            )

            _reset_encryption_state()

            # Create encrypted content
            content = [{"type": "reasoning_text", "text": "Secret reasoning"}]
            encrypted = encrypt_reasoning_content("rs_encrypted_123", content)

            input_data = {
                "model": "test-model",
                "input": [
                    {
                        "type": "reasoning",
                        "summary": [],
                        "content": content,
                        "encrypted_content": encrypted,
                    }
                ],
            }

            validated = ResponsesRequest.function_call_parsing(input_data)
            processed_item = validated["input"][0]

            assert processed_item["id"] == "rs_encrypted_123"
