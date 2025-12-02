# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from openai_harmony import Role

from vllm.entrypoints.harmony_utils import (
    has_custom_tools,
    parse_input_to_harmony_message,
)


class TestParseInputToHarmonyMessage:
    """Tests for parse_input_to_harmony_message function."""

    def test_assistant_message_with_tool_calls(self):
        """Test parsing assistant message with tool calls."""
        chat_msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "San Francisco"}',
                    }
                },
                {
                    "function": {
                        "name": "search_web",
                        "arguments": '{"query": "latest news"}',
                    }
                },
            ],
        }

        messages = parse_input_to_harmony_message(chat_msg)

        assert len(messages) == 2

        # First tool call
        assert messages[0].author.role == Role.ASSISTANT
        assert messages[0].content[0].text == '{"location": "San Francisco"}'
        assert messages[0].channel == "commentary"
        assert messages[0].recipient == "functions.get_weather"
        assert messages[0].content_type == "json"

        # Second tool call
        assert messages[1].author.role == Role.ASSISTANT
        assert messages[1].content[0].text == '{"query": "latest news"}'
        assert messages[1].channel == "commentary"
        assert messages[1].recipient == "functions.search_web"
        assert messages[1].content_type == "json"

    def test_assistant_message_with_empty_tool_call_arguments(self):
        """Test parsing assistant message with tool call having None arguments."""
        chat_msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_current_time",
                        "arguments": None,
                    }
                }
            ],
        }

        messages = parse_input_to_harmony_message(chat_msg)

        assert len(messages) == 1
        assert messages[0].content[0].text == ""
        assert messages[0].recipient == "functions.get_current_time"

    def test_tool_message_with_string_content(self):
        """Test parsing tool message with string content."""
        chat_msg = {
            "role": "tool",
            "name": "get_weather",
            "content": "The weather in San Francisco is sunny, 72°F",
        }

        messages = parse_input_to_harmony_message(chat_msg)

        assert len(messages) == 1
        assert messages[0].author.role == Role.TOOL
        assert messages[0].author.name == "functions.get_weather"
        assert (
            messages[0].content[0].text == "The weather in San Francisco is sunny, 72°F"
        )
        assert messages[0].channel == "commentary"

    def test_tool_message_with_array_content(self):
        """Test parsing tool message with array content."""
        chat_msg = {
            "role": "tool",
            "name": "search_results",
            "content": [
                {"type": "text", "text": "Result 1: "},
                {"type": "text", "text": "Result 2: "},
                {
                    "type": "image",
                    "url": "http://example.com/img.png",
                },  # Should be ignored
                {"type": "text", "text": "Result 3"},
            ],
        }

        messages = parse_input_to_harmony_message(chat_msg)

        assert len(messages) == 1
        assert messages[0].author.role == Role.TOOL
        assert messages[0].content[0].text == "Result 1: Result 2: Result 3"

    def test_tool_message_with_empty_content(self):
        """Test parsing tool message with None content."""
        chat_msg = {
            "role": "tool",
            "name": "empty_tool",
            "content": None,
        }

        messages = parse_input_to_harmony_message(chat_msg)

        assert len(messages) == 1
        assert messages[0].content[0].text == ""

    def test_system_message(self):
        """Test parsing system message."""
        chat_msg = {
            "role": "system",
            "content": "You are a helpful assistant",
        }

        messages = parse_input_to_harmony_message(chat_msg)

        assert len(messages) == 1
        # System messages are converted using Message.from_dict
        # which should preserve the role
        assert messages[0].author.role == Role.SYSTEM

    def test_developer_message(self):
        """Test parsing developer message."""
        chat_msg = {
            "role": "developer",
            "content": "Use concise language",
        }

        messages = parse_input_to_harmony_message(chat_msg)

        assert len(messages) == 1
        assert messages[0].author.role == Role.DEVELOPER

    def test_user_message_with_string_content(self):
        """Test parsing user message with string content."""
        chat_msg = {
            "role": "user",
            "content": "What's the weather in San Francisco?",
        }

        messages = parse_input_to_harmony_message(chat_msg)

        assert len(messages) == 1
        assert messages[0].author.role == Role.USER
        assert messages[0].content[0].text == "What's the weather in San Francisco?"

    def test_user_message_with_array_content(self):
        """Test parsing user message with array content."""
        chat_msg = {
            "role": "user",
            "content": [
                {"text": "What's in this image? "},
                {"text": "Please describe it."},
            ],
        }

        messages = parse_input_to_harmony_message(chat_msg)

        assert len(messages) == 1
        assert messages[0].author.role == Role.USER
        assert len(messages[0].content) == 2
        assert messages[0].content[0].text == "What's in this image? "
        assert messages[0].content[1].text == "Please describe it."

    def test_assistant_message_with_string_content(self):
        """Test parsing assistant message with string content (no tool calls)."""
        chat_msg = {
            "role": "assistant",
            "content": "Hello! How can I help you today?",
        }

        messages = parse_input_to_harmony_message(chat_msg)

        assert len(messages) == 1
        assert messages[0].author.role == Role.ASSISTANT
        assert messages[0].content[0].text == "Hello! How can I help you today?"

    def test_pydantic_model_input(self):
        """Test parsing Pydantic model input (has model_dump method)."""

        class MockPydanticModel:
            def model_dump(self, exclude_none=True):
                return {
                    "role": "user",
                    "content": "Test message",
                }

        chat_msg = MockPydanticModel()
        messages = parse_input_to_harmony_message(chat_msg)

        assert len(messages) == 1
        assert messages[0].author.role == Role.USER
        assert messages[0].content[0].text == "Test message"

    def test_message_with_empty_content(self):
        """Test parsing message with empty string content."""
        chat_msg = {
            "role": "user",
            "content": "",
        }

        messages = parse_input_to_harmony_message(chat_msg)

        assert len(messages) == 1
        assert messages[0].content[0].text == ""

    def test_tool_call_with_missing_function_fields(self):
        """Test parsing tool call with missing name or arguments."""
        chat_msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {}  # Missing both name and arguments
                }
            ],
        }

        messages = parse_input_to_harmony_message(chat_msg)

        assert len(messages) == 1
        assert messages[0].recipient == "functions."
        assert messages[0].content[0].text == ""

    def test_array_content_with_missing_text(self):
        """Test parsing array content where text field is missing."""
        chat_msg = {
            "role": "user",
            "content": [
                {},  # Missing text field
                {"text": "actual text"},
            ],
        }

        messages = parse_input_to_harmony_message(chat_msg)

        assert len(messages) == 1
        assert len(messages[0].content) == 2
        assert messages[0].content[0].text == ""
        assert messages[0].content[1].text == "actual text"


def test_has_custom_tools() -> None:
    assert not has_custom_tools(set())
    assert not has_custom_tools({"web_search_preview", "code_interpreter", "container"})
    assert has_custom_tools({"others"})
    assert has_custom_tools(
        {"web_search_preview", "code_interpreter", "container", "others"}
    )
