# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from openai.types.responses import ResponseFunctionToolCall, ResponseReasoningItem
from openai.types.responses.response_output_item import McpCall
from openai_harmony import Author, Message, Role, TextContent

from tests.entrypoints.openai.utils import verify_harmony_messages
from vllm.entrypoints.openai.parser.harmony_utils import (
    auto_drop_analysis_messages,
    get_encoding,
    has_custom_tools,
    parse_chat_input_to_harmony_message,
    parse_chat_output,
    parse_input_to_harmony_message,
    parse_output_message,
)


class TestCommonParseInputToHarmonyMessage:
    """
    Tests for scenarios that are common to both Chat Completion
    parse_chat_input_to_harmony_message and Responsees API
    parse_input_to_harmony_message functions.
    """

    @pytest.fixture(
        params=[parse_chat_input_to_harmony_message, parse_input_to_harmony_message]
    )
    def parse_function(self, request):
        return request.param

    def test_assistant_message_with_tool_calls(self, parse_function):
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

        messages = parse_function(chat_msg)

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

    def test_assistant_message_with_empty_tool_call_arguments(self, parse_function):
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

        messages = parse_function(chat_msg)

        assert len(messages) == 1
        assert messages[0].content[0].text == ""
        assert messages[0].recipient == "functions.get_current_time"

    def test_system_message(self, parse_function):
        """Test parsing system message."""
        chat_msg = {
            "role": "system",
            "content": "You are a helpful assistant",
        }

        messages = parse_function(chat_msg)

        assert len(messages) == 1
        # System messages are converted using Message.from_dict
        # which should preserve the role
        assert messages[0].author.role == Role.SYSTEM

    def test_developer_message(self, parse_function):
        """Test parsing developer message."""
        chat_msg = {
            "role": "developer",
            "content": "Use concise language",
        }

        messages = parse_function(chat_msg)

        assert len(messages) == 1
        assert messages[0].author.role == Role.DEVELOPER

    def test_user_message_with_string_content(self, parse_function):
        """Test parsing user message with string content."""
        chat_msg = {
            "role": "user",
            "content": "What's the weather in San Francisco?",
        }

        messages = parse_function(chat_msg)

        assert len(messages) == 1
        assert messages[0].author.role == Role.USER
        assert messages[0].content[0].text == "What's the weather in San Francisco?"

    def test_user_message_with_array_content(self, parse_function):
        """Test parsing user message with array content."""
        chat_msg = {
            "role": "user",
            "content": [
                {"text": "What's in this image? "},
                {"text": "Please describe it."},
            ],
        }

        messages = parse_function(chat_msg)

        assert len(messages) == 1
        assert messages[0].author.role == Role.USER
        assert len(messages[0].content) == 2
        assert messages[0].content[0].text == "What's in this image? "
        assert messages[0].content[1].text == "Please describe it."

    def test_assistant_message_with_string_content(self, parse_function):
        """Test parsing assistant message with string content (no tool calls)."""
        chat_msg = {
            "role": "assistant",
            "content": "Hello! How can I help you today?",
        }

        messages = parse_function(chat_msg)

        assert len(messages) == 1
        assert messages[0].author.role == Role.ASSISTANT
        assert messages[0].content[0].text == "Hello! How can I help you today?"

    def test_pydantic_model_input(self, parse_function):
        """Test parsing Pydantic model input (has model_dump method)."""

        class MockPydanticModel:
            def model_dump(self, exclude_none=True):
                return {
                    "role": "user",
                    "content": "Test message",
                }

        chat_msg = MockPydanticModel()
        messages = parse_function(chat_msg)

        assert len(messages) == 1
        assert messages[0].author.role == Role.USER
        assert messages[0].content[0].text == "Test message"

    def test_tool_call_with_missing_function_fields(self, parse_function):
        """Test parsing tool call with missing name or arguments."""
        chat_msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {}  # Missing both name and arguments
                }
            ],
        }

        messages = parse_function(chat_msg)

        assert len(messages) == 1
        assert messages[0].recipient == "functions."
        assert messages[0].content[0].text == ""

    def test_array_content_with_missing_text(self, parse_function):
        """Test parsing array content where text field is missing."""
        chat_msg = {
            "role": "user",
            "content": [
                {},  # Missing text field
                {"text": "actual text"},
            ],
        }

        messages = parse_function(chat_msg)

        assert len(messages) == 1
        assert len(messages[0].content) == 2
        assert messages[0].content[0].text == ""
        assert messages[0].content[1].text == "actual text"


class TestParseInputToHarmonyMessage:
    """
    Tests for scenarios that are specific to the Responses API
    parse_input_to_harmony_message function.
    """

    def test_message_with_empty_content(self):
        """Test parsing message with empty string content."""
        chat_msg = {
            "role": "user",
            "content": "",
        }

        messages = parse_input_to_harmony_message(chat_msg)

        assert len(messages) == 1
        assert messages[0].content[0].text == ""

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
        assert messages[0].author.name == "functions.search_results"
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
        assert messages[0].author.role == Role.TOOL
        assert messages[0].author.name == "functions.empty_tool"
        assert messages[0].content[0].text == ""


class TestParseChatInputToHarmonyMessage:
    """
    Tests for scenarios that are specific to the Chat Completion API
    parse_chat_input_to_harmony_message function.
    """

    def test_user_message_with_empty_content(self):
        chat_msg = {
            "role": "user",
            "content": "",
        }

        messages = parse_chat_input_to_harmony_message(chat_msg)

        verify_harmony_messages(
            messages,
            [
                {
                    "role": "user",
                    "content": "",
                },
            ],
        )

    def test_user_message_with_none_content(self):
        chat_msg = {
            "role": "user",
            "content": None,
        }

        messages = parse_chat_input_to_harmony_message(chat_msg)

        verify_harmony_messages(
            messages,
            [
                {
                    "role": "user",
                    "content": "",
                },
            ],
        )

    def test_assistant_message_with_empty_content(self):
        chat_msg = {
            "role": "assistant",
            "content": "",
        }

        messages = parse_chat_input_to_harmony_message(chat_msg)

        assert len(messages) == 0

    def test_assistant_message_with_none_content(self):
        chat_msg = {
            "role": "assistant",
            "content": None,
        }

        messages = parse_chat_input_to_harmony_message(chat_msg)

        assert len(messages) == 0

    def test_assistant_message_with_content_but_empty_reasoning(self):
        chat_msg = {
            "role": "assistant",
            "content": "The answer is 4.",
            "reasoning": "",
        }

        messages = parse_chat_input_to_harmony_message(chat_msg)

        verify_harmony_messages(
            messages,
            [
                {
                    "role": "assistant",
                    "channel": "final",
                    "content": "The answer is 4.",
                },
            ],
        )

    def test_assistant_message_with_reasoning_but_empty_content(self):
        chat_msg = {
            "role": "assistant",
            "reasoning": "I'm thinking about the user's question.",
            "content": "",
        }

        messages = parse_chat_input_to_harmony_message(chat_msg)

        verify_harmony_messages(
            messages,
            [
                {
                    "role": "assistant",
                    "channel": "analysis",
                    "content": "I'm thinking about the user's question.",
                },
            ],
        )

    def test_assistant_message_with_reasoning_but_none_content(self):
        chat_msg = {
            "role": "assistant",
            "reasoning": "I'm thinking about the user's question.",
            "content": None,
        }

        messages = parse_chat_input_to_harmony_message(chat_msg)

        verify_harmony_messages(
            messages,
            [
                {
                    "role": "assistant",
                    "channel": "analysis",
                    "content": "I'm thinking about the user's question.",
                },
            ],
        )

    def test_assistant_message_with_tool_calls_but_no_content(self):
        chat_msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "San Francisco"}',
                    }
                }
            ],
        }

        messages = parse_chat_input_to_harmony_message(chat_msg)

        verify_harmony_messages(
            messages,
            [
                {
                    "role": "assistant",
                    "channel": "commentary",
                    "recipient": "functions.get_weather",
                    "content": '{"location": "San Francisco"}',
                    "content_type": "json",
                },
            ],
        )

    def test_assistant_message_with_tool_calls_and_content(self):
        chat_msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "San Francisco"}',
                    }
                }
            ],
            "content": "I'll call the tool.",
        }

        messages = parse_chat_input_to_harmony_message(chat_msg)

        verify_harmony_messages(
            messages,
            [
                {
                    "role": "assistant",
                    "channel": "commentary",
                    "content": "I'll call the tool.",
                },
                {
                    "role": "assistant",
                    "channel": "commentary",
                    "recipient": "functions.get_weather",
                    "content": '{"location": "San Francisco"}',
                    "content_type": "json",
                },
            ],
        )

    def test_assistant_message_with_tool_calls_and_reasoning(self):
        chat_msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "San Francisco"}',
                    }
                }
            ],
            "reasoning": "I should use the get_weather tool.",
        }

        messages = parse_chat_input_to_harmony_message(chat_msg)

        verify_harmony_messages(
            messages,
            [
                {
                    "role": "assistant",
                    "channel": "analysis",
                    "content": "I should use the get_weather tool.",
                },
                {
                    "role": "assistant",
                    "channel": "commentary",
                    "recipient": "functions.get_weather",
                    "content": '{"location": "San Francisco"}',
                    "content_type": "json",
                },
            ],
        )

    def test_assistant_message_with_tool_calls_and_reasoning_and_content(self):
        chat_msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "San Francisco"}',
                    }
                }
            ],
            "reasoning": "I should use the get_weather tool.",
            "content": "I'll call the tool.",
        }

        messages = parse_chat_input_to_harmony_message(chat_msg)

        verify_harmony_messages(
            messages,
            [
                {
                    "role": "assistant",
                    "channel": "commentary",
                    "content": "I'll call the tool.",
                },
                {
                    "role": "assistant",
                    "channel": "analysis",
                    "content": "I should use the get_weather tool.",
                },
                {
                    "role": "assistant",
                    "channel": "commentary",
                    "recipient": "functions.get_weather",
                    "content": '{"location": "San Francisco"}',
                    "content_type": "json",
                },
            ],
        )

    def test_tool_message_with_string_content(self):
        tool_id_names = {
            "call_123": "get_weather",
        }
        chat_msg = {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": "The weather in San Francisco is sunny, 72°F",
        }

        messages = parse_chat_input_to_harmony_message(
            chat_msg, tool_id_names=tool_id_names
        )

        verify_harmony_messages(
            messages,
            [
                {
                    "role": "tool",
                    "name": "functions.get_weather",
                    "content": "The weather in San Francisco is sunny, 72°F",
                    "channel": "commentary",
                },
            ],
        )

    def test_tool_message_with_array_content(self):
        tool_id_names = {
            "call_123": "search_results",
        }
        chat_msg = {
            "role": "tool",
            "tool_call_id": "call_123",
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

        messages = parse_chat_input_to_harmony_message(
            chat_msg, tool_id_names=tool_id_names
        )

        verify_harmony_messages(
            messages,
            [
                {
                    "role": "tool",
                    "name": "functions.search_results",
                    "content": "Result 1: Result 2: Result 3",
                    "channel": "commentary",
                },
            ],
        )

    def test_tool_message_with_empty_content(self):
        tool_id_names = {
            "call_123": "empty_tool",
        }
        chat_msg = {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": "",
        }

        messages = parse_chat_input_to_harmony_message(
            chat_msg, tool_id_names=tool_id_names
        )

        verify_harmony_messages(
            messages,
            [
                {
                    "role": "tool",
                    "name": "functions.empty_tool",
                    "content": "",
                    "channel": "commentary",
                },
            ],
        )

    def test_tool_message_with_none_content(self):
        tool_id_names = {
            "call_123": "empty_tool",
        }
        chat_msg = {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": None,
        }

        messages = parse_chat_input_to_harmony_message(
            chat_msg, tool_id_names=tool_id_names
        )

        verify_harmony_messages(
            messages,
            [
                {
                    "role": "tool",
                    "name": "functions.empty_tool",
                    "content": "",
                    "channel": "commentary",
                },
            ],
        )


class TestAutoDropAnalysisMessages:
    def test_no_analysis_messages(self) -> None:
        messages = [
            Message.from_role_and_content(
                Role.ASSISTANT, "The answer is 4."
            ).with_channel("final"),
        ]
        cleaned_messages = auto_drop_analysis_messages(messages)
        assert cleaned_messages == messages

    def test_only_analysis_message(self) -> None:
        messages = [
            Message.from_role_and_content(
                Role.ASSISTANT, "I'm thinking about the user's question."
            ).with_channel("analysis"),
        ]
        cleaned_messages = auto_drop_analysis_messages(messages)
        assert cleaned_messages == messages

    def test_multiple_analysis_messages_without_final_message(self) -> None:
        messages = [
            Message.from_role_and_content(
                Role.ASSISTANT, "I'm thinking about the user's question."
            ).with_channel("analysis"),
            Message.from_role_and_content(
                Role.ASSISTANT, "I'm thinking more."
            ).with_channel("analysis"),
            Message.from_role_and_content(
                Role.ASSISTANT, "I'm thinking even more."
            ).with_channel("analysis"),
        ]
        cleaned_messages = auto_drop_analysis_messages(messages)
        assert cleaned_messages == messages

    def test_only_final_message(self) -> None:
        messages = [
            Message.from_role_and_content(
                Role.ASSISTANT, "The answer is 4."
            ).with_channel("final"),
        ]
        cleaned_messages = auto_drop_analysis_messages(messages)
        assert cleaned_messages == messages

    def test_drops_one_analysis_messages_before_final_message(self) -> None:
        messages = [
            Message.from_role_and_content(
                Role.ASSISTANT, "I'm thinking about the user's question."
            ).with_channel("analysis"),
            Message.from_role_and_content(
                Role.ASSISTANT, "The answer is 4."
            ).with_channel("final"),
            Message.from_role_and_content(
                Role.ASSISTANT, "I should think harder."
            ).with_channel("analysis"),
        ]
        cleaned_messages = auto_drop_analysis_messages(messages)
        # Should have dropped the first analysis message
        assert cleaned_messages == messages[1:]

    def test_drops_all_analysis_messages_before_final_message(self) -> None:
        messages = [
            Message.from_role_and_content(
                Role.ASSISTANT, "I'm thinking about the user's question."
            ).with_channel("analysis"),
            Message.from_role_and_content(
                Role.ASSISTANT, "I'm thinking more."
            ).with_channel("analysis"),
            Message.from_role_and_content(
                Role.ASSISTANT, "I'm thinking even more."
            ).with_channel("analysis"),
            Message.from_role_and_content(
                Role.ASSISTANT, "The answer is 4."
            ).with_channel("final"),
            Message.from_role_and_content(
                Role.ASSISTANT, "I should think harder."
            ).with_channel("analysis"),
        ]
        cleaned_messages = auto_drop_analysis_messages(messages)
        # Should have dropped the first 3 analysis messages
        assert cleaned_messages == messages[3:]

    def test_multiple_analysis_messages_with_multiple_final_messages(self) -> None:
        messages = [
            Message.from_role_and_content(
                Role.ASSISTANT, "I'm thinking about the user's question."
            ).with_channel("analysis"),
            Message.from_role_and_content(
                Role.ASSISTANT, "I'm thinking more."
            ).with_channel("analysis"),
            Message.from_role_and_content(
                Role.ASSISTANT, "I'm thinking even more."
            ).with_channel("analysis"),
            Message.from_role_and_content(
                Role.ASSISTANT, "The answer is 4."
            ).with_channel("final"),
            Message.from_role_and_content(
                Role.ASSISTANT, "I should think harder."
            ).with_channel("analysis"),
            Message.from_role_and_content(
                Role.ASSISTANT, "The answer is 5."
            ).with_channel("final"),
        ]
        cleaned_messages = auto_drop_analysis_messages(messages)
        # Should have dropped all those analysis messages
        assert len(cleaned_messages) == 2
        assert cleaned_messages[0].content[0].text == "The answer is 4."
        assert cleaned_messages[1].content[0].text == "The answer is 5."

    def test_drops_non_assistant_analysis_messages(self) -> None:
        messages = [
            Message.from_role_and_content(
                Role.TOOL, "The tool thinks we should think harder."
            ).with_channel("analysis"),
            Message.from_role_and_content(
                Role.ASSISTANT, "The answer is 4."
            ).with_channel("final"),
        ]
        cleaned_messages = auto_drop_analysis_messages(messages)
        # Should have dropped the analysis message
        assert cleaned_messages == messages[1:]


class TestParseChatOutput:
    def test_parse_chat_output_interrupted_first_message(self) -> None:
        harmony_str = "<|channel|>final<|message|>I'm in the middle of answering"
        token_ids = get_encoding().encode(harmony_str, allowed_special="all")
        reasoning, final_content, _ = parse_chat_output(token_ids)
        assert reasoning is None
        assert final_content == "I'm in the middle of answering"

    def test_parse_chat_output_interrupted_reasoning_first_message(self) -> None:
        harmony_str = "<|channel|>analysis<|message|>I'm in the middle of thinking"
        token_ids = get_encoding().encode(harmony_str, allowed_special="all")
        reasoning, final_content, _ = parse_chat_output(token_ids)
        assert reasoning == "I'm in the middle of thinking"
        assert final_content is None

    def test_parse_chat_output_complete_reasoning_interrupted_content(self) -> None:
        harmony_str = (
            "<|channel|>analysis<|message|>I'm thinking.<|end|>"
            "<|start|>assistant<|channel|>final"
            "<|message|>I'm in the middle of answering"
        )
        token_ids = get_encoding().encode(harmony_str, allowed_special="all")
        reasoning, final_content, _ = parse_chat_output(token_ids)
        assert reasoning == "I'm thinking."
        assert final_content == "I'm in the middle of answering"

    def test_parse_chat_output_complete_content(self) -> None:
        harmony_str = "<|channel|>final<|message|>The answer is 4.<|end|>"
        token_ids = get_encoding().encode(harmony_str, allowed_special="all")
        reasoning, final_content, _ = parse_chat_output(token_ids)
        assert reasoning is None
        assert final_content == "The answer is 4."

    def test_parse_chat_output_complete_commentary(self) -> None:
        harmony_str = (
            "<|channel|>commentary<|message|>I need to call some tools.<|end|>"
        )
        token_ids = get_encoding().encode(harmony_str, allowed_special="all")
        reasoning, final_content, _ = parse_chat_output(token_ids)
        assert reasoning is None
        assert final_content == "I need to call some tools."

    def test_parse_chat_output_complete_reasoning(self) -> None:
        harmony_str = (
            "<|channel|>analysis<|message|>I've thought hard about this.<|end|>"
        )
        token_ids = get_encoding().encode(harmony_str, allowed_special="all")
        reasoning, final_content, _ = parse_chat_output(token_ids)
        assert reasoning == "I've thought hard about this."
        assert final_content is None

    def test_parse_chat_output_complete_reasoning_and_content(self) -> None:
        harmony_str = (
            "<|channel|>analysis<|message|>I've thought hard about this.<|end|>"
            "<|start|>assistant<|channel|>final<|message|>The answer is 4.<|end|>"
        )
        token_ids = get_encoding().encode(harmony_str, allowed_special="all")
        reasoning, final_content, _ = parse_chat_output(token_ids)
        assert reasoning == "I've thought hard about this."
        assert final_content == "The answer is 4."


class TestParseOutputMessage:
    """Tests for parse_output_message function."""

    def test_commentary_with_no_recipient_creates_reasoning(self):
        """Test that commentary with recipient=None (preambles) creates reasoning items.

        Per Harmony format, commentary channel can contain preambles to calling
        multiple functions - explanatory text with no recipient.
        """
        message = Message.from_role_and_content(
            Role.ASSISTANT, "I will now search for the weather information."
        )
        message = message.with_channel("commentary")
        # recipient is None by default, representing a preamble

        output_items = parse_output_message(message)

        assert len(output_items) == 1
        assert isinstance(output_items[0], ResponseReasoningItem)
        assert output_items[0].type == "reasoning"
        assert (
            output_items[0].content[0].text
            == "I will now search for the weather information."
        )
        assert output_items[0].content[0].type == "reasoning_text"

    def test_commentary_with_function_recipient_creates_function_call(self):
        """Test commentary with recipient='functions.X' creates function calls."""
        message = Message.from_role_and_content(
            Role.ASSISTANT, '{"location": "San Francisco", "units": "celsius"}'
        )
        message = message.with_channel("commentary")
        message = message.with_recipient("functions.get_weather")

        output_items = parse_output_message(message)

        assert len(output_items) == 1
        assert isinstance(output_items[0], ResponseFunctionToolCall)
        assert output_items[0].type == "function_call"
        assert output_items[0].name == "get_weather"
        assert (
            output_items[0].arguments
            == '{"location": "San Francisco", "units": "celsius"}'
        )
        assert output_items[0].call_id.startswith("call_")
        assert output_items[0].id.startswith("fc_")

    def test_commentary_with_python_recipient_creates_reasoning(self):
        """Test that commentary with recipient='python' creates reasoning items."""
        message = Message.from_role_and_content(
            Role.ASSISTANT, "import numpy as np\nprint(np.array([1, 2, 3]))"
        )
        message = message.with_channel("commentary")
        message = message.with_recipient("python")

        output_items = parse_output_message(message)

        assert len(output_items) == 1
        assert isinstance(output_items[0], ResponseReasoningItem)
        assert output_items[0].type == "reasoning"
        assert (
            output_items[0].content[0].text
            == "import numpy as np\nprint(np.array([1, 2, 3]))"
        )

    def test_commentary_with_browser_recipient_creates_reasoning(self):
        """Test that commentary with recipient='browser' creates reasoning items."""
        message = Message.from_role_and_content(
            Role.ASSISTANT, "Navigating to the specified URL"
        )
        message = message.with_channel("commentary")
        message = message.with_recipient("browser")

        output_items = parse_output_message(message)

        assert len(output_items) == 1
        assert isinstance(output_items[0], ResponseReasoningItem)
        assert output_items[0].type == "reasoning"
        assert output_items[0].content[0].text == "Navigating to the specified URL"

    def test_commentary_with_container_recipient_creates_reasoning(self):
        """Test that commentary with recipient='container' creates reasoning items."""
        message = Message.from_role_and_content(
            Role.ASSISTANT, "Running command in container"
        )
        message = message.with_channel("commentary")
        message = message.with_recipient("container")

        output_items = parse_output_message(message)

        assert len(output_items) == 1
        assert isinstance(output_items[0], ResponseReasoningItem)
        assert output_items[0].type == "reasoning"
        assert output_items[0].content[0].text == "Running command in container"

    def test_commentary_with_empty_content_and_no_recipient(self):
        """Test edge case: empty commentary with recipient=None."""
        message = Message.from_role_and_content(Role.ASSISTANT, "")
        message = message.with_channel("commentary")

        output_items = parse_output_message(message)

        assert len(output_items) == 1
        assert isinstance(output_items[0], ResponseReasoningItem)
        assert output_items[0].content[0].text == ""

    def test_commentary_with_multiple_contents_and_no_recipient(self):
        """Test multiple content items in commentary with no recipient."""
        contents = [
            TextContent(text="Step 1: Analyze the request"),
            TextContent(text="Step 2: Prepare to call functions"),
        ]
        message = Message.from_role_and_contents(Role.ASSISTANT, contents)
        message = message.with_channel("commentary")

        output_items = parse_output_message(message)

        assert len(output_items) == 2
        assert all(isinstance(item, ResponseReasoningItem) for item in output_items)
        assert output_items[0].content[0].text == "Step 1: Analyze the request"
        assert output_items[1].content[0].text == "Step 2: Prepare to call functions"

    def test_commentary_with_multiple_function_calls(self):
        """Test multiple function calls in commentary channel."""
        contents = [
            TextContent(text='{"location": "San Francisco"}'),
            TextContent(text='{"location": "New York"}'),
        ]
        message = Message.from_role_and_contents(Role.ASSISTANT, contents)
        message = message.with_channel("commentary")
        message = message.with_recipient("functions.get_weather")

        output_items = parse_output_message(message)

        assert len(output_items) == 2
        assert all(isinstance(item, ResponseFunctionToolCall) for item in output_items)
        assert output_items[0].name == "get_weather"
        assert output_items[1].name == "get_weather"
        assert output_items[0].arguments == '{"location": "San Francisco"}'
        assert output_items[1].arguments == '{"location": "New York"}'

    def test_commentary_with_unknown_recipient_creates_mcp_call(self):
        """Test that commentary with unknown recipient creates MCP call."""
        message = Message.from_role_and_content(Role.ASSISTANT, '{"arg": "value"}')
        message = message.with_channel("commentary")
        message = message.with_recipient("custom_tool")

        output_items = parse_output_message(message)

        assert len(output_items) == 1
        assert isinstance(output_items[0], McpCall)
        assert output_items[0].type == "mcp_call"
        assert output_items[0].name == "custom_tool"
        assert output_items[0].server_label == "custom_tool"

    def test_analysis_channel_creates_reasoning(self):
        """Test that analysis channel creates reasoning items."""
        message = Message.from_role_and_content(
            Role.ASSISTANT, "Analyzing the problem step by step..."
        )
        message = message.with_channel("analysis")

        output_items = parse_output_message(message)

        assert len(output_items) == 1
        assert isinstance(output_items[0], ResponseReasoningItem)
        assert output_items[0].type == "reasoning"
        assert (
            output_items[0].content[0].text == "Analyzing the problem step by step..."
        )

    def test_non_assistant_message_returns_empty(self):
        """Test that non-assistant messages return empty list.

        Per the implementation, tool messages to assistant (e.g., search results)
        are not included in final output to align with OpenAI behavior.
        """
        message = Message.from_author_and_content(
            Author.new(Role.TOOL, "functions.get_weather"),
            "The weather is sunny, 72°F",
        )

        output_items = parse_output_message(message)

        assert len(output_items) == 0


def test_has_custom_tools() -> None:
    assert not has_custom_tools(set())
    assert not has_custom_tools({"web_search_preview", "code_interpreter", "container"})
    assert has_custom_tools({"others"})
    assert has_custom_tools(
        {"web_search_preview", "code_interpreter", "container", "others"}
    )


def test_parse_mcp_call_basic() -> None:
    """Test that MCP calls are parsed with correct type and server_label."""
    message = Message.from_role_and_content(Role.ASSISTANT, '{"path": "/tmp"}')
    message = message.with_recipient("filesystem")
    message = message.with_channel("commentary")

    output_items = parse_output_message(message)

    assert len(output_items) == 1
    assert isinstance(output_items[0], McpCall)
    assert output_items[0].type == "mcp_call"
    assert output_items[0].name == "filesystem"
    assert output_items[0].server_label == "filesystem"
    assert output_items[0].arguments == '{"path": "/tmp"}'
    assert output_items[0].status == "completed"


def test_parse_mcp_call_dotted_recipient() -> None:
    """Test that dotted recipients extract the tool name correctly."""
    message = Message.from_role_and_content(Role.ASSISTANT, '{"cmd": "ls"}')
    message = message.with_recipient("repo_browser.list")
    message = message.with_channel("commentary")

    output_items = parse_output_message(message)

    assert len(output_items) == 1
    assert isinstance(output_items[0], McpCall)
    assert output_items[0].name == "list"
    assert output_items[0].server_label == "repo_browser"


def test_mcp_vs_function_call() -> None:
    """Test that function calls are not parsed as MCP calls."""
    func_message = Message.from_role_and_content(Role.ASSISTANT, '{"arg": "value"}')
    func_message = func_message.with_recipient("functions.my_tool")
    func_message = func_message.with_channel("commentary")

    func_items = parse_output_message(func_message)

    assert len(func_items) == 1
    assert not isinstance(func_items[0], McpCall)
    assert func_items[0].type == "function_call"


def test_mcp_vs_builtin_tools() -> None:
    """Test that built-in tools (python, container) are not parsed as MCP calls."""
    # Test python (built-in tool) - should be reasoning, not MCP
    python_message = Message.from_role_and_content(Role.ASSISTANT, "print('hello')")
    python_message = python_message.with_recipient("python")
    python_message = python_message.with_channel("commentary")

    python_items = parse_output_message(python_message)

    assert len(python_items) == 1
    assert not isinstance(python_items[0], McpCall)
    assert python_items[0].type == "reasoning"


def test_parse_remaining_state_commentary_channel() -> None:
    """Test parse_remaining_state with commentary channel and various recipients."""
    from unittest.mock import Mock

    from vllm.entrypoints.openai.parser.harmony_utils import parse_remaining_state

    # Test 1: functions.* recipient → should return function tool call
    parser_func = Mock()
    parser_func.current_content = '{"arg": "value"}'
    parser_func.current_role = Role.ASSISTANT
    parser_func.current_channel = "commentary"
    parser_func.current_recipient = "functions.my_tool"

    func_items = parse_remaining_state(parser_func)

    assert len(func_items) == 1
    assert not isinstance(func_items[0], McpCall)
    assert func_items[0].type == "function_call"
    assert func_items[0].name == "my_tool"
    assert func_items[0].status == "in_progress"

    # Test 2: MCP tool (not builtin) → should return MCP call
    parser_mcp = Mock()
    parser_mcp.current_content = '{"path": "/tmp"}'
    parser_mcp.current_role = Role.ASSISTANT
    parser_mcp.current_channel = "commentary"
    parser_mcp.current_recipient = "filesystem"

    mcp_items = parse_remaining_state(parser_mcp)

    assert len(mcp_items) == 1
    assert isinstance(mcp_items[0], McpCall)
    assert mcp_items[0].type == "mcp_call"
    assert mcp_items[0].name == "filesystem"
    assert mcp_items[0].server_label == "filesystem"
    assert mcp_items[0].status == "in_progress"

    # Test 3: Built-in tool (python)
    # should NOT return MCP call, falls through to reasoning
    parser_builtin = Mock()
    parser_builtin.current_content = "print('hello')"
    parser_builtin.current_role = Role.ASSISTANT
    parser_builtin.current_channel = "commentary"
    parser_builtin.current_recipient = "python"

    builtin_items = parse_remaining_state(parser_builtin)

    # Should fall through to reasoning logic
    assert len(builtin_items) == 1
    assert not isinstance(builtin_items[0], McpCall)
    assert builtin_items[0].type == "reasoning"


def test_parse_remaining_state_analysis_channel() -> None:
    """Test parse_remaining_state with analysis channel and various recipients."""
    from unittest.mock import Mock

    from vllm.entrypoints.openai.parser.harmony_utils import parse_remaining_state

    # Test 1: functions.* recipient → should return function tool call
    parser_func = Mock()
    parser_func.current_content = '{"arg": "value"}'
    parser_func.current_role = Role.ASSISTANT
    parser_func.current_channel = "analysis"
    parser_func.current_recipient = "functions.my_tool"

    func_items = parse_remaining_state(parser_func)

    assert len(func_items) == 1
    assert not isinstance(func_items[0], McpCall)
    assert func_items[0].type == "function_call"
    assert func_items[0].name == "my_tool"
    assert func_items[0].status == "in_progress"

    # Test 2: MCP tool (not builtin) → should return MCP call
    parser_mcp = Mock()
    parser_mcp.current_content = '{"query": "test"}'
    parser_mcp.current_role = Role.ASSISTANT
    parser_mcp.current_channel = "analysis"
    parser_mcp.current_recipient = "database"

    mcp_items = parse_remaining_state(parser_mcp)

    assert len(mcp_items) == 1
    assert isinstance(mcp_items[0], McpCall)
    assert mcp_items[0].type == "mcp_call"
    assert mcp_items[0].name == "database"
    assert mcp_items[0].server_label == "database"
    assert mcp_items[0].status == "in_progress"

    # Test 3: Built-in tool (container)
    # should NOT return MCP call, falls through to reasoning
    parser_builtin = Mock()
    parser_builtin.current_content = "docker run"
    parser_builtin.current_role = Role.ASSISTANT
    parser_builtin.current_channel = "analysis"
    parser_builtin.current_recipient = "container"

    builtin_items = parse_remaining_state(parser_builtin)

    # Should fall through to reasoning logic
    assert len(builtin_items) == 1
    assert not isinstance(builtin_items[0], McpCall)
    assert builtin_items[0].type == "reasoning"
