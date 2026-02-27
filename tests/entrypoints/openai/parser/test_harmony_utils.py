# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from openai_harmony import Message, Role

from tests.entrypoints.openai.utils import verify_harmony_messages
from vllm.entrypoints.openai.parser.harmony_utils import (
    auto_drop_analysis_messages,
    get_encoding,
    get_system_message,
    has_custom_tools,
    parse_chat_input_to_harmony_message,
    parse_chat_output,
)
from vllm.entrypoints.openai.responses.harmony import (
    response_previous_input_to_harmony,
)


class TestCommonParseInputToHarmonyMessage:
    """
    Tests for scenarios that are common to both Chat Completion
    parse_chat_input_to_harmony_message and Responses API
    response_previous_input_to_harmony functions.
    """

    @pytest.fixture(
        params=[parse_chat_input_to_harmony_message, response_previous_input_to_harmony]
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

    def test_parse_chat_output_commentary_with_recipient_excluded(self) -> None:
        """Commentary with a recipient (tool call) should not appear in
        final_content — those are handled separately by the tool parser.

        The first message is a preamble (visible), the second is a tool
        call (excluded). Only the preamble should appear in final_content.
        """
        harmony_str = (
            "<|channel|>commentary"
            "<|message|>Let me check the weather.<|end|>"
            "<|start|>assistant to=functions.get_weather"
            "<|channel|>commentary"
            '<|message|>{"location": "SF"}<|end|>'
        )
        token_ids = get_encoding().encode(harmony_str, allowed_special="all")
        reasoning, final_content, _ = parse_chat_output(token_ids)
        assert reasoning is None
        assert final_content == "Let me check the weather."

    def test_parse_chat_output_interrupted_preamble(self) -> None:
        """Partial/interrupted preamble (commentary without recipient) should
        appear in final_content, not reasoning."""
        harmony_str = "<|channel|>commentary<|message|>I'll search for that"
        token_ids = get_encoding().encode(harmony_str, allowed_special="all")
        reasoning, final_content, _ = parse_chat_output(token_ids)
        assert reasoning is None
        assert final_content == "I'll search for that"

    def test_parse_chat_output_preamble_then_final(self) -> None:
        """Preamble followed by a final message should both appear in
        final_content, joined by newline."""
        harmony_str = (
            "<|channel|>commentary"
            "<|message|>Let me look that up.<|end|>"
            "<|start|>assistant<|channel|>final"
            "<|message|>The answer is 42.<|end|>"
        )
        token_ids = get_encoding().encode(harmony_str, allowed_special="all")
        reasoning, final_content, _ = parse_chat_output(token_ids)
        assert reasoning is None
        assert final_content == "Let me look that up.\nThe answer is 42."


def test_has_custom_tools() -> None:
    assert not has_custom_tools(set())
    assert not has_custom_tools({"web_search_preview", "code_interpreter", "container"})
    assert has_custom_tools({"others"})
    assert has_custom_tools(
        {"web_search_preview", "code_interpreter", "container", "others"}
    )


class TestGetSystemMessage:
    """Tests for get_system_message channel configuration."""

    def test_commentary_channel_present_without_custom_tools(self) -> None:
        """Commentary channel must be valid even without custom tools."""
        sys_msg = get_system_message(with_custom_tools=False)
        valid_channels = sys_msg.content[0].channel_config.valid_channels
        assert "commentary" in valid_channels

    def test_commentary_channel_present_with_custom_tools(self) -> None:
        """Commentary channel present when custom tools are enabled."""
        sys_msg = get_system_message(with_custom_tools=True)
        valid_channels = sys_msg.content[0].channel_config.valid_channels
        assert "commentary" in valid_channels

    def test_all_standard_channels_present(self) -> None:
        """All three standard Harmony channels should always be valid."""
        for with_tools in (True, False):
            sys_msg = get_system_message(with_custom_tools=with_tools)
            valid_channels = sys_msg.content[0].channel_config.valid_channels
            for channel in ("analysis", "commentary", "final"):
                assert channel in valid_channels, (
                    f"{channel} missing when with_custom_tools={with_tools}"
                )
