# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest
from openai_harmony import Author, Message, Role

from tests.entrypoints.openai.utils import verify_harmony_messages
from vllm.entrypoints.openai.parser.harmony_utils import (
    auto_drop_analysis_messages,
    get_developer_message,
    get_encoding,
    get_streamable_parser_for_assistant,
    get_system_message,
    has_custom_tools,
    inject_response_formats,
    parse_chat_input_to_harmony_message,
    parse_chat_output,
    render_for_completion,
    sanitize_harmony_name,
    sanitize_harmony_recipient,
)
from vllm.entrypoints.openai.responses.harmony import (
    response_input_to_harmony,
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


class TestRenderForCompletion:
    def test_preserves_analysis(self):
        """render_for_completion must not strip analysis messages —
        vLLM handles that via auto_drop_analysis_messages()."""
        messages = [
            get_system_message(),
            Message.from_role_and_content(Role.USER, "What is 2+2?"),
            Message.from_role_and_content(Role.ASSISTANT, "Let me think.").with_channel(
                "analysis"
            ),
            Message.from_role_and_content(
                Role.ASSISTANT, "The answer is 4."
            ).with_channel("final"),
        ]
        token_ids = render_for_completion(messages)
        decoded = get_encoding().decode(token_ids)
        assert "Let me think." in decoded

    def test_preserves_reasoning_across_tool_turns(self):
        """Reasoning before a tool call must survive rendering even when
        the conversation ends with a final message (which triggers the
        encoder's auto_drop_analysis)."""
        messages = [
            get_system_message(),
            Message.from_role_and_content(Role.USER, "What's the weather?"),
            Message.from_role_and_content(
                Role.ASSISTANT, "I should call the weather API."
            ).with_channel("analysis"),
            Message.from_role_and_content(Role.ASSISTANT, '{"location": "SF"}')
            .with_channel("commentary")
            .with_recipient("functions.get_weather")
            .with_content_type("json"),
            Message.from_author_and_content(
                Author.new(Role.TOOL, "functions.get_weather"), "72F, sunny"
            )
            .with_channel("commentary")
            .with_recipient("assistant"),
            # Final message triggers the encoder's auto_drop_analysis
            Message.from_role_and_content(
                Role.ASSISTANT, "It is 72F and sunny in SF."
            ).with_channel("final"),
        ]
        token_ids = render_for_completion(messages)
        decoded = get_encoding().decode(token_ids)
        assert "I should call the weather API." in decoded


class TestResponseInputToHarmonyReasoningItem:
    """Tests for response_input_to_harmony handling of reasoning input items.

    Per the OpenAI spec, ResponseReasoningItem.content is
    Optional[List[Content]] = None. Clients like langchain-openai may omit
    this field when constructing multi-turn input from previous responses.

    Reasoning items with content are converted to Harmony messages on the
    'analysis' channel. All content items are concatenated. Items without
    content return None (skipped by the caller).
    """

    def test_reasoning_with_single_content(self):
        """Test reasoning item with a single content entry."""
        item = {
            "type": "reasoning",
            "id": "rs_123",
            "content": [{"type": "reasoning_text", "text": "Thinking step by step"}],
        }

        msg = response_input_to_harmony(item, prev_responses=[])

        assert msg is not None
        assert msg.author.role == Role.ASSISTANT
        assert msg.content[0].text == "Thinking step by step"
        assert msg.channel == "analysis"

    def test_reasoning_with_multiple_content_items(self):
        """Test reasoning item with multiple content entries concatenated."""
        item = {
            "type": "reasoning",
            "id": "rs_123",
            "content": [
                {"type": "reasoning_text", "text": "First, let me analyze"},
                {"type": "reasoning_text", "text": "Second, I should consider"},
                {"type": "reasoning_text", "text": "Finally, the answer is"},
            ],
        }

        msg = response_input_to_harmony(item, prev_responses=[])

        assert msg is not None
        assert msg.author.role == Role.ASSISTANT
        assert msg.content[0].text == (
            "First, let me analyze\nSecond, I should consider\nFinally, the answer is"
        )
        assert msg.channel == "analysis"

    def test_reasoning_without_content_returns_none(self):
        """Test reasoning item without content field returns None."""
        item = {
            "type": "reasoning",
            "id": "rs_123",
            "summary": [{"type": "summary_text", "text": "Thinking about math"}],
        }

        msg = response_input_to_harmony(item, prev_responses=[])

        assert msg is None

    def test_reasoning_with_none_content_returns_none(self):
        """Test reasoning item with content=None returns None."""
        item = {
            "type": "reasoning",
            "id": "rs_123",
            "content": None,
            "summary": [{"type": "summary_text", "text": "Thinking about math"}],
        }

        msg = response_input_to_harmony(item, prev_responses=[])

        assert msg is None

    def test_reasoning_with_empty_content_returns_none(self):
        """Test reasoning item with empty content list returns None."""
        item = {
            "type": "reasoning",
            "id": "rs_123",
            "content": [],
        }

        msg = response_input_to_harmony(item, prev_responses=[])

        assert msg is None


class TestSanitizeHarmonyName:
    """Tests for sanitize_harmony_name()."""

    def test_clean_name_unchanged(self) -> None:
        assert sanitize_harmony_name("get_weather") == "get_weather"

    def test_strip_channel_token(self) -> None:
        assert (
            sanitize_harmony_name("manage_cart<|channel|>commentary") == "manage_cart"
        )

    def test_strip_constrain_token(self) -> None:
        assert sanitize_harmony_name("<|constrain|>json") == ""

    def test_pure_control_token_returns_empty(self) -> None:
        assert sanitize_harmony_name("<|start|>") == ""

    def test_multiple_tokens_earliest_wins(self) -> None:
        assert sanitize_harmony_name("foo<|channel|>bar<|constrain|>baz") == "foo"

    def test_empty_string(self) -> None:
        assert sanitize_harmony_name("") == ""

    def test_trailing_whitespace_stripped(self) -> None:
        assert sanitize_harmony_name("tool_name  <|end|>") == "tool_name"


class TestSanitizeHarmonyRecipient:
    """Tests for sanitize_harmony_recipient()."""

    def test_clean_dotted_name_unchanged(self) -> None:
        assert sanitize_harmony_recipient("browser.search") == "browser.search"

    def test_clean_simple_name_unchanged(self) -> None:
        assert sanitize_harmony_recipient("python") == "python"

    def test_contaminated_first_part_preserved_structure(self) -> None:
        """browser<|channel|>.search → browser.search"""
        assert (
            sanitize_harmony_recipient("browser<|channel|>.search") == "browser.search"
        )

    def test_contaminated_second_part(self) -> None:
        """browser.search<|end|>garbage → browser.search"""
        assert (
            sanitize_harmony_recipient("browser.search<|end|>garbage")
            == "browser.search"
        )

    def test_pure_control_token_returns_empty(self) -> None:
        assert sanitize_harmony_recipient("<|constrain|>json") == ""

    def test_functions_dotted_contaminated(self) -> None:
        """functions.get_weather<|channel|>commentary → functions.get_weather"""
        assert (
            sanitize_harmony_recipient("functions.get_weather<|channel|>commentary")
            == "functions.get_weather"
        )

    def test_empty_string(self) -> None:
        assert sanitize_harmony_recipient("") == ""

    def test_container_dotted_contaminated(self) -> None:
        """container<|channel|>.exec → container.exec"""
        assert (
            sanitize_harmony_recipient("container<|channel|>.exec") == "container.exec"
        )

    def test_full_component_contamination_returns_empty(self) -> None:
        """functions.<|constrain|>json → "" (not "functions")"""
        assert sanitize_harmony_recipient("functions.<|constrain|>json") == ""

    def test_container_full_component_contamination_returns_empty(self) -> None:
        """container.<|channel|>commentary → "" (not "container")"""
        assert sanitize_harmony_recipient("container.<|channel|>commentary") == ""


class TestResilientStreamableParser:
    """Tests for ResilientStreamableParser error recovery."""

    def test_normal_sequence_unchanged(self) -> None:
        """Normal token sequence should produce same results as raw parser."""
        encoding = get_encoding()
        harmony_str = "<|channel|>final<|message|>Hello world<|end|>"
        token_ids = encoding.encode(harmony_str, allowed_special="all")

        parser = get_streamable_parser_for_assistant()
        for tok in token_ids:
            parser.process(tok)

        assert len(parser.messages) == 1
        assert parser.messages[0].content[0].text == "Hello world"
        assert parser.messages[0].channel == "final"

    def test_missing_start_recovery(self) -> None:
        """Parser should recover when <|start|> is missing between messages."""
        encoding = get_encoding()
        # First message completes normally, second is missing <|start|>
        first_msg = "<|channel|>final<|message|>First.<|end|>"
        second_msg = "<|channel|>final<|message|>Second.<|end|>"
        first_tokens = encoding.encode(first_msg, allowed_special="all")
        second_tokens = encoding.encode(second_msg, allowed_special="all")

        parser = get_streamable_parser_for_assistant()
        for tok in first_tokens:
            parser.process(tok)
        # Feed second message tokens directly (missing <|start|>assistant)
        for tok in second_tokens:
            parser.process(tok)

        assert len(parser.messages) == 2
        assert parser.messages[0].content[0].text == "First."
        assert parser.messages[1].content[0].text == "Second."

    def test_constrain_in_header_skipped(self) -> None:
        """<|constrain|> in HEADER state should be skipped gracefully."""
        encoding = get_encoding()
        # First, complete a normal message so parser goes to EXPECT_START
        first_msg = "<|channel|>final<|message|>First.<|end|>"
        first_tokens = encoding.encode(first_msg, allowed_special="all")

        # Build a second message where <|constrain|> appears in the header
        # after <|start|>assistant, before <|channel|>
        start_tok = encoding.encode("<|start|>", allowed_special="all")
        role_toks = encoding.encode("assistant", allowed_special="all")
        constrain_tok = encoding.encode("<|constrain|>", allowed_special="all")
        # Garbage tokens that should be skipped
        json_toks = encoding.encode("json", allowed_special="all")
        message_tok = encoding.encode("<|message|>", allowed_special="all")
        text_toks = encoding.encode("Second.", allowed_special="all")
        end_tok = encoding.encode("<|end|>", allowed_special="all")

        parser = get_streamable_parser_for_assistant()
        # Complete first message
        for tok in first_tokens:
            parser.process(tok)
        assert len(parser.messages) == 1

        # Feed: <|start|>assistant → puts parser in HEADER state
        for tok in start_tok:
            parser.process(tok)
        for tok in role_toks:
            parser.process(tok)
        # Feed: <|constrain|> → should enter skip mode
        for tok in constrain_tok:
            parser.process(tok)
        # Feed: json tokens → should be discarded in skip mode
        for tok in json_toks:
            parser.process(tok)
        # Feed: <|message|> → should exit skip mode and resume
        for tok in message_tok:
            parser.process(tok)
        # Feed: text + <|end|>
        for tok in text_toks:
            parser.process(tok)
        for tok in end_tok:
            parser.process(tok)

        # Should have produced two messages despite the malformed sequence
        assert len(parser.messages) == 2
        assert parser.messages[0].content[0].text == "First."
        assert parser.messages[1].content[0].text == "Second."

    def test_messages_recipients_sanitized(self) -> None:
        """Messages returned by .messages should have sanitized recipients,
        preventing contaminated history in multi-turn interactions."""
        encoding = get_encoding()
        # Build a tool call message with a contaminated recipient
        harmony_str = (
            "<|channel|>commentary"
            "<|message|>Let me search.<|end|>"
            "<|start|>assistant to=functions.get_weather<|channel|>commentary"
            '<|constrain|>json<|message|>{"loc": "SF"}<|end|>'
        )
        token_ids = encoding.encode(harmony_str, allowed_special="all")

        parser = get_streamable_parser_for_assistant()
        for tok in token_ids:
            parser.process(tok)

        msgs = parser.messages
        # All recipients should be clean (no control tokens)
        for msg in msgs:
            if msg.recipient is not None:
                for tok_str in (
                    "<|channel|>",
                    "<|constrain|>",
                    "<|start|>",
                    "<|end|>",
                    "<|message|>",
                ):
                    assert tok_str not in msg.recipient, (
                        f"Leaked control token {tok_str!r} "
                        f"in message recipient: {msg.recipient!r}"
                    )

    def test_last_consumed_token_tracks_normal_processing(self) -> None:
        """Normal tokens forwarded to inner parser update last_consumed_token."""
        encoding = get_encoding()
        harmony_str = "<|channel|>final<|message|>Hello world<|end|>"
        token_ids = encoding.encode(harmony_str, allowed_special="all")

        parser = get_streamable_parser_for_assistant()
        assert parser.last_consumed_token is None

        for tok in token_ids:
            parser.process(tok)

        # After processing, last_consumed_token should be the last token
        assert parser.last_consumed_token == token_ids[-1]

    def test_pattern3_discarded_tokens_not_in_last_consumed(self) -> None:
        """Free-text tokens in EXPECT_START don't update last_consumed_token."""
        encoding = get_encoding()
        # Complete a message to reach EXPECT_START state
        first_msg = "<|channel|>final<|message|>First.<|end|>"
        first_tokens = encoding.encode(first_msg, allowed_special="all")

        parser = get_streamable_parser_for_assistant()
        for tok in first_tokens:
            parser.process(tok)

        last_consumed_after_first = parser.last_consumed_token
        assert last_consumed_after_first is not None

        # Now feed free-text tokens (not <|start|>) — these should be discarded
        garbage_tokens = encoding.encode("some free text", allowed_special="all")
        for tok in garbage_tokens:
            parser.process(tok)

        # last_consumed_token should NOT have changed
        assert parser.last_consumed_token == last_consumed_after_first

    def test_pattern2_skip_mode_discarded_tokens_not_in_last_consumed(self) -> None:
        """Tokens skipped during Pattern 2 don't update last_consumed_token."""
        encoding = get_encoding()
        # Complete a first message
        first_msg = "<|channel|>final<|message|>First.<|end|>"
        first_tokens = encoding.encode(first_msg, allowed_special="all")

        # Build second message with <|constrain|> in header
        start_tok = encoding.encode("<|start|>", allowed_special="all")
        role_toks = encoding.encode("assistant", allowed_special="all")
        constrain_tok = encoding.encode("<|constrain|>", allowed_special="all")
        json_toks = encoding.encode("json", allowed_special="all")
        message_tok = encoding.encode("<|message|>", allowed_special="all")

        parser = get_streamable_parser_for_assistant()
        for tok in first_tokens:
            parser.process(tok)

        last_consumed_after_first = parser.last_consumed_token

        # Feed <|start|>assistant to enter HEADER state
        for tok in start_tok:
            parser.process(tok)
        for tok in role_toks:
            parser.process(tok)

        last_consumed_after_header = parser.last_consumed_token

        # Feed <|constrain|> to enter skip mode
        for tok in constrain_tok:
            parser.process(tok)

        # last_consumed should not change (constrain triggers skip, not forwarded)
        assert parser.last_consumed_token == last_consumed_after_header

        # Feed garbage tokens in skip mode — should not update
        for tok in json_toks:
            parser.process(tok)
        assert parser.last_consumed_token == last_consumed_after_header

        # Feed <|message|> to exit skip mode — this IS forwarded
        for tok in message_tok:
            parser.process(tok)
        assert parser.last_consumed_token != last_consumed_after_first


class TestInjectResponseFormats:
    def test_appends_to_existing_instructions(self):
        result = inject_response_formats("You are helpful.", {"type": "object"})
        assert result.startswith("You are helpful.")
        assert "# Response Formats" in result
        assert '{"type":"object"}' in result

    def test_none_instructions_creates_section(self):
        result = inject_response_formats(None, {"type": "object"})
        assert result.startswith("# Response Formats")
        assert '{"type":"object"}' in result

    def test_custom_format_name(self):
        result = inject_response_formats(None, {"type": "object"}, format_name="order")
        assert "## order" in result

    def test_compact_json_no_spaces(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = inject_response_formats(None, schema)
        assert '{"type":"object","properties":{"name":{"type":"string"}}}' in result

    def test_section_separated_by_blank_lines(self):
        result = inject_response_formats("Instructions here.", {"type": "object"})
        assert "\n\n# Response Formats\n\n## structured_output\n\n" in result


class TestGetDeveloperMessageResponseFormats:
    """Tests for response_format_section parameter in get_developer_message."""

    ENV_VAR = (
        "vllm.entrypoints.openai.parser.harmony_utils"
        ".envs.VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS"
    )

    def _extract_instructions_text(self, dev_msg: Message) -> str | None:
        """Extract the raw text from a developer message's instructions."""
        for content_item in dev_msg.content:
            instructions = getattr(content_item, "instructions", None)
            if instructions is not None:
                return instructions
        return None

    def test_response_format_preserved_with_system_instructions(self):
        """When VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS is True,
        user instructions should be dropped but response format schema
        should still appear in the developer message."""
        schema_section = "# Response Formats\n\n## structured_output\n\n{}"
        with patch(self.ENV_VAR, True):
            dev_msg = get_developer_message(
                instructions="Be concise.",
                response_format_section=schema_section,
            )
        text = self._extract_instructions_text(dev_msg)
        assert text is not None
        assert "# Response Formats" in text
        # User instructions should NOT be present
        assert "Be concise." not in text

    def test_response_format_and_instructions_without_system_instructions(self):
        """When VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS is False,
        both instructions and response format schema should appear."""
        schema_section = "# Response Formats\n\n## structured_output\n\n{}"
        with patch(self.ENV_VAR, False):
            dev_msg = get_developer_message(
                instructions="Be concise.",
                response_format_section=schema_section,
            )
        text = self._extract_instructions_text(dev_msg)
        assert text is not None
        assert "Be concise." in text
        assert "# Response Formats" in text

    def test_response_format_only_no_instructions(self):
        """With instructions=None, only the response format section appears."""
        schema_section = "# Response Formats\n\n## structured_output\n\n{}"
        with patch(self.ENV_VAR, False):
            dev_msg = get_developer_message(
                instructions=None,
                response_format_section=schema_section,
            )
        text = self._extract_instructions_text(dev_msg)
        assert text is not None
        assert "# Response Formats" in text

    def test_backward_compat_no_response_format(self):
        """Without response_format_section, behavior matches the original."""
        with patch(self.ENV_VAR, False):
            dev_msg = get_developer_message(
                instructions="Be concise.",
            )
        text = self._extract_instructions_text(dev_msg)
        assert text is not None
        assert "Be concise." in text
        assert "# Response Formats" not in text
