# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from openai_harmony import Message, Role

from tests.entrypoints.openai.utils import verify_harmony_messages
from vllm.entrypoints.openai.chat_completion.harmony import (
    auto_drop_analysis_messages,
    chat_input_to_harmony,
    parse_chat_output,
)
from vllm.entrypoints.openai.harmony import (
    get_encoding,
)


class TestParseChatInputToHarmonyMessage:
    """
    Tests for scenarios that are specific to the Chat Completion API
    chat_input_to_harmony function.
    """

    def test_user_message_with_empty_content(self):
        chat_msg = {
            "role": "user",
            "content": "",
        }

        messages = chat_input_to_harmony(chat_msg)

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

        messages = chat_input_to_harmony(chat_msg)

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

        messages = chat_input_to_harmony(chat_msg)

        assert len(messages) == 0

    def test_assistant_message_with_none_content(self):
        chat_msg = {
            "role": "assistant",
            "content": None,
        }

        messages = chat_input_to_harmony(chat_msg)

        assert len(messages) == 0

    def test_assistant_message_with_content_but_empty_reasoning(self):
        chat_msg = {
            "role": "assistant",
            "content": "The answer is 4.",
            "reasoning": "",
        }

        messages = chat_input_to_harmony(chat_msg)

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

        messages = chat_input_to_harmony(chat_msg)

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

        messages = chat_input_to_harmony(chat_msg)

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

        messages = chat_input_to_harmony(chat_msg)

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

        messages = chat_input_to_harmony(chat_msg)

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

        messages = chat_input_to_harmony(chat_msg)

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

        messages = chat_input_to_harmony(chat_msg)

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

        messages = chat_input_to_harmony(chat_msg, tool_id_names=tool_id_names)

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

        messages = chat_input_to_harmony(chat_msg, tool_id_names=tool_id_names)

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

        messages = chat_input_to_harmony(chat_msg, tool_id_names=tool_id_names)

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

        messages = chat_input_to_harmony(chat_msg, tool_id_names=tool_id_names)

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
