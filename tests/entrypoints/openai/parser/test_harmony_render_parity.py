# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-API render parity tests.

Verifies that the chat completion input path (parse_chat_input_to_harmony_message)
and the responses API input path (response_input_to_harmony) produce identical
Harmony messages and identical rendered token sequences when given equivalent
conversation representations.

The chat completion API encodes reasoning and tool calls as fields on a single
assistant message dict; the responses API encodes them as separate typed items
in request.input. Both paths must converge on the same Harmony message list and
therefore the same rendered prompt.

Each test:
  1. Builds Harmony messages from each path for a single message or sequence.
  2. Asserts message-level properties (role, channel, recipient, content)
     using verify_harmony_messages.
  3. Asserts that render_for_completion produces identical token sequences.
"""

from openai.types.responses import ResponseFunctionToolCall

from tests.entrypoints.openai.utils import verify_harmony_messages
from vllm.entrypoints.openai.parser.harmony_utils import (
    get_system_message,
    parse_chat_input_to_harmony_message,
    render_for_completion,
)
from vllm.entrypoints.openai.responses.harmony import response_input_to_harmony

# Use a fixed date so the system message is deterministic across both paths.
_DATE = "2025-01-01"


def _system():
    return get_system_message(start_date=_DATE)


class TestResponseInputToHarmonyRenderParity:
    """Each test drives the same conversation through both APIs and asserts
    identical Harmony messages and rendered token sequences."""

    # -----------------------------------------------------------------------
    # Single-message cases
    # -----------------------------------------------------------------------

    def test_user_message(self):
        chat_msgs = parse_chat_input_to_harmony_message(
            {"role": "user", "content": "What's the weather in Paris?"}
        )
        resp_msgs = [
            response_input_to_harmony(
                {"type": "message", "role": "user", "content": "What's the weather in Paris?"},
                prev_responses=[],
            )
        ]

        expected = [{"role": "user", "content": "What's the weather in Paris?"}]
        verify_harmony_messages(chat_msgs, expected)
        verify_harmony_messages(resp_msgs, expected)

        assert render_for_completion([_system()] + chat_msgs) == render_for_completion(
            [_system()] + resp_msgs
        )

    def test_assistant_final_message(self):
        chat_msgs = parse_chat_input_to_harmony_message(
            {"role": "assistant", "content": "It is 18°C in Paris."}
        )
        resp_msgs = [
            response_input_to_harmony(
                {"type": "message", "role": "assistant", "content": "It is 18°C in Paris."},
                prev_responses=[],
            )
        ]

        expected = [
            {"role": "assistant", "channel": "final", "content": "It is 18°C in Paris."}
        ]
        verify_harmony_messages(chat_msgs, expected)
        verify_harmony_messages(resp_msgs, expected)

        assert render_for_completion([_system()] + chat_msgs) == render_for_completion(
            [_system()] + resp_msgs
        )

    def test_reasoning_item(self):
        # Chat path: assistant message with only a reasoning field and no content.
        chat_msgs = parse_chat_input_to_harmony_message(
            {"role": "assistant", "reasoning": "I should call get_weather.", "content": ""}
        )
        resp_msgs = [
            response_input_to_harmony(
                {
                    "type": "reasoning",
                    "content": [
                        {"type": "reasoning_text", "text": "I should call get_weather."}
                    ],
                },
                prev_responses=[],
            )
        ]

        expected = [
            {
                "role": "assistant",
                "channel": "analysis",
                "content": "I should call get_weather.",
            }
        ]
        verify_harmony_messages(chat_msgs, expected)
        verify_harmony_messages(resp_msgs, expected)

        assert render_for_completion([_system()] + chat_msgs) == render_for_completion(
            [_system()] + resp_msgs
        )

    def test_function_call(self):
        chat_msgs = parse_chat_input_to_harmony_message(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris"}',
                        },
                    }
                ],
            }
        )
        resp_msgs = [
            response_input_to_harmony(
                {
                    "type": "function_call",
                    "name": "get_weather",
                    "arguments": '{"location": "Paris"}',
                },
                prev_responses=[],
            )
        ]

        expected = [
            {
                "role": "assistant",
                "channel": "commentary",
                "recipient": "functions.get_weather",
                "content": '{"location": "Paris"}',
                "content_type": "json",
            }
        ]
        verify_harmony_messages(chat_msgs, expected)
        verify_harmony_messages(resp_msgs, expected)

        assert render_for_completion([_system()] + chat_msgs) == render_for_completion(
            [_system()] + resp_msgs
        )

    def test_tool_output(self):
        prev_call = ResponseFunctionToolCall(
            id="fc_1",
            call_id="call_1",
            name="get_weather",
            arguments='{"location": "Paris"}',
            type="function_call",
        )

        chat_msgs = parse_chat_input_to_harmony_message(
            {"role": "tool", "tool_call_id": "call_1", "content": "18°C, clear skies."},
            tool_id_names={"call_1": "get_weather"},
        )
        resp_msgs = [
            response_input_to_harmony(
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "18°C, clear skies.",
                },
                prev_responses=[prev_call],
            )
        ]

        expected = [
            {
                "role": "tool",
                "author_name": "functions.get_weather",
                "channel": "commentary",
                "recipient": "assistant",
                "content": "18°C, clear skies.",
            }
        ]
        verify_harmony_messages(chat_msgs, expected)
        verify_harmony_messages(resp_msgs, expected)

        assert render_for_completion([_system()] + chat_msgs) == render_for_completion(
            [_system()] + resp_msgs
        )

    # -----------------------------------------------------------------------
    # Combined and multi-turn cases
    # -----------------------------------------------------------------------

    def test_reasoning_combined_with_function_call(self):
        """Chat API packs reasoning + tool_calls into one dict; responses API
        represents them as two separate items. Both must produce the same two
        Harmony messages in the same order: analysis then commentary."""
        chat_msgs = parse_chat_input_to_harmony_message(
            {
                "role": "assistant",
                "reasoning": "I should get the weather for Paris.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris"}',
                        },
                    }
                ],
            }
        )
        resp_msgs = [
            response_input_to_harmony(
                {
                    "type": "reasoning",
                    "content": [
                        {
                            "type": "reasoning_text",
                            "text": "I should get the weather for Paris.",
                        }
                    ],
                },
                prev_responses=[],
            ),
            response_input_to_harmony(
                {
                    "type": "function_call",
                    "name": "get_weather",
                    "arguments": '{"location": "Paris"}',
                },
                prev_responses=[],
            ),
        ]

        expected = [
            {
                "role": "assistant",
                "channel": "analysis",
                "content": "I should get the weather for Paris.",
            },
            {
                "role": "assistant",
                "channel": "commentary",
                "recipient": "functions.get_weather",
                "content": '{"location": "Paris"}',
                "content_type": "json",
            },
        ]
        verify_harmony_messages(chat_msgs, expected)
        verify_harmony_messages(resp_msgs, expected)

        assert render_for_completion([_system()] + chat_msgs) == render_for_completion(
            [_system()] + resp_msgs
        )

    def test_full_multi_turn_tool_call_conversation(self):
        """Full conversation: user -> reasoning + tool_call -> tool_output -> final.

        Both APIs must render the complete conversation to identical token sequences.
        This exercises the entire input pipeline including all message types and
        the Rust harmony encoder.
        """
        prev_call = ResponseFunctionToolCall(
            id="fc_1",
            call_id="call_1",
            name="get_weather",
            arguments='{"location": "Paris"}',
            type="function_call",
        )

        # --- Chat completion API path ---
        tool_id_names = {"call_1": "get_weather"}
        chat_msgs = []
        chat_msgs += parse_chat_input_to_harmony_message(
            {"role": "user", "content": "What's the weather in Paris?"}
        )
        chat_msgs += parse_chat_input_to_harmony_message(
            {
                "role": "assistant",
                "reasoning": "I should call get_weather for Paris.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris"}',
                        },
                    }
                ],
            }
        )
        chat_msgs += parse_chat_input_to_harmony_message(
            {"role": "tool", "tool_call_id": "call_1", "content": "18°C, clear skies."},
            tool_id_names=tool_id_names,
        )
        chat_msgs += parse_chat_input_to_harmony_message(
            {"role": "assistant", "content": "It is currently 18°C in Paris with clear skies."}
        )

        # --- Responses API path ---
        resp_input = [
            {"type": "message", "role": "user", "content": "What's the weather in Paris?"},
            {
                "type": "reasoning",
                "content": [
                    {
                        "type": "reasoning_text",
                        "text": "I should call get_weather for Paris.",
                    }
                ],
            },
            {
                "type": "function_call",
                "name": "get_weather",
                "arguments": '{"location": "Paris"}',
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "18°C, clear skies."},
            {
                "type": "message",
                "role": "assistant",
                "content": "It is currently 18°C in Paris with clear skies.",
            },
        ]
        resp_msgs = [
            response_input_to_harmony(item, prev_responses=[prev_call])
            for item in resp_input
        ]

        assert render_for_completion([_system()] + chat_msgs) == render_for_completion(
            [_system()] + resp_msgs
        )

    def test_multi_turn_two_tool_calls_with_reasoning_between(self):
        """Validates parity for a chain of two tool calls, each with its own
        reasoning trace. Reasoning traces in between commentary-channel tool
        calls must survive as analysis-channel messages in both paths.
        """
        prev_call_1 = ResponseFunctionToolCall(
            id="fc_1",
            call_id="call_1",
            name="get_weather",
            arguments='{"location": "Paris"}',
            type="function_call",
        )
        prev_call_2 = ResponseFunctionToolCall(
            id="fc_2",
            call_id="call_2",
            name="get_forecast",
            arguments='{"location": "Paris", "days": 7}',
            type="function_call",
        )

        # --- Chat completion API path ---
        tool_id_names = {"call_1": "get_weather", "call_2": "get_forecast"}
        chat_msgs = []
        chat_msgs += parse_chat_input_to_harmony_message(
            {"role": "user", "content": "What's the weather and forecast for Paris?"}
        )
        # First reasoning + tool call
        chat_msgs += parse_chat_input_to_harmony_message(
            {
                "role": "assistant",
                "reasoning": "I need current weather first.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris"}',
                        },
                    }
                ],
            }
        )
        chat_msgs += parse_chat_input_to_harmony_message(
            {"role": "tool", "tool_call_id": "call_1", "content": "18°C, clear skies."},
            tool_id_names=tool_id_names,
        )
        # Second reasoning + tool call
        chat_msgs += parse_chat_input_to_harmony_message(
            {
                "role": "assistant",
                "reasoning": "Now I need the weekly forecast.",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "function": {
                            "name": "get_forecast",
                            "arguments": '{"location": "Paris", "days": 7}',
                        },
                    }
                ],
            }
        )
        chat_msgs += parse_chat_input_to_harmony_message(
            {
                "role": "tool",
                "tool_call_id": "call_2",
                "content": "Mon 17°C, Tue 19°C, Wed 16°C",
            },
            tool_id_names=tool_id_names,
        )

        # --- Responses API path ---
        prev_responses = [prev_call_1, prev_call_2]
        resp_input = [
            {
                "type": "message",
                "role": "user",
                "content": "What's the weather and forecast for Paris?",
            },
            # First reasoning + tool call
            {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "I need current weather first."}],
            },
            {
                "type": "function_call",
                "name": "get_weather",
                "arguments": '{"location": "Paris"}',
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "18°C, clear skies."},
            # Second reasoning + tool call
            {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "Now I need the weekly forecast."}],
            },
            {
                "type": "function_call",
                "name": "get_forecast",
                "arguments": '{"location": "Paris", "days": 7}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_2",
                "output": "Mon 17°C, Tue 19°C, Wed 16°C",
            },
        ]
        resp_msgs = [
            response_input_to_harmony(item, prev_responses=prev_responses)
            for item in resp_input
        ]

        assert render_for_completion([_system()] + chat_msgs) == render_for_completion(
            [_system()] + resp_msgs
        )
