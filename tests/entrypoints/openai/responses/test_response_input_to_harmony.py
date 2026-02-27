# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for response_input_to_harmony.

Covers every type branch in the function and verifies that each produced
Harmony Message has the correct role, channel, recipient, content_type,
author name, and text content.
"""

import pytest
from openai.types.responses import ResponseFunctionToolCall, ResponseReasoningItem
from openai.types.responses.response_reasoning_item import (
    Content as ReasoningTextContent,
)
from openai_harmony import Role

from vllm.entrypoints.openai.responses.harmony import response_input_to_harmony

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_PREV_CALL = ResponseFunctionToolCall(
    id="fc_test",
    call_id="call_test",
    name="get_weather",
    arguments='{"location": "Paris"}',
    type="function_call",
)

_REASONING_ITEM = ResponseReasoningItem(
    id="rs_test",
    type="reasoning",
    content=[ReasoningTextContent(type="reasoning_text", text="Thinking hard.")],
    summary=[],
    status=None,
)


class TestResponseInputToHarmonyMessage:
    """Unit tests for every message type handled by response_input_to_harmony."""

    # -----------------------------------------------------------------------
    # type="message" (or no type key)
    # -----------------------------------------------------------------------

    def test_user_message_string_content(self):
        msg = response_input_to_harmony(
            {"type": "message", "role": "user", "content": "Hello"},
            prev_responses=[],
        )

        assert msg.author.role == Role.USER
        assert msg.content[0].text == "Hello"
        assert msg.channel is None

    def test_no_type_key_defaults_to_message_branch(self):
        """Omitting 'type' should fall through to the message branch."""
        msg = response_input_to_harmony(
            {"role": "user", "content": "Hello"},
            prev_responses=[],
        )

        assert msg.author.role == Role.USER
        assert msg.content[0].text == "Hello"

    def test_system_message(self):
        msg = response_input_to_harmony(
            {"type": "message", "role": "system", "content": "Be helpful."},
            prev_responses=[],
        )

        assert msg.author.role == Role.SYSTEM
        assert msg.content[0].text == "Be helpful."
        assert msg.channel is None

    def test_assistant_message_gets_final_channel(self):
        msg = response_input_to_harmony(
            {"type": "message", "role": "assistant", "content": "The answer is 42."},
            prev_responses=[],
        )

        assert msg.author.role == Role.ASSISTANT
        assert msg.channel == "final"
        assert msg.content[0].text == "The answer is 42."

    def test_developer_message_gets_instructions_prefix(self):
        msg = response_input_to_harmony(
            {"type": "message", "role": "developer", "content": "Be concise."},
            prev_responses=[],
        )

        assert msg.author.role == Role.DEVELOPER
        assert msg.content[0].text == "Instructions:\nBe concise."
        assert msg.channel is None

    def test_message_with_array_content(self):
        msg = response_input_to_harmony(
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "text", "text": "Part one. "},
                    {"type": "text", "text": "Part two."},
                ],
            },
            prev_responses=[],
        )

        assert msg.author.role == Role.USER
        assert len(msg.content) == 2
        assert msg.content[0].text == "Part one. "
        assert msg.content[1].text == "Part two."

    def test_developer_message_array_content_gets_prefix_on_each_part(self):
        msg = response_input_to_harmony(
            {
                "type": "message",
                "role": "developer",
                "content": [
                    {"type": "text", "text": "Rule 1."},
                    {"type": "text", "text": "Rule 2."},
                ],
            },
            prev_responses=[],
        )

        assert msg.content[0].text == "Instructions:\nRule 1."
        assert msg.content[1].text == "Instructions:\nRule 2."

    # -----------------------------------------------------------------------
    # type="reasoning"
    # -----------------------------------------------------------------------

    def test_reasoning_gets_analysis_channel(self):
        msg = response_input_to_harmony(
            {
                "type": "reasoning",
                "content": [
                    {"type": "reasoning_text", "text": "I should call get_weather."}
                ],
            },
            prev_responses=[],
        )

        assert msg.author.role == Role.ASSISTANT
        assert msg.channel == "analysis"
        assert msg.content[0].text == "I should call get_weather."

    def test_reasoning_pydantic_model_input(self):
        """A Pydantic ResponseReasoningItem should be model_dump()'d before parsing."""
        msg = response_input_to_harmony(_REASONING_ITEM, prev_responses=[])

        assert msg.author.role == Role.ASSISTANT
        assert msg.channel == "analysis"
        assert msg.content[0].text == "Thinking hard."

    # -----------------------------------------------------------------------
    # type="function_call"
    # -----------------------------------------------------------------------

    def test_function_call_channel_recipient_and_content_type(self):
        msg = response_input_to_harmony(
            {
                "type": "function_call",
                "name": "get_weather",
                "arguments": '{"location": "Paris"}',
            },
            prev_responses=[],
        )

        assert msg.author.role == Role.ASSISTANT
        assert msg.channel == "commentary"
        assert msg.recipient == "functions.get_weather"
        assert msg.content_type == "json"
        assert msg.content[0].text == '{"location": "Paris"}'

    def test_function_call_empty_arguments(self):
        msg = response_input_to_harmony(
            {"type": "function_call", "name": "ping", "arguments": ""},
            prev_responses=[],
        )

        assert msg.recipient == "functions.ping"
        assert msg.content[0].text == ""

    # -----------------------------------------------------------------------
    # type="function_call_output"
    # -----------------------------------------------------------------------

    def test_function_call_output_channel_recipient_and_author_name(self):
        msg = response_input_to_harmony(
            {"type": "function_call_output", "call_id": "call_test", "output": "18°C"},
            prev_responses=[_PREV_CALL],
        )

        assert msg.author.role == Role.TOOL
        assert msg.author.name == "functions.get_weather"
        assert msg.channel == "commentary"
        assert msg.recipient == "assistant"
        assert msg.content[0].text == "18°C"

    def test_function_call_output_uses_most_recent_matching_call(self):
        """When multiple prev_responses share a call_id, the last one wins
        because the search is reversed."""
        earlier = ResponseFunctionToolCall(
            id="fc_old",
            call_id="call_test",
            name="old_func",
            arguments="{}",
            type="function_call",
        )
        later = ResponseFunctionToolCall(
            id="fc_new",
            call_id="call_test",
            name="get_weather",
            arguments="{}",
            type="function_call",
        )

        msg = response_input_to_harmony(
            {
                "type": "function_call_output",
                "call_id": "call_test",
                "output": "result",
            },
            prev_responses=[earlier, later],
        )

        assert msg.author.name == "functions.get_weather"

    def test_function_call_output_skips_non_function_call_items_in_prev_responses(
        self,
    ):
        """ResponseReasoningItem entries in prev_responses should be ignored."""
        msg = response_input_to_harmony(
            {
                "type": "function_call_output",
                "call_id": "call_test",
                "output": "18°C",
            },
            prev_responses=[_REASONING_ITEM, _PREV_CALL],
        )

        assert msg.author.name == "functions.get_weather"

    def test_function_call_output_raises_if_no_matching_call(self):
        with pytest.raises(ValueError, match="No call message found for"):
            response_input_to_harmony(
                {
                    "type": "function_call_output",
                    "call_id": "no_such_id",
                    "output": "x",
                },
                prev_responses=[_PREV_CALL],
            )

    def test_function_call_output_raises_on_empty_prev_responses(self):
        with pytest.raises(ValueError, match="No call message found for"):
            response_input_to_harmony(
                {"type": "function_call_output", "call_id": "call_test", "output": "x"},
                prev_responses=[],
            )

    # -----------------------------------------------------------------------
    # Error cases
    # -----------------------------------------------------------------------

    def test_unknown_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown input type"):
            response_input_to_harmony(
                {"type": "image_url", "url": "https://example.com/img.png"},
                prev_responses=[],
            )

