# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from openai.types.responses.response_reasoning_item import (
    Content,
    ResponseReasoningItem,
    Summary,
)

from vllm.entrypoints.responses_utils import (
    construct_chat_message_with_tool_call,
    convert_tool_responses_to_completions_format,
)


class TestResponsesUtils:
    """Tests for convert_tool_responses_to_completions_format function."""

    def test_convert_tool_responses_to_completions_format(self):
        """Test basic conversion of a flat tool schema to nested format."""
        input_tool = {
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location", "unit"],
            },
        }

        result = convert_tool_responses_to_completions_format(input_tool)

        assert result == {"type": "function", "function": input_tool}

    def test_construct_chat_message_with_tool_call(self):
        item = ResponseReasoningItem(
            id="lol",
            summary=[],
            type="reasoning",
            content=[
                Content(
                    text="Leroy Jenkins",
                    type="reasoning_text",
                )
            ],
            encrypted_content=None,
            status=None,
        )
        formatted_item = construct_chat_message_with_tool_call(item)
        assert formatted_item["role"] == "assistant"
        assert formatted_item["reasoning"] == "Leroy Jenkins"

        item = ResponseReasoningItem(
            id="lol",
            summary=[
                Summary(
                    text='Hmm, the user has just started with a simple "Hello,"',
                    type="summary_text",
                )
            ],
            type="reasoning",
            content=None,
            encrypted_content=None,
            status=None,
        )

        formatted_item = construct_chat_message_with_tool_call(item)
        assert formatted_item["role"] == "assistant"
        assert (
            formatted_item["reasoning"]
            == 'Hmm, the user has just started with a simple "Hello,"'
        )

        item = ResponseReasoningItem(
            id="lol",
            summary=[],
            type="reasoning",
            content=None,
            encrypted_content="TOP_SECRET_MESSAGE",
            status=None,
        )
        with pytest.raises(ValueError):
            construct_chat_message_with_tool_call(item)
