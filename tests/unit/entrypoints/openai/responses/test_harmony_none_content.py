# SPDX-License-Identifier: Apache-2.0
"""Tests for response_input_to_harmony handling None content."""

from vllm.entrypoints.openai.responses.harmony import response_input_to_harmony


class TestResponseInputToHarmonyNoneContent:

    def test_assistant_message_none_content(self):
        """Assistant message with content=None (tool-call-only) should be skipped."""
        msg_dict = {
            "type": "message",
            "role": "assistant",
            "content": None,
        }
        result = response_input_to_harmony(msg_dict, [])
        assert result is None

    def test_user_message_string_content(self):
        """Normal string content should still work after the fix."""
        msg_dict = {
            "type": "message",
            "role": "user",
            "content": "Hello world",
        }
        result = response_input_to_harmony(msg_dict, [])
        assert result is not None
        assert result.content[0].text == "Hello world"
