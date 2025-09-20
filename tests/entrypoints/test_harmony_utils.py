# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest

from vllm.entrypoints.harmony_utils import parse_output_message


class TestHarmonyUtils:
    """Test cases for harmony_utils.py functions."""

    def test_commentary_channel_with_none_recipient(self):
        """Test commentary channel with recipient=None is handled correctly."""
        message = Mock()
        message.author.role = "assistant"
        message.channel = "commentary"
        message.recipient = None
        message.content = [Mock(text="reasoning content")]

        result = parse_output_message(message)

        assert len(result) == 1
        assert result[0].type == "reasoning"

    def test_commentary_channel_with_custom_function(self):
        """Test custom function recipient is supported."""
        message = Mock()
        message.author.role = "assistant"
        message.channel = "commentary"
        message.recipient = "functions.my_custom_function"
        message.content = [Mock(text='{"arg": "value"}')]
        
        result = parse_output_message(message)
        
        assert len(result) == 1
        assert result[0].type == "function_call"
        assert result[0].name == "my_custom_function"

    def test_commentary_channel_with_python_tool(self):
        """Test python tool recipient is supported for reasoning."""
        message = Mock()
        message.author.role = "assistant"
        message.channel = "commentary"
        message.recipient = "python"
        message.content = [Mock(text="python tool reasoning")]
        
        result = parse_output_message(message)
        
        assert len(result) == 1
        assert result[0].type == "reasoning"

    def test_unknown_channel_raises_error(self):
        """Test unknown channels raise appropriate errors."""
        message = Mock()
        message.author.role = "assistant"
        message.channel = "unknown_channel"
        message.recipient = None
        message.content = [Mock(text="Some content")]

        with pytest.raises(ValueError) as exc_info:
            parse_output_message(message)

        assert "Unknown channel: unknown_channel" in str(exc_info.value)
        assert "Supported channels are" in str(exc_info.value)

    def test_unknown_recipient_raises_error(self):
        """Test unknown recipient in commentary channel raises error."""
        message = Mock()
        message.author.role = "assistant"
        message.channel = "commentary"
        message.recipient = "unknown_recipient"
        message.content = [Mock(text="Some content")]

        with pytest.raises(ValueError) as exc_info:
            parse_output_message(message)

        assert "Unknown recipient: unknown_recipient" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__])
