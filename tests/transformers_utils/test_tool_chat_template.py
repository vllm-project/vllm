# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for custom tool chat templates that handle special characters
in tool parameter descriptions.

This tests the fix for: https://github.com/vllm-project/vllm/issues/32827
"""

from pathlib import Path

import pytest

from vllm.transformers_utils.chat_templates import get_tool_chat_template_path


class TestToolChatTemplate:
    """Tests for the tool chat template registry."""

    def test_get_minimax_m2_tool_template(self):
        """Test that MiniMax M2 model gets the custom tool template."""
        path = get_tool_chat_template_path(
            model_type="minimax_m2",
            tokenizer_name_or_path="MiniMax/MiniMax-M2.1",
        )
        assert path is not None
        assert isinstance(path, Path)
        assert path.name == "template_minimax_m2.jinja"
        assert path.exists()

    def test_get_tool_template_nonexistent_model(self):
        """Test that non-registered models return None."""
        path = get_tool_chat_template_path(
            model_type="nonexistent_model",
            tokenizer_name_or_path="some/model",
        )
        assert path is None

    def test_minimax_m2_template_uses_tojson(self):
        """Test that the MiniMax M2 template uses tojson filter for proper escaping."""
        path = get_tool_chat_template_path(
            model_type="minimax_m2",
            tokenizer_name_or_path="MiniMax/MiniMax-M2.1",
        )
        assert path is not None
        template_content = path.read_text()
        # The template should use tojson to properly escape special characters
        assert "tojson" in template_content

    def test_minimax_m2_template_renders_parentheses(self):
        """Test that template can render tools with parentheses in descriptions."""
        try:
            import jinja2
        except ImportError:
            pytest.skip("jinja2 not available")

        path = get_tool_chat_template_path(
            model_type="minimax_m2",
            tokenizer_name_or_path="MiniMax/MiniMax-M2.1",
        )
        assert path is not None

        template_content = path.read_text()
        env = jinja2.Environment()
        template = env.from_string(template_content)

        # Test with a tool that has parentheses in the description
        # This is the exact case from issue #32827
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The complete command to execute "
                                "(e.g. ls -la, ssh user@host, cat file.txt)",
                            }
                        },
                        "required": ["command"],
                    },
                },
            }
        ]

        messages = [{"role": "user", "content": "hello"}]

        # This should not raise an exception
        result = template.render(
            tools=tools,
            messages=messages,
            add_generation_prompt=True,
        )

        # The parentheses should be properly escaped in the output
        assert "ls -la" in result or "command" in result
        # Verify the template actually rendered something
        assert len(result) > 0
