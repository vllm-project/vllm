# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for tool_choice handling in the Harmony-based Responses API.

These tests verify that:
- Developer instructions are preserved when tool_choice="none" (Bug 1)
- Builtin tool descriptions are suppressed when tool_choice="none" (Bug 2)
"""

from __future__ import annotations

from unittest.mock import Mock

from openai_harmony import Role, ToolNamespaceConfig

from vllm.entrypoints.openai.parser.harmony_utils import (
    get_developer_message,
    get_system_message,
)


class TestToolChoiceNoneInstructions:
    """Bug 1: Developer instructions must not be dropped when
    tool_choice='none' causes tools to be hidden."""

    def test_developer_message_with_instructions_no_tools(self):
        """get_developer_message must include instructions even when
        tools=None (the condition that arises from tool_choice='none'
        with no custom tools)."""
        dev_msg = get_developer_message(
            instructions="Be helpful and concise", tools=None
        )
        assert dev_msg.author.role == Role.DEVELOPER
        rendered = str(dev_msg)
        assert "Be helpful and concise" in rendered

    def test_developer_message_with_instructions_and_tools(self):
        """Baseline: instructions + tools both appear in the developer
        message when tools are visible."""
        tool = Mock()
        tool.type = "function"
        tool.name = "get_weather"
        tool.description = "Get weather"
        tool.parameters = {"type": "object", "properties": {}}

        dev_msg = get_developer_message(instructions="Be helpful", tools=[tool])
        rendered = str(dev_msg)
        assert "Be helpful" in rendered
        assert "get_weather" in rendered

    def test_developer_message_no_instructions_no_tools(self):
        """When neither instructions nor tools are provided, the
        developer message is still valid (just empty content)."""
        dev_msg = get_developer_message(instructions=None, tools=None)
        assert dev_msg.author.role == Role.DEVELOPER


class TestToolChoiceNoneSystemMessage:
    """Bug 2: Builtin tool descriptions in the system message must be
    suppressed when tool_choice='none'."""

    def test_system_message_no_tool_descriptions(self):
        """When all tool descriptions are None (as happens when
        tools_visible=False), the system message must not contain
        tool descriptions."""
        sys_msg = get_system_message(
            browser_description=None,
            python_description=None,
            container_description=None,
            with_custom_tools=False,
        )
        assert sys_msg.author.role == Role.SYSTEM
        # tools should be None or empty when no descriptions are provided
        assert not sys_msg.content[0].tools

    def test_system_message_with_browser_description(self):
        """Baseline: when a ToolNamespaceConfig is provided, it appears
        in the system message tools."""
        browser_ns = ToolNamespaceConfig.browser()
        sys_msg = get_system_message(
            browser_description=browser_ns,
            python_description=None,
            container_description=None,
            with_custom_tools=False,
        )
        assert sys_msg.author.role == Role.SYSTEM
        assert "browser" in sys_msg.content[0].tools

    def test_system_message_with_python_description(self):
        """Python tool description appears in system message when provided."""
        python_ns = ToolNamespaceConfig.python()
        sys_msg = get_system_message(
            browser_description=None,
            python_description=python_ns,
            container_description=None,
            with_custom_tools=False,
        )
        assert sys_msg.author.role == Role.SYSTEM
        assert "python" in sys_msg.content[0].tools

    def test_none_descriptions_mean_no_tools(self):
        """Passing None for all tool descriptions (as happens when
        tools_visible=False) must result in no tools in the system msg."""
        sys_msg = get_system_message(
            browser_description=None,
            python_description=None,
            container_description=None,
            with_custom_tools=False,
        )
        assert not sys_msg.content[0].tools
