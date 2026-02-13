# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for GPT-OSS structural tag support in reasoning (PR #25515)."""

import json
from unittest.mock import Mock

import pytest

from vllm.entrypoints.mcp.tool_server import ToolServer
from vllm.reasoning.gptoss_reasoning_parser import (
    GptOssReasoningParser,
    from_builtin_tool_to_tag,
    no_func_reaonsing_tag,
    tag_with_builtin_funcs,
)


class TestGptOssReasoningParser:
    """Test cases for GptOssReasoningParser structural tag functionality."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
        return tokenizer

    @pytest.fixture
    def reasoning_parser(self, mock_tokenizer):
        """Create a GptOssReasoningParser instance."""
        return GptOssReasoningParser(mock_tokenizer)

    @pytest.fixture
    def mock_tool_server_empty(self):
        """Create a mock ToolServer with no tools."""
        tool_server = Mock(spec=ToolServer)
        tool_server.has_tool = Mock(return_value=False)
        return tool_server

    @pytest.fixture
    def mock_tool_server_with_browser(self):
        """Create a mock ToolServer with browser tool."""
        tool_server = Mock(spec=ToolServer)
        tool_server.has_tool = Mock(side_effect=lambda tool: tool == "browser")
        return tool_server

    @pytest.fixture
    def mock_tool_server_with_all_tools(self):
        """Create a mock ToolServer with all builtin tools."""
        tool_server = Mock(spec=ToolServer)
        tool_server.has_tool = Mock(
            side_effect=lambda tool: tool in ["browser", "python", "container"]
        )
        return tool_server

    def test_prepare_structured_tag_no_tool_server(self, reasoning_parser):
        """Test prepare_structured_tag with no tool server."""
        result = reasoning_parser.prepare_structured_tag(None, None)
        expected = json.dumps(no_func_reaonsing_tag)

        assert result == expected

        # Verify the structure is correct
        parsed = json.loads(result)
        assert parsed["type"] == "structural_tag"
        assert parsed["format"]["type"] == "triggered_tags"
        assert len(parsed["format"]["tags"]) == 1
        assert parsed["format"]["tags"][0]["begin"] == "<|channel|>analysis<|message|>"
        assert parsed["format"]["triggers"] == ["<|channel|>analysis"]

    def test_prepare_structured_tag_with_all_tools(
        self, reasoning_parser, mock_tool_server_with_all_tools
    ):
        """Test prepare_structured_tag with all builtin tools."""
        result = reasoning_parser.prepare_structured_tag(
            None, mock_tool_server_with_all_tools
        )
        parsed = json.loads(result)

        # Should have analysis tag + tags for all 3 tools (2 tags each)
        assert len(parsed["format"]["tags"]) == 7  # 1 analysis + 6 tool tags

        # Check all tool tags are present
        tag_begins = [tag["begin"] for tag in parsed["format"]["tags"]]
        for tool in ["browser", "python", "container"]:
            assert f"<|channel|>commentary to={tool}" in tag_begins
            assert f"<|channel|>analysis to={tool}" in tag_begins

    def test_prepare_structured_tag_with_original_tag(self, reasoning_parser):
        """Test prepare_structured_tag when original_tag is provided."""
        original_tag = '{"custom": "tag"}'
        result = reasoning_parser.prepare_structured_tag(original_tag, None)

        # Should return the original tag unchanged
        assert result == original_tag

    def test_from_builtin_tool_to_tag(self):
        """Test from_builtin_tool_to_tag function."""
        tags = from_builtin_tool_to_tag("python")

        assert len(tags) == 2
        assert tags[0]["begin"] == "<|channel|>commentary to=python"
        assert tags[0]["content"]["type"] == "any_text"
        assert tags[0]["end"] == "<|end|>"

        assert tags[1]["begin"] == "<|channel|>analysis to=python"
        assert tags[1]["content"]["type"] == "any_text"
        assert tags[1]["end"] == "<|end|>"

    def test_tag_with_builtin_funcs(self):
        """Test tag_with_builtin_funcs function."""
        builtin_tools = ["browser", "python"]
        result = tag_with_builtin_funcs(no_func_reaonsing_tag, builtin_tools)

        assert result["type"] == "structural_tag"
        # Should have original analysis tag + 2 tags per tool
        assert len(result["format"]["tags"]) == 5  # 1 + 2*2

        # Should have added commentary trigger
        assert "<|channel|>commentary to=" in result["format"]["triggers"]
        assert "<|channel|>analysis" in result["format"]["triggers"]

    def test_tag_structure_invariants(self):
        """Test that the basic tag structure follows expected format."""
        # Test the base no_func_reaonsing_tag structure
        assert no_func_reaonsing_tag["type"] == "structural_tag"
        assert no_func_reaonsing_tag["format"]["type"] == "triggered_tags"
        assert no_func_reaonsing_tag["format"]["stop_after_first"] is False

        # Verify analysis tag structure
        analysis_tag = no_func_reaonsing_tag["format"]["tags"][0]
        assert analysis_tag["begin"] == "<|channel|>analysis<|message|>"
        assert analysis_tag["content"]["type"] == "any_text"
        assert analysis_tag["end"] == "<|end|>"

    def test_json_serialization_valid(
        self, reasoning_parser, mock_tool_server_with_all_tools
    ):
        """Test that all generated tags produce valid JSON."""
        # Test with no tool server
        result1 = reasoning_parser.prepare_structured_tag(None, None)
        json.loads(result1)  # Should not raise

        # Test with empty tool server
        empty_server = Mock(spec=ToolServer)
        empty_server.has_tool = Mock(return_value=False)
        result2 = reasoning_parser.prepare_structured_tag(None, empty_server)
        json.loads(result2)  # Should not raise

        # Test with tools
        result3 = reasoning_parser.prepare_structured_tag(
            None, mock_tool_server_with_all_tools
        )
        json.loads(result3)  # Should not raise

    @pytest.mark.parametrize("tool_name", ["browser", "python", "container"])
    def test_single_tool_integration(self, reasoning_parser, tool_name):
        """Test integration with individual tools."""
        tool_server = Mock(spec=ToolServer)
        tool_server.has_tool = Mock(side_effect=lambda tool: tool == tool_name)

        result = reasoning_parser.prepare_structured_tag(None, tool_server)
        parsed = json.loads(result)

        # Should have 1 analysis + 2 tool-specific tags
        assert len(parsed["format"]["tags"]) == 3

        tag_begins = [tag["begin"] for tag in parsed["format"]["tags"]]
        assert f"<|channel|>commentary to={tool_name}" in tag_begins
        assert f"<|channel|>analysis to={tool_name}" in tag_begins
