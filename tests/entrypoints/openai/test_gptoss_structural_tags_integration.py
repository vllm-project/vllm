# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Integration tests for GPT-OSS structural tags functionality (PR #25515)."""

import json
from unittest.mock import Mock

import pytest

from vllm.entrypoints.openai.protocol import (
    StructuredOutputsParams,
)
from vllm.entrypoints.tool_server import ToolServer
from vllm.reasoning.gptoss_reasoning_parser import (
    GptOssReasoningParser,
)


class TestGptOssStructuralTagsIntegration:
    """Integration tests for structural tags in GPT-OSS tool calls."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
        return tokenizer

    @pytest.fixture
    def gptoss_parser(self, mock_tokenizer):
        """Create a real GptOssReasoningParser instance."""
        return GptOssReasoningParser(mock_tokenizer)

    @pytest.fixture
    def tool_server_with_python(self):
        """Create a tool server with Python tool enabled."""
        tool_server = Mock(spec=ToolServer)
        tool_server.has_tool = Mock(side_effect=lambda tool: tool == "python")
        return tool_server

    @pytest.fixture
    def tool_server_empty(self):
        """Create a tool server with no tools."""
        tool_server = Mock(spec=ToolServer)
        tool_server.has_tool = Mock(return_value=False)
        return tool_server

    def test_end_to_end_no_tools(self, gptoss_parser):
        """Test end-to-end flow when no tools are available."""
        # Test the parser directly
        result = gptoss_parser.prepare_structured_tag(None, None)
        parsed_result = json.loads(result)

        # Verify basic structure
        assert parsed_result["type"] == "structural_tag"
        assert parsed_result["format"]["type"] == "triggered_tags"
        assert len(parsed_result["format"]["tags"]) == 1

        # Verify only analysis channel is allowed
        analysis_tag = parsed_result["format"]["tags"][0]
        assert analysis_tag["begin"] == "<|channel|>analysis<|message|>"
        assert analysis_tag["content"]["type"] == "any_text"
        assert analysis_tag["end"] == "<|end|>"

        # Verify triggers
        assert parsed_result["format"]["triggers"] == ["<|channel|>analysis"]
        assert parsed_result["format"]["stop_after_first"] is False

    def test_end_to_end_with_python_tool(self, gptoss_parser, tool_server_with_python):
        """Test end-to-end flow with Python tool enabled."""
        result = gptoss_parser.prepare_structured_tag(None, tool_server_with_python)
        parsed_result = json.loads(result)

        # Should have analysis tag + 2 python tags
        assert len(parsed_result["format"]["tags"]) == 3

        # Verify all expected tags are present
        tag_begins = [tag["begin"] for tag in parsed_result["format"]["tags"]]
        expected_begins = [
            "<|channel|>analysis<|message|>",
            "<|channel|>commentary to=python",
            "<|channel|>analysis to=python",
        ]

        for expected in expected_begins:
            assert expected in tag_begins

        # Verify triggers include commentary
        assert "<|channel|>analysis" in parsed_result["format"]["triggers"]
        assert "<|channel|>commentary to=" in parsed_result["format"]["triggers"]

    def test_structured_outputs_params_integration(
        self, gptoss_parser, tool_server_with_python
    ):
        """Test integration with StructuredOutputsParams."""
        # Generate structural tag
        structural_tag = gptoss_parser.prepare_structured_tag(
            None, tool_server_with_python
        )

        # Create StructuredOutputsParams
        params = StructuredOutputsParams(structural_tag=structural_tag)

        # Verify the tag is properly stored and accessible
        assert params.structural_tag == structural_tag

        # Verify the tag is valid JSON
        parsed_tag = json.loads(params.structural_tag)
        assert parsed_tag["type"] == "structural_tag"

    def test_tool_server_interaction_flow(self, gptoss_parser):
        """Test the complete tool server interaction flow."""
        # Create a comprehensive tool server
        tool_server = Mock(spec=ToolServer)

        # Test different tool combinations
        test_cases = [
            # No tools
            {"browser": False, "python": False, "container": False, "expected_tags": 1},
            # Single tool
            {"browser": True, "python": False, "container": False, "expected_tags": 3},
            # Multiple tools
            {"browser": True, "python": True, "container": False, "expected_tags": 5},
            # All tools
            {"browser": True, "python": True, "container": True, "expected_tags": 7},
        ]

        for case in test_cases:
            tool_server.has_tool = Mock(side_effect=lambda tool: case.get(tool, False))

            result = gptoss_parser.prepare_structured_tag(None, tool_server)
            parsed_result = json.loads(result)

            assert len(parsed_result["format"]["tags"]) == case["expected_tags"]

            # Verify tool-specific tags are present for enabled tools
            tag_begins = [tag["begin"] for tag in parsed_result["format"]["tags"]]
            for tool in ["browser", "python", "container"]:
                if case[tool]:
                    assert f"<|channel|>commentary to={tool}" in tag_begins
                    assert f"<|channel|>analysis to={tool}" in tag_begins

    def test_original_tag_preservation(self, gptoss_parser, tool_server_with_python):
        """Test that original tags are preserved when provided."""
        original_tag = '{"type": "custom_tag", "data": "preserved"}'

        result = gptoss_parser.prepare_structured_tag(
            original_tag, tool_server_with_python
        )

        # Should return original tag unchanged
        assert result == original_tag

    def test_json_validity_comprehensive(self, gptoss_parser):
        """Test JSON validity across all possible tool combinations."""
        tool_combinations = [
            [],
            ["browser"],
            ["python"],
            ["container"],
            ["browser", "python"],
            ["browser", "container"],
            ["python", "container"],
            ["browser", "python", "container"],
        ]

        for tools in tool_combinations:
            tool_server = Mock(spec=ToolServer)
            tool_server.has_tool = Mock(side_effect=lambda tool: tool in tools)

            result = gptoss_parser.prepare_structured_tag(None, tool_server)

            # Should be valid JSON
            parsed_result = json.loads(result)

            # Should have correct structure
            assert parsed_result["type"] == "structural_tag"
            assert "format" in parsed_result
            assert "tags" in parsed_result["format"]
            assert "triggers" in parsed_result["format"]

            # Tag count should be: 1 (analysis) + 2 * len(tools)
            expected_tag_count = 1 + (2 * len(tools))
            assert len(parsed_result["format"]["tags"]) == expected_tag_count

    def test_error_handling_invalid_tool_server(self, gptoss_parser):
        """Test error handling with invalid tool server."""
        # Tool server that raises exceptions
        tool_server = Mock(spec=ToolServer)
        tool_server.has_tool = Mock(side_effect=Exception("Tool server error"))

        # Should handle gracefully and still return a valid tag
        with pytest.raises(Exception, match="Tool server error"):
            gptoss_parser.prepare_structured_tag(None, tool_server)

    def test_concurrent_requests_isolation(self, gptoss_parser):
        """Test that concurrent requests don't interfere with each other."""
        # Simulate concurrent requests with different tool servers
        tool_server_1 = Mock(spec=ToolServer)
        tool_server_1.has_tool = Mock(side_effect=lambda tool: tool == "python")

        tool_server_2 = Mock(spec=ToolServer)
        tool_server_2.has_tool = Mock(side_effect=lambda tool: tool == "browser")

        # Generate tags concurrently
        result_1 = gptoss_parser.prepare_structured_tag(None, tool_server_1)
        result_2 = gptoss_parser.prepare_structured_tag(None, tool_server_2)

        # Parse results
        parsed_1 = json.loads(result_1)
        parsed_2 = json.loads(result_2)

        # Verify they have different tool configurations
        tags_1 = [tag["begin"] for tag in parsed_1["format"]["tags"]]
        tags_2 = [tag["begin"] for tag in parsed_2["format"]["tags"]]

        # Result 1 should have python tags
        assert "<|channel|>commentary to=python" in tags_1
        assert "<|channel|>commentary to=browser" not in tags_1

        # Result 2 should have browser tags
        assert "<|channel|>commentary to=browser" in tags_2
        assert "<|channel|>commentary to=python" not in tags_2

    def test_tag_format_consistency(self, gptoss_parser):
        """Test that all generated tags follow consistent format."""
        tool_server = Mock(spec=ToolServer)
        tool_server.has_tool = Mock(
            side_effect=lambda tool: tool in ["python", "browser"]
        )

        result = gptoss_parser.prepare_structured_tag(None, tool_server)
        parsed_result = json.loads(result)

        # Verify all tags have required fields
        for tag in parsed_result["format"]["tags"]:
            assert "begin" in tag
            assert "content" in tag
            assert "end" in tag
            assert tag["content"]["type"] == "any_text"
            assert tag["end"] == "<|end|>"

            # Verify begin format
            assert tag["begin"].startswith("<|channel|>")

    def test_trigger_configuration(self, gptoss_parser):
        """Test trigger configuration for different tool setups."""
        # Test with no tools
        result_no_tools = gptoss_parser.prepare_structured_tag(None, None)
        parsed_no_tools = json.loads(result_no_tools)
        assert parsed_no_tools["format"]["triggers"] == ["<|channel|>analysis"]

        # Test with tools
        tool_server = Mock(spec=ToolServer)
        tool_server.has_tool = Mock(side_effect=lambda tool: tool == "python")

        result_with_tools = gptoss_parser.prepare_structured_tag(None, tool_server)
        parsed_with_tools = json.loads(result_with_tools)

        expected_triggers = ["<|channel|>analysis", "<|channel|>commentary to="]
        assert set(parsed_with_tools["format"]["triggers"]) == set(expected_triggers)
