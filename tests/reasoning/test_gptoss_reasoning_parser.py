# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import Mock

import pytest
from transformers import AutoTokenizer

from vllm.entrypoints.mcp.tool_server import ToolServer
from vllm.reasoning.gptoss_reasoning_parser import (
    GptOssReasoningParser,
    from_builtin_tool_to_tag,
    no_func_reasoning_tag,
)

REASONING_MODEL_NAME = "openai/gpt-oss-120b"


@pytest.fixture(scope="module")
def gpt_oss_tokenizer():
    return AutoTokenizer.from_pretrained(REASONING_MODEL_NAME)


def test_gptoss_reasoning_ended_is_true(gpt_oss_tokenizer):
    parser = GptOssReasoningParser(gpt_oss_tokenizer)
    assert parser.is_reasoning_end([]) is True
    assert parser.is_reasoning_end_streaming([], []) is True


class TestGptOssStructuralTags:
    """Test cases for GptOssReasoningParser structural tag functionality."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
        tokenizer.get_vocab = Mock(return_value={"<|end|>": 6})
        return tokenizer

    @pytest.fixture
    def reasoning_parser(self, mock_tokenizer):
        """Create a GptOssReasoningParser instance."""
        return GptOssReasoningParser(mock_tokenizer)

    def test_prepare_structured_tag_no_tool_server(self, reasoning_parser):
        """Test prepare_structured_tag with no tool server."""
        result = reasoning_parser.prepare_structured_tag(None, None)
        expected = json.dumps(no_func_reasoning_tag)

        assert result == expected

        # Verify the structure is correct
        parsed = json.loads(result)
        assert parsed["type"] == "structural_tag"
        assert parsed["format"]["type"] == "triggered_tags"
        assert len(parsed["format"]["tags"]) == 1
        assert parsed["format"]["tags"][0]["begin"] == "<|channel|>analysis<|message|>"
        assert parsed["format"]["triggers"] == ["<|channel|>analysis"]

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

    @pytest.mark.parametrize(
        "tools",
        [
            [],
            ["browser"],
            ["python"],
            ["container"],
            ["browser", "python"],
            ["browser", "container"],
            ["python", "container"],
            ["browser", "python", "container"],
        ],
    )
    def test_json_validity_comprehensive(self, reasoning_parser, tools):
        """Test JSON validity across all possible tool combinations."""
        tool_server = Mock(spec=ToolServer)
        tool_server.has_tool = Mock(side_effect=lambda tool: tool in tools)

        result = reasoning_parser.prepare_structured_tag(None, tool_server)
        parsed_result = json.loads(result)

        assert parsed_result["type"] == "structural_tag"
        assert "format" in parsed_result
        assert "tags" in parsed_result["format"]
        assert "triggers" in parsed_result["format"]

        # Tag count should be: 1 (analysis) + 2 * len(tools)
        expected_tag_count = 1 + (2 * len(tools))
        assert len(parsed_result["format"]["tags"]) == expected_tag_count

        # Verify triggers are correctly configured
        expected_triggers = ["<|channel|>analysis"]
        if tools:
            expected_triggers.append("<|channel|>commentary to=")
        assert set(parsed_result["format"]["triggers"]) == set(expected_triggers)

    def test_no_cross_request_state_pollution(self, reasoning_parser):
        """Test that sequential calls with different tool servers produce
        independent results, guarding against shared mutable state
        (e.g. missing deepcopy in tag_with_builtin_funcs)."""
        tool_server_1 = Mock(spec=ToolServer)
        tool_server_1.has_tool = Mock(side_effect=lambda tool: tool == "python")

        tool_server_2 = Mock(spec=ToolServer)
        tool_server_2.has_tool = Mock(side_effect=lambda tool: tool == "browser")

        result_1 = reasoning_parser.prepare_structured_tag(None, tool_server_1)
        result_2 = reasoning_parser.prepare_structured_tag(None, tool_server_2)

        tags_1 = [tag["begin"] for tag in json.loads(result_1)["format"]["tags"]]
        tags_2 = [tag["begin"] for tag in json.loads(result_2)["format"]["tags"]]

        assert "<|channel|>commentary to=python" in tags_1
        assert "<|channel|>commentary to=browser" not in tags_1

        assert "<|channel|>commentary to=browser" in tags_2
        assert "<|channel|>commentary to=python" not in tags_2

    def test_tag_format_consistency(self, reasoning_parser):
        """Test that all generated tags follow consistent format,
        catching malformed tags from from_builtin_tool_to_tag."""
        tool_server = Mock(spec=ToolServer)
        tool_server.has_tool = Mock(
            side_effect=lambda tool: tool in ["python", "browser"]
        )

        result = reasoning_parser.prepare_structured_tag(None, tool_server)
        parsed_result = json.loads(result)

        for tag in parsed_result["format"]["tags"]:
            assert "begin" in tag
            assert "content" in tag
            assert "end" in tag
            assert tag["content"]["type"] == "any_text"
            assert tag["end"] == "<|end|>"
            assert tag["begin"].startswith("<|channel|>")
