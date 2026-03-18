# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import Mock

import pytest
from transformers import AutoTokenizer

from vllm.entrypoints.mcp.tool_server import ToolServer
from vllm.reasoning import ReasoningParser
from vllm.reasoning.gptoss_reasoning_parser import (
    GptOssReasoningParser,
    from_builtin_tool_to_tag,
    from_function_tool_to_tag,
    no_func_reasoning_tag,
    tag_with_builtin_funcs,
)

REASONING_MODEL_NAME = "openai/gpt-oss-120b"


@pytest.fixture(scope="module")
def gpt_oss_tokenizer():
    return AutoTokenizer.from_pretrained(REASONING_MODEL_NAME)


USER_MESSAGE_START = "<|start|>user<|message|>"
REASONING_SECTION_START = "<|end|><|start|>assistant<|channel|>analysis<|message|>"
END = "<|end|>"
ASSISTANT_START = "<|start|>assistant"
ASSISTANT_CONTENT_START_PREFIX = END + ASSISTANT_START + "<|channel|>final"
ASSISTANT_CONTENT_START_SUFFIX = "<|message|>"
ASSISTANT_CONTENT_START = (
    ASSISTANT_CONTENT_START_PREFIX + ASSISTANT_CONTENT_START_SUFFIX
)

BASIC_CONTENT = {
    "output": REASONING_SECTION_START
    + "This is reasoning"
    + ASSISTANT_CONTENT_START
    + "This is the rest",
    "is_reasoning_end": True,
}

BASIC_REASONING_ONLY = {
    "output": REASONING_SECTION_START + "This is reasoning" + "<|end|>",
    "is_reasoning_end": False,
}
BASIC_NO_REASONING_NO_ASSISTANT = {
    "output": USER_MESSAGE_START + "This is a user message",
    "is_reasoning_end": False,
}

# Edge-case where the model omits the assistant tag entirely.
BASIC_NO_REASONING_ASSISTANT = {
    "output": USER_MESSAGE_START + "This is a user message<|end|><|channel|>final",
    "is_reasoning_end": True,
}

COMPLEX_CONTENT_INCOMPLETE_PREFIX_ONLY = {
    "output": REASONING_SECTION_START
    + "This is reasoning"
    + ASSISTANT_CONTENT_START_PREFIX,
    "is_reasoning_end": False,
}

COMPLEX_CONTENT_SUFFIX_ONLY = {
    "output": REASONING_SECTION_START
    + "This is reasoning"
    + ASSISTANT_CONTENT_START_SUFFIX,
    "is_reasoning_end": False,
}

COMPLEX_CONTENT_1_NO_SUFFIX = {
    "output": REASONING_SECTION_START
    + "This is reasoning"
    + ASSISTANT_CONTENT_START_PREFIX
    + "<|constrain|> JSON ",
    "is_reasoning_end": False,
}

COMPLEX_CONTENT_1 = {
    "output": REASONING_SECTION_START
    + "This is reasoning"
    + ASSISTANT_CONTENT_START_PREFIX
    + "<|constrain|> JSON "
    + ASSISTANT_CONTENT_START_SUFFIX,
    "is_reasoning_end": True,
}

COMPLEX_CONTENT_1_WITH_CONTENT = {
    "output": REASONING_SECTION_START
    + "This is reasoning"
    + ASSISTANT_CONTENT_START_PREFIX
    + "<|constrain|> JSON "
    + ASSISTANT_CONTENT_START_SUFFIX
    + "This is the rest",
    "is_reasoning_end": True,
}

COMPLEX_CONTENT_2 = {
    "output": REASONING_SECTION_START
    + "This is reasoning"
    + ASSISTANT_CONTENT_START_PREFIX
    + "<|constrain|>ReplyAction "
    + ASSISTANT_CONTENT_START_SUFFIX
    + "This is the rest",
    "is_reasoning_end": True,
}

MULTI_TURN_CONTENT = {
    "output": USER_MESSAGE_START
    + "1st turn user message"
    + REASONING_SECTION_START
    + "1st turn reasoning"
    + ASSISTANT_CONTENT_START
    + "1st turn response"
    + END
    + USER_MESSAGE_START
    + "2nd turn user message"
    + END
    + ASSISTANT_START,
    "is_reasoning_end": False,
}
TEST_CASES = [
    BASIC_CONTENT,
    BASIC_REASONING_ONLY,
    COMPLEX_CONTENT_INCOMPLETE_PREFIX_ONLY,
    COMPLEX_CONTENT_SUFFIX_ONLY,
    COMPLEX_CONTENT_1_NO_SUFFIX,
    COMPLEX_CONTENT_1,
    COMPLEX_CONTENT_1_WITH_CONTENT,
    COMPLEX_CONTENT_2,
    MULTI_TURN_CONTENT,
]


@pytest.mark.parametrize(
    "output, is_reasoning_end",
    [(t["output"], t["is_reasoning_end"]) for t in TEST_CASES],
)
def test_gptoss_is_reasoning_end(
    output,
    is_reasoning_end,
    gpt_oss_tokenizer,
):
    output = gpt_oss_tokenizer.tokenize(output)
    parser: ReasoningParser = GptOssReasoningParser(gpt_oss_tokenizer)

    # Test is_reasoning_end
    output_ids = gpt_oss_tokenizer.convert_tokens_to_ids(output)
    actual_is_reasoning_end = parser.is_reasoning_end(output_ids)
    assert is_reasoning_end == actual_is_reasoning_end


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

    # --- Fixtures for tool_choice / function_tools tests ---

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

    # --- Tests from structured output PR ---

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

    def test_tag_with_builtin_funcs(self):
        """Test tag_with_builtin_funcs function."""
        builtin_tools = ["browser", "python"]
        result = tag_with_builtin_funcs(no_func_reasoning_tag, builtin_tools)

        assert result["type"] == "structural_tag"
        # Should have original analysis tag + 2 tags per tool
        assert len(result["format"]["tags"]) == 5  # 1 + 2*2

        # Should have added commentary trigger
        assert "<|channel|>commentary to=" in result["format"]["triggers"]
        assert "<|channel|>analysis" in result["format"]["triggers"]

    def test_tag_structure_invariants(self):
        """Test that the basic tag structure follows expected format."""
        assert no_func_reasoning_tag["type"] == "structural_tag"
        assert no_func_reasoning_tag["format"]["type"] == "triggered_tags"
        assert no_func_reasoning_tag["format"]["stop_after_first"] is False

        # Verify analysis tag structure
        analysis_tag = no_func_reasoning_tag["format"]["tags"][0]
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

    # --- final_content_format tests ---

    def test_prepare_structured_tag_with_json_schema(self, reasoning_parser):
        """Test that final channel tag has json_schema content constraint."""
        content_format = {
            "type": "json_schema",
            "json_schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
        }
        result = reasoning_parser.prepare_structured_tag(
            None, None, final_content_format=content_format
        )
        parsed = json.loads(result)

        # Should have analysis tag + final channel tag
        assert len(parsed["format"]["tags"]) == 2

        # Verify analysis tag is unchanged
        assert parsed["format"]["tags"][0]["begin"] == "<|channel|>analysis<|message|>"
        assert parsed["format"]["tags"][0]["content"]["type"] == "any_text"

        # Verify final channel tag has the json_schema content constraint
        final_tag = parsed["format"]["tags"][1]
        assert final_tag["begin"] == "<|channel|>final<|message|>"
        assert final_tag["end"] == "<|end|>"
        assert final_tag["content"] == content_format

        # Verify triggers include both analysis and final
        assert "<|channel|>analysis" in parsed["format"]["triggers"]
        assert "<|channel|>final" in parsed["format"]["triggers"]

    def test_prepare_structured_tag_original_tag_ignores_constraint(
        self, reasoning_parser
    ):
        """When original_tag is provided, final_content_format is ignored."""
        original_tag = '{"custom": "tag"}'
        content_format = {
            "type": "json_schema",
            "json_schema": {"type": "object"},
        }
        result = reasoning_parser.prepare_structured_tag(
            original_tag, None, final_content_format=content_format
        )

        # Should return the original tag unchanged
        assert result == original_tag

    def test_prepare_structured_tag_with_tools_and_constraint(
        self, reasoning_parser, mock_tool_server_with_browser
    ):
        """Test that tools and content constraint coexist in the tag."""
        content_format = {
            "type": "json_schema",
            "json_schema": {"type": "object"},
        }
        result = reasoning_parser.prepare_structured_tag(
            None,
            mock_tool_server_with_browser,
            final_content_format=content_format,
        )
        parsed = json.loads(result)

        # Should have analysis + 2 browser tags + final channel tag = 4
        assert len(parsed["format"]["tags"]) == 4

        tag_begins = [tag["begin"] for tag in parsed["format"]["tags"]]
        assert "<|channel|>analysis<|message|>" in tag_begins
        assert "<|channel|>commentary to=browser" in tag_begins
        assert "<|channel|>analysis to=browser" in tag_begins
        assert "<|channel|>final<|message|>" in tag_begins

        # Verify final tag has the constraint
        final_tag = next(
            t
            for t in parsed["format"]["tags"]
            if t["begin"] == "<|channel|>final<|message|>"
        )
        assert final_tag["content"] == content_format

    # --- Function tool and tool_choice tests ---

    def test_function_tool_tags_on_both_channels(self):
        """Verify from_function_tool_to_tag creates commentary + analysis."""
        tags = from_function_tool_to_tag("get_weather", None)

        assert len(tags) == 2
        assert (
            tags[0]["begin"]
            == "<|channel|>commentary to=functions.get_weather<|message|>"
        )
        assert (
            tags[1]["begin"]
            == "<|channel|>analysis to=functions.get_weather<|message|>"
        )
        assert tags[0]["end"] == "<|end|>"
        assert tags[1]["end"] == "<|end|>"
        # No parameters -> any_text
        assert tags[0]["content"] == {"type": "any_text"}
        assert tags[1]["content"] == {"type": "any_text"}

    def test_function_tool_json_schema_content(self):
        """Verify JSON schema from tool parameters is used as content."""
        schema = {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }
        tags = from_function_tool_to_tag("get_weather", schema)

        expected_content = {"type": "json_schema", "json_schema": schema}
        assert tags[0]["content"] == expected_content
        assert tags[1]["content"] == expected_content

    def test_tool_choice_required_blocks_final(self, reasoning_parser):
        """No final trigger/tag when tool_choice=required (no tools)."""
        result = reasoning_parser.prepare_structured_tag(
            None, None, tool_choice="required"
        )
        parsed = json.loads(result)

        tag_begins = [t["begin"] for t in parsed["format"]["tags"]]
        assert not any("final" in b for b in tag_begins)
        assert "<|channel|>final" not in parsed["format"]["triggers"]

    def test_tool_choice_required_with_function_tools(self, reasoning_parser):
        """Tool tags present but no final when tool_choice=required."""
        fn_tools = [
            {"name": "get_weather", "parameters": {"type": "object"}},
        ]
        result = reasoning_parser.prepare_structured_tag(
            None, None, tool_choice="required", function_tools=fn_tools
        )
        parsed = json.loads(result)

        tag_begins = [t["begin"] for t in parsed["format"]["tags"]]
        # Function tool tags present
        assert "<|channel|>commentary to=functions.get_weather<|message|>" in tag_begins
        assert "<|channel|>analysis to=functions.get_weather<|message|>" in tag_begins
        # No final
        assert not any("final" in b for b in tag_begins)
        assert "<|channel|>final" not in parsed["format"]["triggers"]

    def test_tool_choice_required_ignores_final_content_format(self, reasoning_parser):
        """Final is blocked even when final_content_format is provided."""
        content_fmt = {
            "type": "json_schema",
            "json_schema": {"type": "object"},
        }
        fn_tools = [{"name": "my_func"}]
        result = reasoning_parser.prepare_structured_tag(
            None,
            None,
            final_content_format=content_fmt,
            tool_choice="required",
            function_tools=fn_tools,
        )
        parsed = json.loads(result)

        tag_begins = [t["begin"] for t in parsed["format"]["tags"]]
        assert not any("final" in b for b in tag_begins)

    def test_tool_choice_auto_with_tools_and_content_format(self, reasoning_parser):
        """Tool tags + final with content constraint for auto."""
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        content_fmt = {"type": "json_schema", "json_schema": schema}
        fn_tools = [{"name": "compute", "parameters": schema}]

        result = reasoning_parser.prepare_structured_tag(
            None,
            None,
            final_content_format=content_fmt,
            tool_choice="auto",
            function_tools=fn_tools,
        )
        parsed = json.loads(result)

        tag_begins = [t["begin"] for t in parsed["format"]["tags"]]
        # Function tool tags
        assert "<|channel|>commentary to=functions.compute<|message|>" in tag_begins
        # Final tag with content constraint
        assert "<|channel|>final<|message|>" in tag_begins
        assert "<|channel|>final" in parsed["format"]["triggers"]

        final_tag = next(
            t
            for t in parsed["format"]["tags"]
            if t["begin"] == "<|channel|>final<|message|>"
        )
        assert final_tag["content"] == content_fmt

    def test_tool_choice_auto_with_tools_final_is_any_text(self, reasoning_parser):
        """auto + function tools but no content format -> final allows free text."""
        fn_tools = [{"name": "get_weather", "parameters": {"type": "object"}}]
        result = reasoning_parser.prepare_structured_tag(
            None,
            None,
            tool_choice="auto",
            function_tools=fn_tools,
        )
        parsed = json.loads(result)

        final_tag = next(
            t
            for t in parsed["format"]["tags"]
            if t["begin"] == "<|channel|>final<|message|>"
        )
        # No content format -> model can respond with any text
        assert final_tag["content"] == {"type": "any_text"}

    def test_tool_choice_none_strips_tool_tags(
        self, reasoning_parser, mock_tool_server_with_all_tools
    ):
        """No tool tags with tool_choice=none, analysis only."""
        fn_tools = [{"name": "get_weather"}]
        result = reasoning_parser.prepare_structured_tag(
            None,
            mock_tool_server_with_all_tools,
            tool_choice="none",
            function_tools=fn_tools,
        )
        parsed = json.loads(result)

        tag_begins = [t["begin"] for t in parsed["format"]["tags"]]
        # Only analysis tag, no tool tags
        assert tag_begins == ["<|channel|>analysis<|message|>"]
        assert parsed["format"]["triggers"] == ["<|channel|>analysis"]

    def test_mixed_builtin_and_function_tools(
        self, reasoning_parser, mock_tool_server_with_browser
    ):
        """Both builtin and function tool tags coexist."""
        fn_tools = [{"name": "get_weather"}]
        result = reasoning_parser.prepare_structured_tag(
            None,
            mock_tool_server_with_browser,
            tool_choice="auto",
            function_tools=fn_tools,
        )
        parsed = json.loads(result)

        tag_begins = [t["begin"] for t in parsed["format"]["tags"]]
        # Builtin tool tags
        assert "<|channel|>commentary to=browser" in tag_begins
        assert "<|channel|>analysis to=browser" in tag_begins
        # Function tool tags
        assert "<|channel|>commentary to=functions.get_weather<|message|>" in tag_begins
        assert "<|channel|>analysis to=functions.get_weather<|message|>" in tag_begins
        # Final tag (auto + function tools)
        assert "<|channel|>final<|message|>" in tag_begins
        # General commentary trigger covers both builtin and function
        assert "<|channel|>commentary to=" in parsed["format"]["triggers"]

    def test_named_tool_choice(self, reasoning_parser):
        """Only the named tool's tags present, final blocked."""
        fn_tools = [
            {"name": "get_weather", "parameters": {"type": "object"}},
            {"name": "get_stock", "parameters": {"type": "object"}},
        ]
        result = reasoning_parser.prepare_structured_tag(
            None,
            None,
            tool_choice={"type": "function", "name": "get_weather"},
            function_tools=fn_tools,
        )
        parsed = json.loads(result)

        tag_begins = [t["begin"] for t in parsed["format"]["tags"]]
        # Only get_weather tags, not get_stock
        assert "<|channel|>commentary to=functions.get_weather<|message|>" in tag_begins
        assert "<|channel|>analysis to=functions.get_weather<|message|>" in tag_begins
        assert not any("get_stock" in b for b in tag_begins)
        # No final (named tool choice blocks final)
        assert not any("final" in b for b in tag_begins)
