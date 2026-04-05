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
    no_func_reasoning_tag,
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
ASSISTANT_CONTENT = "This is the rest"

BASIC_CONTENT = {
    "output": REASONING_SECTION_START
    + "This is reasoning"
    + ASSISTANT_CONTENT_START
    + ASSISTANT_CONTENT,
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
    + ASSISTANT_CONTENT,
    "is_reasoning_end": True,
}

COMPLEX_CONTENT_2 = {
    "output": REASONING_SECTION_START
    + "This is reasoning"
    + ASSISTANT_CONTENT_START_PREFIX
    + "<|constrain|>ReplyAction "
    + ASSISTANT_CONTENT_START_SUFFIX
    + ASSISTANT_CONTENT,
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


@pytest.mark.parametrize(
    "output, is_reasoning_end",
    [(t["output"], t["is_reasoning_end"]) for t in TEST_CASES],
)
def test_gptoss_reasoning_end_index(
    output,
    is_reasoning_end,
    gpt_oss_tokenizer,
):
    output_tokens = gpt_oss_tokenizer.tokenize(output)
    parser: ReasoningParser = GptOssReasoningParser(gpt_oss_tokenizer)

    output_ids = gpt_oss_tokenizer.convert_tokens_to_ids(output_tokens)

    output_ids_len = len(output_ids)
    if ASSISTANT_CONTENT in output:
        output_ids_len -= len(
            gpt_oss_tokenizer.convert_tokens_to_ids(
                gpt_oss_tokenizer.tokenize(ASSISTANT_CONTENT)
            )
        )

    ans = -1 if not is_reasoning_end else output_ids_len - 1
    assert parser.reasoning_end_index(output_ids) == ans


@pytest.mark.parametrize(
    "output, is_reasoning_end",
    [(t["output"], t["is_reasoning_end"]) for t in TEST_CASES],
)
def test_gptoss_reasoning_end_delta_index(
    output,
    is_reasoning_end,
    gpt_oss_tokenizer,
):
    output_tokens = gpt_oss_tokenizer.tokenize(output)
    # Assume the last 10 tokes as delta tokens
    delta_tokens = output_tokens[-10:]
    output_tokens = output_tokens[:-10]

    parser: ReasoningParser = GptOssReasoningParser(gpt_oss_tokenizer)

    output_ids = gpt_oss_tokenizer.convert_tokens_to_ids(output_tokens)
    delta_ids = gpt_oss_tokenizer.convert_tokens_to_ids(delta_tokens)

    delta_ids_len = len(delta_tokens)
    if ASSISTANT_CONTENT in output:
        delta_ids_len -= len(
            gpt_oss_tokenizer.convert_tokens_to_ids(
                gpt_oss_tokenizer.tokenize(ASSISTANT_CONTENT)
            )
        )

    ans = -1 if not is_reasoning_end else delta_ids_len - 1
    assert parser.reasoning_end_delta_index(output_ids, delta_ids) == ans


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
