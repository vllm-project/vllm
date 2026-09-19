# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fenced code examples in Qwen3 output must stay content.

A fenced example documents the tool call format itself, so the XML
markers inside it are data, not calls (vLLM issue #57541).
"""

import json
from unittest.mock import MagicMock

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.streaming_helpers import (
    collect_content,
    collect_function_name,
    simulate_tool_streaming,
)
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.qwen3 import (
    TOOL_CALL_END,
    TOOL_CALL_START,
    qwen3_config,
)


@pytest.fixture
def mock_tokenizer():
    return make_mock_tokenizer(
        {
            TOOL_CALL_START: 100,
            TOOL_CALL_END: 101,
        }
    )


@pytest.fixture
def parser(mock_tokenizer):
    return ParserEngine(
        mock_tokenizer,
        parser_engine_config=qwen3_config(thinking=False),
    )


@pytest.fixture
def tools():
    bash = MagicMock()
    bash.function.name = "Bash"
    return [bash]


@pytest.fixture
def parser_with_tools(mock_tokenizer, tools):
    return ParserEngine(
        mock_tokenizer,
        tools=tools,
        parser_engine_config=qwen3_config(thinking=False),
    )


class TestFencedExamplesInContent:
    """Fenced code examples in content must stay text (vLLM issue #57541)."""

    FENCED_EXAMPLE = (
        "Here is what a tool call looks like:\n"
        "\n"
        "```xml\n"
        "<tool_call>\n"
        "<function=Bash>\n"
        "<parameter=command>ls -la</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        "```\n"
        "\n"
        "That is the format."
    )

    def test_fenced_example_kept_as_content(self, parser, mock_request):
        result = parser.extract_tool_calls(self.FENCED_EXAMPLE, mock_request)

        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == self.FENCED_EXAMPLE

    @pytest.mark.parametrize("chunk_size", [1, 7])
    def test_fenced_example_streaming_kept_as_content(
        self, parser, mock_request, chunk_size
    ):
        text = self.FENCED_EXAMPLE
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
        results = simulate_tool_streaming(parser, mock_request, chunks)

        assert collect_content(results) == text
        assert collect_function_name(results) is None

    def test_xml_without_fence_still_promotes_tool_call(self, parser, mock_request):
        text = (
            "Here is what a tool call looks like:\n"
            "\n"
            "<tool_call>\n"
            "<function=Bash>\n"
            "<parameter=command>ls -la</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "Bash"

    def test_real_tool_call_after_fenced_example(self, parser, mock_request):
        text = (
            "Here is what a tool call looks like:\n"
            "\n"
            "```xml\n"
            "<tool_call>\n"
            "<function=Example>\n"
            "<parameter=x>1</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "```\n"
            "\n"
            "Now run it for real:\n"
            "<tool_call>\n"
            "<function=Bash>\n"
            "<parameter=command>pwd</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "Bash"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"command": "pwd"}
        assert "```xml" in result.content

    def test_code_fence_inside_parameter_value_is_preserved(self, parser, mock_request):
        text = (
            "<tool_call>\n"
            "<function=Bash>\n"
            "<parameter=command>echo `python` | cat</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert "python" in args["command"]

    def test_fenced_example_after_think_end(self, mock_tokenizer, mock_request):
        thinking_parser = ParserEngine(
            mock_tokenizer,
            parser_engine_config=qwen3_config(thinking=True),
        )
        fenced = "```xml\ncode\n```\n"
        text = "<think>\nreasoning here\n</think>\n" + fenced
        result = thinking_parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == "\n" + fenced

    def test_streaming_matches_non_streaming(self, parser, mock_request):
        text = self.FENCED_EXAMPLE
        chunks = [text[i : i + 7] for i in range(0, len(text), 7)]
        results = simulate_tool_streaming(parser, mock_request, chunks)
        assert collect_content(results) == text
        assert collect_function_name(results) is None
