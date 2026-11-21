# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fenced code examples in Qwen3 output must stay content.

A fenced example documents the tool call format itself, so the XML markers
inside it are data, not calls (vLLM issue #57541).  Markdown/CommonMark
fence rules matter here: the closing run must use the same delimiter
character as the opener and be at least as long, it must sit at the start
of a line (up to three spaces of indentation), and inline code spans must
never open a block fence.  Fences that open inside the reasoning channel
must stay in that channel.
"""

import json
from unittest.mock import MagicMock

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.streaming_helpers import (
    collect_content,
    collect_function_name,
    collect_tool_arguments,
    simulate_reasoning_streaming,
    simulate_tool_streaming,
)
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.qwen3 import (
    FENCE,
    TOOL_CALL_END,
    TOOL_CALL_START,
    qwen3_config,
)

# Built from pieces so the literal reasoning markers survive tooling.
THINK_START = "<" + "think" + ">"
THINK_END = "<" + "/think" + ">"

TICK = FENCE[0]
TILDE = "~" * 3
FOUR_TICKS = FENCE + TICK
FOUR_TILDES = TILDE + "~"

CALL = (
    "<tool_call>\n"
    "<function=Bash>\n"
    "<parameter=command>ls -la</parameter>\n"
    "</function>\n"
    "</tool_call>\n"
)


def bash_tool():
    bash = MagicMock()
    bash.function.name = "Bash"
    return bash


def chunks_for(text: str, chunk_size: int | None) -> list[str]:
    if chunk_size is None:
        return [text]
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


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
def thinking_parser(mock_tokenizer):
    return ParserEngine(
        mock_tokenizer,
        parser_engine_config=qwen3_config(thinking=True),
    )


@pytest.fixture
def parser_with_tools(mock_tokenizer):
    return ParserEngine(
        mock_tokenizer,
        tools=[bash_tool()],
        parser_engine_config=qwen3_config(thinking=False),
    )


FENCED_EXAMPLE = (
    "Here is what a tool call looks like:\n"
    "\n" + FENCE + "xml\n" + CALL + FENCE + "\n"
    "\n"
    "That is the format."
)


class TestFencedExamplesInContent:
    """Fenced code examples in content must stay text (vLLM issue #57541)."""

    def test_fenced_example_kept_as_content(self, parser, mock_request):
        result = parser.extract_tool_calls(FENCED_EXAMPLE, mock_request)

        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == FENCED_EXAMPLE

    @pytest.mark.parametrize("chunk_size", [1, 7, None])
    def test_fenced_example_streaming_kept_as_content(
        self, parser, mock_request, chunk_size
    ):
        chunks = chunks_for(FENCED_EXAMPLE, chunk_size)
        results = simulate_tool_streaming(parser, mock_request, chunks)

        assert collect_content(results) == FENCED_EXAMPLE
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
            "\n" + FENCE + "xml\n"
            "<tool_call>\n"
            "<function=Example>\n"
            "<parameter=x>1</parameter>\n"
            "</function>\n"
            "</tool_call>\n" + FENCE + "\n"
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
        assert FENCE + "xml" in result.content

    def test_code_fence_inside_parameter_value_is_preserved(self, parser, mock_request):
        text = (
            "<tool_call>\n"
            "<function=Bash>\n"
            "<parameter=command>echo " + TICK + "python" + TICK + " | cat</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert "python" in args["command"]

    @pytest.mark.parametrize("fence", [FENCE, TILDE])
    @pytest.mark.parametrize("chunk_size", [1, 7, None])
    def test_fenced_block_inside_parameter_value_is_preserved(
        self, parser, mock_request, fence, chunk_size
    ):
        text = (
            "<tool_call>\n"
            "<function=write_file>\n"
            "<parameter=content>\n" + fence + "python\n"
            "print(1)\n" + fence + "\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        results = simulate_tool_streaming(
            parser, mock_request, chunks_for(text, chunk_size)
        )

        assert "print(1)" in collect_tool_arguments(results)

    def test_fenced_example_after_think_end(self, thinking_parser, mock_request):
        fenced = FENCE + "xml\ncode\n" + FENCE + "\n"
        text = THINK_START + "\nreasoning here\n" + THINK_END + "\n" + fenced
        result = thinking_parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == "\n" + fenced

    def test_streaming_matches_non_streaming(self, parser, mock_request):
        chunks = chunks_for(FENCED_EXAMPLE, 7)
        results = simulate_tool_streaming(parser, mock_request, chunks)
        assert collect_content(results) == FENCED_EXAMPLE
        assert collect_function_name(results) is None


class TestFenceDelimiters:
    """Both backtick and tilde fences are recognized, and they do not mix."""

    def test_tilde_fence_kept_as_content(self, parser, mock_request):
        text = "Example:\n" + TILDE + "xml\n" + CALL + TILDE + "\ndone"
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is False
        assert result.content == text

    @pytest.mark.parametrize("chunk_size", [1, 7, None])
    def test_tilde_fence_streaming_kept_as_content(
        self, parser, mock_request, chunk_size
    ):
        text = "Example:\n" + TILDE + "xml\n" + CALL + TILDE + "\ndone"
        results = simulate_tool_streaming(
            parser, mock_request, chunks_for(text, chunk_size)
        )

        assert collect_content(results) == text
        assert collect_function_name(results) is None

    def test_tilde_run_does_not_close_backtick_fence(self, parser, mock_request):
        text = FENCE + "xml\n" + TILDE + "\n" + CALL + FENCE + "\ndone"
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is False
        assert result.content == text

    def test_backtick_run_does_not_close_tilde_fence(self, parser, mock_request):
        text = TILDE + "xml\n" + FENCE + "\n" + CALL + TILDE + "\ndone"
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is False
        assert result.content == text

    def test_real_tool_call_after_tilde_fence(self, parser, mock_request):
        text = (
            TILDE + "xml\n" + CALL + TILDE + "\n"
            "<tool_call>\n"
            "<function=Bash>\n"
            "<parameter=command>pwd</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "Bash"


class TestFenceLengths:
    """A closing run must use the same delimiter and be at least as long."""

    def test_shorter_inner_run_does_not_close_longer_fence(self, parser, mock_request):
        text = (
            "before\n"
            + FOUR_TICKS
            + "xml\n"
            + FENCE
            + "\n"
            + CALL
            + FOUR_TICKS
            + "\nafter"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is False
        assert result.content == text

    @pytest.mark.parametrize("chunk_size", [1, 7, None])
    def test_longer_fence_streaming_kept_as_content(
        self, parser, mock_request, chunk_size
    ):
        text = (
            "before\n"
            + FOUR_TICKS
            + "xml\n"
            + FENCE
            + "\n"
            + CALL
            + FOUR_TICKS
            + "\nafter"
        )
        results = simulate_tool_streaming(
            parser, mock_request, chunks_for(text, chunk_size)
        )

        assert collect_content(results) == text
        assert collect_function_name(results) is None

    def test_longer_run_closes_shorter_fence(self, parser, mock_request):
        text = (
            "before\n" + FENCE + "xml\n" + FOUR_TICKS + "\n"
            "<tool_call>\n"
            "<function=Bash>\n"
            "<parameter=command>pwd</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "Bash"

    def test_tilde_fence_lengths(self, parser, mock_request):
        text = (
            "before\n"
            + FOUR_TILDES
            + "xml\n"
            + TILDE
            + "\n"
            + CALL
            + FOUR_TILDES
            + "\nafter"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is False
        assert result.content == text


class TestLineAndIndentationRules:
    """Fences need a line start; inline code must never open one."""

    def test_three_space_indent_is_a_fence(self, parser, mock_request):
        text = "   " + TILDE + "xml\n" + CALL + TILDE + "\ndone"
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is False
        assert result.content == text

    def test_four_space_indent_is_not_a_fence(self, parser, mock_request):
        text = (
            "    "
            + TILDE
            + "xml is indented code, not a fence\n"
            + "<tool_call>\n"
            + "<function=Bash>\n"
            + "<parameter=command>pwd</parameter>\n"
            + "</function>\n"
            + "</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "Bash"

    def test_inline_backticks_in_prose_do_not_open_fence(self, parser, mock_request):
        text = (
            "run " + TICK + "echo" + TICK + " then:\n"
            "<tool_call>\n"
            "<function=Bash>\n"
            "<parameter=command>pwd</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "Bash"

    def test_inline_triple_backticks_do_not_open_fence(self, parser, mock_request):
        text = (
            "inline " + FENCE + "code" + FENCE + " then:\n"
            "<tool_call>\n"
            "<function=Bash>\n"
            "<parameter=command>pwd</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "Bash"

    def test_closing_fence_with_trailing_text_does_not_close(
        self, parser, mock_request
    ):
        text = (
            FENCE + "xml\n" + CALL + FENCE + " trailing info\n" + CALL + FENCE + "\nend"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is False
        assert result.content == text

    @pytest.mark.parametrize("chunk_size", [1, 7, None])
    def test_inline_code_streaming_still_promotes_real_call(
        self, parser, mock_request, chunk_size
    ):
        text = (
            "run " + TICK + "echo" + TICK + " then:\n"
            "<tool_call>\n"
            "<function=Bash>\n"
            "<parameter=command>pwd</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        results = simulate_tool_streaming(
            parser, mock_request, chunks_for(text, chunk_size)
        )

        assert collect_function_name(results) == "Bash"


class TestReasoningChannel:
    """A fence opened in reasoning stays in reasoning and never promotes."""

    REASONING_TEXT = (
        "thinking about the format\n"
        + FENCE
        + "xml\n"
        + CALL
        + FENCE
        + "\nstill thinking"
        + THINK_END
        + "answer"
    )

    def test_fenced_example_in_reasoning_stays_reasoning(
        self, thinking_parser, mock_request
    ):
        reasoning, content = thinking_parser.extract_reasoning(
            self.REASONING_TEXT, mock_request
        )

        assert content == "answer"
        assert reasoning is not None
        assert FENCE + "xml" in reasoning
        assert "tool_call>\n<function" in reasoning

    def test_fenced_example_in_reasoning_no_tool_call(
        self, thinking_parser, mock_request
    ):
        result = thinking_parser.extract_tool_calls(self.REASONING_TEXT, mock_request)

        assert result.tools_called is False
        assert result.content is None or "function=Bash" not in result.content

    def test_tilde_fence_in_reasoning_stays_reasoning(
        self, thinking_parser, mock_request
    ):
        text = (
            "thinking\n"
            + TILDE
            + "xml\n"
            + CALL
            + TILDE
            + "\nmore"
            + THINK_END
            + "answer"
        )
        reasoning, content = thinking_parser.extract_reasoning(text, mock_request)

        assert content == "answer"
        assert reasoning is not None
        assert TILDE + "xml" in reasoning

    @pytest.mark.parametrize("chunk_size", [1, 7, None])
    def test_reasoning_fence_streaming_preserved(
        self, thinking_parser, mock_request, chunk_size
    ):
        chunks = chunks_for(self.REASONING_TEXT, chunk_size)
        reasoning, content = simulate_reasoning_streaming(thinking_parser, chunks)

        assert content == "answer"
        assert FENCE + "xml" in reasoning
        assert "function=Bash" not in content

    @pytest.mark.parametrize("chunk_size", [1, 7, None])
    def test_reasoning_fence_streaming_no_tool_call(
        self, thinking_parser, mock_request, chunk_size
    ):
        chunks = chunks_for(self.REASONING_TEXT, chunk_size)
        results = simulate_tool_streaming(thinking_parser, mock_request, chunks)

        assert collect_function_name(results) is None


class TestUnterminatedFence:
    """An unterminated fence normalizes to its enclosing state at EOF."""

    def test_unterminated_content_fence_stays_content(self, parser, mock_request):
        text = "before\n" + TILDE + "xml\n" + CALL
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is False
        assert result.content == text

    def test_unterminated_reasoning_fence_stays_reasoning(
        self, thinking_parser, mock_request
    ):
        text = "thinking\n" + FENCE + "xml\n" + CALL
        reasoning, content = thinking_parser.extract_reasoning(text, mock_request)

        assert reasoning is not None
        assert FENCE + "xml" in reasoning
        assert content is None

        result = thinking_parser.extract_tool_calls(text, mock_request)
        assert result.tools_called is False

    @pytest.mark.parametrize("chunk_size", [1, 7, None])
    def test_unterminated_fence_streaming(self, parser, mock_request, chunk_size):
        text = "before\n" + FENCE + "xml\n" + CALL
        results = simulate_tool_streaming(
            parser, mock_request, chunks_for(text, chunk_size)
        )

        assert collect_function_name(results) is None
        assert collect_content(results) == text

    def test_unterminated_reasoning_fence_closes_on_think_end_non_streaming(
        self, thinking_parser, mock_request
    ):
        text = (
            THINK_START + "\nthinking\n" + FENCE + "xml\n" + CALL + THINK_END + "\n"
            "<tool_call>\n"
            "<function=Bash>\n"
            "<parameter=command>ls -la</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = thinking_parser.extract_tool_calls(text, mock_request)
        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "Bash"

    @pytest.mark.parametrize("chunk_size", [1, 7, None])
    def test_unterminated_reasoning_fence_closes_on_think_end_streaming(
        self, thinking_parser, mock_request, chunk_size
    ):
        text = (
            THINK_START + "\nthinking\n" + FENCE + "xml\n" + CALL + THINK_END + "\n"
            "<tool_call>\n"
            "<function=Bash>\n"
            "<parameter=command>ls -la</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        chunks = chunks_for(text, chunk_size)
        results = simulate_tool_streaming(thinking_parser, mock_request, chunks)
        assert collect_function_name(results) == "Bash"
