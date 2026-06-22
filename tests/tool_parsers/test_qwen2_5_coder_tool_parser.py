# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.tool_parsers.common_tests import ToolParserTestConfig, ToolParserTests
from tests.tool_parsers.utils import run_tool_extraction_streaming
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.qwen2_5_coder_tool_parser import Qwen25CoderToolParser


class TestQwen25CoderToolParser(ToolParserTests):
    @pytest.fixture
    def test_config(self) -> ToolParserTestConfig:
        return ToolParserTestConfig(
            parser_name="qwen2_5_coder",
            no_tool_calls_output=("This is a regular response without any tool calls."),
            single_tool_call_output=(
                '<tools>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tools>'
            ),
            parallel_tool_calls_output=(
                '<tools>{"name": "get_weather", "arguments": {"city": "Tokyo"}}'
                '</tools><tools>{"name": "get_time", "arguments": '
                '{"timezone": "Asia/Tokyo"}}</tools>'
            ),
            various_data_types_output=(
                '<tools>{"name": "test_function", "arguments": {'
                '"string_field": "hello", '
                '"int_field": 42, '
                '"float_field": 3.14, '
                '"bool_field": true, '
                '"null_field": null, '
                '"array_field": ["a", "b", "c"], '
                '"object_field": {"nested": "value"}'
                "}}</tools>"
            ),
            empty_arguments_output=(
                '<tools>{"name": "refresh", "arguments": {}}</tools>'
            ),
            surrounding_text_output=(
                "Let me check the weather for you.\n\n"
                '<tools>{"name": "get_weather", "arguments": {"city": "Tokyo"}}'
                "</tools>\n\n"
                "I will get that information."
            ),
            escaped_strings_output=(
                '<tools>{"name": "test_function", "arguments": {'
                '"quoted": "He said \\"hello\\"", '
                '"path": "C:\\\\Users\\\\file.txt", '
                '"newline": "line1\\nline2"'
                "}}</tools>"
            ),
            malformed_input_outputs=[
                "<tools>not valid json</tools>",
                '<tools>{"name": "no_close",',
                "<tools></tools>",
            ],
        )


# =============================================================================
# Streaming regression tests for partial-tag handling fixes
# =============================================================================


@pytest.fixture
def parser(default_tokenizer: TokenizerLike) -> Qwen25CoderToolParser:
    return ToolParserManager.get_tool_parser("qwen2_5_coder")(default_tokenizer)


def _emit_one_char_at_a_time(parser: Qwen25CoderToolParser, text: str) -> str:
    """Drive the streaming API with one-character deltas and join emitted content."""
    reconstructor = run_tool_extraction_streaming(parser, list(text))
    return reconstructor.other_content


def test_streaming_partial_start_tag_does_not_leak(
    parser: Qwen25CoderToolParser,
) -> None:
    """When '<tools>' arrives one character at a time, the parser must hold
    back the partial prefix and only emit it (or not) once the full tag
    state is known. Plain text before the tag must still be emitted.
    """
    text = 'Hello <tools>{"name": "f", "arguments": {}}</tools>'
    emitted = _emit_one_char_at_a_time(parser, text)

    assert "Hello " in emitted
    for k in range(1, len("<tools>")):
        assert "<tools>"[:k] not in emitted, (
            f"partial prefix {'<tools>'[:k]!r} leaked as content"
        )


def test_streaming_trailing_text_in_same_delta_as_close_tag(
    parser: Qwen25CoderToolParser,
) -> None:
    """When '</tools>' and following text arrive in the same delta
    (e.g. '}</tools> Done!'), the parser must emit both the completed tool
    call and the trailing text — neither may be dropped.
    """
    deltas = [
        '<tools>{"name": "foo", "arguments": {"a": 1}',
        "}</tools> Done!",
    ]
    reconstructor = run_tool_extraction_streaming(parser, deltas)

    assert len(reconstructor.tool_calls) == 1
    assert reconstructor.tool_calls[0].function.name == "foo"
    assert reconstructor.other_content == " Done!"
