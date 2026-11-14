# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import MagicMock

import pytest

from tests.tool_parsers.common_tests import (
    ToolParserTestConfig,
    ToolParserTests,
)
from tests.tool_parsers.utils import (
    run_tool_extraction_nonstreaming,
    run_tool_extraction_streaming,
)
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.internlm2_tool_parser import Internlm2ToolParser


@pytest.fixture
def internlm2_tokenizer(
    default_tokenizer: TokenizerLike, monkeypatch: pytest.MonkeyPatch
) -> TokenizerLike:
    """Add some internlm2 specific tokens to the default vocab."""
    tokenizer_vocab = default_tokenizer.get_vocab()
    tokenizer_vocab.update(
        {
            "<|action_start|>": 92540,
            "<|plugin|>": 92541,
            "<|action_end|>": 92542,
        }
    )
    monkeypatch.setattr(
        default_tokenizer, "get_vocab", MagicMock(return_value=tokenizer_vocab)
    )
    return default_tokenizer


class TestInternLM2ToolParser(ToolParserTests):
    @pytest.fixture
    def tokenizer(self, internlm2_tokenizer: TokenizerLike) -> TokenizerLike:
        return internlm2_tokenizer

    @pytest.fixture
    def test_config(self) -> ToolParserTestConfig:
        return ToolParserTestConfig(
            parser_name="internlm",
            # Test data
            no_tool_calls_output="This is a regular response without any tool calls.",
            single_tool_call_output=(
                '<|action_start|><|plugin|>{"name": "get_weather", '
                '"parameters": {"city": "Tokyo"}}<|action_end|>'
            ),
            # A single InternLM2 turn normally holds one tool call, but the
            # parser extracts every block the model emits.
            parallel_tool_calls_output=(
                '<|action_start|><|plugin|>{"name": "get_weather", '
                '"parameters": {"city": "Tokyo"}}<|action_end|>'
                '<|action_start|><|plugin|>{"name": "get_time", '
                '"parameters": {"timezone": "UTC"}}<|action_end|>'
            ),
            various_data_types_output="""<|action_start|><|plugin|>{
  "name": "test_function",
  "parameters": {
    "string_field": "hello",
    "int_field": 42,
    "float_field": 3.14,
    "bool_field": true,
    "null_field": null,
    "array_field": ["a", "b", "c"],
    "object_field": {"nested": "value"},
    "empty_array": [],
    "empty_object": {}
  }
}<|action_end|>""",
            empty_arguments_output=(
                '<|action_start|><|plugin|>{"name": "refresh", '
                '"parameters": {}}<|action_end|>'
            ),
            surrounding_text_output=(
                "Let me check the weather for you. "
                '<|action_start|><|plugin|>{"name": "get_weather", '
                '"parameters": {"city": "Tokyo"}}<|action_end|>'
            ),
            escaped_strings_output="""<|action_start|><|plugin|>{
  "name": "test_function",
  "parameters": {
    "quoted": "He said \\"hello\\"",
    "path": "C:\\\\Users\\\\file.txt",
    "newline": "line1\\nline2",
    "unicode": "emoji: 🎉"
  }
}<|action_end|>""",
            malformed_input_outputs=[
                '<|action_start|><|plugin|>{"name": "func", "parameters": {',
                (
                    '<|action_start|><|plugin|>{"name": "func", '
                    '"parameters": "not a dict"}<|action_end|>'
                ),
                "<|action_start|><|plugin|>not json<|action_end|>",
                "<|action_start|><|plugin|>",
                '<|action_start|>{"name": "func"}',
            ],
            # Expected results
            single_tool_call_expected_name="get_weather",
            single_tool_call_expected_args={"city": "Tokyo"},
            single_tool_call_expected_content=None,
            parallel_tool_calls_count=2,
            parallel_tool_calls_names=["get_weather", "get_time"],
            # Parser-specific settings
            allow_empty_or_json_empty_args=True,
            # xfail markers
            xfail_streaming={
                "test_single_tool_call_simple_args": (
                    "InternLM2 streaming not fully implemented"
                ),
                "test_parallel_tool_calls": (
                    "InternLM2 streaming not fully implemented"
                ),
                "test_various_data_types": (
                    "InternLM2 streaming not fully implemented"
                ),
                "test_empty_arguments": ("InternLM2 streaming not fully implemented"),
                "test_surrounding_text": ("InternLM2 streaming not fully implemented"),
                "test_escaped_strings": ("InternLM2 streaming not fully implemented"),
                "test_streaming_reconstruction": (
                    "InternLM2 streaming parser returns '<|action_start|' as "
                    "content instead of None - streaming/non-streaming inconsistency"
                ),
            },
            xfail_nonstreaming={
                "test_malformed_input": (
                    "InternLM2 parser raises JSONDecodeError on malformed JSON "
                    "instead of gracefully handling it"
                ),
            },
        )


def test_streaming_arguments_in_single_delta(
    internlm2_tokenizer: TokenizerLike,
) -> None:
    """Arguments arriving whole in one delta must not be dropped."""
    parser = Internlm2ToolParser(internlm2_tokenizer)

    deltas = [
        '<|action_start|><|plugin|>{"name": "get_weather"',
        ', "parameters": {"city": "Dallas", "state": "TX"}}<|action_end|>',
    ]

    streamed = ""
    current_text = ""
    for delta_text in deltas:
        previous_text = current_text
        current_text += delta_text
        delta_message = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=None,
        )
        if delta_message and delta_message.tool_calls:
            assert delta_message.tool_calls[0].function is not None
            arguments = delta_message.tool_calls[0].function.arguments
            if arguments:
                streamed += arguments

    assert json.loads(streamed) == {"city": "Dallas", "state": "TX"}


@pytest.mark.parametrize(
    "arguments_deltas",
    [
        ['{"location": "Beijing', '", "days": 3}}<|action_end|>'],
        [*'{"location": "Beijing", "days": 3}}', "<|action_end|>"],
    ],
    ids=["split_inside_string", "char_by_char"],
)
def test_streaming_arguments_match_final_arguments(
    default_tokenizer: TokenizerLike, arguments_deltas: list[str]
) -> None:
    """Streamed argument deltas must add up to the final arguments, wherever
    the chunk boundaries fall."""
    parser = Internlm2ToolParser(default_tokenizer)
    deltas = [
        '<|action_start|><|plugin|>{"name": "get_weather", "parameters": ',
        *arguments_deltas,
    ]

    reconstructor = run_tool_extraction_streaming(parser, deltas)

    assert len(reconstructor.tool_calls) == 1
    streamed = reconstructor.tool_calls[0].function.arguments
    streamed += parser.get_remaining_unstreamed_args()
    assert streamed == '{"location": "Beijing", "days": 3}'


TWO_TOOL_CALLS_OUTPUT = (
    "Let me check both for you. "
    '<|action_start|><|plugin|>{"name": "get_weather", '
    '"parameters": {"city": "Tokyo"}}<|action_end|>'
    '<|action_start|><|plugin|>{"name": "get_time", '
    '"parameters": {"timezone": "UTC"}}<|action_end|>'
)


def test_multiple_blocks_extract_every_tool_call(
    internlm2_tokenizer: TokenizerLike,
) -> None:
    """Regression for #44139.

    Two `<|action_start|><|plugin|>` blocks used to raise
    `ValueError: too many values to unpack`; guarding the split with a
    `maxsplit` still silently dropped the second call. Every block must
    become a tool call, and the leading text must stay content.
    """
    parser = Internlm2ToolParser(internlm2_tokenizer)

    extracted = run_tool_extraction_nonstreaming(parser, TWO_TOOL_CALLS_OUTPUT)

    assert extracted.tools_called
    assert [call.function.name for call in extracted.tool_calls] == [
        "get_weather",
        "get_time",
    ]
    assert json.loads(extracted.tool_calls[0].function.arguments) == {"city": "Tokyo"}
    assert json.loads(extracted.tool_calls[1].function.arguments) == {"timezone": "UTC"}
    assert extracted.content == "Let me check both for you. "


def test_multiple_blocks_stream_across_deltas(
    internlm2_tokenizer: TokenizerLike,
) -> None:
    """Each block arriving in its own delta must yield its own tool call."""
    parser = Internlm2ToolParser(internlm2_tokenizer)

    reconstructor = run_tool_extraction_streaming(
        parser,
        [
            '<|action_start|><|plugin|>{"name": "get_weather", '
            '"parameters": {"city": "Tokyo"}}<|action_end|>',
            '<|action_start|><|plugin|>{"name": "get_time", '
            '"parameters": {"timezone": "UTC"}}<|action_end|>',
        ],
    )

    assert [call.function.name for call in reconstructor.tool_calls] == [
        "get_weather",
        "get_time",
    ]
    assert json.loads(reconstructor.tool_calls[0].function.arguments) == {
        "city": "Tokyo"
    }
    assert json.loads(reconstructor.tool_calls[1].function.arguments) == {
        "timezone": "UTC"
    }


def test_multiple_blocks_stream_in_one_delta(
    internlm2_tokenizer: TokenizerLike,
) -> None:
    """Both blocks arriving in one delta must still yield both tool calls."""
    parser = Internlm2ToolParser(internlm2_tokenizer)

    reconstructor = run_tool_extraction_streaming(
        parser,
        [TWO_TOOL_CALLS_OUTPUT],
        assert_one_tool_per_delta=False,
    )

    assert [call.function.name for call in reconstructor.tool_calls] == [
        "get_weather",
        "get_time",
    ]
    assert json.loads(reconstructor.tool_calls[1].function.arguments) == {
        "timezone": "UTC"
    }
    assert reconstructor.other_content == "Let me check both for you. "
