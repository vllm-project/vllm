# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest

from tests.tool_parsers.common_tests import (
    ToolParserTestConfig,
    ToolParserTests,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
)
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.internlm2_tool_parser import Internlm2ToolParser

_GET_WEATHER_TOOL = ChatCompletionToolsParam(
    type="function",
    function={
        "name": "get_weather",
        "description": "Get the current weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
        },
    },
)


def _weather_output(name: str) -> str:
    return (
        f'<|action_start|><|plugin|>{{"name": "{name}", '
        '"parameters": {"city": "Tokyo"}}<|action_end|>'
    )


def test_extract_unknown_tool_name_not_emitted(
    default_tokenizer: TokenizerLike,
) -> None:
    # A tool name the client never offered must NOT be surfaced as a tool call.
    # Regression test: the whitelist guard was dead code (missing `return`), so
    # any hallucinated tool name was emitted as a real tool call anyway.
    parser = Internlm2ToolParser(default_tokenizer, [_GET_WEATHER_TOOL])
    result = parser.extract_tool_calls(
        _weather_output("delete_everything"), MagicMock()
    )
    assert result.tools_called is False
    assert result.tool_calls == []


def test_extract_known_tool_name_emitted(default_tokenizer: TokenizerLike) -> None:
    # A tool name present in the request must still be emitted.
    parser = Internlm2ToolParser(default_tokenizer, [_GET_WEATHER_TOOL])
    result = parser.extract_tool_calls(_weather_output("get_weather"), MagicMock())
    assert result.tools_called is True
    assert result.tool_calls[0].function.name == "get_weather"


def test_extract_without_tools_not_over_rejected(
    default_tokenizer: TokenizerLike,
) -> None:
    # When no tools are provided (e.g. the batch API constructs the parser with
    # tools=None) there is no whitelist to validate against, so a tool call must
    # still be emitted rather than dropped.
    parser = Internlm2ToolParser(default_tokenizer, None)
    result = parser.extract_tool_calls(_weather_output("get_weather"), MagicMock())
    assert result.tools_called is True
    assert result.tool_calls[0].function.name == "get_weather"


class TestInternLM2ToolParser(ToolParserTests):
    @pytest.fixture
    def tokenizer(self, default_tokenizer: TokenizerLike) -> TokenizerLike:
        """Add some internlm2 specific tokens to the default vocab."""

        tokenizer_vocab = default_tokenizer.get_vocab()
        default_tokenizer.get_vocab = MagicMock()
        tokenizer_vocab.update(
            {
                "<|action_start|>": 92540,
                "<|plugin|>": 92541,
                "<|action_end|>": 92542,
            }
        )
        default_tokenizer.get_vocab.return_value = tokenizer_vocab
        return default_tokenizer

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
            # InternLM2 doesn't support parallel calls
            parallel_tool_calls_output=(
                '<|action_start|><|plugin|>{"name": "get_weather", '
                '"parameters": {"city": "Tokyo"}}<|action_end|>'
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
            parallel_tool_calls_count=1,  # InternLM2 only supports single tool calls
            parallel_tool_calls_names=["get_weather"],
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
