# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest

import vllm.envs as envs
from tests.tool_parsers.common_tests import (
    ToolParserTestConfig,
    ToolParserTests,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.tokenizers import TokenizerLike, get_tokenizer
from vllm.tool_parsers.deepseekv3_tool_parser import DeepSeekV3ToolParser


class TestDeepSeekV3ToolParser(ToolParserTests):
    @pytest.fixture(scope="class")
    def tokenizer(self) -> TokenizerLike:
        return get_tokenizer("deepseek-ai/DeepSeek-V3")

    @pytest.fixture
    def test_config(self) -> ToolParserTestConfig:
        return ToolParserTestConfig(
            parser_name="deepseek_v3",
            # Test data
            no_tool_calls_output=(
                "How can I help you today? I can check weather for you."
            ),
            single_tool_call_output="""<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
```json
{"city": "Tokyo", "unit": "celsius"}
```<｜tool▁call▁end｜><｜tool▁calls▁end｜>""",
            parallel_tool_calls_output="""<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
```json
{"city": "Tokyo", "unit": "celsius"}
```<｜tool▁call▁end｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>search_hotels
```json
{"location": "Tokyo", "check_in": "2025-01-15"}
```<｜tool▁call▁end｜><｜tool▁calls▁end｜>""",
            various_data_types_output=(
                """<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>test_function
```json
"""
                """{"string_field": "hello", "int_field": 42, "float_field": 3.14, """
                """"bool_field": true, "null_field": null, """
                """"array_field": ["a", "b", "c"], """
                """"object_field": {"nested": "value"}, """
                """"empty_array": [], "empty_object": {}}
```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"""
            ),
            empty_arguments_output="""<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_current_time
```json
{}
```<｜tool▁call▁end｜><｜tool▁calls▁end｜>""",
            surrounding_text_output=(
                """Let me check the weather for you."""
                """<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
```json
{"city": "Paris"}
```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"""
            ),
            escaped_strings_output=(
                """<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>send_message
```json
"""
                """{"text": "He said \\"hello\\"", "path": "C:\\\\Users\\\\file", """
                """"newline": "line1\\nline2"}
```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"""
            ),
            malformed_input_outputs=[
                """<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
```json
{"city": "Tokyo"
```<｜tool▁call▁end｜><｜tool▁calls▁end｜>""",
                """<｜tool▁calls▁begin｜>function<｜tool▁sep｜>get_weather
```json
{"city": "Tokyo"}
```<｜tool▁calls▁end｜>""",
            ],
            # Expected results
            single_tool_call_expected_name="get_weather",
            single_tool_call_expected_args={"city": "Tokyo", "unit": "celsius"},
            single_tool_call_expected_content=None,
            parallel_tool_calls_count=2,
            parallel_tool_calls_names=["get_weather", "search_hotels"],
            # xfail markers
            xfail_streaming={},
            xfail_nonstreaming={
                "test_malformed_input": (
                    "Parser sets tools_called=True even when tool_calls is "
                    "empty (detects start token but fails to parse)"
                ),
            },
        )


_TOOL_CALLS_BEGIN = "<｜tool▁calls▁begin｜>"
_TOOL_CALLS_END = "<｜tool▁calls▁end｜>"
_TOOL_CALL_BEGIN = "<｜tool▁call▁begin｜>"
_TOOL_CALL_END = "<｜tool▁call▁end｜>"
_TOOL_SEP = "<｜tool▁sep｜>"


def _fake_deepseek_tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {
        _TOOL_CALLS_BEGIN: 1,
        _TOOL_CALLS_END: 2,
        _TOOL_CALL_BEGIN: 3,
        _TOOL_CALL_END: 4,
    }
    return tokenizer


def _timeout_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(messages=[], model="test-model")


def test_regex_timeout_treated_as_no_tool_call():
    parser = DeepSeekV3ToolParser(_fake_deepseek_tokenizer())
    model_output = f"{_TOOL_CALLS_BEGIN}{_TOOL_CALL_BEGIN}incomplete"
    mock_regex = MagicMock()
    mock_regex.findall.side_effect = TimeoutError("Regex timeout")

    with patch.object(parser, "tool_call_regex", mock_regex):
        result = parser.extract_tool_calls(model_output, _timeout_request())

    assert result.tools_called is False
    assert result.tool_calls == []
    assert result.content == model_output
    mock_regex.findall.assert_called_once()
    assert (
        mock_regex.findall.call_args.kwargs["timeout"]
        == envs.VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS
    )


def test_streaming_portion_regex_timeout_skips_delta():
    parser = DeepSeekV3ToolParser(_fake_deepseek_tokenizer())
    mock_regex = MagicMock()
    mock_regex.match.side_effect = TimeoutError("Regex timeout")
    current_text = (
        f"{_TOOL_CALLS_BEGIN}{_TOOL_CALL_BEGIN}function{_TOOL_SEP}get_weather"
    )
    token_ids = [1, 3]

    with patch.object(parser, "stream_tool_call_portion_regex", mock_regex):
        result = parser.extract_tool_calls_streaming(
            current_text,
            current_text,
            "",
            token_ids,
            token_ids,
            [],
            _timeout_request(),
        )

    assert result is None
    mock_regex.match.assert_called_once()
    assert (
        mock_regex.match.call_args.kwargs["timeout"]
        == envs.VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS
    )


def test_streaming_name_regex_timeout_skips_delta():
    parser = DeepSeekV3ToolParser(_fake_deepseek_tokenizer())
    portion_regex = MagicMock()
    portion_regex.match.return_value = None
    name_regex = MagicMock()
    name_regex.match.side_effect = TimeoutError("Regex timeout")
    current_text = f"{_TOOL_CALLS_BEGIN}{_TOOL_CALL_BEGIN}function{_TOOL_SEP}"
    token_ids = [1, 3]

    with (
        patch.object(parser, "stream_tool_call_portion_regex", portion_regex),
        patch.object(parser, "stream_tool_call_name_regex", name_regex),
    ):
        result = parser.extract_tool_calls_streaming(
            current_text,
            current_text,
            "",
            token_ids,
            token_ids,
            [],
            _timeout_request(),
        )

    assert result is None
    name_regex.match.assert_called_once()
    assert (
        name_regex.match.call_args.kwargs["timeout"]
        == envs.VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS
    )
