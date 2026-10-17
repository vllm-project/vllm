# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest

from tests.tool_parsers.common_tests import (
    ToolParserTestConfig,
    ToolParserTests,
)
from tests.tool_parsers.utils import (
    run_tool_extraction,
    run_tool_extraction_streaming,
)
from vllm.tokenizers import TokenizerLike, get_tokenizer


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

    def test_streaming_whole_tool_call_in_one_delta(
        self,
        tool_parser,
        test_config: ToolParserTestConfig,
    ):
        """A complete tool call delivered in a single streaming delta must be
        emitted, matching the non-streaming result.

        Chunk boundaries can batch a whole tool call (begin tag, name,
        arguments, and end tag) into one delta -- e.g. under speculative
        decoding or scheduler batching. Feeding the full output as a single
        delta reproduces that case; before the fix the call was silently
        dropped (streamed result was empty).
        """
        _, tools_non = run_tool_extraction(
            tool_parser, test_config.single_tool_call_output, streaming=False
        )

        reconstructor = run_tool_extraction_streaming(
            tool_parser, [test_config.single_tool_call_output]
        )

        assert len(reconstructor.tool_calls) == 1
        assert reconstructor.tool_calls[0].function.name == (
            test_config.single_tool_call_expected_name
        )
        assert (
            reconstructor.tool_calls[0].function.arguments
            == tools_non[0].function.arguments
        )

    def test_streaming_parallel_tool_calls_in_one_delta(
        self,
        tool_parser,
        test_config: ToolParserTestConfig,
    ):
        """Multiple complete tool calls in a single delta must all be emitted."""
        _, tools_non = run_tool_extraction(
            tool_parser, test_config.parallel_tool_calls_output, streaming=False
        )

        reconstructor = run_tool_extraction_streaming(
            tool_parser,
            [test_config.parallel_tool_calls_output],
            assert_one_tool_per_delta=False,
        )

        assert len(reconstructor.tool_calls) == test_config.parallel_tool_calls_count
        for streamed, expected_name in zip(
            reconstructor.tool_calls, test_config.parallel_tool_calls_names
        ):
            assert streamed.function.name == expected_name
        for streamed, non_streamed in zip(reconstructor.tool_calls, tools_non):
            assert streamed.function.arguments == non_streamed.function.arguments
