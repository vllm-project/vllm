# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import json

import pytest

from tests.tool_parsers.common_tests import (
    ToolParserTestConfig,
    ToolParserTests,
)
from tests.tool_parsers.utils import run_tool_extraction
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

    def test_pretty_printed_arguments(self, tool_parser, streaming: bool):
        """Arguments the model indents over several lines are still parsed,
        and each call keeps its own arguments."""
        expected = [
            ("get_weather", {"city": "Tokyo", "unit": "celsius"}),
            ("search_hotels", {"location": "Tokyo", "check_in": "2025-01-15"}),
        ]
        calls = "".join(
            f"<｜tool▁call▁begin｜>function<｜tool▁sep｜>{name}\n"
            f"```json\n{json.dumps(args, indent=2)}\n```<｜tool▁call▁end｜>"
            for name, args in expected
        )
        model_output = f"<｜tool▁calls▁begin｜>{calls}<｜tool▁calls▁end｜>"

        _, tool_calls = run_tool_extraction(
            tool_parser, model_output, streaming=streaming
        )

        assert len(tool_calls) == len(expected)
        for tool_call, (name, args) in zip(tool_calls, expected):
            assert tool_call.function.name == name
            assert json.loads(tool_call.function.arguments) == args
