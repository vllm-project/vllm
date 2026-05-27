# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import json

import pytest

from tests.tool_parsers.common_tests import (
    ToolParserTestConfig,
    ToolParserTests,
)
from tests.tool_parsers.utils import run_tool_extraction

# All current streaming xfails for this parser fail with the same assertion:
# `StreamingToolReconstructor` in tests/tool_parsers/utils.py requires `id`
# (and `name`) only on the first delta of each tool_call slot, but the
# parser sets id=self.current_call_id on every DeltaToolCall it emits,
# including continuation/args deltas. Fix is localized to the parser's
# delta-emit sites; once done, remove this constant and these entries.
_STREAMING_ID_REEMITTED = (
    "Parser sets id on every delta; OpenAI protocol requires id only on the "
    "first delta per slot"
)


class TestQwen3xmlToolParser(ToolParserTests):
    @pytest.fixture
    def test_config(self) -> ToolParserTestConfig:
        return ToolParserTestConfig(
            parser_name="qwen3_xml",
            # Test data
            no_tool_calls_output="This is a regular response without any tool calls.",
            single_tool_call_output="<tool_call>\n<function=get_weather>\n<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>",
            parallel_tool_calls_output="<tool_call>\n<function=get_weather>\n<parameter=city>Tokyo</parameter>\n</function>\n</tool_call><tool_call>\n<function=get_time>\n<parameter=timezone>Asia/Tokyo</parameter>\n</function>\n</tool_call>",
            various_data_types_output=(
                "<tool_call>\n<function=test_function>\n"
                "<parameter=string_field>hello</parameter>\n"
                "<parameter=int_field>42</parameter>\n"
                "<parameter=float_field>3.14</parameter>\n"
                "<parameter=bool_field>true</parameter>\n"
                "<parameter=null_field>null</parameter>\n"
                '<parameter=array_field>["a", "b", "c"]</parameter>\n'
                '<parameter=object_field>{"nested": "value"}</parameter>\n'
                "</function>\n</tool_call>"
            ),
            empty_arguments_output="<tool_call>\n<function=refresh>\n</function>\n</tool_call>",
            surrounding_text_output=(
                "Let me check the weather for you.\n\n"
                "<tool_call>\n<function=get_weather>\n"
                "<parameter=city>Tokyo</parameter>\n"
                "</function>\n</tool_call>\n\n"
                "I will get that information."
            ),
            escaped_strings_output=(
                "<tool_call>\n<function=test_function>\n"
                '<parameter=quoted>He said "hello"</parameter>\n'
                "<parameter=path>C:\\Users\\file.txt</parameter>\n"
                "<parameter=newline>line1\nline2</parameter>\n"
                "</function>\n</tool_call>"
            ),
            malformed_input_outputs=[
                "<tool_call><function=func>",
                "<tool_call><function=></function></tool_call>",
            ],
            # Expected results
            single_tool_call_expected_name="get_weather",
            single_tool_call_expected_args={"city": "Tokyo"},
            parallel_tool_calls_count=2,
            parallel_tool_calls_names=["get_weather", "get_time"],
            # xfail markers - Qwen3XML has systematic streaming issues
            xfail_streaming=dict.fromkeys(
                [
                    "test_single_tool_call_simple_args",
                    "test_parallel_tool_calls",
                    "test_various_data_types",
                    "test_empty_arguments",
                    "test_surrounding_text",
                    "test_escaped_strings",
                    "test_streaming_reconstruction",
                    "test_multiple_functions_in_one_tool_call",
                    "test_multiple_empty_functions_in_one_tool_call",
                    "test_three_functions_in_one_tool_call",
                    "test_multi_function_state_resets_between_tool_calls",
                ],
                _STREAMING_ID_REEMITTED,
            ),
            supports_typed_arguments=False,
        )

    def test_multiple_functions_in_one_tool_call(
        self,
        request: pytest.FixtureRequest,
        tool_parser,
        test_config: ToolParserTestConfig,
        streaming: bool,
    ):
        """Multiple <function=...> in one <tool_call> -> distinct slots."""
        self.apply_xfail_mark(
            request,
            test_config,
            "test_multiple_functions_in_one_tool_call",
            streaming,
        )
        model_output = (
            "<tool_call>\n"
            "<function=read>\n<parameter=path>README.md</parameter>\n</function>\n"
            "<function=read>\n<parameter=path>AGENTS.md</parameter>\n</function>\n"
            "</tool_call>"
        )
        _, tool_calls = run_tool_extraction(
            tool_parser, model_output, streaming=streaming
        )

        assert len(tool_calls) == 2, (
            f"Expected 2 tool calls (one per <function=...>), got {len(tool_calls)}"
        )
        for i, tc in enumerate(tool_calls):
            assert tc.function.name == "read", (
                f"tool_calls[{i}].name expected 'read', got {tc.function.name!r}"
            )
            json.loads(tc.function.arguments)  # must be valid JSON
        assert json.loads(tool_calls[0].function.arguments) == {"path": "README.md"}
        assert json.loads(tool_calls[1].function.arguments) == {"path": "AGENTS.md"}

    def test_multiple_empty_functions_in_one_tool_call(
        self,
        request: pytest.FixtureRequest,
        tool_parser,
        test_config: ToolParserTestConfig,
        streaming: bool,
    ):
        """Empty-args variant: pre-fix produced `{}{}` (or `{}{}{}` for N=3)."""
        self.apply_xfail_mark(
            request,
            test_config,
            "test_multiple_empty_functions_in_one_tool_call",
            streaming,
        )
        model_output = (
            "<tool_call>\n"
            "<function=ping>\n</function>\n"
            "<function=ping>\n</function>\n"
            "</tool_call>"
        )
        _, tool_calls = run_tool_extraction(
            tool_parser, model_output, streaming=streaming
        )

        assert len(tool_calls) == 2, (
            f"Expected 2 tool calls, got {len(tool_calls)}"
        )
        for tc in tool_calls:
            assert tc.function.name == "ping"
            # Empty args render as "{}" or "" depending on path; never `{}{}`.
            assert tc.function.arguments in ("{}", ""), (
                f"Expected empty args, got {tc.function.arguments!r}"
            )

    def test_three_functions_in_one_tool_call(
        self,
        request: pytest.FixtureRequest,
        tool_parser,
        test_config: ToolParserTestConfig,
        streaming: bool,
    ):
        """Guards against an off-by-one that would only break for N>2."""
        self.apply_xfail_mark(
            request,
            test_config,
            "test_three_functions_in_one_tool_call",
            streaming,
        )
        model_output = (
            "<tool_call>\n"
            "<function=read>\n<parameter=path>A</parameter>\n</function>\n"
            "<function=read>\n<parameter=path>B</parameter>\n</function>\n"
            "<function=read>\n<parameter=path>C</parameter>\n</function>\n"
            "</tool_call>"
        )
        _, tool_calls = run_tool_extraction(
            tool_parser, model_output, streaming=streaming
        )
        assert [tc.function.name for tc in tool_calls] == ["read", "read", "read"]
        assert [json.loads(tc.function.arguments) for tc in tool_calls] == [
            {"path": "A"},
            {"path": "B"},
            {"path": "C"},
        ]

    def test_multi_function_state_resets_between_tool_calls(
        self,
        request: pytest.FixtureRequest,
        tool_parser,
        test_config: ToolParserTestConfig,
        streaming: bool,
    ):
        """Per-<tool_call> function counter must reset on a new <tool_call>."""
        self.apply_xfail_mark(
            request,
            test_config,
            "test_multi_function_state_resets_between_tool_calls",
            streaming,
        )
        model_output = (
            "<tool_call>\n"
            "<function=read>\n<parameter=path>A</parameter>\n</function>\n"
            "<function=read>\n<parameter=path>B</parameter>\n</function>\n"
            "</tool_call>\n"
            "<tool_call>\n"
            "<function=read>\n<parameter=path>C</parameter>\n</function>\n"
            "<function=read>\n<parameter=path>D</parameter>\n</function>\n"
            "</tool_call>"
        )
        _, tool_calls = run_tool_extraction(
            tool_parser, model_output, streaming=streaming
        )
        assert [json.loads(tc.function.arguments) for tc in tool_calls] == [
            {"path": "A"},
            {"path": "B"},
            {"path": "C"},
            {"path": "D"},
        ]

    def test_malformed_function_close_without_open_does_not_crash(
        self, tool_parser
    ):
        """Out-of-scope contract: parser may return garbage, must not raise."""
        model_output = (
            "<tool_call>\n"
            "<parameter=q>x</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        run_tool_extraction(tool_parser, model_output, streaming=False)
