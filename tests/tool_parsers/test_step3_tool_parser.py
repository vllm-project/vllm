# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import json

import pytest

from tests.tool_parsers.common_tests import (
    ToolParserTestConfig,
    ToolParserTests,
)
from vllm.tokenizers import TokenizerLike, get_tokenizer
from vllm.tool_parsers.step3_tool_parser import Step3ToolParser


class TestStep3ToolParser(ToolParserTests):
    @pytest.fixture(scope="class")
    def tokenizer(self) -> TokenizerLike:
        return get_tokenizer("stepfun-ai/step3")

    @pytest.fixture
    def test_config(self) -> ToolParserTestConfig:
        return ToolParserTestConfig(
            parser_name="step3",
            # Test data
            no_tool_calls_output="This is a regular response without any tool calls.",
            single_tool_call_output=(
                "<｜tool_calls_begin｜><｜tool_call_begin｜>"
                '<steptml:invoke name="get_weather">'
                '<steptml:parameter name="city">Tokyo</steptml:parameter>'
                "</steptml:invoke><｜tool_call_end｜><｜tool_calls_end｜>"
            ),
            parallel_tool_calls_output=(
                "<｜tool_calls_begin｜><｜tool_call_begin｜>"
                '<steptml:invoke name="get_weather">'
                '<steptml:parameter name="city">Tokyo</steptml:parameter>'
                "</steptml:invoke><｜tool_call_end｜><｜tool_sep｜>"
                '<｜tool_call_begin｜><steptml:invoke name="get_time">'
                '<steptml:parameter name="timezone">Asia/Tokyo</steptml:parameter>'
                "</steptml:invoke><｜tool_call_end｜><｜tool_calls_end｜>"
            ),
            various_data_types_output=(
                "<｜tool_calls_begin｜><｜tool_call_begin｜>"
                '<steptml:invoke name="test_function">'
                '<steptml:parameter name="string_field">hello</steptml:parameter>'
                '<steptml:parameter name="int_field">42</steptml:parameter>'
                '<steptml:parameter name="float_field">3.14</steptml:parameter>'
                '<steptml:parameter name="bool_field">true</steptml:parameter>'
                '<steptml:parameter name="null_field">null</steptml:parameter>'
                '<steptml:parameter name="array_field">'
                '["a", "b", "c"]</steptml:parameter>'
                '<steptml:parameter name="object_field">'
                '{"nested": "value"}</steptml:parameter>'
                "</steptml:invoke><｜tool_call_end｜><｜tool_calls_end｜>"
            ),
            empty_arguments_output=(
                "<｜tool_calls_begin｜><｜tool_call_begin｜>"
                '<steptml:invoke name="refresh"></steptml:invoke>'
                "<｜tool_call_end｜><｜tool_calls_end｜>"
            ),
            surrounding_text_output=(
                "Let me check the weather for you.\n\n"
                "<｜tool_calls_begin｜><｜tool_call_begin｜>"
                '<steptml:invoke name="get_weather">'
                '<steptml:parameter name="city">Tokyo</steptml:parameter>'
                "</steptml:invoke><｜tool_call_end｜><｜tool_calls_end｜>\n\n"
                "I'll get that information."
            ),
            escaped_strings_output=(
                "<｜tool_calls_begin｜><｜tool_call_begin｜>"
                '<steptml:invoke name="test_function">'
                '<steptml:parameter name="quoted">He said "hello"</steptml:parameter>'
                '<steptml:parameter name="path">C:\\Users\\file.txt</steptml:parameter>'
                '<steptml:parameter name="newline">line1\nline2</steptml:parameter>'
                "</steptml:invoke><｜tool_call_end｜><｜tool_calls_end｜>"
            ),
            malformed_input_outputs=[
                (
                    "<｜tool_calls_begin｜><｜tool_call_begin｜>"
                    '<steptml:invoke name="func">'
                ),
                (
                    '<｜tool_call_begin｜><steptml:invoke name="func">'
                    "</steptml:invoke><｜tool_call_end｜>"
                ),
            ],
            # Expected results
            single_tool_call_expected_name="get_weather",
            single_tool_call_expected_args={"city": "Tokyo"},
            parallel_tool_calls_count=2,
            parallel_tool_calls_names=["get_weather", "get_time"],
            # xfail markers
            xfail_nonstreaming={
                "test_single_tool_call_simple_args": (
                    "Step3 parser non-streaming has bugs"
                ),
                "test_parallel_tool_calls": ("Step3 parser non-streaming has bugs"),
                "test_various_data_types": "Step3 parser non-streaming has bugs",
                "test_empty_arguments": "Step3 parser non-streaming has bugs",
                "test_surrounding_text": "Step3 parser non-streaming has bugs",
                "test_escaped_strings": "Step3 parser non-streaming has bugs",
            },
            xfail_streaming={
                "test_parallel_tool_calls": (
                    "Step3 parser has significant bugs in both streaming "
                    "and non-streaming"
                ),
                "test_streaming_reconstruction": (
                    "Step3 parser non-streaming has bugs, so streaming "
                    "doesn't match non-streaming"
                ),
            },
            supports_typed_arguments=False,
        )


class TestStep3RecordsStreamedArgs:
    """step3 must record what it streamed, or the finalizer re-sends it.

    ``DelegatingParser.finalize_generation`` runs ``_append_unstreamed_tool_args``
    on every finished stream, appending ``get_remaining_unstreamed_args()`` to
    the last tool call. That helper diffs ``prev_tool_call_arr[-1]["arguments"]``
    against ``streamed_args_for_tool[-1]``, so a parser that emits arguments
    without recording them gets the payload appended a second time and the
    client sees concatenated JSON.
    """

    SINGLE_CALL = (
        "<｜tool_calls_begin｜><｜tool_call_begin｜>"
        '<steptml:invoke name="get_weather">'
        '<steptml:parameter name="city">Paris</steptml:parameter>'
        "</steptml:invoke><｜tool_call_end｜><｜tool_calls_end｜>"
    )

    @pytest.fixture(scope="class")
    def tokenizer(self) -> TokenizerLike:
        return get_tokenizer("stepfun-ai/step3")

    @pytest.fixture
    def parser(self, tokenizer) -> Step3ToolParser:
        return Step3ToolParser(tokenizer)

    @staticmethod
    def _stream(parser: Step3ToolParser, text: str) -> str:
        """Feed ``text`` one character at a time; return the arguments emitted."""
        emitted: list[str] = []
        previous = ""
        for i in range(1, len(text) + 1):
            current = text[:i]
            delta = parser.extract_tool_calls_streaming(
                previous_text=previous,
                current_text=current,
                delta_text=current[len(previous) :],
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=None,
            )
            if delta is not None and delta.tool_calls:
                for tool_call in delta.tool_calls:
                    if tool_call.function and tool_call.function.arguments:
                        emitted.append(tool_call.function.arguments)
            previous = current
        return "".join(emitted)

    def test_streamed_arguments_are_recorded(self, parser):
        emitted = self._stream(parser, self.SINGLE_CALL)

        assert emitted, "parser emitted no arguments"
        assert parser.streamed_args_for_tool == [emitted]

    def test_nothing_is_owed_after_a_complete_call(self, parser):
        self._stream(parser, self.SINGLE_CALL)

        assert parser.get_remaining_unstreamed_args() == ""

    def test_finalizer_does_not_duplicate_arguments(self, parser):
        # What the client ends up with: the streamed deltas plus whatever the
        # finalizer decides is still owed.
        emitted = self._stream(parser, self.SINGLE_CALL)

        final = emitted + parser.get_remaining_unstreamed_args()

        assert json.loads(final) == {"city": "Paris"}
