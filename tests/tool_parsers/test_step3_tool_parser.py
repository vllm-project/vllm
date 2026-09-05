# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import json

import pytest

from tests.tool_parsers.common_tests import (
    ToolParserTestConfig,
    ToolParserTests,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tokenizers import TokenizerLike, get_tokenizer
from vllm.tool_parsers import ToolParserManager


def _stream_in_deltas(
    tokenizer: TokenizerLike, output: str, boundaries: list[tuple[int, int]]
) -> tuple[str, list[tuple[str, str]]]:
    """Stream `output` through a fresh parser, batching tokens as given.

    Returns the reconstructed content and (name, arguments) of each tool call.
    """
    parser = ToolParserManager.get_tool_parser("step3")(tokenizer)
    request = ChatCompletionRequest(messages=[], model="test")

    token_ids = tokenizer.encode(output, add_special_tokens=False)
    token_texts, decoded = [], ""
    for i in range(1, len(token_ids) + 1):
        current = tokenizer.decode(token_ids[:i])
        token_texts.append(current[len(decoded) :])
        decoded = current

    previous_text = ""
    previous_ids: list[int] = []
    content = ""
    calls: dict[int, dict[str, str]] = {}
    for start, end in boundaries:
        delta_text = "".join(token_texts[start:end])
        delta_ids = token_ids[start:end]
        current_text = previous_text + delta_text
        current_ids = previous_ids + delta_ids

        message = parser.extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_ids,
            current_ids,
            delta_ids,
            request,
        )
        if message is not None:
            content += message.content or ""
            for tool_call in message.tool_calls or []:
                call = calls.setdefault(tool_call.index, {"name": "", "arguments": ""})
                call["name"] += tool_call.function.name or ""
                call["arguments"] += tool_call.function.arguments or ""

        previous_text, previous_ids = current_text, current_ids

    return content, [(c["name"], c["arguments"]) for c in calls.values()]


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

    @pytest.mark.parametrize(
        "output_attr", ["single_tool_call_output", "surrounding_text_output"]
    )
    def test_streaming_is_independent_of_delta_boundaries(
        self,
        test_config: ToolParserTestConfig,
        tokenizer: TokenizerLike,
        output_attr: str,
    ):
        """Streaming must not depend on how tokens are batched into deltas.

        The name and the arguments used to be emitted from separate calls, so
        a delta carrying a whole tool call yielded empty arguments, and a delta
        carrying both leading text and a tool call yielded no tool call at all.
        Multi-token deltas are normal in production (``--stream-interval``,
        speculative decoding, scheduler batching).
        """
        output = getattr(test_config, output_attr)
        num_tokens = len(tokenizer.encode(output, add_special_tokens=False))

        baseline = _stream_in_deltas(
            tokenizer, output, [(i, i + 1) for i in range(num_tokens)]
        )
        assert baseline[1] == [
            (
                test_config.single_tool_call_expected_name,
                json.dumps(test_config.single_tool_call_expected_args),
            )
        ]

        for split in range(1, num_tokens):
            batched = _stream_in_deltas(
                tokenizer, output, [(0, split), (split, num_tokens)]
            )
            assert batched == baseline, f"differs when split after token {split}"

        assert _stream_in_deltas(tokenizer, output, [(0, num_tokens)]) == baseline
