# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Generator

import partial_json_parser
import pytest
from partial_json_parser.core.options import Allow

from vllm.entrypoints.openai.engine.protocol import DeltaMessage, FunctionCall, ToolCall
from vllm.tokenizers import TokenizerLike, get_tokenizer
from vllm.tokenizers.detokenizer_utils import detokenize_incrementally
from vllm.tool_parsers.jamba_tool_parser import JambaToolParser

MODEL = "ai21labs/Jamba-tiny-dev"


@pytest.fixture(scope="module")
def jamba_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)


@pytest.fixture
def jamba_tool_parser(jamba_tokenizer):
    return JambaToolParser(jamba_tokenizer)


def assert_tool_calls(
    actual_tool_calls: list[ToolCall], expected_tool_calls: list[ToolCall]
):
    assert len(actual_tool_calls) == len(expected_tool_calls)

    for actual_tool_call, expected_tool_call in zip(
        actual_tool_calls, expected_tool_calls
    ):
        assert isinstance(actual_tool_call.id, str)
        assert len(actual_tool_call.id) > 16

        assert actual_tool_call.type == "function"
        assert actual_tool_call.function == expected_tool_call.function


def stream_delta_message_generator(
    jamba_tool_parser: JambaToolParser,
    jamba_tokenizer: TokenizerLike,
    model_output: str,
) -> Generator[DeltaMessage, None, None]:
    all_token_ids = jamba_tokenizer.encode(model_output, add_special_tokens=False)

    previous_text = ""
    previous_tokens = None
    prefix_offset = 0
    read_offset = 0
    for i, delta_token in enumerate(all_token_ids):
        delta_token_ids = [delta_token]
        previous_token_ids = all_token_ids[:i]
        current_token_ids = all_token_ids[: i + 1]

        (new_tokens, delta_text, new_prefix_offset, new_read_offset) = (
            detokenize_incrementally(
                tokenizer=jamba_tokenizer,
                all_input_ids=current_token_ids,
                prev_tokens=previous_tokens,
                prefix_offset=prefix_offset,
                read_offset=read_offset,
                skip_special_tokens=False,
                spaces_between_special_tokens=True,
            )
        )

        current_text = previous_text + delta_text

        delta_message = jamba_tool_parser.extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request=None,  # type: ignore[arg-type]
        )
        if delta_message:
            yield delta_message

        previous_text = current_text
        previous_tokens = (
            previous_tokens + new_tokens if previous_tokens else new_tokens
        )
        prefix_offset = new_prefix_offset
        read_offset = new_read_offset


def test_extract_tool_calls_no_tools(jamba_tool_parser):
    model_output = "This is a test"
    extracted_tool_calls = jamba_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]
    assert not extracted_tool_calls.tools_called
    assert extracted_tool_calls.tool_calls == []
    assert extracted_tool_calls.content == model_output


@pytest.mark.parametrize(
    ids=[
        "single_tool",
        "single_tool_with_content",
        "parallel_tools",
    ],
    argnames=["model_output", "expected_tool_calls", "expected_content"],
    argvalues=[
        (
            """ <tool_calls>[\n    {"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}\n]</tool_calls>""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}
                        ),
                    )
                )
            ],
            None,
        ),
        (
            """ Sure! let me call the tool for you.<tool_calls>[\n    {"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}\n]</tool_calls>""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}
                        ),
                    )
                )
            ],
            " Sure! let me call the tool for you.",
        ),
        (
            """ <tool_calls>[\n    {"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}},\n    {"name": "get_current_weather", "arguments": {"city": "Orlando", "state": "FL", "unit": "fahrenheit"}}\n]</tool_calls>""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}
                        ),
                    )
                ),
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {"city": "Orlando", "state": "FL", "unit": "fahrenheit"}
                        ),
                    )
                ),
            ],
            None,
        ),
    ],
)
def test_extract_tool_calls(
    jamba_tool_parser, model_output, expected_tool_calls, expected_content
):
    extracted_tool_calls = jamba_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]
    assert extracted_tool_calls.tools_called

    assert_tool_calls(extracted_tool_calls.tool_calls, expected_tool_calls)

    assert extracted_tool_calls.content == expected_content


@pytest.mark.parametrize(
    ids=[
        "no_tools",
        "single_tool",
        "single_tool_with_content",
        "parallel_tools",
    ],
    argnames=["model_output", "expected_tool_calls", "expected_content"],
    argvalues=[
        ("""This is a test""", [], """This is a test"""),
        (
            """ <tool_calls>[\n    {"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}\n]</tool_calls>""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}
                        ),
                    )
                )
            ],
            " ",
        ),
        (
            """ Sure! let me call the tool for you.<tool_calls>[\n    {"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}\n]</tool_calls>""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}
                        ),
                    )
                )
            ],
            " Sure! let me call the tool for you.",
        ),
        (
            """ <tool_calls>[\n    {"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}},\n    {"name": "get_current_weather", "arguments": {"city": "Orlando", "state": "FL", "unit": "fahrenheit"}}\n]</tool_calls>""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}
                        ),
                    )
                ),
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {"city": "Orlando", "state": "FL", "unit": "fahrenheit"}
                        ),
                    )
                ),
            ],
            " ",
        ),
    ],
)
def test_extract_tool_calls_streaming(
    jamba_tool_parser,
    jamba_tokenizer,
    model_output,
    expected_tool_calls,
    expected_content,
):
    other_content: str = ""
    function_names: list[str] = []
    function_args_strs: list[str] = []
    tool_call_idx: int = -1
    tool_call_ids: list[str | None] = []

    for delta_message in stream_delta_message_generator(
        jamba_tool_parser, jamba_tokenizer, model_output
    ):
        # role should never be streamed from tool parser
        assert not delta_message.role

        if delta_message.content:
            other_content += delta_message.content

        streamed_tool_calls = delta_message.tool_calls

        if streamed_tool_calls and len(streamed_tool_calls) > 0:
            # make sure only one diff is present - correct even for parallel
            assert len(streamed_tool_calls) == 1
            tool_call = streamed_tool_calls[0]

            # if a new tool is being called, set up empty arguments
            if tool_call.index != tool_call_idx:
                tool_call_idx = tool_call.index
                function_args_strs.append("")
                tool_call_ids.append(None)

            # if a tool call ID is streamed, make sure one hasn't been already
            if tool_call.id and not tool_call_ids[tool_call.index]:
                tool_call_ids[tool_call.index] = tool_call.id

            # if parts of the function start being streamed
            if tool_call.function:
                # if the function name is defined, set it. it should be streamed
                # IN ENTIRETY, exactly one time.
                if tool_call.function.name:
                    assert isinstance(tool_call.function.name, str)
                    function_names.append(tool_call.function.name)

                if tool_call.function.arguments:
                    # make sure they're a string and then add them to the list
                    assert isinstance(tool_call.function.arguments, str)

                    function_args_strs[tool_call.index] += tool_call.function.arguments

    assert other_content == expected_content

    actual_tool_calls = [
        ToolCall(
            id=tool_call_id,
            function=FunctionCall(
                name=function_name,
                arguments=partial_json_parser.ensure_json(
                    function_args_str, Allow.OBJ | Allow.STR
                ),
            ),
        )
        for tool_call_id, function_name, function_args_str in zip(
            tool_call_ids, function_names, function_args_strs
        )
    ]
    assert_tool_calls(actual_tool_calls, expected_tool_calls)


def test_flush_on_new_tool_replaces_only_leading_occurrence(jamba_tool_parser):
    # Regression test for the "starting a new tool" flush path: the leftover,
    # not-yet-streamed tail of the PREVIOUS tool's arguments is computed with
    # `json.dumps(diff).replace(streamed_prefix, "")`. Without `count=1`, that
    # `.replace()` removes EVERY occurrence of `streamed_prefix`, not just the
    # leading one -- so if the streamed prefix's text recurs elsewhere in the
    # arguments (e.g. a nested object that repeats a top-level key/value, a
    # realistic pattern for find/replace-style tools), the recurrence is
    # silently deleted from the flushed diff too, corrupting the JSON. State is
    # set up manually to isolate this one code path.
    jamba_tool_parser.current_tool_id = 0
    jamba_tool_parser.current_tool_name_sent = True
    jamba_tool_parser.streamed_args_for_tool = ['{"find": "cat"']
    jamba_tool_parser.prev_tool_call_arr = [
        {"name": "find_replace", "arguments": {"find": "cat"}}
    ]

    previous_text = (
        ' <tool_calls>[\n    {"name": "find_replace", "arguments": '
        '{"find": "cat", "options": {"find": "cat", "replace": "dog"}}},\n'
        '    {"name": "get_current_weather", "arguments": '
    )
    current_text = previous_text + "{"

    result = jamba_tool_parser.extract_tool_calls_streaming(
        previous_text=previous_text,
        current_text=current_text,
        delta_text="{",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=None,  # type: ignore[arg-type]
    )

    assert result is not None
    assert result.tool_calls
    flushed = result.tool_calls[0]
    assert flushed.index == 0
    assert flushed.function is not None and flushed.function.arguments

    assert json.loads(jamba_tool_parser.streamed_args_for_tool[0]) == {
        "find": "cat",
        "options": {"find": "cat", "replace": "dog"},
    }
