# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Generator
from typing import Any, Literal
from unittest.mock import MagicMock, PropertyMock, patch

import partial_json_parser
import pytest
from mistral_common.protocol.instruct.messages import AssistantMessage
from mistral_common.protocol.instruct.request import InstructRequest
from mistral_common.protocol.instruct.tool_calls import FunctionCall, ToolCall
from partial_json_parser.core.options import Allow

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    DeltaToolCall,
    FunctionDefinition,
)
from vllm.tokenizers import TokenizerLike, get_tokenizer
from vllm.tokenizers.detokenizer_utils import detokenize_incrementally
from vllm.tokenizers.mistral import MistralTokenizer
from vllm.tool_parsers.mistral_tool_parser import (
    ChatCompletionToolsParam,
    MistralGrammarFactory,
    MistralToolParser,
    PostV11ToolsLarkConverter,
)


@pytest.fixture(scope="module")
def mistral_pre_v11_tokenizer():
    MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
    return get_tokenizer(tokenizer_name=MODEL)


@pytest.fixture(scope="module")
def mistral_tokenizer():
    MODEL = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    return get_tokenizer(tokenizer_name=MODEL, tokenizer_mode="mistral")


@pytest.fixture
def mistral_pre_v11_tool_parser(mistral_pre_v11_tokenizer):
    return MistralToolParser(mistral_pre_v11_tokenizer)


@pytest.fixture
def mistral_tool_parser(mistral_tokenizer):
    return MistralToolParser(mistral_tokenizer)


def assert_tool_calls(
    actual_tool_calls: list[ToolCall] | list[DeltaToolCall],
    expected_tool_calls: list[ToolCall],
):
    assert len(actual_tool_calls) == len(expected_tool_calls)

    for actual_tool_call, expected_tool_call in zip(
        actual_tool_calls, expected_tool_calls
    ):
        assert isinstance(actual_tool_call.id, str)
        assert len(actual_tool_call.id) == 9

        if isinstance(actual_tool_call, ToolCall):
            assert actual_tool_call.type == "function"
        elif isinstance(actual_tool_call, DeltaToolCall):
            assert actual_tool_call.function is not None
            assert actual_tool_call.function.name is not None
            assert actual_tool_call.function.arguments is not None
        assert actual_tool_call.function is not None
        assert actual_tool_call.function.name == expected_tool_call.function.name, (
            f"got wrong function name:${actual_tool_call.function.name}"
        )
        assert (
            actual_tool_call.function.arguments == expected_tool_call.function.arguments
        ), f"got wrong function argument:${actual_tool_call.function.arguments}"


def fix_tool_call_tokenization(
    tokens: list[int],
    mistral_tool_parser: MistralToolParser,
    mistral_tokenizer: TokenizerLike,
):
    """
    Replaces the textual token sequence for [TOOL_CALLS]
    with its single special token ID.
    """
    textual_tool_call_token_ids = mistral_tokenizer.encode(
        text=mistral_tool_parser.bot_token,
        add_special_tokens=False,
    )
    # textual_tool_call_token_ids must not contain special tokens like bos, eos etc
    special_tool_call_token_ids = [mistral_tool_parser.bot_token_id]

    # If the input is too short to contain the sequence, no replacement is possible
    if not tokens or len(tokens) < len(textual_tool_call_token_ids):
        return tokens

    result_tokens = []
    i = 0
    target_len = len(textual_tool_call_token_ids)

    while i < len(tokens):
        # Check if the slice from the current position matches the target sequence
        if tokens[i : i + target_len] == textual_tool_call_token_ids:
            # If it matches, add the replacement and jump the index forward
            result_tokens.extend(special_tool_call_token_ids)
            i += target_len
        else:
            # Otherwise, just add the current token and move to the next one
            result_tokens.append(tokens[i])
            i += 1

    return result_tokens


def stream_delta_message_generator(
    mistral_tool_parser: MistralToolParser,
    mistral_tokenizer: TokenizerLike,
    model_output: str | None,
    tools: list[tuple[str, str]] | None,
) -> Generator[DeltaMessage, None, None]:
    if (
        isinstance(mistral_tokenizer, MistralTokenizer)
        and mistral_tokenizer.version >= 11
    ):
        # With the newer versions of the tokenizer,
        # we cannot tokenize free text
        # so we need to create a list of messages to get tokenized
        assert tools is not None
        assistant_msg = AssistantMessage(
            tool_calls=[
                ToolCall(
                    function=FunctionCall(
                        name=name,
                        arguments=arg,
                    )
                )
                for (name, arg) in tools
            ],
        )
        request = InstructRequest(
            messages=[assistant_msg],
        )
        all_token_ids = mistral_tokenizer.instruct.encode_instruct(request).tokens
    else:
        # Older versions of the tokenizer are
        # able to encode directly the model's output (free text) into tokens
        assert model_output is not None
        all_token_ids = mistral_tokenizer.encode(model_output, add_special_tokens=False)

    all_token_ids = fix_tool_call_tokenization(
        all_token_ids, mistral_tool_parser, mistral_tokenizer
    )

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
                tokenizer=mistral_tokenizer,
                all_input_ids=current_token_ids,
                prev_tokens=previous_tokens,
                prefix_offset=prefix_offset,
                read_offset=read_offset,
                skip_special_tokens=isinstance(mistral_tokenizer, MistralTokenizer),
                spaces_between_special_tokens=True,
            )
        )

        current_text = previous_text + delta_text

        delta_message = mistral_tool_parser.extract_tool_calls_streaming(
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


def test_extract_tool_calls_no_tools(mistral_pre_v11_tool_parser):
    model_output = "This is a test"
    extracted_tool_calls = mistral_pre_v11_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]
    assert not extracted_tool_calls.tools_called
    assert extracted_tool_calls.tool_calls == []
    assert extracted_tool_calls.content == model_output


@pytest.mark.parametrize(
    ids=[
        "single_tool_add",
        "single_tool_weather",
        "argument_before_name",
        "argument_before_name_and_name_in_argument",
    ],
    argnames=["model_output", "expected_tool_calls", "expected_content"],
    argvalues=[
        (
            """[TOOL_CALLS][{"name": "add", "arguments":{"a": 3.5, "b": 4}}]""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="add", arguments=json.dumps({"a": 3.5, "b": 4})
                    )
                )
            ],
            None,
        ),
        (
            """[TOOL_CALLS] [{"name": "get_current_weather", "arguments":{"city": "San Francisco", "state": "CA", "unit": "celsius"}}]""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {"city": "San Francisco", "state": "CA", "unit": "celsius"}
                        ),
                    )
                )
            ],
            None,
        ),
        (
            """[TOOL_CALLS] [{"arguments":{"city": "San Francisco", "state": "CA", "unit": "celsius"}, "name": "get_current_weather"}]""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {"city": "San Francisco", "state": "CA", "unit": "celsius"}
                        ),
                    )
                )
            ],
            None,
        ),
        (
            """[TOOL_CALLS] [{"arguments":{"name": "John Doe"}, "name": "get_age"}]""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_age",
                        arguments=json.dumps(
                            {
                                "name": "John Doe",
                            }
                        ),
                    )
                )
            ],
            None,
        ),
    ],
)
def test_extract_tool_calls_pre_v11_tokenizer(
    mistral_pre_v11_tool_parser, model_output, expected_tool_calls, expected_content
):
    extracted_tool_calls = mistral_pre_v11_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]
    assert extracted_tool_calls.tools_called

    assert_tool_calls(extracted_tool_calls.tool_calls, expected_tool_calls)

    assert extracted_tool_calls.content == expected_content


@pytest.mark.parametrize(
    ids=[
        "single_tool_add",
        "single_tool_weather",
        "multiple_tool_calls",
        "complex",
        "wrong_json",
    ],
    argnames=["model_output", "expected_tool_calls", "expected_content"],
    argvalues=[
        (
            """[TOOL_CALLS]add_this_and_that{"a": 3.5, "b": 4}""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="add_this_and_that",
                        arguments=json.dumps({"a": 3.5, "b": 4}),
                    )
                )
            ],
            None,
        ),
        (
            """[TOOL_CALLS]get_current_weather{"city": "San Francisco", "state": "CA", "unit": "celsius"}""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {"city": "San Francisco", "state": "CA", "unit": "celsius"}
                        ),
                    )
                )
            ],
            None,
        ),
        (
            """[TOOL_CALLS]add{"a": 3.5, "b": 4}[TOOL_CALLS]multiply{"a": 3, "b": 6}""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="add", arguments=json.dumps({"a": 3.5, "b": 4})
                    )
                ),
                ToolCall(
                    function=FunctionCall(
                        name="multiply", arguments=json.dumps({"a": 3, "b": 6})
                    )
                ),
            ],
            None,
        ),
        (
            # Complex
            """hi{hi[TOOL_CALLS]bash{"command": "print(\\"hello world!\\")\\nre.compile(r\'{}\')""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="bash",
                        arguments=json.dumps(
                            {"command": "print(\"hello world!\")\nre.compile(r'{}')"}
                        )[:-2],
                    )
                )
            ],
            "hi{hi",
        ),
        (
            # Wrong json
            """hi{hi[TOOL_CALLS]bash{"command": "print(\\"hello world!\\")\\nre.compile(r\'{}\')"}""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="bash",
                        arguments=json.dumps(
                            {"command": "print(\"hello world!\")\nre.compile(r'{}')"}
                        ),
                    )
                )
            ],
            "hi{hi",
        ),
    ],
)
def test_extract_tool_calls(
    mistral_tool_parser, model_output, expected_tool_calls, expected_content
):
    extracted_tool_calls = mistral_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]
    assert extracted_tool_calls.tools_called

    assert_tool_calls(extracted_tool_calls.tool_calls, expected_tool_calls)

    assert extracted_tool_calls.content == expected_content


def _test_extract_tool_calls_streaming(
    tool_parser, tokenizer, model_output, tools, expected_tool_calls, expected_content
):
    other_content: str = ""
    function_names: list[str] = []
    function_args_strs: list[str] = []
    tool_call_idx: int = -1
    tool_call_ids: list[str | None] = []

    for delta_message in stream_delta_message_generator(
        tool_parser, tokenizer, model_output, tools
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

            assert len(tool_parser.prev_tool_call_arr) > 0

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


@pytest.mark.parametrize(
    ids=[
        "no_tools",
        "single_tool_add",
        "single_tool_add_strings",
        "single_tool_weather",
        "argument_before_name",
        "argument_before_name_and_name_in_argument",
        "multiple_tools",
    ],
    argnames=["model_output", "expected_tool_calls", "expected_content"],
    argvalues=[
        ("""This is a test""", [], """This is a test"""),
        (
            """[TOOL_CALLS]  [ {"name":"add" , "arguments" : {"a": 3, "b": 4} } ]""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="add", arguments=json.dumps({"a": 3, "b": 4})
                    )
                )
            ],
            "",
        ),
        (
            """[TOOL_CALLS] [{"name": "add", "arguments":{"a": "3", "b": "4"}}]""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="add", arguments=json.dumps({"a": "3", "b": "4"})
                    )
                )
            ],
            "",
        ),
        (
            """[TOOL_CALLS] [{"name": "get_current_weather", "arguments": {"city": "San Francisco", "state": "CA", "unit": "celsius"}}]""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {"city": "San Francisco", "state": "CA", "unit": "celsius"}
                        ),
                    )
                )
            ],
            "",
        ),
        (
            """[TOOL_CALLS] [{"arguments": {"city": "San Francisco", "state": "CA", "unit": "celsius"}, "name": "get_current_weather"}]""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {"city": "San Francisco", "state": "CA", "unit": "celsius"}
                        ),
                    )
                )
            ],
            "",
        ),
        (
            """[TOOL_CALLS] [{"arguments": {"name": "John Doe"}, "name": "get_age"}]""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_age",
                        arguments=json.dumps(
                            {
                                "name": "John Doe",
                            }
                        ),
                    )
                )
            ],
            "",
        ),
        (
            """[TOOL_CALLS] [{"name": "add", "arguments": {"a": 3.5, "b": 4}}, {"name": "get_current_weather", "arguments":{"city": "San Francisco", "state": "CA", "unit": "celsius"}}]""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="add", arguments=json.dumps({"a": 3.5, "b": 4})
                    )
                ),
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {"city": "San Francisco", "state": "CA", "unit": "celsius"}
                        ),
                    )
                ),
            ],
            "",
        ),
    ],
)
def test_extract_tool_calls_streaming_pre_v11_tokenizer(
    mistral_pre_v11_tool_parser,
    mistral_pre_v11_tokenizer,
    model_output,
    expected_tool_calls,
    expected_content,
):
    _test_extract_tool_calls_streaming(
        mistral_pre_v11_tool_parser,
        mistral_pre_v11_tokenizer,
        model_output,
        None,
        expected_tool_calls,
        expected_content,
    )


@pytest.mark.parametrize(
    ids=[
        "single_tool_add",
        "single_tool_add_strings",
        "multiple_tools",
    ],
    argnames=["tools", "expected_tool_calls", "expected_content"],
    argvalues=[
        (
            [("add", '{"a": 3, "b": 4}')],
            # [TOOL_CALLS]add{"a": 3, "b": 4}
            [
                ToolCall(
                    function=FunctionCall(
                        name="add", arguments=json.dumps({"a": 3, "b": 4})
                    )
                )
            ],
            "",
        ),
        (
            [("add_two_strings", '{"a": "3", "b": "4"}')],
            # [TOOL_CALLS]add_two_strings{"a": "3", "b": "4"}
            [
                ToolCall(
                    function=FunctionCall(
                        name="add_two_strings",
                        arguments=json.dumps({"a": "3", "b": "4"}),
                    )
                )
            ],
            "",
        ),
        (
            [
                ("add", '{"a": 3.5, "b": 4}'),
                (
                    "get_current_weather",
                    '{"city": "San Francisco", "state": "CA", "unit": "celsius"}',  # noqa: E501
                ),
            ],
            # [TOOL_CALLS]add{"a": 3.5, "b": 4}[TOOL_CALLS]get_current_weather{"city": "San Francisco", "state": "CA", "unit": "celsius"}  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="add", arguments=json.dumps({"a": 3.5, "b": 4})
                    )
                ),
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {"city": "San Francisco", "state": "CA", "unit": "celsius"}
                        ),
                    )
                ),
            ],
            "",
        ),
    ],
)
def test_extract_tool_calls_streaming(
    mistral_tool_parser,
    mistral_tokenizer,
    tools,
    expected_tool_calls,
    expected_content,
):
    _test_extract_tool_calls_streaming(
        mistral_tool_parser,
        mistral_tokenizer,
        None,
        tools,
        expected_tool_calls,
        expected_content,
    )


@pytest.mark.parametrize(
    ids=[
        "single_tool_add",
        "single_tool_weather",
        "multiple_tool_calls",
        "content_before_tool",
        "complex",
    ],
    argnames=["model_output", "expected_tool_calls", "expected_content"],
    argvalues=[
        (
            """[TOOL_CALLS]add_this_and_that{"a": 3.5, "b": 4}""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="add_this_and_that",
                        arguments=json.dumps({"a": 3.5, "b": 4}),
                    )
                )
            ],
            "",
        ),
        (
            """[TOOL_CALLS]get_current_weather{"city": "San Francisco", "state": "CA", "unit": "celsius"}""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {"city": "San Francisco", "state": "CA", "unit": "celsius"}
                        ),
                    )
                )
            ],
            "",
        ),
        (
            """[TOOL_CALLS]add{"a": 3.5, "b": 4}[TOOL_CALLS]multiply{"a": 3, "b": 6}""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="add", arguments=json.dumps({"a": 3.5, "b": 4})
                    )
                ),
                ToolCall(
                    function=FunctionCall(
                        name="multiply", arguments=json.dumps({"a": 3, "b": 6})
                    )
                ),
            ],
            "",
        ),
        (
            # Additional content should not be after the tool calls
            """bla[TOOL_CALLS]add_this_and_that{"a": 3.5, "b": 4}""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="add_this_and_that",
                        arguments=json.dumps({"a": 3.5, "b": 4}),
                    )
                )
            ],
            "bla",
        ),
        (
            # Complex
            """hi{hi[TOOL_CALLS]bash{"command": "print(\\"hello world!\\")\\nre.compile(r\'{}\')"}""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="bash",
                        arguments=json.dumps(
                            {"command": "print(\"hello world!\")\nre.compile(r'{}')"}
                        ),
                    )
                )
            ],
            "hi{hi",
        ),
    ],
)
def test_extract_tool_calls_streaming_one_chunk(
    mistral_tool_parser,
    mistral_tokenizer,
    model_output,
    expected_tool_calls,
    expected_content,
):
    if isinstance(mistral_tokenizer, MistralTokenizer):
        all_token_ids = mistral_tokenizer.encode(model_output)
    else:
        all_token_ids = mistral_tokenizer.encode(model_output, add_special_tokens=False)
    all_token_ids = fix_tool_call_tokenization(
        all_token_ids, mistral_tool_parser, mistral_tokenizer
    )

    delta_message = mistral_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=model_output,
        delta_text=model_output,
        previous_token_ids=[],
        current_token_ids=all_token_ids,
        delta_token_ids=all_token_ids,
        request=None,
    )  # type: ignore[arg-type]
    assert isinstance(delta_message, DeltaMessage)
    assert len(delta_message.tool_calls) == len(expected_tool_calls)

    assert_tool_calls(delta_message.tool_calls, expected_tool_calls)

    if delta_message.content is None:
        assert expected_content == ""
    else:
        assert delta_message.content == expected_content


@pytest.mark.parametrize(
    ids=[
        "no_tools",
        "single_tool_add",
        "single_tool_add_strings",
        "single_tool_weather",
        "argument_before_name",
        "argument_before_name_and_name_in_argument",
        "multiple_tools",
    ],
    argnames=["model_output", "expected_tool_calls", "expected_content"],
    argvalues=[
        ("""This is a test""", [], """This is a test"""),
        (
            """[TOOL_CALLS]  [ {"name":"add" , "arguments" : {"a": 3, "b": 4} } ]""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="add", arguments=json.dumps({"a": 3, "b": 4})
                    )
                )
            ],
            "",
        ),
        (
            """[TOOL_CALLS] [{"name": "add", "arguments":{"a": "3", "b": "4"}}]""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="add", arguments=json.dumps({"a": "3", "b": "4"})
                    )
                )
            ],
            "",
        ),
        (
            """[TOOL_CALLS] [{"name": "get_current_weather", "arguments": {"city": "San Francisco", "state": "CA", "unit": "celsius"}}]""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {"city": "San Francisco", "state": "CA", "unit": "celsius"}
                        ),
                    )
                )
            ],
            "",
        ),
        (
            """[TOOL_CALLS] [{"arguments": {"city": "San Francisco", "state": "CA", "unit": "celsius"}, "name": "get_current_weather"}]""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {"city": "San Francisco", "state": "CA", "unit": "celsius"}
                        ),
                    )
                )
            ],
            "",
        ),
        (
            """[TOOL_CALLS] [{"arguments": {"name": "John Doe"}, "name": "get_age"}]""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_age",
                        arguments=json.dumps(
                            {
                                "name": "John Doe",
                            }
                        ),
                    )
                )
            ],
            "",
        ),
        (
            """[TOOL_CALLS] [{"arguments": {"a": 3.5, "b": 4}, "name": "add"}, {"arguments":{"city": "San Francisco", "state": "CA", "unit": "celsius"}, "name": "get_current_weather"}]""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="add", arguments=json.dumps({"a": 3.5, "b": 4})
                    )
                ),
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {"city": "San Francisco", "state": "CA", "unit": "celsius"}
                        ),
                    )
                ),
            ],
            "",
        ),
    ],
)
def test_extract_tool_calls_streaming_pre_v11_tokenizer_one_chunk(
    mistral_pre_v11_tool_parser,
    mistral_pre_v11_tokenizer,
    model_output,
    expected_tool_calls,
    expected_content,
):
    if isinstance(mistral_pre_v11_tokenizer, MistralTokenizer):
        all_token_ids = mistral_pre_v11_tokenizer.encode(model_output)
    else:
        all_token_ids = mistral_pre_v11_tokenizer.encode(
            model_output, add_special_tokens=False
        )
    all_token_ids = fix_tool_call_tokenization(
        all_token_ids, mistral_pre_v11_tool_parser, mistral_pre_v11_tokenizer
    )

    delta_message = mistral_pre_v11_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=model_output,
        delta_text=model_output,
        previous_token_ids=[],
        current_token_ids=all_token_ids,
        delta_token_ids=all_token_ids,
        request=None,
    )  # type: ignore[arg-type]
    assert isinstance(delta_message, DeltaMessage)
    assert len(delta_message.tool_calls) == len(expected_tool_calls)

    assert_tool_calls(delta_message.tool_calls, expected_tool_calls)

    if delta_message.content is None:
        assert expected_content == ""
    else:
        assert delta_message.content == expected_content


def _create_tool(
    name: str, parameters: dict[str, Any], strict: bool = False
) -> ChatCompletionToolsParam:
    """Helper function to create a tool with optional strict attribute"""
    func = FunctionDefinition(
        name=name,
        description="test",
        parameters=parameters,
    )
    if strict:
        func.strict = True  # type: ignore[attr-defined]
    return ChatCompletionToolsParam(function=func)


class TestToolsLarkConverter:
    @pytest.mark.parametrize(
        ("tool", "expected"),
        [
            (
                _create_tool("function", {"parameter": 1}, strict=True),
                {"parameter": 1},
            ),
            (
                _create_tool("function", {"parameter": 1}, strict=False),
                {"type": "object"},
            ),
            (
                _create_tool("function", {}, strict=True),
                {"type": "object", "properties": {}, "additionalProperties": False},
            ),
        ],
    )
    def test_get_args_json(
        self, tool: ChatCompletionToolsParam, expected: dict[str, Any]
    ) -> None:
        assert PostV11ToolsLarkConverter().get_args_json(tool=tool) == expected

    @pytest.mark.parametrize(
        "tools,tokenizer_version,mode,parallel_tool_calls,expected",
        [
            # mode="none" always returns empty string
            (
                [_create_tool("non_strict_func", {"param": "value"}, strict=False)],
                7,
                "none",
                True,
                "",
            ),
            # Pre-v11: non-strict parallel vs no-parallel (maxItems difference)
            (
                [_create_tool("non_strict_func", {"param": "value"}, strict=False)],
                7,
                "auto",
                True,
                '<TOOL_CALLS> SAFE_WS? fcall_array SAFE_WS?\nfcall_array: %json {"minItems": 1, "type": "array", "items": {"type": "object", "additionalProperties": false, "properties": {"name": {"type": "string", "enum": ["non_strict_func"]}, "arguments": {"type": "object"}}, "required": ["name", "arguments"]}}\n',  # noqa: E501
            ),
            (
                [_create_tool("non_strict_func", {"param": "value"}, strict=False)],
                7,
                "auto",
                False,
                '<TOOL_CALLS> SAFE_WS? fcall_array SAFE_WS?\nfcall_array: %json {"minItems": 1, "type": "array", "items": {"type": "object", "additionalProperties": false, "properties": {"name": {"type": "string", "enum": ["non_strict_func"]}, "arguments": {"type": "object"}}, "required": ["name", "arguments"]}, "maxItems": 1}\n',  # noqa: E501
            ),
            # Pre-v11: strict tool uses anyOf with const name
            (
                [_create_tool("strict_func", {"param": "value"}, strict=True)],
                7,
                "auto",
                True,
                '<TOOL_CALLS> SAFE_WS? fcall_array SAFE_WS?\nfcall_array: %json {"minItems": 1, "type": "array", "items": {"anyOf": [{"type": "object", "required": ["name", "arguments"], "additionalProperties": false, "properties": {"name": {"const": "strict_func"}, "arguments": {"param": "value"}}}]}}\n',  # noqa: E501
            ),
            # Pre-v11: empty strict params get default schema
            (
                [_create_tool("empty_strict_func", {}, strict=True)],
                7,
                "auto",
                True,
                '<TOOL_CALLS> SAFE_WS? fcall_array SAFE_WS?\nfcall_array: %json {"minItems": 1, "type": "array", "items": {"anyOf": [{"type": "object", "required": ["name", "arguments"], "additionalProperties": false, "properties": {"name": {"const": "empty_strict_func"}, "arguments": {"type": "object", "properties": {}, "additionalProperties": false}}}]}}\n',  # noqa: E501
            ),
            # Pre-v11: mixed strict/non-strict
            (
                [
                    _create_tool("strict_func", {"param": "value"}, strict=True),
                    _create_tool("non_strict_func", {"param": "value"}, strict=False),
                ],
                7,
                "auto",
                True,
                '<TOOL_CALLS> SAFE_WS? fcall_array SAFE_WS?\nfcall_array: %json {"minItems": 1, "type": "array", "items": {"anyOf": [{"type": "object", "required": ["name", "arguments"], "additionalProperties": false, "properties": {"name": {"const": "strict_func"}, "arguments": {"param": "value"}}}, {"type": "object", "required": ["name", "arguments"], "additionalProperties": false, "properties": {"name": {"const": "non_strict_func"}, "arguments": {"type": "object"}}}]}}\n',  # noqa: E501
            ),
            # Post-v11: non-strict parallel vs no-parallel (+ suffix difference)
            (
                [_create_tool("non_strict_func", {"param": "value"}, strict=False)],
                11,
                "auto",
                True,
                '((<TOOL_CALLS> SAFE_WS? "non_strict_func" <ARGS> SAFE_WS? %json {"type": "object"} SAFE_WS?))+',  # noqa: E501
            ),
            (
                [_create_tool("non_strict_func", {"param": "value"}, strict=False)],
                11,
                "auto",
                False,
                '(<TOOL_CALLS> SAFE_WS? "non_strict_func" <ARGS> SAFE_WS? %json {"type": "object"} SAFE_WS?)',  # noqa: E501
            ),
            # Post-v11: strict tool embeds schema directly
            (
                [_create_tool("strict_func", {"param": "value"}, strict=True)],
                11,
                "auto",
                True,
                '((<TOOL_CALLS> SAFE_WS? "strict_func" <ARGS> SAFE_WS? %json {"param": "value"} SAFE_WS?))+',  # noqa: E501
            ),
            # Post-v11: multiple tools use | separator
            (
                [
                    _create_tool("strict_func", {"param": "value"}, strict=True),
                    _create_tool("non_strict_func", {"param": "value"}, strict=False),
                ],
                11,
                "auto",
                True,
                '((<TOOL_CALLS> SAFE_WS? "strict_func" <ARGS> SAFE_WS? %json {"param": "value"} SAFE_WS?) | (<TOOL_CALLS> SAFE_WS? "non_strict_func" <ARGS> SAFE_WS? %json {"type": "object"} SAFE_WS?))+',  # noqa: E501
            ),
            # mode="required" uses same grammar as "auto"
            (
                [_create_tool("test_func", {"param": "value"}, strict=True)],
                7,
                "required",
                True,
                '<TOOL_CALLS> SAFE_WS? fcall_array SAFE_WS?\nfcall_array: %json {"minItems": 1, "type": "array", "items": {"anyOf": [{"type": "object", "required": ["name", "arguments"], "additionalProperties": false, "properties": {"name": {"const": "test_func"}, "arguments": {"param": "value"}}}]}}\n',  # noqa: E501
            ),
        ],
        ids=[
            "none_returns_empty",
            "pre_v11_non_strict_parallel",
            "pre_v11_non_strict_no_parallel",
            "pre_v11_strict",
            "pre_v11_empty_strict",
            "pre_v11_mixed",
            "post_v11_non_strict_parallel",
            "post_v11_non_strict_no_parallel",
            "post_v11_strict",
            "post_v11_mixed",
            "required_same_as_auto",
        ],
    )
    def test_convert(
        self,
        tools: list[ChatCompletionToolsParam] | None,
        tokenizer_version: int,
        mode: Literal["auto", "required", "none"],
        parallel_tool_calls: bool,
        expected: str,
    ):
        assert (
            PostV11ToolsLarkConverter().convert(
                tools=tools,
                tokenizer_version=tokenizer_version,
                mode=mode,
                parallel_tool_calls=parallel_tool_calls,
            )
            == expected
        )


class TestMistralGrammarFactory:
    @pytest.mark.parametrize(
        "tokenizer_version,reasoning,expected_body_rule,expect_think",
        [
            # version < 13: BASE_LARK_GRAMMAR regardless of reasoning
            (11, True, "body: content | (content? fcalls)", False),
            # version >= 13, reasoning=False: BASE_LARK_GRAMMAR
            (13, False, "body: content | (content? fcalls)", False),
            # 13 <= version < 15, reasoning=True: OPTIONAL_THINK_LARK_GRAMMAR
            (13, True, "body: think? (content | fcalls)", True),
            # version >= 15, reasoning=False: BASE_LARK_GRAMMAR
            (15, False, "body: content | (content? fcalls)", False),
            # version >= 15, reasoning=True: THINK_LARK_GRAMMAR
            (15, True, "body: (think (content | content? fcalls)) | fcalls", True),
        ],
        ids=[
            "pre_v13_ignores_reasoning",
            "v13_no_reasoning_base",
            "v13_reasoning_optional_think",
            "v15_no_reasoning_base",
            "v15_reasoning_required_think",
        ],
    )
    def test_get_lark_from_jinja_template_selection(
        self,
        tokenizer_version: int,
        reasoning: bool,
        expected_body_rule: str,
        expect_think: bool,
    ):
        tools = [_create_tool("func", {}, strict=False)]

        with patch(
            "vllm.tool_parsers.mistral_tool_parser.is_mistral_tokenizer",
            return_value=True,
        ):
            factory = MistralGrammarFactory.__new__(MistralGrammarFactory)
            factory._tokenizer_version = tokenizer_version

            grammar = factory.get_lark_from_jinja(
                mode="auto",
                tools=tools,
                reasoning=reasoning,
                parallel_tool_calls=True,
            )

        assert expected_body_rule in grammar

        if expect_think:
            assert "think: <THINK> content </THINK>" in grammar
        else:
            assert "think:" not in grammar

    @pytest.mark.parametrize(
        "mode,expected_body_rule",
        [
            ("auto", "body: content | (content? fcalls)"),
            ("required", "body: content? fcalls"),
            ("none", "body: content"),
        ],
        ids=["auto", "required", "none"],
    )
    def test_get_lark_from_jinja_mode_handling(
        self,
        mode: str,
        expected_body_rule: str,
    ):
        tools = [_create_tool("func", {}, strict=False)]

        with patch(
            "vllm.tool_parsers.mistral_tool_parser.is_mistral_tokenizer",
            return_value=True,
        ):
            factory = MistralGrammarFactory.__new__(MistralGrammarFactory)
            factory._tokenizer_version = 11

            grammar = factory.get_lark_from_jinja(
                mode=mode,
                tools=tools,
                reasoning=False,
                parallel_tool_calls=True,
            )

        assert expected_body_rule in grammar

        # "none" mode should not have fcalls rule
        if mode == "none":
            assert "fcalls:" not in grammar
        else:
            assert "fcalls:" in grammar

    def test_get_lark_from_jinja_none_mode_defaults_to_auto(self):
        tools = [_create_tool("func", {}, strict=False)]

        with patch(
            "vllm.tool_parsers.mistral_tool_parser.is_mistral_tokenizer",
            return_value=True,
        ):
            factory = MistralGrammarFactory.__new__(MistralGrammarFactory)
            factory._tokenizer_version = 11

            grammar_none = factory.get_lark_from_jinja(
                mode=None,
                tools=tools,
                reasoning=False,
                parallel_tool_calls=True,
            )
            grammar_auto = factory.get_lark_from_jinja(
                mode="auto",
                tools=tools,
                reasoning=False,
                parallel_tool_calls=True,
            )

        assert grammar_none == grammar_auto


_UNSET = object()


def _create_request(
    tools: list[ChatCompletionToolsParam] | None = None,
    tool_choice: str | None | object = _UNSET,
    reasoning_effort: str | None = None,
) -> ChatCompletionRequest:
    if tool_choice is _UNSET:
        tool_choice = "auto" if tools else "none"
    return ChatCompletionRequest(
        messages=[{"role": "user", "content": "test"}],
        model="test-model",
        tools=tools,
        tool_choice=tool_choice,
        reasoning_effort=reasoning_effort,
    )


def _create_mock_tool_parser(tokenizer_version: int, has_reasoning_parser: bool):
    mock_tokenizer = MagicMock(spec=MistralTokenizer)
    type(mock_tokenizer).version = PropertyMock(return_value=tokenizer_version)

    # Set up a mock vocab with the TOOL_CALLS token
    mock_tokenizer.get_vocab.return_value = {"[TOOL_CALLS]": 999}

    with patch(
        "vllm.tool_parsers.mistral_tool_parser.is_mistral_tokenizer",
        return_value=True,
    ):
        parser = MistralToolParser.__new__(MistralToolParser)
        parser.model_tokenizer = mock_tokenizer
        parser.vocab = {"[TOOL_CALLS]": 999}
        parser.bot_token = "[TOOL_CALLS]"
        parser.bot_token_id = 999
        parser._is_pre_v11 = tokenizer_version < 11
        parser.prev_tool_call_arr = []
        parser.current_tool_id = -1
        parser._has_reasoning_parser = has_reasoning_parser

        # Set up the grammar factory
        factory = MistralGrammarFactory.__new__(MistralGrammarFactory)
        factory._tokenizer = mock_tokenizer
        factory._tokenizer_version = tokenizer_version
        parser.grammar_factory = factory

    return parser


class TestAdjustRequestReasoningEffort:
    @pytest.mark.parametrize(
        "tokenizer_version,reasoning_effort",
        [
            (11, "high"),
            (15, "high"),
        ],
        ids=["pre_v15", "v15"],
    )
    def test_should_reason_no_reasoning_parser(
        self,
        tokenizer_version: int,
        reasoning_effort: str | None,
    ):
        parser = _create_mock_tool_parser(tokenizer_version, has_reasoning_parser=False)
        request = _create_request(reasoning_effort=reasoning_effort)
        assert parser._should_reason(request) is False

    @pytest.mark.parametrize(
        "tokenizer_version,reasoning_effort,expected",
        [
            # Pre-v15: always True regardless of effort
            (13, None, True),
            (13, "high", True),
            (13, "none", True),
            # V15+: depends on reasoning_effort
            (15, None, False),
            (15, "none", False),
            (15, "high", True),
        ],
        ids=[
            "pre_v15_none",
            "pre_v15_high",
            "pre_v15_effort_none",
            "v15_none",
            "v15_effort_none",
            "v15_high",
        ],
    )
    def test_should_reason_with_reasoning_parser(
        self,
        tokenizer_version: int,
        reasoning_effort: str | None,
        expected: bool,
    ):
        parser = _create_mock_tool_parser(tokenizer_version, has_reasoning_parser=True)
        request = _create_request(reasoning_effort=reasoning_effort)
        assert parser._should_reason(request) is expected

    @pytest.mark.parametrize(
        "tokenizer_version,reasoning_effort,has_reasoning_parser,"
        "expected_body_rule,expect_think",
        [
            # Pre-v13 with parser: BASE (reasoning ignored below v13)
            (11, "high", True, "body: content | (content? fcalls)", False),
            # V13 with parser: OPTIONAL_THINK
            (13, "high", True, "body: think? (content | fcalls)", True),
            # V15 with parser + high effort: THINK (required)
            (
                15,
                "high",
                True,
                "body: (think (content | content? fcalls)) | fcalls",
                True,
            ),
            # V15 without parser: BASE regardless of effort
            (15, "high", False, "body: content | (content? fcalls)", False),
        ],
        ids=[
            "pre_v13_base",
            "v13_optional_think",
            "v15_required_think",
            "v15_no_parser_base",
        ],
    )
    def test_adjust_request_grammar_selection(
        self,
        tokenizer_version: int,
        reasoning_effort: str | None,
        has_reasoning_parser: bool,
        expected_body_rule: str,
        expect_think: bool,
    ):
        tools = [_create_tool("func", {"param": "value"}, strict=False)]
        parser = _create_mock_tool_parser(
            tokenizer_version, has_reasoning_parser=has_reasoning_parser
        )

        request = _create_request(
            tools=tools,
            tool_choice="auto",
            reasoning_effort=reasoning_effort,
        )

        with patch(
            "vllm.tool_parsers.mistral_tool_parser.is_mistral_tokenizer",
            return_value=True,
        ):
            result = parser.adjust_request(request)

        assert result.structured_outputs is not None
        grammar = result.structured_outputs.lark
        assert grammar is not None
        assert expected_body_rule in grammar

        if expect_think:
            assert "think: <THINK> content </THINK>" in grammar
        else:
            assert "think:" not in grammar

        assert result.response_format is None

    def test_adjust_request_no_tools_skips_grammar(self):
        """When no tools are provided, adjust_request should return
        early without setting up any grammar."""
        parser = _create_mock_tool_parser(
            tokenizer_version=11, has_reasoning_parser=False
        )

        request = _create_request(
            tools=None,
            tool_choice="none",
            reasoning_effort=None,
        )

        with patch(
            "vllm.tool_parsers.mistral_tool_parser.is_mistral_tokenizer",
            return_value=True,
        ):
            result = parser.adjust_request(request)

        assert result.tool_choice == "none"
        assert result.structured_outputs is None

    def test_adjust_request_none_tool_choice_defaults_to_auto(self):
        """When tools are provided but tool_choice is None, it should
        default to 'auto'."""
        tools = [_create_tool("func", {"param": "value"}, strict=False)]
        parser = _create_mock_tool_parser(
            tokenizer_version=11, has_reasoning_parser=False
        )

        request = _create_request(
            tools=tools,
            tool_choice=None,
            reasoning_effort=None,
        )
        assert request.tool_choice is None

        with patch(
            "vllm.tool_parsers.mistral_tool_parser.is_mistral_tokenizer",
            return_value=True,
        ):
            result = parser.adjust_request(request)

        assert result.tool_choice == "auto"

        assert result.structured_outputs is not None
        grammar = result.structured_outputs.lark
        assert grammar is not None
        assert "body: content | (content? fcalls)" in grammar

    def test_adjust_request_none_mode_with_tools_raises(self):
        """When tool_choice='none' is used with tools, it should raise
        because only 'auto' tool_choice is currently supported."""
        tools = [_create_tool("func", {"param": "value"}, strict=False)]
        parser = _create_mock_tool_parser(
            tokenizer_version=11, has_reasoning_parser=False
        )

        request = _create_request(
            tools=tools,
            tool_choice="none",
            reasoning_effort=None,
        )

        with (
            patch(
                "vllm.tool_parsers.mistral_tool_parser.is_mistral_tokenizer",
                return_value=True,
            ),
            pytest.raises(ValueError, match="only 'auto' tool choice"),
        ):
            parser.adjust_request(request)
