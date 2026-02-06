# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Generator
from typing import Any, Literal
from unittest.mock import MagicMock, Mock

import partial_json_parser
import pytest
from mistral_common.protocol.instruct.messages import AssistantMessage
from mistral_common.protocol.instruct.request import InstructRequest
from mistral_common.protocol.instruct.tool_calls import FunctionCall, ToolCall
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from partial_json_parser.core.options import Allow

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
    MistralToolCall,
    MistralToolParser,
    ToolsLarkConverter,
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

    actual_tool_calls = []
    for tool_call_id, function_name, function_args_str in zip(
        tool_call_ids, function_names, function_args_strs
    ):
        actual_id: str = (
            tool_call_id
            if tool_call_id is not None
            else MistralToolCall.generate_random_id()
        )
        actual_tool_calls.append(
            ToolCall(
                id=actual_id,
                function=FunctionCall(
                    name=function_name,
                    arguments=partial_json_parser.ensure_json(
                        function_args_str, Allow.OBJ | Allow.STR
                    ),
                ),
            )
        )
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
        assert ToolsLarkConverter().get_args_json(tool=tool) == expected

    @pytest.mark.parametrize(
        "tools,tokenizer_version,mode,parallel_tool_calls,expected",
        [
            # Test cases for mode="none" - should always return empty string
            (
                [_create_tool("non_strict_func", {"param": "value"}, strict=False)],
                7,
                "none",
                True,
                "",
            ),
            (
                [_create_tool("non_strict_func", {"param": "value"}, strict=False)],
                11,
                "none",
                False,
                "",
            ),
            # Test cases for mode="auto" with tokenizer version 7 (pre-v11)
            # Single non-strict tool
            (
                [_create_tool("non_strict_func", {"param": "value"}, strict=False)],
                7,
                "auto",
                True,
                '<TOOL_CALLS> SAFE_WS? fcall_array SAFE_WS?\nfcall_array: %json {"minItems": 1, "type": "array", "items": {"type": "object", "additionalProperties": false, "properties": {"name": {"type": "string", "minLength": 1}, "arguments": {"type": "object"}}, "required": ["name", "arguments"]}}\n',
            ),
            (
                [_create_tool("non_strict_func", {"param": "value"}, strict=False)],
                7,
                "auto",
                False,
                '<TOOL_CALLS> SAFE_WS? fcall_array SAFE_WS?\nfcall_array: %json {"minItems": 1, "type": "array", "items": {"type": "object", "additionalProperties": false, "properties": {"name": {"type": "string", "minLength": 1}, "arguments": {"type": "object"}}, "required": ["name", "arguments"]}, "maxItems": 1}\n',
            ),
            # Single strict tool
            (
                [_create_tool("strict_func", {"param": "value"}, strict=True)],
                7,
                "auto",
                True,
                '<TOOL_CALLS> SAFE_WS? fcall_array SAFE_WS?\nfcall_array: %json {"minItems": 1, "type": "array", "items": {"anyOf": [{"type": "object", "required": ["name", "arguments"], "additionalProperties": false, "properties": {"name": {"const": "strict_func"}, "arguments": {"param": "value"}}}]}}\n',
            ),
            (
                [_create_tool("strict_func", {"param": "value"}, strict=True)],
                7,
                "auto",
                False,
                '<TOOL_CALLS> SAFE_WS? fcall_array SAFE_WS?\nfcall_array: %json {"minItems": 1, "type": "array", "items": {"anyOf": [{"type": "object", "required": ["name", "arguments"], "additionalProperties": false, "properties": {"name": {"const": "strict_func"}, "arguments": {"param": "value"}}}]}, "maxItems": 1}\n',
            ),
            # Single strict tool with empty parameters
            (
                [_create_tool("empty_strict_func", {}, strict=True)],
                7,
                "auto",
                True,
                '<TOOL_CALLS> SAFE_WS? fcall_array SAFE_WS?\nfcall_array: %json {"minItems": 1, "type": "array", "items": {"anyOf": [{"type": "object", "required": ["name", "arguments"], "additionalProperties": false, "properties": {"name": {"const": "empty_strict_func"}, "arguments": {"type": "object", "properties": {}, "additionalProperties": false}}}]}}\n',
            ),
            # Multiple non-strict tools
            (
                [
                    _create_tool("func1", {"param1": "value"}, strict=False),
                    _create_tool("func2", {"param2": "value"}, strict=False),
                ],
                7,
                "auto",
                True,
                '<TOOL_CALLS> SAFE_WS? fcall_array SAFE_WS?\nfcall_array: %json {"minItems": 1, "type": "array", "items": {"type": "object", "additionalProperties": false, "properties": {"name": {"type": "string", "minLength": 1}, "arguments": {"type": "object"}}, "required": ["name", "arguments"]}}\n',
            ),
            # Multiple mixed tools (some strict, some not)
            (
                [
                    _create_tool("strict_func", {"param": "value"}, strict=True),
                    _create_tool("non_strict_func", {"param": "value"}, strict=False),
                ],
                7,
                "auto",
                True,
                '<TOOL_CALLS> SAFE_WS? fcall_array SAFE_WS?\nfcall_array: %json {"minItems": 1, "type": "array", "items": {"anyOf": [{"type": "object", "required": ["name", "arguments"], "additionalProperties": false, "properties": {"name": {"const": "strict_func"}, "arguments": {"param": "value"}}}, {"type": "object", "required": ["name", "arguments"], "additionalProperties": false, "properties": {"name": {"const": "non_strict_func"}, "arguments": {"type": "object"}}}]}}\n',
            ),
            # Test cases for mode="auto" with tokenizer version 11 (post-v11)
            # Single non-strict tool
            (
                [_create_tool("non_strict_func", {"param": "value"}, strict=False)],
                11,
                "auto",
                True,
                '(<TOOL_CALLS> SAFE_WS? /.+/ <ARGS> SAFE_WS? %json {"type": "object"} SAFE_WS?)+',
            ),
            (
                [_create_tool("non_strict_func", {"param": "value"}, strict=False)],
                11,
                "auto",
                False,
                '<TOOL_CALLS> SAFE_WS? /.+/ <ARGS> SAFE_WS? %json {"type": "object"} SAFE_WS?',
            ),
            # Single strict tool
            (
                [_create_tool("strict_func", {"param": "value"}, strict=True)],
                11,
                "auto",
                True,
                '((<TOOL_CALLS> SAFE_WS? "strict_func" <ARGS> SAFE_WS? %json {"param": "value"} SAFE_WS?))+',
            ),
            (
                [_create_tool("strict_func", {"param": "value"}, strict=True)],
                11,
                "auto",
                False,
                '(<TOOL_CALLS> SAFE_WS? "strict_func" <ARGS> SAFE_WS? %json {"param": "value"} SAFE_WS?)',
            ),
            # Single strict tool with empty parameters
            (
                [_create_tool("empty_strict_func", {}, strict=True)],
                11,
                "auto",
                True,
                '((<TOOL_CALLS> SAFE_WS? "empty_strict_func" <ARGS> SAFE_WS? %json {"type": "object", "properties": {}, "additionalProperties": false} SAFE_WS?))+',
            ),
            # Multiple non-strict tools
            (
                [
                    _create_tool("func1", {"param1": "value"}, strict=False),
                    _create_tool("func2", {"param2": "value"}, strict=False),
                ],
                11,
                "auto",
                True,
                '(<TOOL_CALLS> SAFE_WS? /.+/ <ARGS> SAFE_WS? %json {"type": "object"} SAFE_WS?)+',
            ),
            # Multiple mixed tools (some strict, some not)
            (
                [
                    _create_tool("strict_func", {"param": "value"}, strict=True),
                    _create_tool("non_strict_func", {"param": "value"}, strict=False),
                ],
                11,
                "auto",
                True,
                '((<TOOL_CALLS> SAFE_WS? "strict_func" <ARGS> SAFE_WS? %json {"param": "value"} SAFE_WS?) | (<TOOL_CALLS> SAFE_WS? "non_strict_func" <ARGS> SAFE_WS? %json {"type": "object"} SAFE_WS?))+',
            ),
            # Test cases for mode="required" - should be same as "auto" for grammar generation
            # Just test a few representative cases to ensure mode is handled correctly
            (
                [_create_tool("test_func", {"param": "value"}, strict=True)],
                7,
                "required",
                True,
                '<TOOL_CALLS> SAFE_WS? fcall_array SAFE_WS?\nfcall_array: %json {"minItems": 1, "type": "array", "items": {"anyOf": [{"type": "object", "required": ["name", "arguments"], "additionalProperties": false, "properties": {"name": {"const": "test_func"}, "arguments": {"param": "value"}}}]}}\n',
            ),
            (
                [_create_tool("test_func", {"param": "value"}, strict=False)],
                11,
                "required",
                False,
                '<TOOL_CALLS> SAFE_WS? /.+/ <ARGS> SAFE_WS? %json {"type": "object"} SAFE_WS?',
            ),
        ],
        ids=[
            # mode="none" cases
            "none_v7_parallel",
            "none_v11_no_parallel",
            # mode="auto" v7 cases
            "auto_v7_single_non_strict_parallel",
            "auto_v7_single_non_strict_no_parallel",
            "auto_v7_single_strict_parallel",
            "auto_v7_single_strict_no_parallel",
            "auto_v7_single_empty_strict_parallel",
            "auto_v7_multiple_non_strict_parallel",
            "auto_v7_multiple_mixed_parallel",
            # mode="auto" v11 cases
            "auto_v11_single_non_strict_parallel",
            "auto_v11_single_non_strict_no_parallel",
            "auto_v11_single_strict_parallel",
            "auto_v11_single_strict_no_parallel",
            "auto_v11_single_empty_strict_parallel",
            "auto_v11_multiple_non_strict_parallel",
            "auto_v11_multiple_mixed_parallel",
            # mode="required" cases
            "required_v7_single_strict_parallel",
            "required_v11_single_non_strict_no_parallel",
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
            ToolsLarkConverter().convert(
                tools=tools,
                tokenizer_version=tokenizer_version,
                mode=mode,
                parallel_tool_calls=parallel_tool_calls,
            )
            == expected
        )


class TestMistralGrammarFactory:
    """Test class for MistralGrammarFactory"""

    @pytest.fixture
    def mock_tokenizer_v7(self):
        mock_tokenizer = Mock(spec=MistralTokenizer)
        mock_tokenizer.tokenizer = MagicMock()
        mock_tokenizer.tokenizer._version = TokenizerVersion.v7
        return mock_tokenizer

    @pytest.fixture
    def mock_tokenizer_v11(self):
        mock_tokenizer = Mock(spec=MistralTokenizer)
        mock_tokenizer.tokenizer = MagicMock()
        mock_tokenizer.tokenizer._version = TokenizerVersion.v11
        return mock_tokenizer

    @pytest.fixture
    def mock_tokenizer_v13(self):
        mock_tokenizer = Mock(spec=MistralTokenizer)
        mock_tokenizer.tokenizer = MagicMock()
        mock_tokenizer.tokenizer._version = TokenizerVersion.v13
        return mock_tokenizer

    @pytest.mark.parametrize(
        "mode,reasoning,parallel_tool_calls,tokenizer_version,tools,expected",
        [
            (
                "auto",
                True,
                True,
                11,
                [
                    _create_tool(
                        "tool1",
                        {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City and state, e.g., 'San Francisco, CA'",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location", "unit"],
                        },
                        strict=True,
                    ),
                    _create_tool("tool2", {}, strict=False),
                ],
                'start: body\nbody: content | (content? fcalls)\nfcalls: ((<TOOL_CALLS> SAFE_WS? "tool1" <ARGS> SAFE_WS? %json {"type": "object", "properties": {"location": {"type": "string", "description": "City and state, e.g., \'San Francisco, CA\'"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location", "unit"]} SAFE_WS?) | (<TOOL_CALLS> SAFE_WS? "tool2" <ARGS> SAFE_WS? %json {"type": "object"} SAFE_WS?))+\ncontent: (/(.|\\n)+/)+\nSAFE_WS: /[ \\t\\r\\n]+/',
            ),
            (
                "auto",
                True,
                True,
                13,
                [
                    _create_tool(
                        "tool1",
                        {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City and state, e.g., 'San Francisco, CA'",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location", "unit"],
                        },
                        strict=True,
                    ),
                    _create_tool("tool2", {}, strict=False),
                ],
                'start: body\nbody: think? (content | fcalls)\nfcalls: content? ((<TOOL_CALLS> SAFE_WS? "tool1" <ARGS> SAFE_WS? %json {"type": "object", "properties": {"location": {"type": "string", "description": "City and state, e.g., \'San Francisco, CA\'"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location", "unit"]} SAFE_WS?) | (<TOOL_CALLS> SAFE_WS? "tool2" <ARGS> SAFE_WS? %json {"type": "object"} SAFE_WS?))+\nthink: <THINK> content </THINK>\ncontent: (/(.|\\n)+/)+\nSAFE_WS: /[ \\t\\r\\n]+/',
            ),
            (
                "auto",
                False,
                True,
                7,
                [
                    _create_tool(
                        "tool1",
                        {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City and state, e.g., 'San Francisco, CA'",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location", "unit"],
                        },
                        strict=True,
                    ),
                    _create_tool("tool2", {}, strict=False),
                ],
                'start: body\nbody: content | (content? fcalls)\nfcalls: <TOOL_CALLS> SAFE_WS? fcall_array SAFE_WS?\nfcall_array: %json {"minItems": 1, "type": "array", "items": {"anyOf": [{"type": "object", "required": ["name", "arguments"], "additionalProperties": false, "properties": {"name": {"const": "tool1"}, "arguments": {"type": "object", "properties": {"location": {"type": "string", "description": "City and state, e.g., \'San Francisco, CA\'"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location", "unit"]}}}, {"type": "object", "required": ["name", "arguments"], "additionalProperties": false, "properties": {"name": {"const": "tool2"}, "arguments": {"type": "object"}}}]}}\n\ncontent: (/(.|\\n)+/)+\nSAFE_WS: /[ \\t\\r\\n]+/',
            ),
            (
                "auto",
                False,
                True,
                11,
                [
                    _create_tool(
                        "tool1",
                        {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City and state, e.g., 'San Francisco, CA'",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location", "unit"],
                        },
                        strict=True,
                    ),
                    _create_tool("tool2", {}, strict=False),
                ],
                'start: body\nbody: content | (content? fcalls)\nfcalls: ((<TOOL_CALLS> SAFE_WS? "tool1" <ARGS> SAFE_WS? %json {"type": "object", "properties": {"location": {"type": "string", "description": "City and state, e.g., \'San Francisco, CA\'"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location", "unit"]} SAFE_WS?) | (<TOOL_CALLS> SAFE_WS? "tool2" <ARGS> SAFE_WS? %json {"type": "object"} SAFE_WS?))+\ncontent: (/(.|\\n)+/)+\nSAFE_WS: /[ \\t\\r\\n]+/',
            ),
            (
                "auto",
                False,
                True,
                13,
                [
                    _create_tool(
                        "tool1",
                        {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City and state, e.g., 'San Francisco, CA'",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location", "unit"],
                        },
                        strict=True,
                    ),
                    _create_tool("tool2", {}, strict=False),
                ],
                'start: body\nbody: content | (content? fcalls)\nfcalls: ((<TOOL_CALLS> SAFE_WS? "tool1" <ARGS> SAFE_WS? %json {"type": "object", "properties": {"location": {"type": "string", "description": "City and state, e.g., \'San Francisco, CA\'"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location", "unit"]} SAFE_WS?) | (<TOOL_CALLS> SAFE_WS? "tool2" <ARGS> SAFE_WS? %json {"type": "object"} SAFE_WS?))+\ncontent: (/(.|\\n)+/)+\nSAFE_WS: /[ \\t\\r\\n]+/',
            ),
            (
                "auto",
                False,
                False,
                7,
                [_create_tool("tool1", {}, strict=False)],
                'start: body\nbody: content | (content? fcalls)\nfcalls: <TOOL_CALLS> SAFE_WS? fcall_array SAFE_WS?\nfcall_array: %json {"minItems": 1, "type": "array", "items": {"type": "object", "additionalProperties": false, "properties": {"name": {"type": "string", "minLength": 1}, "arguments": {"type": "object"}}, "required": ["name", "arguments"]}, "maxItems": 1}\n\ncontent: (/(.|\\n)+/)+\nSAFE_WS: /[ \\t\\r\\n]+/',
            ),
            (
                "auto",
                False,
                False,
                7,
                [
                    _create_tool(
                        "tool1",
                        {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City and state, e.g., 'San Francisco, CA'",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location", "unit"],
                        },
                        strict=True,
                    )
                ],
                'start: body\nbody: content | (content? fcalls)\nfcalls: <TOOL_CALLS> SAFE_WS? fcall_array SAFE_WS?\nfcall_array: %json {"minItems": 1, "type": "array", "items": {"anyOf": [{"type": "object", "required": ["name", "arguments"], "additionalProperties": false, "properties": {"name": {"const": "tool1"}, "arguments": {"type": "object", "properties": {"location": {"type": "string", "description": "City and state, e.g., \'San Francisco, CA\'"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location", "unit"]}}}]}, "maxItems": 1}\n\ncontent: (/(.|\\n)+/)+\nSAFE_WS: /[ \\t\\r\\n]+/',
            ),
            (
                "auto",
                False,
                False,
                7,
                [
                    _create_tool(
                        "tool1",
                        {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City and state, e.g., 'San Francisco, CA'",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location", "unit"],
                        },
                        strict=True,
                    ),
                    _create_tool("tool2", {}, strict=False),
                ],
                'start: body\nbody: content | (content? fcalls)\nfcalls: <TOOL_CALLS> SAFE_WS? fcall_array SAFE_WS?\nfcall_array: %json {"minItems": 1, "type": "array", "items": {"anyOf": [{"type": "object", "required": ["name", "arguments"], "additionalProperties": false, "properties": {"name": {"const": "tool1"}, "arguments": {"type": "object", "properties": {"location": {"type": "string", "description": "City and state, e.g., \'San Francisco, CA\'"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location", "unit"]}}}, {"type": "object", "required": ["name", "arguments"], "additionalProperties": false, "properties": {"name": {"const": "tool2"}, "arguments": {"type": "object"}}}]}, "maxItems": 1}\n\ncontent: (/(.|\\n)+/)+\nSAFE_WS: /[ \\t\\r\\n]+/',
            ),
            (
                "auto",
                False,
                False,
                11,
                [_create_tool("tool1", {}, strict=False)],
                'start: body\nbody: content | (content? fcalls)\nfcalls: <TOOL_CALLS> SAFE_WS? /.+/ <ARGS> SAFE_WS? %json {"type": "object"} SAFE_WS?\ncontent: (/(.|\\n)+/)+\nSAFE_WS: /[ \\t\\r\\n]+/',
            ),
            (
                "auto",
                False,
                False,
                11,
                [
                    _create_tool(
                        "tool1",
                        {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City and state, e.g., 'San Francisco, CA'",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location", "unit"],
                        },
                        strict=True,
                    )
                ],
                'start: body\nbody: content | (content? fcalls)\nfcalls: (<TOOL_CALLS> SAFE_WS? "tool1" <ARGS> SAFE_WS? %json {"type": "object", "properties": {"location": {"type": "string", "description": "City and state, e.g., \'San Francisco, CA\'"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location", "unit"]} SAFE_WS?)\ncontent: (/(.|\\n)+/)+\nSAFE_WS: /[ \\t\\r\\n]+/',
            ),
            (
                "auto",
                False,
                False,
                11,
                [
                    _create_tool(
                        "tool1",
                        {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City and state, e.g., 'San Francisco, CA'",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location", "unit"],
                        },
                        strict=True,
                    ),
                    _create_tool("tool2", {}, strict=False),
                ],
                'start: body\nbody: content | (content? fcalls)\nfcalls: (<TOOL_CALLS> SAFE_WS? "tool1" <ARGS> SAFE_WS? %json {"type": "object", "properties": {"location": {"type": "string", "description": "City and state, e.g., \'San Francisco, CA\'"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location", "unit"]} SAFE_WS?) | (<TOOL_CALLS> SAFE_WS? "tool2" <ARGS> SAFE_WS? %json {"type": "object"} SAFE_WS?)\ncontent: (/(.|\\n)+/)+\nSAFE_WS: /[ \\t\\r\\n]+/',
            ),
            (
                "auto",
                False,
                False,
                13,
                [_create_tool("tool1", {}, strict=False)],
                'start: body\nbody: content | (content? fcalls)\nfcalls: <TOOL_CALLS> SAFE_WS? /.+/ <ARGS> SAFE_WS? %json {"type": "object"} SAFE_WS?\ncontent: (/(.|\\n)+/)+\nSAFE_WS: /[ \\t\\r\\n]+/',
            ),
            (
                "auto",
                False,
                False,
                13,
                [
                    _create_tool(
                        "tool1",
                        {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City and state, e.g., 'San Francisco, CA'",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location", "unit"],
                        },
                        strict=True,
                    )
                ],
                'start: body\nbody: content | (content? fcalls)\nfcalls: (<TOOL_CALLS> SAFE_WS? "tool1" <ARGS> SAFE_WS? %json {"type": "object", "properties": {"location": {"type": "string", "description": "City and state, e.g., \'San Francisco, CA\'"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location", "unit"]} SAFE_WS?)\ncontent: (/(.|\\n)+/)+\nSAFE_WS: /[ \\t\\r\\n]+/',
            ),
            (
                "auto",
                False,
                False,
                13,
                [
                    _create_tool(
                        "tool1",
                        {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City and state, e.g., 'San Francisco, CA'",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location", "unit"],
                        },
                        strict=True,
                    ),
                    _create_tool("tool2", {}, strict=False),
                ],
                'start: body\nbody: content | (content? fcalls)\nfcalls: (<TOOL_CALLS> SAFE_WS? "tool1" <ARGS> SAFE_WS? %json {"type": "object", "properties": {"location": {"type": "string", "description": "City and state, e.g., \'San Francisco, CA\'"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location", "unit"]} SAFE_WS?) | (<TOOL_CALLS> SAFE_WS? "tool2" <ARGS> SAFE_WS? %json {"type": "object"} SAFE_WS?)\ncontent: (/(.|\\n)+/)+\nSAFE_WS: /[ \\t\\r\\n]+/',
            ),
            (
                "required",
                True,
                True,
                11,
                [
                    _create_tool(
                        "tool1",
                        {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City and state, e.g., 'San Francisco, CA'",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location", "unit"],
                        },
                        strict=True,
                    ),
                    _create_tool("tool2", {}, strict=False),
                ],
                'start: body\nbody: content? fcalls\nfcalls: ((<TOOL_CALLS> SAFE_WS? "tool1" <ARGS> SAFE_WS? %json {"type": "object", "properties": {"location": {"type": "string", "description": "City and state, e.g., \'San Francisco, CA\'"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location", "unit"]} SAFE_WS?) | (<TOOL_CALLS> SAFE_WS? "tool2" <ARGS> SAFE_WS? %json {"type": "object"} SAFE_WS?))+\ncontent: (/(.|\\n)+/)+\nSAFE_WS: /[ \\t\\r\\n]+/',
            ),
            (
                "required",
                True,
                True,
                13,
                [
                    _create_tool(
                        "tool1",
                        {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City and state, e.g., 'San Francisco, CA'",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location", "unit"],
                        },
                        strict=True,
                    ),
                    _create_tool("tool2", {}, strict=False),
                ],
                'start: body\nbody: think? fcalls\nfcalls: content? ((<TOOL_CALLS> SAFE_WS? "tool1" <ARGS> SAFE_WS? %json {"type": "object", "properties": {"location": {"type": "string", "description": "City and state, e.g., \'San Francisco, CA\'"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location", "unit"]} SAFE_WS?) | (<TOOL_CALLS> SAFE_WS? "tool2" <ARGS> SAFE_WS? %json {"type": "object"} SAFE_WS?))+\nthink: <THINK> content </THINK>\ncontent: (/(.|\\n)+/)+\nSAFE_WS: /[ \\t\\r\\n]+/',
            ),
            (
                "required",
                False,
                True,
                7,
                [
                    _create_tool(
                        "tool1",
                        {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City and state, e.g., 'San Francisco, CA'",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location", "unit"],
                        },
                        strict=True,
                    ),
                    _create_tool("tool2", {}, strict=False),
                ],
                'start: body\nbody: content? fcalls\nfcalls: <TOOL_CALLS> SAFE_WS? fcall_array SAFE_WS?\nfcall_array: %json {"minItems": 1, "type": "array", "items": {"anyOf": [{"type": "object", "required": ["name", "arguments"], "additionalProperties": false, "properties": {"name": {"const": "tool1"}, "arguments": {"type": "object", "properties": {"location": {"type": "string", "description": "City and state, e.g., \'San Francisco, CA\'"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location", "unit"]}}}, {"type": "object", "required": ["name", "arguments"], "additionalProperties": false, "properties": {"name": {"const": "tool2"}, "arguments": {"type": "object"}}}]}}\n\ncontent: (/(.|\\n)+/)+\nSAFE_WS: /[ \\t\\r\\n]+/',
            ),
            (
                "required",
                False,
                True,
                11,
                [
                    _create_tool(
                        "tool1",
                        {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City and state, e.g., 'San Francisco, CA'",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location", "unit"],
                        },
                        strict=True,
                    ),
                    _create_tool("tool2", {}, strict=False),
                ],
                'start: body\nbody: content? fcalls\nfcalls: ((<TOOL_CALLS> SAFE_WS? "tool1" <ARGS> SAFE_WS? %json {"type": "object", "properties": {"location": {"type": "string", "description": "City and state, e.g., \'San Francisco, CA\'"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location", "unit"]} SAFE_WS?) | (<TOOL_CALLS> SAFE_WS? "tool2" <ARGS> SAFE_WS? %json {"type": "object"} SAFE_WS?))+\ncontent: (/(.|\\n)+/)+\nSAFE_WS: /[ \\t\\r\\n]+/',
            ),
            (
                "required",
                False,
                True,
                13,
                [
                    _create_tool(
                        "tool1",
                        {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City and state, e.g., 'San Francisco, CA'",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location", "unit"],
                        },
                        strict=True,
                    ),
                    _create_tool("tool2", {}, strict=False),
                ],
                'start: body\nbody: content? fcalls\nfcalls: ((<TOOL_CALLS> SAFE_WS? "tool1" <ARGS> SAFE_WS? %json {"type": "object", "properties": {"location": {"type": "string", "description": "City and state, e.g., \'San Francisco, CA\'"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location", "unit"]} SAFE_WS?) | (<TOOL_CALLS> SAFE_WS? "tool2" <ARGS> SAFE_WS? %json {"type": "object"} SAFE_WS?))+\ncontent: (/(.|\\n)+/)+\nSAFE_WS: /[ \\t\\r\\n]+/',
            ),
            (
                "required",
                False,
                False,
                7,
                [_create_tool("tool1", {}, strict=False)],
                'start: body\nbody: content? fcalls\nfcalls: <TOOL_CALLS> SAFE_WS? fcall_array SAFE_WS?\nfcall_array: %json {"minItems": 1, "type": "array", "items": {"type": "object", "additionalProperties": false, "properties": {"name": {"type": "string", "minLength": 1}, "arguments": {"type": "object"}}, "required": ["name", "arguments"]}, "maxItems": 1}\n\ncontent: (/(.|\\n)+/)+\nSAFE_WS: /[ \\t\\r\\n]+/',
            ),
            (
                "required",
                False,
                False,
                7,
                [
                    _create_tool(
                        "tool1",
                        {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City and state, e.g., 'San Francisco, CA'",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location", "unit"],
                        },
                        strict=True,
                    )
                ],
                'start: body\nbody: content? fcalls\nfcalls: <TOOL_CALLS> SAFE_WS? fcall_array SAFE_WS?\nfcall_array: %json {"minItems": 1, "type": "array", "items": {"anyOf": [{"type": "object", "required": ["name", "arguments"], "additionalProperties": false, "properties": {"name": {"const": "tool1"}, "arguments": {"type": "object", "properties": {"location": {"type": "string", "description": "City and state, e.g., \'San Francisco, CA\'"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location", "unit"]}}}]}, "maxItems": 1}\n\ncontent: (/(.|\\n)+/)+\nSAFE_WS: /[ \\t\\r\\n]+/',
            ),
            (
                "required",
                False,
                False,
                7,
                [
                    _create_tool(
                        "tool1",
                        {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City and state, e.g., 'San Francisco, CA'",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location", "unit"],
                        },
                        strict=True,
                    ),
                    _create_tool("tool2", {}, strict=False),
                ],
                'start: body\nbody: content? fcalls\nfcalls: <TOOL_CALLS> SAFE_WS? fcall_array SAFE_WS?\nfcall_array: %json {"minItems": 1, "type": "array", "items": {"anyOf": [{"type": "object", "required": ["name", "arguments"], "additionalProperties": false, "properties": {"name": {"const": "tool1"}, "arguments": {"type": "object", "properties": {"location": {"type": "string", "description": "City and state, e.g., \'San Francisco, CA\'"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location", "unit"]}}}, {"type": "object", "required": ["name", "arguments"], "additionalProperties": false, "properties": {"name": {"const": "tool2"}, "arguments": {"type": "object"}}}]}, "maxItems": 1}\n\ncontent: (/(.|\\n)+/)+\nSAFE_WS: /[ \\t\\r\\n]+/',
            ),
            (
                "required",
                False,
                False,
                11,
                [_create_tool("tool1", {}, strict=False)],
                'start: body\nbody: content? fcalls\nfcalls: <TOOL_CALLS> SAFE_WS? /.+/ <ARGS> SAFE_WS? %json {"type": "object"} SAFE_WS?\ncontent: (/(.|\\n)+/)+\nSAFE_WS: /[ \\t\\r\\n]+/',
            ),
            (
                "required",
                False,
                False,
                11,
                [
                    _create_tool(
                        "tool1",
                        {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City and state, e.g., 'San Francisco, CA'",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location", "unit"],
                        },
                        strict=True,
                    )
                ],
                'start: body\nbody: content? fcalls\nfcalls: (<TOOL_CALLS> SAFE_WS? "tool1" <ARGS> SAFE_WS? %json {"type": "object", "properties": {"location": {"type": "string", "description": "City and state, e.g., \'San Francisco, CA\'"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location", "unit"]} SAFE_WS?)\ncontent: (/(.|\\n)+/)+\nSAFE_WS: /[ \\t\\r\\n]+/',
            ),
            (
                "required",
                False,
                False,
                11,
                [
                    _create_tool(
                        "tool1",
                        {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City and state, e.g., 'San Francisco, CA'",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location", "unit"],
                        },
                        strict=True,
                    ),
                    _create_tool("tool2", {}, strict=False),
                ],
                'start: body\nbody: content? fcalls\nfcalls: (<TOOL_CALLS> SAFE_WS? "tool1" <ARGS> SAFE_WS? %json {"type": "object", "properties": {"location": {"type": "string", "description": "City and state, e.g., \'San Francisco, CA\'"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location", "unit"]} SAFE_WS?) | (<TOOL_CALLS> SAFE_WS? "tool2" <ARGS> SAFE_WS? %json {"type": "object"} SAFE_WS?)\ncontent: (/(.|\\n)+/)+\nSAFE_WS: /[ \\t\\r\\n]+/',
            ),
            (
                "required",
                False,
                False,
                13,
                [_create_tool("tool1", {}, strict=False)],
                'start: body\nbody: content? fcalls\nfcalls: <TOOL_CALLS> SAFE_WS? /.+/ <ARGS> SAFE_WS? %json {"type": "object"} SAFE_WS?\ncontent: (/(.|\\n)+/)+\nSAFE_WS: /[ \\t\\r\\n]+/',
            ),
            (
                "required",
                False,
                False,
                13,
                [
                    _create_tool(
                        "tool1",
                        {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City and state, e.g., 'San Francisco, CA'",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location", "unit"],
                        },
                        strict=True,
                    )
                ],
                'start: body\nbody: content? fcalls\nfcalls: (<TOOL_CALLS> SAFE_WS? "tool1" <ARGS> SAFE_WS? %json {"type": "object", "properties": {"location": {"type": "string", "description": "City and state, e.g., \'San Francisco, CA\'"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location", "unit"]} SAFE_WS?)\ncontent: (/(.|\\n)+/)+\nSAFE_WS: /[ \\t\\r\\n]+/',
            ),
            (
                "required",
                False,
                False,
                13,
                [
                    _create_tool(
                        "tool1",
                        {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City and state, e.g., 'San Francisco, CA'",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location", "unit"],
                        },
                        strict=True,
                    ),
                    _create_tool("tool2", {}, strict=False),
                ],
                'start: body\nbody: content? fcalls\nfcalls: (<TOOL_CALLS> SAFE_WS? "tool1" <ARGS> SAFE_WS? %json {"type": "object", "properties": {"location": {"type": "string", "description": "City and state, e.g., \'San Francisco, CA\'"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location", "unit"]} SAFE_WS?) | (<TOOL_CALLS> SAFE_WS? "tool2" <ARGS> SAFE_WS? %json {"type": "object"} SAFE_WS?)\ncontent: (/(.|\\n)+/)+\nSAFE_WS: /[ \\t\\r\\n]+/',
            ),
            (
                "none",
                False,
                False,
                7,
                [_create_tool("tool1", {}, strict=False)],
                "start: body\nbody: content\ncontent: (/(.|\\n)+/)+\nSAFE_WS: /[ \\t\\r\\n]+/",
            ),
            (
                "none",
                False,
                False,
                7,
                [
                    _create_tool(
                        "tool1",
                        {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City and state, e.g., 'San Francisco, CA'",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location", "unit"],
                        },
                        strict=True,
                    )
                ],
                "start: body\nbody: content\ncontent: (/(.|\\n)+/)+\nSAFE_WS: /[ \\t\\r\\n]+/",
            ),
            (
                "none",
                False,
                False,
                11,
                [_create_tool("tool1", {}, strict=False)],
                "start: body\nbody: content\ncontent: (/(.|\\n)+/)+\nSAFE_WS: /[ \\t\\r\\n]+/",
            ),
            (
                "none",
                False,
                False,
                11,
                [
                    _create_tool(
                        "tool1",
                        {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City and state, e.g., 'San Francisco, CA'",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location", "unit"],
                        },
                        strict=True,
                    )
                ],
                "start: body\nbody: content\ncontent: (/(.|\\n)+/)+\nSAFE_WS: /[ \\t\\r\\n]+/",
            ),
            (
                "none",
                False,
                False,
                13,
                [_create_tool("tool1", {}, strict=False)],
                "start: body\nbody: content\ncontent: (/(.|\\n)+/)+\nSAFE_WS: /[ \\t\\r\\n]+/",
            ),
            (
                "none",
                False,
                False,
                13,
                [
                    _create_tool(
                        "tool1",
                        {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City and state, e.g., 'San Francisco, CA'",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location", "unit"],
                        },
                        strict=True,
                    )
                ],
                "start: body\nbody: content\ncontent: (/(.|\\n)+/)+\nSAFE_WS: /[ \\t\\r\\n]+/",
            ),
        ],
    )
    def test_get_lark_from_jinja(
        self,
        mock_tokenizer_v7,
        mock_tokenizer_v11,
        mock_tokenizer_v13,
        mode: str,
        reasoning: bool,
        parallel_tool_calls: bool,
        tokenizer_version: int,
        tools: list[str],
        expected: str,
    ):
        # Create the grammar factory
        match tokenizer_version:
            case 7:
                tokenizer = mock_tokenizer_v7
            case 11:
                tokenizer = mock_tokenizer_v11
            case 13:
                tokenizer = mock_tokenizer_v13
            case _:
                raise AssertionError(f"wrong {tokenizer_version=}")
        factory = MistralGrammarFactory(tokenizer)

        # Generate the grammar
        grammar = factory.get_lark_from_jinja(
            mode=mode,
            tools=tools,
            reasoning=reasoning,
            parallel_tool_calls=parallel_tool_calls,
        )

        # Verify that the grammar matches exactly
        assert grammar == expected, (
            f"Grammar mismatch:\nExpected:\n{expected}\n\nGot:\n{grammar}"
        )
