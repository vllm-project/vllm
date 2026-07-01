# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import partial_json_parser
import pytest
from mistral_common.protocol.instruct.messages import AssistantMessage
from mistral_common.protocol.instruct.request import InstructRequest
from mistral_common.protocol.instruct.tool_calls import (
    FunctionCall,
    ToolCall,
)
from mistral_common.protocol.instruct.tool_calls import (
    NamedToolChoice as MistralNamedToolChoice,
)
from mistral_common.protocol.instruct.tool_calls import (
    ToolChoice as MistralToolChoice,
)
from mistral_common.protocol.instruct.tool_calls import (
    ToolChoiceEnum as MistralToolChoiceEnum,
)
from partial_json_parser.core.options import Allow

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    StructuralTagResponseFormat,
)
from vllm.sampling_params import StructuredOutputsParams
from vllm.tokenizers import TokenizerLike, get_tokenizer
from vllm.tokenizers.detokenizer_utils import detokenize_incrementally
from vllm.tokenizers.mistral import MistralTokenizer
from vllm.tool_parsers.mistral_tool_parser import (
    _DEFAULT_JSON_SCHEMA,
    MistralToolParser,
)

_DUMMY_REQUEST = ChatCompletionRequest(messages=[], model="test")


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


@pytest.fixture
def non_mistral_parser() -> MistralToolParser:
    mock_tokenizer = MagicMock()
    mock_tokenizer.get_vocab.return_value = {"[TOOL_CALLS]": 1}
    return MistralToolParser(mock_tokenizer)


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
            request=_DUMMY_REQUEST,
        )
        if delta_message:
            yield delta_message

        previous_text = current_text
        previous_tokens = (
            previous_tokens + new_tokens if previous_tokens else new_tokens
        )
        prefix_offset = new_prefix_offset
        read_offset = new_read_offset


@pytest.mark.parametrize(
    "parser_fixture",
    ["mistral_pre_v11_tool_parser", "mistral_tool_parser"],
    ids=["pre_v11", "v11"],
)
def test_extract_tool_calls_no_tools(parser_fixture, request):
    parser = request.getfixturevalue(parser_fixture)
    model_output = "This is a test"
    result = parser.extract_tool_calls(model_output, request=_DUMMY_REQUEST)
    assert result == ExtractedToolCallInformation(
        tools_called=False, tool_calls=[], content=model_output
    )


@pytest.mark.parametrize(
    ids=[
        "single_tool_add",
        "single_tool_weather",
        "argument_before_name",
        "argument_before_name_and_name_in_argument",
        "multiple_tools",
        "content_before_tool",
        "trailing_data_after_json",
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
            None,
        ),
        (
            """Hello[TOOL_CALLS] [{"name": "add", "arguments":{"a": 1, "b": 2}}]""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="add", arguments=json.dumps({"a": 1, "b": 2})
                    )
                )
            ],
            "Hello",
        ),
        (
            """[TOOL_CALLS] [{"name": "get_current_weather", "arguments":{"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}]\nextra trailing data""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {
                                "city": "Dallas",
                                "state": "TX",
                                "unit": "fahrenheit",
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
        model_output, request=_DUMMY_REQUEST
    )
    assert extracted_tool_calls.tools_called

    assert_tool_calls(extracted_tool_calls.tool_calls, expected_tool_calls)

    assert extracted_tool_calls.content == expected_content


def test_extract_tool_calls_pre_v11_multiple_bot_tokens_raises(
    mistral_pre_v11_tool_parser,
):
    model_output = (
        '[TOOL_CALLS] [{"name": "add", "arguments":{"a": 1}}]'
        '[TOOL_CALLS] [{"name": "sub", "arguments":{"b": 2}}]'
    )
    with pytest.raises(ValueError, match="Only one BOT token"):
        mistral_pre_v11_tool_parser.extract_tool_calls(
            model_output, request=_DUMMY_REQUEST
        )


def test_extract_tool_calls_pre_v11_regex_fallback(
    mistral_pre_v11_tool_parser,
):
    """The regex fallback path finds valid JSON via regex when the primary
    raw_decode fails on leading junk. It should re-serialize arguments
    and return a valid tool call."""
    model_output = (
        '[TOOL_CALLS]  junk [{"name": "add", "arguments":{"a": 1, "b": 2}}] trail'
    )
    result = mistral_pre_v11_tool_parser.extract_tool_calls(
        model_output, request=_DUMMY_REQUEST
    )
    assert result.tools_called
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "add"
    assert result.tool_calls[0].function.arguments == json.dumps({"a": 1, "b": 2})


def test_extract_tool_calls_pre_v11_regex_fallback_fails(
    mistral_pre_v11_tool_parser,
):
    model_output = "[TOOL_CALLS] not json at all"
    result = mistral_pre_v11_tool_parser.extract_tool_calls(
        model_output, request=_DUMMY_REQUEST
    )
    assert result == ExtractedToolCallInformation(
        tools_called=False, tool_calls=[], content="not json at all"
    )


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
        model_output, request=_DUMMY_REQUEST
    )
    assert extracted_tool_calls.tools_called

    assert_tool_calls(extracted_tool_calls.tool_calls, expected_tool_calls)

    assert extracted_tool_calls.content == expected_content


def test_extract_tool_calls_v11_without_args_skipped(mistral_tool_parser):
    model_output = "[TOOL_CALLS]toolname_no_args"
    result = mistral_tool_parser.extract_tool_calls(
        model_output, request=_DUMMY_REQUEST
    )
    assert result == ExtractedToolCallInformation(
        tools_called=True, tool_calls=[], content=None
    )


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

    if expected_tool_calls:
        assert len(tool_parser.streamed_args_for_tool) == len(expected_tool_calls)
        assert len(tool_parser.prev_tool_call_arr) == len(expected_tool_calls)
        for i in range(len(expected_tool_calls)):
            assert (
                tool_parser.prev_tool_call_arr[i]["arguments"]
                == tool_parser.streamed_args_for_tool[i]
            )
            assert tool_parser.streamed_args_for_tool[i] == function_args_strs[i]
            assert (
                tool_parser.prev_tool_call_arr[i]["name"]
                == expected_tool_calls[i].function.name
            )

        # Simulate the serving layer's unstreamed-args check
        index = len(tool_parser.prev_tool_call_arr) - 1
        args = tool_parser.prev_tool_call_arr[index].get("arguments", {})
        expected_call = (
            args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
        )
        actual_call = tool_parser.streamed_args_for_tool[index]
        remaining_call = expected_call.replace(actual_call, "", 1)
        assert remaining_call == ""
    else:
        assert len(tool_parser.streamed_args_for_tool) == 0
        assert len(tool_parser.prev_tool_call_arr) == 0


@pytest.mark.parametrize(
    ids=[
        "no_tools",
        "single_tool_add",
        "single_tool_add_strings",
        "single_tool_weather",
        "argument_before_name",
        "argument_before_name_and_name_in_argument",
        "multiple_tools",
        "trailing_data_after_json",
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
        (
            """[TOOL_CALLS] [{"name": "get_current_weather", "arguments":{"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}]\nextra trailing data""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments=json.dumps(
                            {
                                "city": "Dallas",
                                "state": "TX",
                                "unit": "fahrenheit",
                            }
                        ),
                    )
                )
            ],
            "\nextra trailing data",
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


def test_extract_tool_calls_streaming_v11_no_tools(
    mistral_tool_parser, mistral_tokenizer
):
    model_output = "This is a test"
    if isinstance(mistral_tokenizer, MistralTokenizer):
        all_token_ids = mistral_tokenizer.encode(model_output)
    else:
        all_token_ids = mistral_tokenizer.encode(model_output, add_special_tokens=False)
    skip_special = isinstance(mistral_tokenizer, MistralTokenizer)
    collected_content = ""
    previous_text = ""
    previous_tokens = None
    prefix_offset = 0
    read_offset = 0
    for i in range(len(all_token_ids)):
        current_token_ids = all_token_ids[: i + 1]
        previous_token_ids = all_token_ids[:i]
        delta_token_ids = [all_token_ids[i]]

        new_tokens, delta_text, prefix_offset, read_offset = detokenize_incrementally(
            tokenizer=mistral_tokenizer,
            all_input_ids=current_token_ids,
            prev_tokens=previous_tokens,
            prefix_offset=prefix_offset,
            read_offset=read_offset,
            skip_special_tokens=skip_special,
            spaces_between_special_tokens=True,
        )
        current_text = previous_text + delta_text
        previous_tokens = (
            previous_tokens + new_tokens if previous_tokens else new_tokens
        )

        delta_message = mistral_tool_parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=previous_token_ids,
            current_token_ids=current_token_ids,
            delta_token_ids=delta_token_ids,
            request=_DUMMY_REQUEST,
        )
        if delta_message and delta_message.content:
            collected_content += delta_message.content
        if delta_message:
            assert not delta_message.tool_calls

        previous_text = current_text

    assert collected_content == model_output
    assert len(mistral_tool_parser.streamed_args_for_tool) == 0
    assert len(mistral_tool_parser.prev_tool_call_arr) == 0


@pytest.mark.parametrize(
    "parser_fixture, tokenizer_fixture, model_output,"
    " expected_tool_calls, expected_content",
    [
        pytest.param(
            "mistral_tool_parser",
            "mistral_tokenizer",
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
            id="v11-single_tool_add",
        ),
        pytest.param(
            "mistral_tool_parser",
            "mistral_tokenizer",
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
            id="v11-single_tool_weather",
        ),
        pytest.param(
            "mistral_tool_parser",
            "mistral_tokenizer",
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
            id="v11-multiple_tool_calls",
        ),
        pytest.param(
            "mistral_tool_parser",
            "mistral_tokenizer",
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
            id="v11-content_before_tool",
        ),
        pytest.param(
            "mistral_tool_parser",
            "mistral_tokenizer",
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
            id="v11-complex",
        ),
        pytest.param(
            "mistral_pre_v11_tool_parser",
            "mistral_pre_v11_tokenizer",
            """This is a test""",
            [],
            """This is a test""",
            id="pre_v11-no_tools",
        ),
        pytest.param(
            "mistral_pre_v11_tool_parser",
            "mistral_pre_v11_tokenizer",
            """[TOOL_CALLS]  [ {"name":"add" , "arguments" : {"a": 3, "b": 4} } ]""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="add", arguments=json.dumps({"a": 3, "b": 4})
                    )
                )
            ],
            "",
            id="pre_v11-single_tool_add",
        ),
        pytest.param(
            "mistral_pre_v11_tool_parser",
            "mistral_pre_v11_tokenizer",
            """[TOOL_CALLS] [{"name": "add", "arguments":{"a": "3", "b": "4"}}]""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="add", arguments=json.dumps({"a": "3", "b": "4"})
                    )
                )
            ],
            "",
            id="pre_v11-single_tool_add_strings",
        ),
        pytest.param(
            "mistral_pre_v11_tool_parser",
            "mistral_pre_v11_tokenizer",
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
            id="pre_v11-single_tool_weather",
        ),
        pytest.param(
            "mistral_pre_v11_tool_parser",
            "mistral_pre_v11_tokenizer",
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
            id="pre_v11-argument_before_name",
        ),
        pytest.param(
            "mistral_pre_v11_tool_parser",
            "mistral_pre_v11_tokenizer",
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
            id="pre_v11-argument_before_name_and_name_in_argument",
        ),
        pytest.param(
            "mistral_pre_v11_tool_parser",
            "mistral_pre_v11_tokenizer",
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
            id="pre_v11-multiple_tools",
        ),
        pytest.param(
            "mistral_pre_v11_tool_parser",
            "mistral_pre_v11_tokenizer",
            """Some text[TOOL_CALLS] [{"name": "add", "arguments":{"a": 1, "b": 2}}]""",  # noqa: E501
            [
                ToolCall(
                    function=FunctionCall(
                        name="add", arguments=json.dumps({"a": 1, "b": 2})
                    )
                )
            ],
            "Some text",
            id="pre_v11-content_before_tool",
        ),
    ],
)
def test_extract_tool_calls_streaming_one_chunk(
    parser_fixture,
    tokenizer_fixture,
    model_output,
    expected_tool_calls,
    expected_content,
    request,
):
    tool_parser = request.getfixturevalue(parser_fixture)
    tokenizer = request.getfixturevalue(tokenizer_fixture)

    if isinstance(tokenizer, MistralTokenizer):
        all_token_ids = tokenizer.encode(model_output)
    else:
        all_token_ids = tokenizer.encode(model_output, add_special_tokens=False)
    all_token_ids = fix_tool_call_tokenization(all_token_ids, tool_parser, tokenizer)

    delta_message = tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=model_output,
        delta_text=model_output,
        previous_token_ids=[],
        current_token_ids=all_token_ids,
        delta_token_ids=all_token_ids,
        request=_DUMMY_REQUEST,
    )
    assert isinstance(delta_message, DeltaMessage)
    assert len(delta_message.tool_calls) == len(expected_tool_calls)

    assert_tool_calls(delta_message.tool_calls, expected_tool_calls)

    if delta_message.content is None:
        assert expected_content == ""
    else:
        assert delta_message.content == expected_content


@pytest.mark.parametrize(
    "parser_fixture, model_output, fake_count, two_phase",
    [
        pytest.param(
            "mistral_tool_parser",
            '[TOOL_CALLS]add{"a": 1, "b": 2}',
            20,
            True,
            id="v11",
        ),
        pytest.param(
            "mistral_pre_v11_tool_parser",
            '[TOOL_CALLS] [{"name": "add", "arguments":{"a": 1, "b": 2}}]',
            30,
            False,
            id="pre_v11",
        ),
    ],
)
def test_fast_detokenization_text_detection(
    parser_fixture, model_output, fake_count, two_phase, request
):
    """Regression: bot_token in text but not token_ids (PR #37209)."""
    parser = request.getfixturevalue(parser_fixture)
    # Token IDs that do NOT contain bot_token_id.
    fake_token_ids = list(range(99, 99 + fake_count))

    if two_phase:
        # First delta: pure content, no bot token yet
        delta_message_before = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="Hello",
            delta_text="Hello",
            previous_token_ids=[],
            current_token_ids=[99],
            delta_token_ids=[99],
            request=_DUMMY_REQUEST,
        )
        assert delta_message_before is not None
        assert delta_message_before.content == "Hello"
        assert not delta_message_before.tool_calls

        previous_text = "Hello"
        current_text = "Hello" + model_output
        previous_token_ids = [99]
        delta_token_ids = fake_token_ids[1:]
    else:
        previous_text = ""
        current_text = model_output
        previous_token_ids = []
        delta_token_ids = fake_token_ids

    delta_message = parser.extract_tool_calls_streaming(
        previous_text=previous_text,
        current_text=current_text,
        delta_text=model_output,
        previous_token_ids=previous_token_ids,
        current_token_ids=fake_token_ids,
        delta_token_ids=delta_token_ids,
        request=_DUMMY_REQUEST,
    )
    assert delta_message is not None
    assert delta_message.tool_calls is not None
    assert len(delta_message.tool_calls) == 1
    assert delta_message.tool_calls[0].function is not None
    assert delta_message.tool_calls[0].function.name == "add"


@pytest.mark.parametrize(
    "parser_fixture, patched_method, current_text",
    [
        (
            "mistral_tool_parser",
            "_extract_tool_calls_streaming",
            "[TOOL_CALLS]add{}",
        ),
        (
            "mistral_pre_v11_tool_parser",
            "_extract_tool_calls_streaming_pre_v11_tokenizer",
            '[TOOL_CALLS] [{"name":"a","arguments":{}}]',
        ),
    ],
    ids=["v11", "pre_v11"],
)
def test_extract_tool_calls_streaming_exception_returns_none(
    parser_fixture, patched_method, current_text, request
):
    parser = request.getfixturevalue(parser_fixture)
    with patch.object(parser, patched_method, side_effect=RuntimeError("boom")):
        result = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=current_text,
            delta_text=current_text,
            previous_token_ids=[],
            current_token_ids=[parser.bot_token_id],
            delta_token_ids=[parser.bot_token_id],
            request=_DUMMY_REQUEST,
        )
    assert result is None


SAMPLE_TOOLS_DICTS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        },
    },
]


def _make_request(**kwargs) -> ChatCompletionRequest:
    defaults: dict = {
        "messages": [],
        "model": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "tools": SAMPLE_TOOLS_DICTS,
        "tool_choice": "auto",
    }
    defaults.update(kwargs)
    return ChatCompletionRequest(**defaults)


@pytest.mark.parametrize(
    "request_kwargs,expected_mode,expected_parallel",
    [
        ({"tool_choice": "auto"}, MistralToolChoiceEnum.auto, True),
        ({"tool_choice": "none"}, MistralToolChoiceEnum.none, True),
        ({"tool_choice": "required"}, MistralToolChoiceEnum.required, True),
        ({"tool_choice": None, "tools": None}, MistralToolChoiceEnum.auto, True),
        (
            {
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "get_weather"},
                }
            },
            MistralNamedToolChoice.model_validate(
                {"type": "function", "function": {"name": "get_weather"}}
            ),
            True,
        ),
        (
            {"tool_choice": "auto", "parallel_tool_calls": False},
            MistralToolChoiceEnum.auto,
            False,
        ),
        (
            {"tool_choice": "auto", "response_format": {"type": "text"}},
            MistralToolChoiceEnum.auto,
            True,
        ),
    ],
    ids=[
        "auto",
        "none",
        "required",
        "null_tool_choice",
        "named_tool_choice",
        "parallel_false",
        "response_format_text",
    ],
)
def test_adjust_request_grammar_factory(
    mistral_tool_parser: MistralToolParser,
    request_kwargs: dict,
    expected_mode: MistralToolChoice,
    expected_parallel: bool,
) -> None:
    request = _make_request(**request_kwargs)
    factory = mistral_tool_parser.model_tokenizer.grammar_factory

    with patch.object(
        factory,
        "get_lark_from_jinja",
        wraps=factory.get_lark_from_jinja,
    ) as mock_get_lark:
        result = mistral_tool_parser.adjust_request(request)

        mock_get_lark.assert_called_once()
        call_kwargs = mock_get_lark.call_args

        assert call_kwargs.kwargs["mode"] == expected_mode
        assert call_kwargs.kwargs["json_schema"] is None
        assert call_kwargs.kwargs["parallel_tool_calls"] == expected_parallel

    assert result.structured_outputs is not None
    assert isinstance(result.structured_outputs.grammar, str)
    assert len(result.structured_outputs.grammar) > 0


def test_adjust_request_unsupported_grammar_for_tokenizer(mistral_tokenizer) -> None:
    with patch.object(
        type(mistral_tokenizer),
        "supports_grammar",
        new_callable=lambda: property(lambda self: False),
    ):
        parser = MistralToolParser(mistral_tokenizer)
        request = _make_request()
        result = parser.adjust_request(request)

        assert result.structured_outputs is None


@pytest.mark.parametrize(
    "tool_choice,expected_skip",
    [("auto", False), ("none", True)],
    ids=["auto_skip_false", "none_skip_true"],
)
def test_adjust_request_non_mistral_tokenizer(
    non_mistral_parser: MistralToolParser,
    tool_choice: str,
    expected_skip: bool,
) -> None:
    request = _make_request(tool_choice=tool_choice)
    result = non_mistral_parser.adjust_request(request)

    assert result.skip_special_tokens is expected_skip


@pytest.mark.parametrize(
    "so_kwargs",
    [
        {"regex": r"\d+"},
        {"choice": ["a", "b"]},
        {
            "structural_tag": json.dumps(
                {
                    "structures": [
                        {
                            "begin": "<tool>",
                            "schema": {"type": "object"},
                            "end": "</tool>",
                        }
                    ],
                    "triggers": ["<tool>"],
                }
            )
        },
        {"grammar": "start: 'hello'"},
    ],
    ids=["regex", "choice", "structural_tag", "grammar"],
)
def test_adjust_request_unsupported_structured_outputs(
    mistral_tool_parser: MistralToolParser,
    so_kwargs: dict,
) -> None:
    request = _make_request(
        structured_outputs=StructuredOutputsParams(**so_kwargs),
    )
    result = mistral_tool_parser.adjust_request(request)

    assert result.structured_outputs == request.structured_outputs


def test_adjust_request_unsupported_response_format(
    mistral_tool_parser: MistralToolParser,
) -> None:
    request = _make_request(
        response_format=StructuralTagResponseFormat(
            type="structural_tag",
            format={
                "type": "triggered_tags",
                "tags": [
                    {
                        "begin": "<tool>",
                        "content": {"type": "any_text"},
                        "end": "</tool>",
                    }
                ],
                "triggers": ["<tool>"],
            },
        ),
    )
    result = mistral_tool_parser.adjust_request(request)
    assert result.structured_outputs is None
    assert result.response_format == request.response_format


@pytest.mark.parametrize(
    "so_kwargs,expected_json_schema",
    [
        ({"json_object": True}, _DEFAULT_JSON_SCHEMA),
        ({"json": '{"type": "object"}'}, {"type": "object"}),
        (
            {"json": {"type": "object", "properties": {"x": {"type": "integer"}}}},
            {"type": "object", "properties": {"x": {"type": "integer"}}},
        ),
    ],
    ids=["json_object", "json_str", "json_dict"],
)
def test_adjust_request_structured_outputs_generates_grammar(
    mistral_tool_parser: MistralToolParser,
    so_kwargs: dict,
    expected_json_schema: str,
) -> None:
    request = _make_request(
        structured_outputs=StructuredOutputsParams(**so_kwargs),
    )
    factory = mistral_tool_parser.model_tokenizer.grammar_factory

    with patch.object(
        factory,
        "get_lark_from_jinja",
        wraps=factory.get_lark_from_jinja,
    ) as mock_get_lark:
        result = mistral_tool_parser.adjust_request(request)

        mock_get_lark.assert_called_once()
        assert mock_get_lark.call_args.kwargs["json_schema"] == expected_json_schema

    assert result.structured_outputs is not None
    assert isinstance(result.structured_outputs.grammar, str)
    assert len(result.structured_outputs.grammar) > 0


@pytest.mark.parametrize(
    "response_format_kwargs,expected_json_schema",
    [
        ({"type": "json_object"}, _DEFAULT_JSON_SCHEMA),
        (
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "my_schema",
                    "schema": {
                        "type": "object",
                        "properties": {"x": {"type": "integer"}},
                    },
                },
            },
            {"type": "object", "properties": {"x": {"type": "integer"}}},
        ),
    ],
    ids=["json_object", "json_schema_with_schema"],
)
def test_adjust_request_response_format_generates_grammar(
    mistral_tool_parser: MistralToolParser,
    response_format_kwargs: dict,
    expected_json_schema: str,
) -> None:
    request = _make_request(response_format=response_format_kwargs)
    factory = mistral_tool_parser.model_tokenizer.grammar_factory

    with patch.object(
        factory,
        "get_lark_from_jinja",
        wraps=factory.get_lark_from_jinja,
    ) as mock_get_lark:
        result = mistral_tool_parser.adjust_request(request)

        mock_get_lark.assert_called_once()
        assert mock_get_lark.call_args.kwargs["json_schema"] == expected_json_schema

    assert result.structured_outputs is not None
    assert isinstance(result.structured_outputs.grammar, str)
    assert len(result.structured_outputs.grammar) > 0


@pytest.mark.parametrize(
    "tool_choice, expected_method, not_called_method",
    [
        ("none", "get_lark_for_json_schema", None),
        ("auto", "get_lark_from_jinja", "get_lark_for_json_schema"),
    ],
    ids=["none_uses_json_schema_factory", "auto_uses_jinja_factory"],
)
def test_adjust_request_tool_choice_with_json_schema_factory_routing(
    mistral_tool_parser: MistralToolParser,
    tool_choice: str,
    expected_method: str,
    not_called_method: str | None,
) -> None:
    request = _make_request(
        tool_choice=tool_choice,
        structured_outputs=StructuredOutputsParams(json='{"type": "object"}'),
    )
    factory = mistral_tool_parser.model_tokenizer.grammar_factory

    patches = {
        expected_method: patch.object(
            factory,
            expected_method,
            wraps=getattr(factory, expected_method),
        ),
    }
    if not_called_method:
        patches[not_called_method] = patch.object(
            factory,
            not_called_method,
            wraps=getattr(factory, not_called_method),
        )

    with patches[expected_method] as mock_expected:
        ctx = patches[not_called_method] if not_called_method else None
        if ctx:
            with ctx as mock_not_called:
                result = mistral_tool_parser.adjust_request(request)
                mock_not_called.assert_not_called()
        else:
            result = mistral_tool_parser.adjust_request(request)

        mock_expected.assert_called_once()
        assert mock_expected.call_args.kwargs["json_schema"] == {"type": "object"}

    assert result.structured_outputs is not None
    assert isinstance(result.structured_outputs.grammar, str)
    assert len(result.structured_outputs.grammar) > 0


def test_grammar_from_tool_parser_default_false() -> None:
    request = _make_request()
    assert request._grammar_from_tool_parser is False


def test_grammar_from_tool_parser_set_by_adjust_request(
    mistral_tool_parser: MistralToolParser,
) -> None:
    request = _make_request()
    result = mistral_tool_parser.adjust_request(request)
    assert result._grammar_from_tool_parser is True
