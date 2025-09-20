# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Generator
from typing import Optional

import partial_json_parser
import pytest
from partial_json_parser.core.options import Allow

from vllm.entrypoints.openai.protocol import (DeltaMessage, FunctionCall,
                                              ToolCall)
from vllm.entrypoints.openai.tool_parsers import MistralToolParser
from vllm.transformers_utils.detokenizer import detokenize_incrementally
from vllm.transformers_utils.tokenizer import AnyTokenizer, get_tokenizer

MODEL = "jeffcookio/Mistral-Small-3.2-24B-Instruct-2506-awq-sym"


@pytest.fixture(scope="module")
def mistral_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)


@pytest.fixture
def mistral_tool_parser(mistral_tokenizer):
    return MistralToolParser(mistral_tokenizer)


def assert_tool_calls(actual_tool_calls: list[ToolCall],
                      expected_tool_calls: list[ToolCall]):
    assert len(actual_tool_calls) == len(expected_tool_calls)

    for actual_tool_call, expected_tool_call in zip(actual_tool_calls,
                                                    expected_tool_calls):
        assert isinstance(actual_tool_call.id, str)
        assert len(actual_tool_call.id) == 9

        assert actual_tool_call.type == "function"
        assert actual_tool_call.function == expected_tool_call.function, (
            f'got ${actual_tool_call.function}')


def stream_delta_message_generator(
        mistral_tool_parser: MistralToolParser,
        mistral_tokenizer: AnyTokenizer,
        model_output: str) -> Generator[DeltaMessage, None, None]:
    all_token_ids = mistral_tokenizer.encode(model_output,
                                             add_special_tokens=False)

    previous_text = ""
    previous_tokens = None
    prefix_offset = 0
    read_offset = 0
    for i, delta_token in enumerate(all_token_ids):
        delta_token_ids = [delta_token]
        previous_token_ids = all_token_ids[:i]
        current_token_ids = all_token_ids[:i + 1]

        (new_tokens, delta_text, new_prefix_offset,
         new_read_offset) = detokenize_incrementally(
             tokenizer=mistral_tokenizer,
             all_input_ids=current_token_ids,
             prev_tokens=previous_tokens,
             prefix_offset=prefix_offset,
             read_offset=read_offset,
             skip_special_tokens=False,
             spaces_between_special_tokens=True,
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
        previous_tokens = previous_tokens + new_tokens if previous_tokens\
            else new_tokens
        prefix_offset = new_prefix_offset
        read_offset = new_read_offset


def test_extract_tool_calls_no_tools(mistral_tool_parser):
    model_output = "This is a test"
    extracted_tool_calls = mistral_tool_parser.extract_tool_calls(
        model_output, request=None)  # type: ignore[arg-type]
    assert not extracted_tool_calls.tools_called
    assert extracted_tool_calls.tool_calls == []
    assert extracted_tool_calls.content == model_output


@pytest.mark.parametrize(
    ids=[
        "single_tool_add", "single_tool_weather", "argument_before_name",
        "argument_before_name_and_name_in_argument"
    ],
    argnames=["model_output", "expected_tool_calls", "expected_content"],
    argvalues=[
        (
            '''[TOOL_CALLS][{"name": "add", "arguments":{"a": 3.5, "b": 4}}]''',  # noqa: E501
            [
                ToolCall(function=FunctionCall(name="add",
                                               arguments=json.dumps({
                                                   "a": 3.5,
                                                   "b": 4
                                               })))
            ],
            None),
        (
            '''[TOOL_CALLS] [{"name": "get_current_weather", "arguments":{"city": "San Francisco", "state": "CA", "unit": "celsius"}}]''',  # noqa: E501
            [
                ToolCall(function=FunctionCall(name="get_current_weather",
                                               arguments=json.dumps(
                                                   {
                                                       "city": "San Francisco",
                                                       "state": "CA",
                                                       "unit": "celsius"
                                                   })))
            ],
            None),
        (
            '''[TOOL_CALLS] [{"arguments":{"city": "San Francisco", "state": "CA", "unit": "celsius"}, "name": "get_current_weather"}]''',  # noqa: E501
            [
                ToolCall(function=FunctionCall(name="get_current_weather",
                                               arguments=json.dumps(
                                                   {
                                                       "city": "San Francisco",
                                                       "state": "CA",
                                                       "unit": "celsius"
                                                   })))
            ],
            None),
        (
            '''[TOOL_CALLS] [{"arguments":{"name": "John Doe"}, "name": "get_age"}]''',  # noqa: E501
            [
                ToolCall(function=FunctionCall(name="get_age",
                                               arguments=json.dumps({
                                                   "name":
                                                   "John Doe",
                                               })))
            ],
            None),
    ],
)
def test_extract_tool_calls(mistral_tool_parser, model_output,
                            expected_tool_calls, expected_content):
    extracted_tool_calls = mistral_tool_parser.extract_tool_calls(
        model_output, request=None)  # type: ignore[arg-type]
    assert extracted_tool_calls.tools_called

    assert_tool_calls(extracted_tool_calls.tool_calls, expected_tool_calls)

    assert extracted_tool_calls.content == expected_content


@pytest.mark.parametrize(
    ids=[
        "no_tools",
        "single_tool_add",
        "single_tool_add_strings",
        "single_tool_weather",
        "argument_before_name",
        "argument_before_name_and_name_in_argument",
        "multiple_tools",
        "v11_single_tool",
        "v11_multiple_tools",
        "v11_nested_json",
        "v11_special_chars",
        "v11_empty_args",
        "v11_complex_nested",
        "v11_with_comma_separator",
        "v11_with_whitespace_separator",
    ],
    argnames=["model_output", "expected_tool_calls", "expected_content"],
    argvalues=[
        ('''This is a test''', [], '''This is a test'''),
        (
            '''[TOOL_CALLS]  [ {"name":"add" , "arguments" : {"a": 3, "b": 4} } ]''',  # noqa: E501
            [
                ToolCall(function=FunctionCall(name="add",
                                               arguments=json.dumps({
                                                   "a": 3,
                                                   "b": 4
                                               })))
            ],
            ""),
        (
            '''[TOOL_CALLS] [{"name": "add", "arguments":{"a": "3", "b": "4"}}]''',  # noqa: E501
            [
                ToolCall(function=FunctionCall(name="add",
                                               arguments=json.dumps({
                                                   "a": "3",
                                                   "b": "4"
                                               })))
            ],
            ""),
        (
            '''[TOOL_CALLS] [{"name": "get_current_weather", "arguments": {"city": "San Francisco", "state": "CA", "unit": "celsius"}}]''',  # noqa: E501
            [
                ToolCall(function=FunctionCall(name="get_current_weather",
                                               arguments=json.dumps(
                                                   {
                                                       "city": "San Francisco",
                                                       "state": "CA",
                                                       "unit": "celsius"
                                                   })))
            ],
            ""),
        (
            '''[TOOL_CALLS] [{"arguments": {"city": "San Francisco", "state": "CA", "unit": "celsius"}, "name": "get_current_weather"}]''',  # noqa: E501
            [
                ToolCall(function=FunctionCall(name="get_current_weather",
                                               arguments=json.dumps(
                                                   {
                                                       "city": "San Francisco",
                                                       "state": "CA",
                                                       "unit": "celsius"
                                                   })))
            ],
            ''),
        (
            '''[TOOL_CALLS] [{"arguments": {"name": "John Doe"}, "name": "get_age"}]''',  # noqa: E501
            [
                ToolCall(function=FunctionCall(name="get_age",
                                               arguments=json.dumps({
                                                   "name":
                                                   "John Doe",
                                               })))
            ],
            ''),
        (
            '''[TOOL_CALLS][{"name": "add", "arguments": {"a": 3.5, "b": 4}}, {"name": "get_current_weather", "arguments":{"city": "San Francisco", "state": "CA", "unit": "celsius"}]''',  # noqa: E501
            [
                ToolCall(function=FunctionCall(name="add",
                                               arguments=json.dumps({
                                                   "a": 3.5,
                                                   "b": 4
                                               }))),
                ToolCall(function=FunctionCall(name="get_current_weather",
                                               arguments=json.dumps(
                                                   {
                                                       "city": "San Francisco",
                                                       "state": "CA",
                                                       "unit": "celsius"
                                                   })))
            ],
            ''),
        # V11 format tests
        (
            '''[TOOL_CALLS] add{"a": 3, "b": 4}''',
            [
                ToolCall(function=FunctionCall(name="add",
                                               arguments=json.dumps({
                                                   "a": 3,
                                                   "b": 4
                                               })))
            ],
            ""),
        (
            '''[TOOL_CALLS] add{"a": 3, "b": 4}, get_weather{"city": "Paris", "unit": "celsius"}''',  # noqa: E501
            [
                ToolCall(function=FunctionCall(name="add",
                                               arguments=json.dumps({
                                                   "a": 3,
                                                   "b": 4
                                               }))),
                ToolCall(function=FunctionCall(name="get_weather",
                                               arguments=json.dumps({
                                                   "city": "Paris",
                                                   "unit": "celsius"
                                               })))
            ],
            ""),
        (
            '''[TOOL_CALLS] process_data{"input": {"nested": {"value": 42, "array": [1, 2, 3]}, "flag": true}}''',  # noqa: E501
            [
                ToolCall(function=FunctionCall(name="process_data",
                                               arguments=json.dumps({
                                                   "input": {
                                                       "nested": {
                                                           "value": 42,
                                                           "array": [1, 2, 3]
                                                       },
                                                       "flag": True
                                                   }
                                               })))
            ],
            ""),
        (
            '''[TOOL_CALLS] send_message{"text": "Hello, it's a nice day!", "recipient": "user@example.com"}''',  # noqa: E501
            [
                ToolCall(function=FunctionCall(name="send_message",
                                               arguments=json.dumps({
                                                   "text": "Hello, it's a nice day!",
                                                   "recipient": "user@example.com"
                                               })))
            ],
            ""),
        (
            '''[TOOL_CALLS] empty_function{}''',
            [
                ToolCall(function=FunctionCall(name="empty_function",
                                               arguments=json.dumps({})))
            ],
            ""),
        (
            '''[TOOL_CALLS] complex_tool{"data": {"items": [{"id": 1, "props": {"key": "value"}}, {"id": 2, "props": {"key": "other"}}], "meta": {"count": 2}}}''',  # noqa: E501
            [
                ToolCall(function=FunctionCall(name="complex_tool",
                                               arguments=json.dumps({
                                                   "data": {
                                                       "items": [
                                                           {"id": 1, "props": {"key": "value"}},
                                                           {"id": 2, "props": {"key": "other"}}
                                                       ],
                                                       "meta": {"count": 2}
                                                   }
                                               })))
            ],
            ""),
        (
            '''[TOOL_CALLS] first_tool{"x": 1}, second_tool{"y": 2}''',
            [
                ToolCall(function=FunctionCall(name="first_tool",
                                               arguments=json.dumps({"x": 1}))),
                ToolCall(function=FunctionCall(name="second_tool",
                                               arguments=json.dumps({"y": 2})))
            ],
            ""),
        (
            '''[TOOL_CALLS] tool_a{"param": "A"} tool_b{"param": "B"}''',
            [
                ToolCall(function=FunctionCall(name="tool_a",
                                               arguments=json.dumps({"param": "A"}))),
                ToolCall(function=FunctionCall(name="tool_b",
                                               arguments=json.dumps({"param": "B"})))
            ],
            ""),
    ],
)
def test_extract_tool_calls_streaming(mistral_tool_parser, mistral_tokenizer,
                                      model_output, expected_tool_calls,
                                      expected_content):
    other_content: str = ''
    function_names: list[str] = []
    function_args_strs: list[str] = []
    tool_call_idx: int = -1
    tool_call_ids: list[Optional[str]] = []

    for delta_message in stream_delta_message_generator(
            mistral_tool_parser, mistral_tokenizer, model_output):
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

                    function_args_strs[
                        tool_call.index] += tool_call.function.arguments

    assert other_content == expected_content

    actual_tool_calls = [
        ToolCall(id=tool_call_id,
                 function=FunctionCall(
                     name=function_name,
                     arguments=partial_json_parser.ensure_json(
                         function_args_str, Allow.OBJ | Allow.STR)))
        for tool_call_id, function_name, function_args_str in zip(
            tool_call_ids, function_names, function_args_strs)
    ]
    assert_tool_calls(actual_tool_calls, expected_tool_calls)


@pytest.mark.parametrize(
    ids=[
        "v11_single_tool",
        "v11_multiple_tools_comma",
        "v11_nested_with_quotes",
        "v11_escaped_chars",
        "v11_mixed_content",
    ],
    argnames=["model_output", "expected_tool_calls", "expected_content"],
    argvalues=[
        (
            '''[TOOL_CALLS] calculate_sum{"numbers": [1, 2, 3, 4, 5]}''',
            [
                ToolCall(function=FunctionCall(name="calculate_sum",
                                               arguments=json.dumps({
                                                   "numbers": [1, 2, 3, 4, 5]
                                               })))
            ],
            None),
        (
            '''[TOOL_CALLS] get_user{"id": 123}, update_profile{"name": "John", "age": 30}''',  # noqa: E501
            [
                ToolCall(function=FunctionCall(name="get_user",
                                               arguments=json.dumps({"id": 123}))),
                ToolCall(function=FunctionCall(name="update_profile",
                                               arguments=json.dumps({
                                                   "name": "John",
                                                   "age": 30
                                               })))
            ],
            None),
        (
            '''[TOOL_CALLS] parse_json{"content": "{\\"key\\": \\"value\\", \\"nested\\": {\\"item\\": 1}}"}''',  # noqa: E501
            [
                ToolCall(function=FunctionCall(name="parse_json",
                                               arguments=json.dumps({
                                                   "content": "{\"key\": \"value\", \"nested\": {\"item\": 1}}"
                                               })))
            ],
            None),
        (
            '''[TOOL_CALLS] format_text{"template": "Hello {name}\\nWelcome!", "vars": {"name": "User"}}''',  # noqa: E501
            [
                ToolCall(function=FunctionCall(name="format_text",
                                               arguments=json.dumps({
                                                   "template": "Hello {name}\nWelcome!",
                                                   "vars": {"name": "User"}
                                               })))
            ],
            None),
        (
            '''Some content before [TOOL_CALLS] analyze_data{"dataset": "sales_2024", "metrics": ["revenue", "growth"]}''',  # noqa: E501
            [
                ToolCall(function=FunctionCall(name="analyze_data",
                                               arguments=json.dumps({
                                                   "dataset": "sales_2024",
                                                   "metrics": ["revenue", "growth"]
                                               })))
            ],
            "Some content before "),
    ],
)
def test_extract_tool_calls_v11_format(mistral_tool_parser, model_output,
                                       expected_tool_calls, expected_content):
    """Test extraction of tool calls in v11 format (non-streaming)"""
    extracted_tool_calls = mistral_tool_parser.extract_tool_calls(
        model_output, request=None)  # type: ignore[arg-type]
    assert extracted_tool_calls.tools_called

    assert_tool_calls(extracted_tool_calls.tool_calls, expected_tool_calls)

    assert extracted_tool_calls.content == expected_content
