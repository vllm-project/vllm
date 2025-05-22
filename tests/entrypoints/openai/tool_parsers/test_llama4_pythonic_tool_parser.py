# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from tests.entrypoints.openai.tool_parsers.utils import (
    run_tool_extraction, run_tool_extraction_streaming)
from vllm.entrypoints.openai.protocol import FunctionCall
from vllm.entrypoints.openai.tool_parsers import ToolParser, ToolParserManager

# Test cases similar to pythonic parser but with Llama4 specific format
SIMPLE_FUNCTION_OUTPUT = "[get_weather(city='LA', metric='C')]"
SIMPLE_FUNCTION_CALL = FunctionCall(
    name="get_weather",
    arguments='{"city": "LA", "metric": "C"}',
)
MORE_TYPES_FUNCTION_OUTPUT = ("[register_user(name='Doe', "
                              "age=9, "
                              "address={'city': 'LA', 'state': 'CA'}, "
                              "role=None, "
                              "passed_test=True, "
                              "aliases=['John', 'Johnny'])]")
MORE_TYPES_FUNCTION_CALL = FunctionCall(
    name="register_user",
    arguments='{"name": "Doe", '
    '"age": 9, '
    '"address": {"city": "LA", "state": "CA"}, '
    '"role": null, '
    '"passed_test": true, '
    '"aliases": ["John", "Johnny"]}',
)
PARAMETERLESS_FUNCTION_OUTPUT = "[get_weather()]"
PARAMETERLESS_FUNCTION_CALL = FunctionCall(
    name="get_weather",
    arguments='{}',
)
EMPTY_DICT_FUNCTION_OUTPUT = "[do_something_cool(additional_data={})]"
EMPTY_DICT_FUNCTION_CALL = FunctionCall(
    name="do_something_cool",
    arguments='{"additional_data": {}}',
)
EMPTY_LIST_FUNCTION_OUTPUT = "[do_something_cool(steps=[])]"
EMPTY_LIST_FUNCTION_CALL = FunctionCall(
    name="do_something_cool",
    arguments='{"steps": []}',
)
ESCAPED_STRING_FUNCTION_OUTPUT = (
    r"[get_weather(city='Martha\'s Vineyard', metric='\"cool units\"')]")
ESCAPED_STRING_FUNCTION_CALL = FunctionCall(
    name="get_weather",
    arguments='{"city": "Martha\'s Vineyard", "metric": "\\"cool units\\""}',
)
PYTHON_TAG_FUNCTION_OUTPUT = (
    "<|python_start|>[get_weather(city='LA', metric='C')]<|python_end|>")


@pytest.mark.parametrize("streaming", [True, False])
def test_no_tool_call(streaming: bool):
    mock_tokenizer = MagicMock()
    tool_parser: ToolParser = ToolParserManager.get_tool_parser(
        "llama4_pythonic")(mock_tokenizer)
    model_output = "How can I help you today?"

    content, tool_calls = run_tool_extraction(tool_parser,
                                              model_output,
                                              streaming=streaming)

    assert content == model_output
    assert len(tool_calls) == 0


test_str = "<|python_start|>"
test_str += "[get_weather(city='LA', metric='C'),"
test_str += "register_user(name='Doe', age=9)]"
TEST_CASES = [
    pytest.param(True,
                 ESCAPED_STRING_FUNCTION_OUTPUT,
                 [ESCAPED_STRING_FUNCTION_CALL],
                 id="simple_streaming"),
    pytest.param(False,
                 SIMPLE_FUNCTION_OUTPUT, [SIMPLE_FUNCTION_CALL],
                 id="simple_nonstreaming"),
    pytest.param(True,
                 MORE_TYPES_FUNCTION_OUTPUT, [MORE_TYPES_FUNCTION_CALL],
                 id="more_types_streaming"),
    pytest.param(False,
                 MORE_TYPES_FUNCTION_OUTPUT, [MORE_TYPES_FUNCTION_CALL],
                 id="more_types_nonstreaming"),
    pytest.param(True,
                 PARAMETERLESS_FUNCTION_OUTPUT, [PARAMETERLESS_FUNCTION_CALL],
                 id="parameterless_streaming"),
    pytest.param(False,
                 PARAMETERLESS_FUNCTION_OUTPUT, [PARAMETERLESS_FUNCTION_CALL],
                 id="parameterless_nonstreaming"),
    pytest.param(True,
                 EMPTY_DICT_FUNCTION_OUTPUT, [EMPTY_DICT_FUNCTION_CALL],
                 id="empty_dict_streaming"),
    pytest.param(False,
                 EMPTY_DICT_FUNCTION_OUTPUT, [EMPTY_DICT_FUNCTION_CALL],
                 id="empty_dict_nonstreaming"),
    pytest.param(True,
                 EMPTY_LIST_FUNCTION_OUTPUT, [EMPTY_LIST_FUNCTION_CALL],
                 id="empty_list_streaming"),
    pytest.param(False,
                 EMPTY_LIST_FUNCTION_OUTPUT, [EMPTY_LIST_FUNCTION_CALL],
                 id="empty_list_nonstreaming"),
    pytest.param(True,
                 ESCAPED_STRING_FUNCTION_OUTPUT,
                 [ESCAPED_STRING_FUNCTION_CALL],
                 id="escaped_string_streaming"),
    pytest.param(False,
                 ESCAPED_STRING_FUNCTION_OUTPUT,
                 [ESCAPED_STRING_FUNCTION_CALL],
                 id="escaped_string_nonstreaming"),
    pytest.param(
        True,
        "[get_weather(city='LA',metric='C'),register_user(name='Doe',age=9)]",
        [
            SIMPLE_FUNCTION_CALL,
            FunctionCall(name="register_user",
                         arguments='{"name": "Doe", "age": 9}')
        ],
        id="parallel_calls_streaming"),
    pytest.param(
        False,
        "[get_weather(city='LA',metric='C'),register_user(name='Doe',age=9)]",
        [
            SIMPLE_FUNCTION_CALL,
            FunctionCall(name="register_user",
                         arguments='{"name": "Doe", "age": 9}')
        ],
        id="parallel_calls_nonstreaming"),
    pytest.param(True,
                 PYTHON_TAG_FUNCTION_OUTPUT, [SIMPLE_FUNCTION_CALL],
                 id="python_tag_streaming"),
    pytest.param(False,
                 PYTHON_TAG_FUNCTION_OUTPUT, [SIMPLE_FUNCTION_CALL],
                 id="python_tag_nonstreaming"),
    pytest.param(True,
                 test_str, [
                     SIMPLE_FUNCTION_CALL,
                     FunctionCall(name="register_user",
                                  arguments='{"name": "Doe", "age": 9}')
                 ],
                 id="parallel_calls_streaming"),
    pytest.param(False,
                 "<|python_start|>[get_weather(city='LA', metric='C'), " +
                 "register_user(name='Doe', age=9)]", [
                     SIMPLE_FUNCTION_CALL,
                     FunctionCall(name="register_user",
                                  arguments='{"name": "Doe", "age": 9}')
                 ],
                 id="parallel_calls_nonstreaming"),
]


@pytest.mark.parametrize("streaming, model_output, expected_tool_calls",
                         TEST_CASES)
def test_tool_call(streaming: bool, model_output: str,
                   expected_tool_calls: list[FunctionCall]):
    mock_tokenizer = MagicMock()
    tool_parser: ToolParser = ToolParserManager.get_tool_parser(
        "llama4_pythonic")(mock_tokenizer)

    content, tool_calls = run_tool_extraction(tool_parser,
                                              model_output,
                                              streaming=streaming)

    assert len(tool_calls) == len(expected_tool_calls)
    for actual, expected in zip(tool_calls, expected_tool_calls):
        assert actual.type == "function"
        assert actual.function == expected


def test_streaming_tool_call_with_large_steps():
    mock_tokenizer = MagicMock()
    tool_parser: ToolParser = ToolParserManager.get_tool_parser(
        "llama4_pythonic")(mock_tokenizer)
    model_output_deltas = [
        "<|python_start|>[get_weather(city='LA', metric='C'), "
        "get_weather(), "
        "do_something_cool(steps=[])]<|python_end|>",
    ]

    reconstructor = run_tool_extraction_streaming(
        tool_parser, model_output_deltas, assert_one_tool_per_delta=False)

    assert reconstructor.other_content == ""
    assert len(reconstructor.tool_calls) == 3
    assert reconstructor.tool_calls[0].function == SIMPLE_FUNCTION_CALL
    assert reconstructor.tool_calls[1].function == PARAMETERLESS_FUNCTION_CALL
    assert reconstructor.tool_calls[2].function == EMPTY_LIST_FUNCTION_CALL
