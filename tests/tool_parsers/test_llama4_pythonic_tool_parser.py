# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest

from tests.tool_parsers.utils import (
    run_tool_extraction,
    run_tool_extraction_streaming,
)
from vllm.entrypoints.openai.engine.protocol import FunctionCall
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser, ToolParserManager

# Test cases similar to pythonic parser but with Llama4 specific format
SIMPLE_FUNCTION_OUTPUT = "[get_weather(city='LA', metric='C')]"
SIMPLE_FUNCTION_CALL = FunctionCall(
    name="get_weather",
    arguments='{"city": "LA", "metric": "C"}',
)
MORE_TYPES_FUNCTION_OUTPUT = (
    "[register_user(name='Doe', "
    "age=9, "
    "address={'city': 'LA', 'state': 'CA'}, "
    "role=None, "
    "passed_test=True, "
    "aliases=['John', 'Johnny'])]"
)
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
    arguments="{}",
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
    r"[get_weather(city='Martha\'s Vineyard', metric='\"cool units\"')]"
)
ESCAPED_STRING_FUNCTION_CALL = FunctionCall(
    name="get_weather",
    arguments='{"city": "Martha\'s Vineyard", "metric": "\\"cool units\\""}',
)
PYTHON_TAG_FUNCTION_OUTPUT = (
    "<|python_start|>[get_weather(city='LA', metric='C')]<|python_end|>"
)

# Some Llama-4 checkpoints (e.g. served with the Llama-3.1 JSON chat template
# rather than the pythonic one) emit tool calls as JSON instead of the
# pythonic form this parser otherwise expects. See issue #46863.
JSON_FUNCTION_OUTPUT = (
    '{"type": "function", "name": "Bash", '
    '"parameters": {"command": "echo hello", "timeout": 120000, '
    '"run_in_background": false}}'
)
JSON_FUNCTION_CALL = FunctionCall(
    name="Bash",
    arguments='{"command": "echo hello", "timeout": 120000, '
    '"run_in_background": false}',
)
JSON_ARGUMENTS_KEY_OUTPUT = (
    '{"name": "get_weather", "arguments": {"city": "LA", "metric": "C"}}'
)
JSON_PARALLEL_FUNCTION_OUTPUT = (
    '{"name": "get_weather", "parameters": {"city": "LA", "metric": "C"}}, '
    '{"name": "register_user", "parameters": {"name": "Doe", "age": 9}}'
)
JSON_ARRAY_FUNCTION_OUTPUT = (
    '[{"name": "get_weather", "parameters": {"city": "LA", "metric": "C"}}, '
    '{"name": "register_user", "parameters": {"name": "Doe", "age": 9}}]'
)
REGISTER_USER_FUNCTION_CALL = FunctionCall(
    name="register_user", arguments='{"name": "Doe", "age": 9}'
)


@pytest.mark.parametrize("streaming", [True, False])
def test_no_tool_call(streaming: bool, default_tokenizer: TokenizerLike):
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("llama4_pythonic")(
        default_tokenizer
    )
    model_output = "How can I help you today?"

    content, tool_calls = run_tool_extraction(
        tool_parser, model_output, streaming=streaming
    )

    assert content == model_output
    assert len(tool_calls) == 0


test_str = "<|python_start|>"
test_str += "[get_weather(city='LA', metric='C'),"
test_str += "register_user(name='Doe', age=9)]"
TEST_CASES = [
    pytest.param(
        True,
        ESCAPED_STRING_FUNCTION_OUTPUT,
        [ESCAPED_STRING_FUNCTION_CALL],
        id="simple_streaming",
    ),
    pytest.param(
        False, SIMPLE_FUNCTION_OUTPUT, [SIMPLE_FUNCTION_CALL], id="simple_nonstreaming"
    ),
    pytest.param(
        True,
        MORE_TYPES_FUNCTION_OUTPUT,
        [MORE_TYPES_FUNCTION_CALL],
        id="more_types_streaming",
    ),
    pytest.param(
        False,
        MORE_TYPES_FUNCTION_OUTPUT,
        [MORE_TYPES_FUNCTION_CALL],
        id="more_types_nonstreaming",
    ),
    pytest.param(
        True,
        PARAMETERLESS_FUNCTION_OUTPUT,
        [PARAMETERLESS_FUNCTION_CALL],
        id="parameterless_streaming",
    ),
    pytest.param(
        False,
        PARAMETERLESS_FUNCTION_OUTPUT,
        [PARAMETERLESS_FUNCTION_CALL],
        id="parameterless_nonstreaming",
    ),
    pytest.param(
        True,
        EMPTY_DICT_FUNCTION_OUTPUT,
        [EMPTY_DICT_FUNCTION_CALL],
        id="empty_dict_streaming",
    ),
    pytest.param(
        False,
        EMPTY_DICT_FUNCTION_OUTPUT,
        [EMPTY_DICT_FUNCTION_CALL],
        id="empty_dict_nonstreaming",
    ),
    pytest.param(
        True,
        EMPTY_LIST_FUNCTION_OUTPUT,
        [EMPTY_LIST_FUNCTION_CALL],
        id="empty_list_streaming",
    ),
    pytest.param(
        False,
        EMPTY_LIST_FUNCTION_OUTPUT,
        [EMPTY_LIST_FUNCTION_CALL],
        id="empty_list_nonstreaming",
    ),
    pytest.param(
        True,
        ESCAPED_STRING_FUNCTION_OUTPUT,
        [ESCAPED_STRING_FUNCTION_CALL],
        id="escaped_string_streaming",
    ),
    pytest.param(
        False,
        ESCAPED_STRING_FUNCTION_OUTPUT,
        [ESCAPED_STRING_FUNCTION_CALL],
        id="escaped_string_nonstreaming",
    ),
    pytest.param(
        True,
        "[get_weather(city='LA',metric='C'),register_user(name='Doe',age=9)]",
        [
            SIMPLE_FUNCTION_CALL,
            FunctionCall(name="register_user", arguments='{"name": "Doe", "age": 9}'),
        ],
        id="parallel_calls_streaming",
    ),
    pytest.param(
        False,
        "[get_weather(city='LA',metric='C'),register_user(name='Doe',age=9)]",
        [
            SIMPLE_FUNCTION_CALL,
            FunctionCall(name="register_user", arguments='{"name": "Doe", "age": 9}'),
        ],
        id="parallel_calls_nonstreaming",
    ),
    pytest.param(
        True,
        PYTHON_TAG_FUNCTION_OUTPUT,
        [SIMPLE_FUNCTION_CALL],
        id="python_tag_streaming",
    ),
    pytest.param(
        False,
        PYTHON_TAG_FUNCTION_OUTPUT,
        [SIMPLE_FUNCTION_CALL],
        id="python_tag_nonstreaming",
    ),
    pytest.param(
        True,
        test_str,
        [
            SIMPLE_FUNCTION_CALL,
            FunctionCall(name="register_user", arguments='{"name": "Doe", "age": 9}'),
        ],
        id="parallel_calls_streaming",
    ),
    pytest.param(
        False,
        "<|python_start|>[get_weather(city='LA', metric='C'), "
        "register_user(name='Doe', age=9)]",
        [
            SIMPLE_FUNCTION_CALL,
            FunctionCall(name="register_user", arguments='{"name": "Doe", "age": 9}'),
        ],
        id="parallel_calls_nonstreaming",
    ),
    pytest.param(True, JSON_FUNCTION_OUTPUT, [JSON_FUNCTION_CALL], id="json_streaming"),
    pytest.param(
        False, JSON_FUNCTION_OUTPUT, [JSON_FUNCTION_CALL], id="json_nonstreaming"
    ),
    pytest.param(
        True,
        JSON_ARGUMENTS_KEY_OUTPUT,
        [SIMPLE_FUNCTION_CALL],
        id="json_arguments_key_streaming",
    ),
    pytest.param(
        False,
        JSON_ARGUMENTS_KEY_OUTPUT,
        [SIMPLE_FUNCTION_CALL],
        id="json_arguments_key_nonstreaming",
    ),
    pytest.param(
        True,
        JSON_PARALLEL_FUNCTION_OUTPUT,
        [SIMPLE_FUNCTION_CALL, REGISTER_USER_FUNCTION_CALL],
        id="json_parallel_streaming",
    ),
    pytest.param(
        False,
        JSON_PARALLEL_FUNCTION_OUTPUT,
        [SIMPLE_FUNCTION_CALL, REGISTER_USER_FUNCTION_CALL],
        id="json_parallel_nonstreaming",
    ),
    pytest.param(
        True,
        JSON_ARRAY_FUNCTION_OUTPUT,
        [SIMPLE_FUNCTION_CALL, REGISTER_USER_FUNCTION_CALL],
        id="json_array_streaming",
    ),
    pytest.param(
        False,
        JSON_ARRAY_FUNCTION_OUTPUT,
        [SIMPLE_FUNCTION_CALL, REGISTER_USER_FUNCTION_CALL],
        id="json_array_nonstreaming",
    ),
]


@pytest.mark.parametrize("streaming, model_output, expected_tool_calls", TEST_CASES)
def test_tool_call(
    streaming: bool,
    model_output: str,
    expected_tool_calls: list[FunctionCall],
    default_tokenizer: TokenizerLike,
):
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("llama4_pythonic")(
        default_tokenizer
    )

    content, tool_calls = run_tool_extraction(
        tool_parser, model_output, streaming=streaming
    )

    assert len(tool_calls) == len(expected_tool_calls)
    for actual, expected in zip(tool_calls, expected_tool_calls):
        assert actual.type == "function"
        assert actual.function == expected


def test_streaming_tool_call_with_large_steps(default_tokenizer: TokenizerLike):
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("llama4_pythonic")(
        default_tokenizer
    )
    model_output_deltas = [
        "<|python_start|>[get_weather(city='LA', metric='C'), "
        "get_weather(), "
        "do_something_cool(steps=[])]<|python_end|>",
    ]

    reconstructor = run_tool_extraction_streaming(
        tool_parser, model_output_deltas, assert_one_tool_per_delta=False
    )

    assert reconstructor.other_content == ""
    assert len(reconstructor.tool_calls) == 3
    assert reconstructor.tool_calls[0].function == SIMPLE_FUNCTION_CALL
    assert reconstructor.tool_calls[1].function == PARAMETERLESS_FUNCTION_CALL
    assert reconstructor.tool_calls[2].function == EMPTY_LIST_FUNCTION_CALL


@pytest.mark.parametrize(
    "model_output",
    [
        # Plain JSON that happens to have a "name" field must not be
        # mistaken for a tool call.
        '{"answer": 42, "note": "not a tool call"}',
        '{"name": "Alice", "age": 30}',
        # Malformed / truncated JSON must fall through as content rather
        # than dropping the tail.
        '{"name": "a", "parameters": {}}, {"name":',
        '[{"name": "a", "parameters": {}}',
        # A valid tool call followed by trailing prose is not a clean
        # payload, so the whole thing stays as content.
        '{"name": "get_weather", "parameters": {"city": "LA"}} then text',
        # Dangling / trailing commas are invalid JSON.
        '{"name": "a", "parameters": {}},',
        '[{"name": "a", "parameters": {}},]',
    ],
)
def test_json_non_tool_content(model_output: str, default_tokenizer: TokenizerLike):
    """JSON that isn't a clean tool-call payload must pass through as
    content instead of being misread as (or losing part of it to) a tool
    call."""
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("llama4_pythonic")(
        default_tokenizer
    )

    content, tool_calls = run_tool_extraction(
        tool_parser, model_output, streaming=False
    )

    assert content == model_output
    assert len(tool_calls) == 0


@pytest.mark.parametrize(
    "model_output",
    [
        '{"answer": 42, "note": "not a tool call"}',
        '{"name": "Alice", "age": 30}',
    ],
)
def test_json_streaming_never_fabricates_tool_call(
    model_output: str, default_tokenizer: TokenizerLike
):
    """Streamed JSON that isn't a tool call must never be reported as one,
    even though (unlike the non-streaming path) the parser can't always
    recover it as content once it has committed to parsing it as JSON."""
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("llama4_pythonic")(
        default_tokenizer
    )

    _, tool_calls = run_tool_extraction(tool_parser, model_output, streaming=True)

    assert len(tool_calls) == 0


@pytest.mark.parametrize("streaming", [False])
def test_regex_timeout_handling(streaming: bool, default_tokenizer: TokenizerLike):
    """test regex timeout is handled gracefully"""
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("llama4_pythonic")(
        default_tokenizer
    )

    fake_problematic_input = "hello world[A(A=" + "\t)A(A=,\t" * 2

    # create a mock regex that raises TimeoutError
    mock_regex = MagicMock()
    mock_regex.match.side_effect = TimeoutError("Regex timeout")

    with patch.object(tool_parser, "TOOL_CALL_REGEX", mock_regex):
        content, tool_calls = run_tool_extraction(
            tool_parser, fake_problematic_input, streaming=streaming
        )

        # should treat as regular text when regex times out
        assert content == fake_problematic_input
        assert len(tool_calls) == 0
        mock_regex.match.assert_called_once()
