# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest

from tests.entrypoints.openai.tool_parsers.utils import (
    run_tool_extraction,
    run_tool_extraction_streaming,
)
from vllm.entrypoints.openai.engine.protocol import FunctionCall
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser, ToolParserManager

# https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/text_prompt_format.md#model-response-format-1
SIMPLE_FUNCTION_OUTPUT = "get_weather(city='San Francisco', metric='celsius')"
SIMPLE_FUNCTION_CALL = FunctionCall(
    name="get_weather",
    arguments='{"city": "San Francisco", "metric": "celsius"}',
)
MORE_TYPES_FUNCTION_OUTPUT = (
    "register_user(name='John Doe', "
    "age=37, "
    "address={'city': 'San Francisco', 'state': 'CA'}, "
    "role=None, "
    "passed_test=True, "
    "aliases=['John', 'Johnny'])"
)
MORE_TYPES_FUNCTION_OUTPUT_JSON_LITERALS = (
    "register_user(name='John Doe', "
    "age=37, "
    "address={'city': 'San Francisco', 'state': 'CA'}, "
    "role=null, "
    "passed_test=true, "
    "aliases=['John', 'Johnny'])"
)
MORE_TYPES_FUNCTION_CALL = FunctionCall(
    name="register_user",
    arguments='{"name": "John Doe", '
    '"age": 37, '
    '"address": {"city": "San Francisco", "state": "CA"}, '
    '"role": null, '
    '"passed_test": true, '
    '"aliases": ["John", "Johnny"]}',
)
PARAMETERLESS_FUNCTION_OUTPUT = "get_weather()"
PARAMETERLESS_FUNCTION_CALL = FunctionCall(
    name="get_weather",
    arguments="{}",
)
EMPTY_DICT_FUNCTION_OUTPUT = "do_something_cool(additional_data={})"
EMPTY_DICT_FUNCTION_CALL = FunctionCall(
    name="do_something_cool",
    arguments='{"additional_data": {}}',
)
EMPTY_LIST_FUNCTION_OUTPUT = "do_something_cool(steps=[])"
EMPTY_LIST_FUNCTION_CALL = FunctionCall(
    name="do_something_cool",
    arguments='{"steps": []}',
)
ESCAPED_STRING_FUNCTION_OUTPUT = (
    r"get_weather(city='Martha\'s Vineyard', metric='\"cool units\"')"
)
ESCAPED_STRING_FUNCTION_CALL = FunctionCall(
    name="get_weather",
    arguments='{"city": "Martha\'s Vineyard", "metric": "\\"cool units\\""}',
)


@pytest.mark.parametrize("streaming", [True, False])
def test_no_tool_call(streaming: bool, default_tokenizer: TokenizerLike):
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("olmo3")(
        default_tokenizer
    )
    model_output = "How can I help you today?"

    content, tool_calls = run_tool_extraction(
        tool_parser, model_output, streaming=streaming
    )

    assert content == model_output
    assert len(tool_calls) == 0


TEST_CASES = [
    pytest.param(
        True,
        f"<function_calls>{SIMPLE_FUNCTION_OUTPUT}</function_calls>",
        [SIMPLE_FUNCTION_CALL],
        id="simple_streaming",
    ),
    pytest.param(
        False,
        f"<function_calls>{SIMPLE_FUNCTION_OUTPUT}</function_calls>",
        [SIMPLE_FUNCTION_CALL],
        id="simple_nonstreaming",
    ),
    pytest.param(
        True,
        f"<function_calls>{MORE_TYPES_FUNCTION_OUTPUT}</function_calls>",
        [MORE_TYPES_FUNCTION_CALL],
        id="more_types_streaming",
    ),
    pytest.param(
        False,
        f"<function_calls>{MORE_TYPES_FUNCTION_OUTPUT}</function_calls>",
        [MORE_TYPES_FUNCTION_CALL],
        id="more_types_nonstreaming",
    ),
    pytest.param(
        True,
        f"<function_calls>{MORE_TYPES_FUNCTION_OUTPUT_JSON_LITERALS}</function_calls>",
        [MORE_TYPES_FUNCTION_CALL],
        id="more_types_streaming_json_literals",
    ),
    pytest.param(
        False,
        f"<function_calls>{MORE_TYPES_FUNCTION_OUTPUT_JSON_LITERALS}</function_calls>",
        [MORE_TYPES_FUNCTION_CALL],
        id="more_types_nonstreaming_json_literals",
    ),
    pytest.param(
        True,
        f"<function_calls>{PARAMETERLESS_FUNCTION_OUTPUT}</function_calls>",
        [PARAMETERLESS_FUNCTION_CALL],
        id="parameterless_streaming",
    ),
    pytest.param(
        False,
        f"<function_calls>{PARAMETERLESS_FUNCTION_OUTPUT}</function_calls>",
        [PARAMETERLESS_FUNCTION_CALL],
        id="parameterless_nonstreaming",
    ),
    pytest.param(
        True,
        f"<function_calls>{EMPTY_DICT_FUNCTION_OUTPUT}</function_calls>",
        [EMPTY_DICT_FUNCTION_CALL],
        id="empty_dict_streaming",
    ),
    pytest.param(
        False,
        f"<function_calls>{EMPTY_DICT_FUNCTION_OUTPUT}</function_calls>",
        [EMPTY_DICT_FUNCTION_CALL],
        id="empty_dict_nonstreaming",
    ),
    pytest.param(
        True,
        f"<function_calls>{EMPTY_LIST_FUNCTION_OUTPUT}</function_calls>",
        [EMPTY_LIST_FUNCTION_CALL],
        id="empty_list_streaming",
    ),
    pytest.param(
        False,
        f"<function_calls>{EMPTY_LIST_FUNCTION_OUTPUT}</function_calls>",
        [EMPTY_LIST_FUNCTION_CALL],
        id="empty_list_nonstreaming",
    ),
    pytest.param(
        True,
        f"<function_calls>{ESCAPED_STRING_FUNCTION_OUTPUT}</function_calls>",
        [ESCAPED_STRING_FUNCTION_CALL],
        id="escaped_string_streaming",
    ),
    pytest.param(
        False,
        f"<function_calls>{ESCAPED_STRING_FUNCTION_OUTPUT}</function_calls>",
        [ESCAPED_STRING_FUNCTION_CALL],
        id="escaped_string_nonstreaming",
    ),
    pytest.param(
        True,
        f"<function_calls>{SIMPLE_FUNCTION_OUTPUT}\n{MORE_TYPES_FUNCTION_OUTPUT}</function_calls>",
        [SIMPLE_FUNCTION_CALL, MORE_TYPES_FUNCTION_CALL],
        id="parallel_calls_streaming",
    ),
    pytest.param(
        False,
        f"<function_calls>{SIMPLE_FUNCTION_OUTPUT}\n{MORE_TYPES_FUNCTION_OUTPUT}</function_calls>",
        [SIMPLE_FUNCTION_CALL, MORE_TYPES_FUNCTION_CALL],
        id="parallel_calls_nonstreaming",
    ),
]


@pytest.mark.parametrize("streaming, model_output, expected_tool_calls", TEST_CASES)
def test_tool_call(
    streaming: bool,
    model_output: str,
    expected_tool_calls: list[FunctionCall],
    default_tokenizer: TokenizerLike,
):
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("olmo3")(
        default_tokenizer
    )

    content, tool_calls = run_tool_extraction(
        tool_parser, model_output, streaming=streaming
    )

    assert content is None
    assert len(tool_calls) == len(expected_tool_calls)
    for actual, expected in zip(tool_calls, expected_tool_calls):
        assert actual.type == "function"
        assert actual.function == expected


def test_streaming_tool_call_with_large_steps(default_tokenizer: TokenizerLike):
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("olmo3")(
        default_tokenizer
    )
    model_output_deltas = [
        "<function_calls>get_weather(city='San",
        " Francisco', metric='celsius')\n"
        f"{PARAMETERLESS_FUNCTION_OUTPUT}\n"
        f"{EMPTY_LIST_FUNCTION_OUTPUT}</function_calls>",
    ]

    reconstructor = run_tool_extraction_streaming(
        tool_parser, model_output_deltas, assert_one_tool_per_delta=False
    )

    assert reconstructor.other_content == ""
    assert len(reconstructor.tool_calls) == 3
    assert reconstructor.tool_calls[0].function == SIMPLE_FUNCTION_CALL
    assert reconstructor.tool_calls[1].function == PARAMETERLESS_FUNCTION_CALL
    assert reconstructor.tool_calls[2].function == EMPTY_LIST_FUNCTION_CALL


@pytest.mark.parametrize("streaming", [False])
def test_regex_timeout_handling(streaming: bool, default_tokenizer: TokenizerLike):
    """test regex timeout is handled gracefully"""
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("olmo3")(
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
