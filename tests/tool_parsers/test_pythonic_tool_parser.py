# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import MagicMock, patch

import pytest

from tests.tool_parsers.utils import (
    run_tool_extraction,
    run_tool_extraction_streaming,
)
from vllm.entrypoints.generate.base.protocol import FunctionCall
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
RESERVED_KEYWORD_FUNCTION_OUTPUT = (
    'send_message(from="alice@example.com", in="inbox", class="urgent", '
    'import=True, return=False, count=42)'
)
RESERVED_KEYWORD_FUNCTION_CALL = FunctionCall(
    name="send_message",
    arguments=(
        '{"from": "alice@example.com", "in": "inbox", "class": "urgent", '
        '"import": true, "return": false, "count": 42}'
    ),
)


@pytest.mark.parametrize("streaming", [True, False])
def test_no_tool_call(streaming: bool, default_tokenizer: TokenizerLike):
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("pythonic")(
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
        f"[{SIMPLE_FUNCTION_OUTPUT}]",
        [SIMPLE_FUNCTION_CALL],
        id="simple_streaming",
    ),
    pytest.param(
        False,
        f"[{SIMPLE_FUNCTION_OUTPUT}]",
        [SIMPLE_FUNCTION_CALL],
        id="simple_nonstreaming",
    ),
    pytest.param(
        True,
        f"[{MORE_TYPES_FUNCTION_OUTPUT}]",
        [MORE_TYPES_FUNCTION_CALL],
        id="more_types_streaming",
    ),
    pytest.param(
        False,
        f"[{MORE_TYPES_FUNCTION_OUTPUT}]",
        [MORE_TYPES_FUNCTION_CALL],
        id="more_types_nonstreaming",
    ),
    pytest.param(
        True,
        f"[{PARAMETERLESS_FUNCTION_OUTPUT}]",
        [PARAMETERLESS_FUNCTION_CALL],
        id="parameterless_streaming",
    ),
    pytest.param(
        False,
        f"[{PARAMETERLESS_FUNCTION_OUTPUT}]",
        [PARAMETERLESS_FUNCTION_CALL],
        id="parameterless_nonstreaming",
    ),
    pytest.param(
        True,
        f"[{EMPTY_DICT_FUNCTION_OUTPUT}]",
        [EMPTY_DICT_FUNCTION_CALL],
        id="empty_dict_streaming",
    ),
    pytest.param(
        False,
        f"[{EMPTY_DICT_FUNCTION_OUTPUT}]",
        [EMPTY_DICT_FUNCTION_CALL],
        id="empty_dict_nonstreaming",
    ),
    pytest.param(
        True,
        f"[{EMPTY_LIST_FUNCTION_OUTPUT}]",
        [EMPTY_LIST_FUNCTION_CALL],
        id="empty_list_streaming",
    ),
    pytest.param(
        False,
        f"[{EMPTY_LIST_FUNCTION_OUTPUT}]",
        [EMPTY_LIST_FUNCTION_CALL],
        id="empty_list_nonstreaming",
    ),
    pytest.param(
        True,
        f"[{ESCAPED_STRING_FUNCTION_OUTPUT}]",
        [ESCAPED_STRING_FUNCTION_CALL],
        id="escaped_string_streaming",
    ),
    pytest.param(
        False,
        f"[{ESCAPED_STRING_FUNCTION_OUTPUT}]",
        [ESCAPED_STRING_FUNCTION_CALL],
        id="escaped_string_nonstreaming",
    ),
    pytest.param(
        True,
        f"[{SIMPLE_FUNCTION_OUTPUT}, {MORE_TYPES_FUNCTION_OUTPUT}]",
        [SIMPLE_FUNCTION_CALL, MORE_TYPES_FUNCTION_CALL],
        id="parallel_calls_streaming",
    ),
    pytest.param(
        False,
        f"[{SIMPLE_FUNCTION_OUTPUT}, {MORE_TYPES_FUNCTION_OUTPUT}]",
        [SIMPLE_FUNCTION_CALL, MORE_TYPES_FUNCTION_CALL],
        id="parallel_calls_nonstreaming",
    ),
    pytest.param(
        True,
        f"[{RESERVED_KEYWORD_FUNCTION_OUTPUT}]",
        [RESERVED_KEYWORD_FUNCTION_CALL],
        id="reserved_keywords_streaming",
    ),
    pytest.param(
        False,
        f"[{RESERVED_KEYWORD_FUNCTION_OUTPUT}]",
        [RESERVED_KEYWORD_FUNCTION_CALL],
        id="reserved_keywords_nonstreaming",
    ),
]


@pytest.mark.parametrize("streaming, model_output, expected_tool_calls", TEST_CASES)
def test_tool_call(
    streaming: bool,
    model_output: str,
    expected_tool_calls: list[FunctionCall],
    default_tokenizer: TokenizerLike,
):
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("pythonic")(
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
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("pythonic")(
        default_tokenizer
    )
    model_output_deltas = [
        "[get_weather(city='San",
        " Francisco', metric='celsius'), "
        f"{PARAMETERLESS_FUNCTION_OUTPUT}, "
        f"{EMPTY_LIST_FUNCTION_OUTPUT}]",
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
    """Test regex timeout is handled gracefully."""
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("pythonic")(
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


def test_reserved_keyword_triple_quotes(default_tokenizer: TokenizerLike):
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("pythonic")(
        default_tokenizer
    )
    # Triple quotes with apostrophes and keyword-like substrings
    output1 = "[f(text='''User\\'s data, from=1''')]"
    content1, tool_calls1 = run_tool_extraction(
        tool_parser, output1, streaming=False
    )
    assert content1 is None
    assert len(tool_calls1) == 1
    assert json.loads(tool_calls1[0].function.arguments) == {
        "text": "User's data, from=1"
    }


@pytest.mark.parametrize("streaming", [True, False])
def test_reserved_keyword_provenance(
    streaming: bool, default_tokenizer: TokenizerLike
):
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("pythonic")(
        default_tokenizer
    )
    # Per-call provenance: parameter with rename suffix must not be mangled
    output2 = "[f(from=1), g(from_pyreservedkw_=2)]"
    content2, tool_calls2 = run_tool_extraction(
        tool_parser, output2, streaming=streaming
    )
    assert content2 is None
    assert len(tool_calls2) == 2
    assert json.loads(tool_calls2[0].function.arguments) == {"from": 1}
    assert json.loads(tool_calls2[1].function.arguments) == {"from_pyreservedkw_": 2}

    # Single call containing both reserved keyword and parameter with suffix
    output3 = "[f(from=1, from_pyreservedkw_=2)]"
    content3, tool_calls3 = run_tool_extraction(
        tool_parser, output3, streaming=streaming
    )
    assert content3 is None
    assert len(tool_calls3) == 1
    assert json.loads(tool_calls3[0].function.arguments) == {
        "from": 1,
        "from_pyreservedkw_": 2,
    }
