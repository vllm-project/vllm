# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest
from transformers import AutoTokenizer

from tests.tool_parsers.utils import (
    run_tool_extraction,
    run_tool_extraction_streaming,
)
from vllm.entrypoints.openai.engine.protocol import FunctionCall
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser, ToolParserManager

TOOL_CALL_START = "<|tool_call_start|>"
TOOL_CALL_END = "<|tool_call_end|>"

SIMPLE_FUNCTION_OUTPUT = "get_candidate_status(candidate_id='12345')"
SIMPLE_FUNCTION_CALL = FunctionCall(
    name="get_candidate_status",
    arguments='{"candidate_id": "12345"}',
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


@pytest.fixture(scope="module")
def lfm2_tokenizer() -> TokenizerLike:
    return AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-1.2B-Instruct")


def _wrap(tool_text: str, content_after: str = "") -> str:
    """Wrap pythonic tool call in LFM2.5 sentinel tokens."""
    result = f"{TOOL_CALL_START}[{tool_text}]{TOOL_CALL_END}"
    if content_after:
        result += f"\n{content_after}"
    return result


@pytest.mark.parametrize("streaming", [True, False])
def test_no_tool_call(streaming: bool, lfm2_tokenizer: TokenizerLike):
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("lfm2")(lfm2_tokenizer)
    model_output = "How can I help you today?"

    content, tool_calls = run_tool_extraction(
        tool_parser, model_output, streaming=streaming
    )

    assert content == model_output
    assert len(tool_calls) == 0


TEST_CASES = [
    pytest.param(
        True,
        _wrap(SIMPLE_FUNCTION_OUTPUT),
        [SIMPLE_FUNCTION_CALL],
        None,
        id="simple_streaming",
    ),
    pytest.param(
        False,
        _wrap(SIMPLE_FUNCTION_OUTPUT),
        [SIMPLE_FUNCTION_CALL],
        None,
        id="simple_nonstreaming",
    ),
    pytest.param(
        True,
        _wrap(MORE_TYPES_FUNCTION_OUTPUT),
        [MORE_TYPES_FUNCTION_CALL],
        None,
        id="more_types_streaming",
    ),
    pytest.param(
        False,
        _wrap(MORE_TYPES_FUNCTION_OUTPUT),
        [MORE_TYPES_FUNCTION_CALL],
        None,
        id="more_types_nonstreaming",
    ),
    pytest.param(
        True,
        _wrap(PARAMETERLESS_FUNCTION_OUTPUT),
        [PARAMETERLESS_FUNCTION_CALL],
        None,
        id="parameterless_streaming",
    ),
    pytest.param(
        False,
        _wrap(PARAMETERLESS_FUNCTION_OUTPUT),
        [PARAMETERLESS_FUNCTION_CALL],
        None,
        id="parameterless_nonstreaming",
    ),
    pytest.param(
        True,
        _wrap(EMPTY_DICT_FUNCTION_OUTPUT),
        [EMPTY_DICT_FUNCTION_CALL],
        None,
        id="empty_dict_streaming",
    ),
    pytest.param(
        False,
        _wrap(EMPTY_DICT_FUNCTION_OUTPUT),
        [EMPTY_DICT_FUNCTION_CALL],
        None,
        id="empty_dict_nonstreaming",
    ),
    pytest.param(
        True,
        _wrap(EMPTY_LIST_FUNCTION_OUTPUT),
        [EMPTY_LIST_FUNCTION_CALL],
        None,
        id="empty_list_streaming",
    ),
    pytest.param(
        False,
        _wrap(EMPTY_LIST_FUNCTION_OUTPUT),
        [EMPTY_LIST_FUNCTION_CALL],
        None,
        id="empty_list_nonstreaming",
    ),
    pytest.param(
        True,
        _wrap(ESCAPED_STRING_FUNCTION_OUTPUT),
        [ESCAPED_STRING_FUNCTION_CALL],
        None,
        id="escaped_string_streaming",
    ),
    pytest.param(
        False,
        _wrap(ESCAPED_STRING_FUNCTION_OUTPUT),
        [ESCAPED_STRING_FUNCTION_CALL],
        None,
        id="escaped_string_nonstreaming",
    ),
    pytest.param(
        True,
        _wrap(f"{SIMPLE_FUNCTION_OUTPUT}, {MORE_TYPES_FUNCTION_OUTPUT}"),
        [SIMPLE_FUNCTION_CALL, MORE_TYPES_FUNCTION_CALL],
        None,
        id="parallel_calls_streaming",
    ),
    pytest.param(
        False,
        _wrap(f"{SIMPLE_FUNCTION_OUTPUT}, {MORE_TYPES_FUNCTION_OUTPUT}"),
        [SIMPLE_FUNCTION_CALL, MORE_TYPES_FUNCTION_CALL],
        None,
        id="parallel_calls_nonstreaming",
    ),
    # LFM2.5 specific: content AFTER tool call
    pytest.param(
        False,
        _wrap(
            SIMPLE_FUNCTION_OUTPUT,
            content_after="Checking the current status of candidate ID 12345.",
        ),
        [SIMPLE_FUNCTION_CALL],
        "Checking the current status of candidate ID 12345.",
        id="content_after_tool_call_nonstreaming",
    ),
]


@pytest.mark.parametrize(
    "streaming, model_output, expected_tool_calls, expected_content",
    TEST_CASES,
)
def test_tool_call(
    streaming: bool,
    model_output: str,
    expected_tool_calls: list[FunctionCall],
    expected_content: str | None,
    lfm2_tokenizer: TokenizerLike,
):
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("lfm2")(lfm2_tokenizer)

    content, tool_calls = run_tool_extraction(
        tool_parser, model_output, streaming=streaming
    )

    if expected_content and not streaming:
        assert content == expected_content
    assert len(tool_calls) == len(expected_tool_calls)
    for actual, expected in zip(tool_calls, expected_tool_calls):
        assert actual.type == "function"
        assert actual.function == expected


def test_streaming_tool_call_with_large_steps(lfm2_tokenizer: TokenizerLike):
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("lfm2")(lfm2_tokenizer)
    model_output_deltas = [
        f"{TOOL_CALL_START}[get_candidate_status(candidate_id='12345'), "
        f"{PARAMETERLESS_FUNCTION_OUTPUT}, "
        f"{EMPTY_LIST_FUNCTION_OUTPUT}]{TOOL_CALL_END}",
    ]

    reconstructor = run_tool_extraction_streaming(
        tool_parser, model_output_deltas, assert_one_tool_per_delta=False
    )

    assert len(reconstructor.tool_calls) == 3
    assert reconstructor.tool_calls[0].function == SIMPLE_FUNCTION_CALL
    assert reconstructor.tool_calls[1].function == PARAMETERLESS_FUNCTION_CALL
    assert reconstructor.tool_calls[2].function == EMPTY_LIST_FUNCTION_CALL


@pytest.mark.parametrize("streaming", [False])
def test_regex_timeout_handling(streaming: bool, lfm2_tokenizer: TokenizerLike):
    """Test regex timeout is handled gracefully."""
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("lfm2")(lfm2_tokenizer)

    fake_input = f"{TOOL_CALL_START}[A(A=" + "\t)A(A=,\t" * 2
    fake_input += f"]{TOOL_CALL_END}"

    mock_regex = MagicMock()
    mock_regex.match.side_effect = TimeoutError("Regex timeout")

    with patch.object(tool_parser, "TOOL_CALL_REGEX", mock_regex):
        content, tool_calls = run_tool_extraction(
            tool_parser, fake_input, streaming=streaming
        )

        assert content == fake_input
        assert len(tool_calls) == 0
        mock_regex.match.assert_called_once()
