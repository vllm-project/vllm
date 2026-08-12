# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest

from tests.tool_parsers.utils import (
    run_tool_extraction,
    run_tool_extraction_nonstreaming,
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
    """test regex timeout is handled gracefully"""
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


@pytest.mark.parametrize("streaming", [True, False])
def test_bad_sibling_call_does_not_drop_good_calls(
    streaming: bool, default_tokenizer: TokenizerLike
):
    """One unconvertible call (bytes argument) used to abort the whole
    conversion, dropping every parseable sibling call in the block."""
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("pythonic")(
        default_tokenizer
    )
    model_output = f"[bad(x=b'z'), {SIMPLE_FUNCTION_OUTPUT}]"

    content, tool_calls = run_tool_extraction(
        tool_parser,
        model_output,
        streaming=streaming,
        assert_one_tool_per_delta=False,
    )
    assert len(tool_calls) == 1
    assert tool_calls[0].function == SIMPLE_FUNCTION_CALL


def test_non_finite_argument_rejected(default_tokenizer: TokenizerLike):
    """A 1e999 literal overflows to inf and used to serialize as Infinity —
    arguments no JSON parser accepts; the call is rejected and every
    emitted arguments string stays valid JSON."""
    import json

    tool_parser: ToolParser = ToolParserManager.get_tool_parser("pythonic")(
        default_tokenizer
    )
    model_output = f"[calc(x=1e999), {SIMPLE_FUNCTION_OUTPUT}]"

    content, tool_calls = run_tool_extraction(
        tool_parser, model_output, streaming=False
    )
    assert len(tool_calls) == 1
    assert tool_calls[0].function == SIMPLE_FUNCTION_CALL
    for call in tool_calls:
        json.loads(call.function.arguments)


def test_good_call_survives_unparsable_sibling(default_tokenizer: TokenizerLike):
    """A broken quote makes the whole block a SyntaxError, so the per-call
    salvage never gets a call list and the parseable sibling died with the
    block. Non-streaming only: a partial streaming block is legitimately
    unparsable and must keep waiting, not be split."""
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("pythonic")(
        default_tokenizer
    )
    model_output = f"[{SIMPLE_FUNCTION_OUTPUT}, f(a='x 'y', b='z')]"

    extracted = run_tool_extraction_nonstreaming(tool_parser, model_output)

    assert extracted.tools_called
    assert len(extracted.tool_calls) == 1
    assert extracted.tool_calls[0].function == SIMPLE_FUNCTION_CALL


def test_unrecoverable_block_still_reports_no_tool_call(
    default_tokenizer: TokenizerLike,
):
    """Splitting must not fabricate calls: when no segment parses, the
    block still falls back to content with tools_called=False."""
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("pythonic")(
        default_tokenizer
    )
    model_output = "[f(a='x 'y' 'z), g(b='p 'q' 'r)]"

    extracted = run_tool_extraction_nonstreaming(tool_parser, model_output)

    assert not extracted.tools_called
    assert extracted.tool_calls == []
    assert extracted.content == model_output


def test_streaming_mismatched_brackets_reported_once(
    default_tokenizer: TokenizerLike,
):
    """Brackets the model got structurally wrong raise at the same offset on
    every chunk of the block, so the failure was logged with a full traceback
    once per chunk and none of them named the offending text. It is reported
    once per request instead."""
    from vllm.tool_parsers import pythonic_tool_parser as parser_module

    tool_parser: ToolParser = ToolParserManager.get_tool_parser("pythonic")(
        default_tokenizer
    )
    model_output = "[foo(x=1])]"

    with (
        patch.object(parser_module.logger, "exception") as logged_exception,
        patch.object(parser_module.logger, "warning") as logged_warning,
    ):
        _, tool_calls = run_tool_extraction(
            tool_parser,
            model_output,
            streaming=True,
            assert_one_tool_per_delta=False,
        )

    assert tool_calls == []
    assert logged_exception.call_count == 0
    assert logged_warning.call_count == 1
