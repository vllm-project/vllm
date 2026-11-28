# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
        (
            " Francisco', metric='celsius'), "
            f"{PARAMETERLESS_FUNCTION_OUTPUT}, "
            f"{EMPTY_LIST_FUNCTION_OUTPUT}]"
        ),
    ]

    reconstructor = run_tool_extraction_streaming(
        tool_parser, model_output_deltas, assert_one_tool_per_delta=False
    )

    assert reconstructor.other_content == ""
    assert len(reconstructor.tool_calls) == 3
    assert reconstructor.tool_calls[0].function == SIMPLE_FUNCTION_CALL
    assert reconstructor.tool_calls[1].function == PARAMETERLESS_FUNCTION_CALL
    assert reconstructor.tool_calls[2].function == EMPTY_LIST_FUNCTION_CALL


def test_streaming_valid_call_before_malformed_call(default_tokenizer: TokenizerLike):
    """The delta that completes a valid call also starts a malformed one. The
    valid call's withheld closing characters must still be streamed."""
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("pythonic")(
        default_tokenizer
    )
    model_output_deltas = [
        "[get_weather(city='San",
        " Francisco', metric='celsius'), get",
        "_weather(city=)]",
    ]

    reconstructor = run_tool_extraction_streaming(
        tool_parser, model_output_deltas, assert_one_tool_per_delta=False
    )

    assert len(reconstructor.tool_calls) == 1
    assert reconstructor.tool_calls[0].function == SIMPLE_FUNCTION_CALL


def test_streaming_malformed_call_with_quoted_separator(
    default_tokenizer: TokenizerLike,
):
    """Regression test for the review on #69342: the separator `"),"` can
    appear inside a quoted argument (`fmt="),"`), so recovery must only
    use a separator between top-level calls. The malformed call is never
    emitted, and no partial arguments with invalid JSON may leak out —
    including when a delta ends inside the quoted argument."""
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("pythonic")(
        default_tokenizer
    )
    model_output_deltas = [
        "[get_weather(city='Par",
        "is'), get_time(fmt=\")",
        '", timezone=)]',
    ]

    reconstructor = run_tool_extraction_streaming(
        tool_parser, model_output_deltas, assert_one_tool_per_delta=False
    )

    assert len(reconstructor.tool_calls) == 1
    assert reconstructor.tool_calls[0].function == FunctionCall(
        name="get_weather",
        arguments='{"city": "Paris"}',
    )


def test_streaming_malformed_call_random_splits(default_tokenizer: TokenizerLike):
    """The valid call is released complete and the malformed call is never
    emitted, regardless of where the stream is split."""
    import random

    text = "[get_weather(city='Paris'), get_time(fmt=\"),\", timezone=)]"
    expected = FunctionCall(
        name="get_weather",
        arguments='{"city": "Paris"}',
    )
    rng = random.Random(20261123)
    for _ in range(20):
        parser: ToolParser = ToolParserManager.get_tool_parser("pythonic")(
            default_tokenizer
        )
        n_deltas = rng.randint(2, 6)
        cuts = sorted(rng.sample(range(1, len(text)), n_deltas - 1))
        deltas = []
        prev = 0
        for cut in cuts + [len(text)]:
            deltas.append(text[prev:cut])
            prev = cut
        reconstructor = run_tool_extraction_streaming(
            parser, deltas, assert_one_tool_per_delta=False
        )
        assert len(reconstructor.tool_calls) == 1, deltas
        assert reconstructor.tool_calls[0].function == expected, deltas


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
