# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from tests.entrypoints.openai.tool_parsers.utils import (
    run_tool_extraction,
    run_tool_extraction_streaming,
)
from vllm.entrypoints.openai.engine.protocol import FunctionCall
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser, ToolParserManager

MSG_SEP_TOKEN = "<|message_sep|>\n\n"
ROLE_SEP_TOKEN = "<|role_sep|>\n"
EOS_TOKEN = "</s>"
TOOL_HEADER = f"function call{ROLE_SEP_TOKEN}"


SIMPLE_ARGS_DICT = {
    "action": "create",
    "id": "preferences",
}
SIMPLE_FUNCTION_JSON = json.dumps(
    {
        "name": "manage_user_memory",
        "arguments": SIMPLE_ARGS_DICT,
    },
    ensure_ascii=False,
)
SIMPLE_FUNCTION_OUTPUT = f"{MSG_SEP_TOKEN}{TOOL_HEADER}{SIMPLE_FUNCTION_JSON}"
SIMPLE_FUNCTION_CALL = FunctionCall(
    name="manage_user_memory",
    arguments=json.dumps(SIMPLE_ARGS_DICT, ensure_ascii=False),
)


PARAMETERLESS_FUNCTION_JSON = json.dumps(
    {
        "name": "manage_user_memory",
        "arguments": {},
    },
    ensure_ascii=False,
)
PARAMETERLESS_FUNCTION_OUTPUT = (
    f"{MSG_SEP_TOKEN}{TOOL_HEADER}{PARAMETERLESS_FUNCTION_JSON}"
)
PARAMETERLESS_FUNCTION_CALL = FunctionCall(
    name="manage_user_memory",
    arguments=json.dumps({}, ensure_ascii=False),
)


COMPLEX_ARGS_DICT = {
    "action": "create",
    "id": "preferences",
    "content": {
        "short_answers": True,
        "hate_emojis": True,
        "english_ui": False,
        "russian_math_explanations": True,
    },
}
COMPLEX_FUNCTION_JSON = json.dumps(
    {
        "name": "manage_user_memory",
        "arguments": COMPLEX_ARGS_DICT,
    },
    ensure_ascii=False,
)
COMPLEX_FUNCTION_OUTPUT = f"{MSG_SEP_TOKEN}{TOOL_HEADER}{COMPLEX_FUNCTION_JSON}"
COMPLEX_FUNCTION_CALL = FunctionCall(
    name="manage_user_memory",
    arguments=json.dumps(COMPLEX_ARGS_DICT, ensure_ascii=False),
)


CONTENT_TEXT = "I'll check that for you."
MIXED_OUTPUT = f"{CONTENT_TEXT}{SIMPLE_FUNCTION_OUTPUT}"


@pytest.fixture(name="gigachat_tokenizer")
def fixture_gigachat_tokenizer(default_tokenizer: TokenizerLike):
    default_tokenizer.add_tokens([MSG_SEP_TOKEN, ROLE_SEP_TOKEN, EOS_TOKEN])
    return default_tokenizer


@pytest.mark.parametrize("streaming", [True, False])
def test_no_tool_call(streaming: bool, gigachat_tokenizer: TokenizerLike):
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("gigachat3")(
        gigachat_tokenizer
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
        SIMPLE_FUNCTION_OUTPUT,
        [SIMPLE_FUNCTION_CALL],
        None,
        id="simple_streaming",
    ),
    pytest.param(
        False,
        SIMPLE_FUNCTION_OUTPUT,
        [SIMPLE_FUNCTION_CALL],
        None,
        id="simple_nonstreaming",
    ),
    pytest.param(
        True,
        PARAMETERLESS_FUNCTION_OUTPUT,
        [PARAMETERLESS_FUNCTION_CALL],
        None,
        id="parameterless_streaming",
    ),
    pytest.param(
        False,
        PARAMETERLESS_FUNCTION_OUTPUT,
        [PARAMETERLESS_FUNCTION_CALL],
        None,
        id="parameterless_nonstreaming",
    ),
    pytest.param(
        True,
        COMPLEX_FUNCTION_OUTPUT,
        [COMPLEX_FUNCTION_CALL],
        None,
        id="complex_streaming",
    ),
    pytest.param(
        False,
        COMPLEX_FUNCTION_OUTPUT,
        [COMPLEX_FUNCTION_CALL],
        None,
        id="complex_nonstreaming",
    ),
    pytest.param(
        True,
        MIXED_OUTPUT,
        [SIMPLE_FUNCTION_CALL],
        CONTENT_TEXT,
        id="mixed_content_streaming",
    ),
    pytest.param(
        False,
        MIXED_OUTPUT,
        [SIMPLE_FUNCTION_CALL],
        CONTENT_TEXT,
        id="mixed_content_nonstreaming",
    ),
    pytest.param(
        True,
        MIXED_OUTPUT + EOS_TOKEN,
        [SIMPLE_FUNCTION_CALL],
        CONTENT_TEXT,
        id="mixed_content_streaming_with_eos",
    ),
    pytest.param(
        False,
        MIXED_OUTPUT + EOS_TOKEN,
        [SIMPLE_FUNCTION_CALL],
        CONTENT_TEXT,
        id="mixed_content_nonstreaming_with_eos",
    ),
]


@pytest.mark.parametrize(
    "streaming, model_output, expected_tool_calls, expected_content", TEST_CASES
)
def test_tool_call(
    streaming: bool,
    model_output: str,
    expected_tool_calls: list[FunctionCall],
    expected_content: str | None,
    gigachat_tokenizer: TokenizerLike,
):
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("gigachat3")(
        gigachat_tokenizer
    )
    content, tool_calls = run_tool_extraction(
        tool_parser, model_output, streaming=streaming
    )
    if content == "":
        content = None
    assert content == expected_content
    assert len(tool_calls) == len(expected_tool_calls)
    for actual, expected in zip(tool_calls, expected_tool_calls):
        assert actual.type == "function"
        assert actual.function.name == expected.name
        actual_args = json.loads(actual.function.arguments)
        expected_args = json.loads(expected.arguments)
        assert actual_args == expected_args


def test_streaming_tool_call_with_large_steps(gigachat_tokenizer: TokenizerLike):
    """
    Test that the closing braces are streamed correctly.
    """
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("gigachat3")(
        gigachat_tokenizer
    )
    model_output_deltas = [
        CONTENT_TEXT[:3],
        CONTENT_TEXT[3:5],
        CONTENT_TEXT[5:],
        MSG_SEP_TOKEN,
        TOOL_HEADER,
        COMPLEX_FUNCTION_JSON[:40],
        COMPLEX_FUNCTION_JSON[40:-1],
        COMPLEX_FUNCTION_JSON[-1],
    ]
    reconstructor = run_tool_extraction_streaming(
        tool_parser,
        model_output_deltas,
        assert_one_tool_per_delta=False,
    )
    assert len(reconstructor.tool_calls) == 1
    call = reconstructor.tool_calls[0]
    assert call.type == "function"
    assert call.function.name == "manage_user_memory"
    args_dict = json.loads(call.function.arguments)
    assert args_dict == COMPLEX_ARGS_DICT
