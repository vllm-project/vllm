# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer

from tests.reasoning.utils import run_reasoning_extraction
from vllm.reasoning import ReasoningParser, ReasoningParserManager

parser_name = "ernie45"

REASONING_MODEL_NAME = "baidu/ERNIE-4.5-21B-A3B-Thinking"


@pytest.fixture(scope="module")
def ernie45_tokenizer():
    return AutoTokenizer.from_pretrained(REASONING_MODEL_NAME)


# 带 </think>，非stream
WITH_THINK = {
    "output": "abc</think>def",
    "reasoning": "abc",
    "content": "def",
}
# 带 </think>，stream
WITH_THINK_STREAM = {
    "output": "abc</think>def",
    "reasoning": "abc",
    "content": "def",
}
# without </think>, all is reasoning
WITHOUT_THINK = {
    "output": "abc",
    "reasoning": "abc",
    "content": None,
}
# without </think>, all is reasoning
WITHOUT_THINK_STREAM = {
    "output": "abc",
    "reasoning": "abc",
    "content": None,
}

COMPLETE_REASONING = {
    "output": "abc</think>",
    "reasoning": "abc",
    "content": None,
}
MULTILINE_REASONING = {
    "output": "abc\nABC</think>def\nDEF",
    "reasoning": "abc\nABC",
    "content": "def\nDEF",
}

TEST_CASES = [
    pytest.param(
        False,
        WITH_THINK,
        id="with_think",
    ),
    pytest.param(
        True,
        WITH_THINK_STREAM,
        id="with_think_stream",
    ),
    pytest.param(
        False,
        WITHOUT_THINK,
        id="without_think",
    ),
    pytest.param(
        True,
        WITHOUT_THINK_STREAM,
        id="without_think_stream",
    ),
    pytest.param(
        False,
        COMPLETE_REASONING,
        id="complete_reasoning",
    ),
    pytest.param(
        True,
        COMPLETE_REASONING,
        id="complete_reasoning_stream",
    ),
    pytest.param(
        False,
        MULTILINE_REASONING,
        id="multiline_reasoning",
    ),
    pytest.param(
        True,
        MULTILINE_REASONING,
        id="multiline_reasoning_stream",
    ),
]


@pytest.mark.parametrize("streaming, param_dict", TEST_CASES)
def test_reasoning(
    streaming: bool,
    param_dict: dict,
    ernie45_tokenizer,
):
    output = ernie45_tokenizer.tokenize(param_dict["output"])
    output_tokens: list[str] = []
    for token in output:
        one_token = ernie45_tokenizer.convert_tokens_to_string([token])
        if one_token:
            output_tokens.append(one_token)

    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        ernie45_tokenizer
    )

    reasoning, content = run_reasoning_extraction(
        parser, output_tokens, streaming=streaming
    )

    print()

    assert reasoning == param_dict["reasoning"]
    assert content == param_dict["content"]
