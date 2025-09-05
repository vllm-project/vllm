# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer

from tests.reasoning.utils import run_reasoning_extraction
from vllm.reasoning import ReasoningParser, ReasoningParserManager

parser_name = "minimax_m1"
start_token = "<think>"
end_token = "</think>"

REASONING_MODEL_NAME = "MiniMaxAI/MiniMax-M1-80k"


@pytest.fixture(scope="module")
def minimax_m1_tokenizer():
    return AutoTokenizer.from_pretrained(REASONING_MODEL_NAME)


# 带 <think></think>，非stream
WITH_THINK = {
    "output": "<think>\nThis is a reasoning section\n</think>\nThis is the rest",
    "reasoning_content": "This is a reasoning section",
    "content": "This is the rest",
}
# 带 <think></think>，stream
WITH_THINK_STREAM = {
    "output": "<think>\nThis is a reasoning section\n</think>\nThis is the rest",
    "reasoning_content": "This is a reasoning section",
    "content": "This is the rest",
}
# 不带 <think></think>，非stream
WITHOUT_THINK = {
    "output": "This is the rest",
    "reasoning_content": None,
    "content": "This is the rest",
}
# 不带 <think></think>，stream
WITHOUT_THINK_STREAM = {
    "output": "This is the rest",
    "reasoning_content": None,
    "content": "This is the rest",
}

COMPLETE_REASONING = {
    "output": "<think>\nThis is a reasoning section\n</think>\n",
    "reasoning_content": "This is a reasoning section",
    "content": None,
}
MULTILINE_REASONING = {
    "output":
    "<think>\nThis is a reasoning\nsection\n</think>\nThis is the rest\nThat",
    "reasoning_content": "This is a reasoning\nsection",
    "content": "This is the rest\nThat",
}
ONLY_OPEN_TAG = {
    "output": "<think>\nThis is a reasoning section",
    "reasoning_content": None,
    "content": "<think>\nThis is a reasoning section",
}

ONLY_OPEN_TAG_STREAM = {
    "output": "<think>\nThis is a reasoning section",
    "reasoning_content": "This is a reasoning section",
    "content": None,
}

PERIOD = {
    "output": "<think>\nHTML换行可以用<br>。\n</think>\nHTML用<br>换行",
    "reasoning_content": "HTML换行可以用<br>。",
    "content": "HTML用<br>换行",
}

PERIOD_STREAM = {
    "output": "<think>\nHTML换行可以用<br>。\n</think>\nHTML用<br>换行",
    "reasoning_content": "HTML换行可以用<br>。",
    "content": "HTML用<br>换行",
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
    pytest.param(
        False,
        ONLY_OPEN_TAG,
        id="only_open_tag",
    ),
    pytest.param(
        True,
        ONLY_OPEN_TAG_STREAM,
        id="only_open_tag_stream",
    ),
    pytest.param(
        False,
        PERIOD,
        id="period",
    ),
    pytest.param(
        True,
        PERIOD_STREAM,
        id="period_stream",
    ),
]


@pytest.mark.parametrize("streaming, param_dict", TEST_CASES)
def test_reasoning(
    streaming: bool,
    param_dict: dict,
    minimax_m1_tokenizer,
):
    output = minimax_m1_tokenizer.tokenize(param_dict["output"])
    output_tokens: list[str] = [
        minimax_m1_tokenizer.convert_tokens_to_string([token]) for token in output
    ]
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(
        parser_name)(minimax_m1_tokenizer)

    reasoning, content = run_reasoning_extraction(parser,
                                                  output_tokens,
                                                  streaming=streaming)

    assert reasoning == param_dict["reasoning_content"]
    assert content == param_dict["content"]
