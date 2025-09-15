# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer

from tests.reasoning.utils import run_reasoning_extraction
from vllm.reasoning import ReasoningParser, ReasoningParserManager

parser_name = "minimax_m1"
start_token = "<think>\n"
end_token = "</think>"

REASONING_MODEL_NAME = "MiniMaxAI/MiniMax-M1-80k"


@pytest.fixture(scope="module")
def minimax_m1_tokenizer():
    return AutoTokenizer.from_pretrained(REASONING_MODEL_NAME)


SIMPLE_REASONING = {
    "output": "This is a reasoning section\n</think>\nThis is the rest",
    "reasoning_content": "This is a reasoning section",
    "content": "This is the rest",
    "is_reasoning_end": True,
}
COMPLETE_REASONING = {
    "output": "This is a reasoning section\n</think>\n",
    "reasoning_content": "This is a reasoning section",
    "content": None,
    "is_reasoning_end": True,
}
NO_CONTENT = {
    "output": "This is content",
    "reasoning_content": "This is content",
    "content": None,
    "is_reasoning_end": False,
}
NO_REASONING_STREAMING = {
    "output": "This is a reasoning section",
    "reasoning_content": "This is a reasoning section",
    "content": None,
    "is_reasoning_end": False,
}
MULTIPLE_LINES = {
    "output": "This\nThat\n</think>\nThis is the rest\nThat",
    "reasoning_content": "This\nThat",
    "content": "This is the rest\nThat",
    "is_reasoning_end": True,
}
SHORTEST_REASONING_NO_STREAMING = {
    "output": "\n</think>\nThis is the rest",
    "reasoning_content": "",
    "content": "This is the rest",
    "is_reasoning_end": True,
}
SHORTEST_REASONING = {
    "output": "\n</think>\nThis is the rest",
    "reasoning_content": None,
    "content": "This is the rest",
    "is_reasoning_end": True,
}
REASONING_WITH_THINK = {
    "output": "<think>\nThis is a reasoning section\n</think>\nThis is the rest",
    "reasoning_content": "This is a reasoning section",
    "content": "This is the rest",
    "is_reasoning_end": True,
}
COMPLETE_REASONING_WITH_THINK = {
    "output": "<think>\nThis is a reasoning section\n</think>\n",
    "reasoning_content": "This is a reasoning section",
    "content": None,
    "is_reasoning_end": True,
}
MULTIPLE_LINES_WITH_THINK = {
    "output": "<think>\nThis\nThat\n</think>\nThis is the rest\nThat",
    "reasoning_content": "This\nThat",
    "content": "This is the rest\nThat",
    "is_reasoning_end": True,
}
SHORTEST_REASONING_NO_STREAMING_WITH_THINK = {
    "output": "\n</think>\nThis is the rest",
    "reasoning_content": "",
    "content": "This is the rest",
    "is_reasoning_end": True,
}
SHORTEST_REASONING_WITH_THINK = {
    "output": "\n</think>\nThis is the rest",
    "reasoning_content": None,
    "content": "This is the rest",
    "is_reasoning_end": True,
}
THINK_NO_END = {
    "output": "<think>\nThis is a reasoning section",
    "reasoning_content": "This is a reasoning section",
    "content": None,
    "is_reasoning_end": False,
}
EMPTY = {
    "output": "",
    "reasoning_content": "",
    "content": None,
    "is_reasoning_end": False,
}
EMPTY_STREAMING = {
    "output": "",
    "reasoning_content": None,
    "content": None,
    "is_reasoning_end": False,
}

TEST_CASES = [
    pytest.param(
        False,
        SIMPLE_REASONING,
        id="simple_reasoning",
    ),
    pytest.param(
        True,
        SIMPLE_REASONING,
        id="simple_reasoning_streaming",
    ),
    pytest.param(
        False,
        COMPLETE_REASONING,
        id="complete_reasoning",
    ),
    pytest.param(
        True,
        COMPLETE_REASONING,
        id="complete_reasoning_streaming",
    ),
    pytest.param(
        False,
        NO_CONTENT,
        id="no_content_token",
    ),
    pytest.param(
        True,
        NO_REASONING_STREAMING,
        id="no_reasoning_token_streaming",
    ),
    pytest.param(
        False,
        MULTIPLE_LINES,
        id="multiple_lines",
    ),
    pytest.param(
        True,
        MULTIPLE_LINES,
        id="multiple_lines_streaming",
    ),
    pytest.param(
        True,
        SHORTEST_REASONING,
        id="shortest_streaming",
    ),
    pytest.param(
        False,
        SHORTEST_REASONING_NO_STREAMING,
        id="shortest",
    ),
    pytest.param(
        False,
        REASONING_WITH_THINK,
        id="reasoning_with_think",
    ),
    pytest.param(
        True,
        REASONING_WITH_THINK,
        id="reasoning_with_think_streaming",
    ),
    pytest.param(
        False,
        COMPLETE_REASONING_WITH_THINK,
        id="complete_reasoning_with_think",
    ),
    pytest.param(
        True,
        COMPLETE_REASONING_WITH_THINK,
        id="complete_reasoning_with_think_streaming",
    ),
    pytest.param(
        False,
        MULTIPLE_LINES_WITH_THINK,
        id="multiple_lines_with_think",
    ),
    pytest.param(
        True,
        MULTIPLE_LINES_WITH_THINK,
        id="multiple_lines_with_think_streaming",
    ),
    pytest.param(
        False,
        SHORTEST_REASONING_NO_STREAMING_WITH_THINK,
        id="shortest_with_think",
    ),
    pytest.param(
        True,
        SHORTEST_REASONING_WITH_THINK,
        id="shortest_with_think_streaming",
    ),
    pytest.param(
        False,
        THINK_NO_END,
        id="think_no_end",
    ),
    pytest.param(
        True,
        THINK_NO_END,
        id="think_no_end_streaming",
    ),
    pytest.param(
        False,
        EMPTY,
        id="empty",
    ),
    pytest.param(
        True,
        EMPTY_STREAMING,
        id="empty_streaming",
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
