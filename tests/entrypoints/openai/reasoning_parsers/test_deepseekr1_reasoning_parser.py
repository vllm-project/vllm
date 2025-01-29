from typing import List

import pytest
from transformers import AutoTokenizer

from tests.entrypoints.openai.reasoning_parsers.utils import (
    run_reasoning_extraction)
from vllm.entrypoints.openai.reasoning_parsers import (ReasoningParser,
                                                       ReasoningParserManager)

parser_name = "deepseek_r1"
start_token = "<think>"
end_token = "</think>"

SIMPLE_REASONING = {
    "output": "<think>This is a reasoning section</think>This is the rest",
    "reasoning_content": "This is a reasoning section",
    "content": "This is the rest",
}
COMPLETE_REASONING = {
    "output": "<think>This is a reasoning section</think>",
    "reasoning_content": "This is a reasoning section",
    "content": None,
}
NO_REASONING = {
    "output": "This is a reasoning section",
    "reasoning_content": None,
    "content": "This is a reasoning section",
}
MULTIPLE_LINES = {
    "output": "<think>This\nThat</think>This is the rest\nThat",
    "reasoning_content": "This\nThat",
    "content": "This is the rest\nThat",
}
SHORTEST_REASONING_NO_STREAMING = {
    "output": "<think></think>This is the rest",
    "reasoning_content": "",
    "content": "This is the rest",
}
SHORTEST_REASONING = {
    "output": "<think></think>This is the rest",
    "reasoning_content": None,
    "content": "This is the rest",
}

TEST_CASES = [
    pytest.param(
        False,
        SIMPLE_REASONING,
        id="simple_streaming",
    ),
    pytest.param(
        True,
        SIMPLE_REASONING,
        id="simple_streaming",
    ),
    pytest.param(
        False,
        COMPLETE_REASONING,
        id="complete_streaming",
    ),
    pytest.param(
        True,
        COMPLETE_REASONING,
        id="complete_streaming",
    ),
    pytest.param(
        False,
        NO_REASONING,
        id="no_streaming",
    ),
    pytest.param(
        True,
        NO_REASONING,
        id="no_streaming",
    ),
    pytest.param(
        False,
        MULTIPLE_LINES,
        id="multiple_lines_streaming",
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
        id="shortest_streaming",
    ),
]


@pytest.mark.parametrize("streaming, param_dict", TEST_CASES)
def test_reasoning(
    streaming: bool,
    param_dict: dict,
):
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    tokenizer.add_tokens([start_token, end_token])
    output = tokenizer.tokenize(param_dict["output"])
    # decode everything to tokens
    output_tokens: List[str] = [
        tokenizer.convert_tokens_to_string([token]) for token in output
    ]
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(
        parser_name)(tokenizer)

    reasoning, content = run_reasoning_extraction(parser,
                                                  output_tokens,
                                                  streaming=streaming)

    assert reasoning == param_dict["reasoning_content"]
    assert content == param_dict["content"]
