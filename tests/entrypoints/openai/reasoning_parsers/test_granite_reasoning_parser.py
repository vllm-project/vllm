# SPDX-License-Identifier: Apache-2.0
from typing import List

import pytest
from transformers import AutoTokenizer

from tests.entrypoints.openai.reasoning_parsers.utils import (
    run_reasoning_extraction)
from vllm.entrypoints.openai.reasoning_parsers import (ReasoningParser,
                                                       ReasoningParserManager)

parser_name = "granite"
START_REASONING = "Here is my thought process:"
START_RESPONSE = "Here is my response:"

SIMPLE_REASONING = {
    "output":
    f"{START_REASONING}This is a reasoning section{START_RESPONSE}This is the rest",  #noqa: E501
    "reasoning_content": "This is a reasoning section",
    "content": "This is the rest",
}
COMPLETE_REASONING = {
    "output": f"{START_REASONING}This is a reasoning section{START_RESPONSE}",
    "reasoning_content": "This is a reasoning section",
    "content": None,
}
NO_CONTENT = {
    "output": "This is content",
    "reasoning_content": "This is content",
    "content": None,
}
MULTIPLE_LINES = {
    "output":
    f"{START_REASONING}This\nThat{START_RESPONSE}This is the rest\nThat",
    "reasoning_content": "This\nThat",
    "content": "This is the rest\nThat",
}
REASONING_WITH_THINK = {
    "output":
    f"{START_REASONING}This is a reasoning section{START_RESPONSE}This is the rest",  #noqa: E501
    "reasoning_content": "This is a reasoning section",
    "content": "This is the rest",
}
COMPLETE_REASONING_WITH_THINK = {
    "output": f"{START_REASONING}This is a reasoning section{START_RESPONSE}",
    "reasoning_content": "This is a reasoning section",
    "content": None,
}
MULTIPLE_LINES_WITH_THINK = {
    "output":
    f"{START_REASONING}This\nThat{START_RESPONSE}This is the rest\nThat",
    "reasoning_content": "This\nThat",
    "content": "This is the rest\nThat",
}

TEST_CASES = [
    pytest.param(False, SIMPLE_REASONING, id="simple_reasoning"),
    pytest.param(False, COMPLETE_REASONING, id="complete_reasoning"),
    pytest.param(False, NO_CONTENT, id="no_content"),
    pytest.param(False, MULTIPLE_LINES, id="multiple_lines"),
    pytest.param(False, REASONING_WITH_THINK, id="reasoning_with_think"),
    pytest.param(False,
                 COMPLETE_REASONING_WITH_THINK,
                 id="complete_reasoning_with_think"),
    pytest.param(False,
                 MULTIPLE_LINES_WITH_THINK,
                 id="multiple_lines_with_think"),
]

# Global tokenizer initialization to avoid repeated loading
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")


@pytest.mark.parametrize("streaming, param_dict", TEST_CASES)
def test_reasoning(
    streaming: bool,
    param_dict: dict,
):
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
