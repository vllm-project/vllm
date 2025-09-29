# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer

from tests.reasoning.utils import run_reasoning_extraction
from vllm.reasoning import ReasoningParser, ReasoningParserManager

parser_name = "olmo3"
START_REASONING = "<think>"
END_REASONING = "</think>"


NO_REASONING = {
    "output": f"{START_REASONING}{END_REASONING}No thoughts, head empty!",
    "reasoning_content": None,
    "content": "No thoughts, head empty!",
}

NO_REASONING_WITH_NEWLINE = {
    "output": f"{START_REASONING}\n{END_REASONING}\n\nNo thoughts, head empty!",
    "reasoning_content": None,
    "content": "No thoughts, head empty!",
}

SIMPLE_REASONING = {
    "output": f"{START_REASONING}This is a reasoning section{END_REASONING}This is the rest",  # noqa: E501
    "reasoning_content": "This is a reasoning section",
    "content": "This is the rest",
}

SIMPLE_REASONING_WITH_NEWLINE = {
    "output": f"{START_REASONING}\n Look!\n\nI'm thinking...{END_REASONING}\nThis is the rest",  # noqa: E501
    "reasoning_content": "Look!\n\nI'm thinking...",
    "content": "This is the rest",
}

SIMPLE_REASONING_WITH_MULTIPLE_NEWLINES = {
    "output": f"{START_REASONING}\n\n Look!\nI'm thinking...\n\n{END_REASONING}\n\n\nThis is the rest",  # noqa: E501
    "reasoning_content": "Look!\nI'm thinking...",
    "content": "This is the rest",
}

TEST_CASES = [
    pytest.param(
        False,
        SIMPLE_REASONING,
        id="simple_reasoning",
    ),
    pytest.param(
        False,
        COMPLETE_REASONING,
        id="complete_reasoning",
    ),
    pytest.param(
        False,
        COMPLETE_REASONING_WITH_SYMBOL,
        id="complete_reasoning_with_symbol",
    ),
    pytest.param(
        False,
        NO_REASONING,
        id="no_reasoning",
    ),
    pytest.param(False, NO_REASONING_QUICK_THROUGHT, id="no_reasoning_quick"),
    pytest.param(
        False,
        MULTIPLE_LINES,
        id="multiple_lines",
    ),
    pytest.param(
        False,
        REASONING_WITH_THINK,
        id="reasoning_with_think",
    ),
    pytest.param(
        False,
        COMPLETE_REASONING_WITH_THINK,
        id="complete_reasoning_with_think",
    ),
    pytest.param(
        False,
        MULTIPLE_LINES_WITH_THINK,
        id="multiple_lines_with_think",
    ),
    pytest.param(
        True,
        SIMPLE_REASONING,
        id="simple_reasoning_streaming",
    ),
    pytest.param(
        True,
        COMPLETE_REASONING,
        id="complete_reasoning_streaming",
    ),
    pytest.param(
        True,
        NO_REASONING,
        id="no_reasoning_streaming",
    ),
    pytest.param(
        True, NO_REASONING_QUICK_THROUGHT, id="no_reasoning_quick_stream"
    ),
    pytest.param(
        True,
        MULTIPLE_LINES,
        id="multiple_lines_streaming",
    ),
    pytest.param(
        True,
        REASONING_WITH_THINK,
        id="reasoning_with_think_streaming",
    ),
    pytest.param(
        True,
        COMPLETE_REASONING_WITH_THINK,
        id="complete_reasoning_with_think_streaming",
    ),
    pytest.param(
        True,
        MULTIPLE_LINES_WITH_THINK,
        id="multiple_lines_with_think_streaming",
    ),
]

# Global tokenizer initialization to avoid repeated loading
tokenizer = AutoTokenizer.from_pretrained(
    "tencent/Hunyuan-A13B-Instruct", trust_remote_code=True
)


@pytest.mark.parametrize("streaming, param_dict", TEST_CASES)
def test_reasoning(
    streaming: bool,
    param_dict: dict,
):
    output = tokenizer.tokenize(param_dict["output"])
    # decode everything to tokens
    output_tokens: list[str] = [
        tokenizer.convert_tokens_to_string([token]) for token in output
    ]
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(
        parser_name
    )(tokenizer)

    reasoning, content = run_reasoning_extraction(
        parser, output_tokens, streaming=streaming
    )

    assert reasoning == param_dict["reasoning_content"]
    assert content == param_dict["content"]
