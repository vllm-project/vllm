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
    "reasoning_content": "\n",
    "content": "\n\nNo thoughts, head empty!",
}

SIMPLE_REASONING = {
    "output": f"{START_REASONING}This is a reasoning section{END_REASONING}This is the rest",  # noqa: E501
    "reasoning_content": "This is a reasoning section",
    "content": "This is the rest",
}

SIMPLE_REASONING_WITH_NEWLINE = {
    "output": f"{START_REASONING} Look!\n\nI'm thinking...{END_REASONING}\nThis is the rest",  # noqa: E501
    "reasoning_content": " Look!\n\nI'm thinking...",
    "content": "\nThis is the rest",
}

SIMPLE_REASONING_WITH_MULTIPLE_NEWLINES = {
    "output": f"{START_REASONING}\nLook!\nI'm thinking...\n\n{END_REASONING}\n\n\nThis is the rest",  # noqa: E501
    "reasoning_content": "\nLook!\nI'm thinking...\n\n",
    "content": "\n\n\nThis is the rest",
}


TEST_CASES = [
    pytest.param(
        False,  # not streaming
        NO_REASONING,
        id="no_reasoning",
    ),
    pytest.param(
        False,  # not streaming
        NO_REASONING_WITH_NEWLINE,
        id="no_reasoning_with_newline",
    ),
    pytest.param(
        False,  # not streaming
        SIMPLE_REASONING,
        id="simple_reasoning",
    ),
    pytest.param(
        False,  # not streaming
        SIMPLE_REASONING_WITH_NEWLINE,
        id="simple_reasoning_with_newline",
    ),
    pytest.param(
        True,  # enamble streaming
        SIMPLE_REASONING_WITH_MULTIPLE_NEWLINES,
        id="simple_reasoning_with_multiple_newlines",
    ),
    pytest.param(
        True,  # enamble streaming
        NO_REASONING,
        id="no_reasoning_streaming",
    ),
    pytest.param(
        True,  # enamble streaming
        NO_REASONING_WITH_NEWLINE,
        id="no_reasoning_with_newline_streaming",
    ),
    pytest.param(
        True,  # enamble streaming
        SIMPLE_REASONING,
        id="simple_reasoning_streaming",
    ),
    pytest.param(
        True,  # enamble streaming
        SIMPLE_REASONING_WITH_NEWLINE,
        id="simple_reasoning_with_newline_streaming",
    ),
    pytest.param(
        True,  # enamble streaming
        SIMPLE_REASONING_WITH_MULTIPLE_NEWLINES,
        id="simple_reasoning_with_multiple_newlines_streaming",
    ),
]

# Global tokenizer initialization to avoid repeated loading
tokenizer = AutoTokenizer.from_pretrained(
    "tencent/Hunyuan-A13B-Instruct", trust_remote_code=True
)


@pytest.mark.parametrize("streaming, param_dict", TEST_CASES)
def test_reasoning(
    streaming: bool,
    param_dict: dict[str, str],
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
