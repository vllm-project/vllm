# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer

from tests.reasoning.utils import run_reasoning_extraction
from vllm.reasoning import ReasoningParser, ReasoningParserManager

parser_name = "minimax_m2_append_think"
end_token = "</think>"

# MiniMax M2 model path
REASONING_MODEL_NAME = "MiniMaxAI/MiniMax-M2"


@pytest.fixture(scope="module")
def minimax_m2_tokenizer():
    return AutoTokenizer.from_pretrained(REASONING_MODEL_NAME)


# =============================================================================
# MiniMaxM2AppendThinkReasoningParser behavior:
# - Prepends <think> to the beginning of the output
# - Does NOT separate reasoning and content
# - Returns everything as content (with <think> prepended)
# - reasoning is always None
#
# This parser is used when you want to keep the raw output with <think> added
# =============================================================================

# Case: simple output with end token
SIMPLE_OUTPUT = {
    "output": "This is reasoning</think>This is response",
    "reasoning": None,
    "content": "<think>This is reasoning</think>This is response",
    "is_reasoning_end": True,
}

# Case: output without end token (reasoning in progress)
NO_END_TOKEN = {
    "output": "This is reasoning in progress",
    "reasoning": None,
    "content": "<think>This is reasoning in progress",
    "is_reasoning_end": False,
}

# Case: only end token
ONLY_END_TOKEN = {
    "output": "</think>This is response",
    "reasoning": None,
    "content": "<think></think>This is response",
    "is_reasoning_end": True,
}

# Case: multiple lines
MULTIPLE_LINES = {
    "output": "Line 1\nLine 2</think>Response 1\nResponse 2",
    "reasoning": None,
    "content": "<think>Line 1\nLine 2</think>Response 1\nResponse 2",
    "is_reasoning_end": True,
}

# Case: empty output (non-streaming prepends <think>)
EMPTY = {
    "output": "",
    "reasoning": None,
    "content": "<think>",
    "is_reasoning_end": False,
}

# Case: empty output streaming (no tokens = no output)
EMPTY_STREAMING = {
    "output": "",
    "reasoning": None,
    "content": None,
    "is_reasoning_end": False,
}

# Case: special characters
SPECIAL_CHARS = {
    "output": "Let me think... 1+1=2</think>Yes!",
    "reasoning": None,
    "content": "<think>Let me think... 1+1=2</think>Yes!",
    "is_reasoning_end": True,
}

# Case: code in output
CODE_OUTPUT = {
    "output": "```python\nprint('hi')\n```</think>Here's the code.",
    "reasoning": None,
    "content": "<think>```python\nprint('hi')\n```</think>Here's the code.",
    "is_reasoning_end": True,
}

TEST_CASES = [
    pytest.param(
        False,
        SIMPLE_OUTPUT,
        id="simple_output",
    ),
    pytest.param(
        True,
        SIMPLE_OUTPUT,
        id="simple_output_streaming",
    ),
    pytest.param(
        False,
        NO_END_TOKEN,
        id="no_end_token",
    ),
    pytest.param(
        True,
        NO_END_TOKEN,
        id="no_end_token_streaming",
    ),
    pytest.param(
        False,
        ONLY_END_TOKEN,
        id="only_end_token",
    ),
    pytest.param(
        True,
        ONLY_END_TOKEN,
        id="only_end_token_streaming",
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
        False,
        EMPTY,
        id="empty",
    ),
    pytest.param(
        True,
        EMPTY_STREAMING,
        id="empty_streaming",
    ),
    pytest.param(
        False,
        SPECIAL_CHARS,
        id="special_chars",
    ),
    pytest.param(
        True,
        SPECIAL_CHARS,
        id="special_chars_streaming",
    ),
    pytest.param(
        False,
        CODE_OUTPUT,
        id="code_output",
    ),
    pytest.param(
        True,
        CODE_OUTPUT,
        id="code_output_streaming",
    ),
]


@pytest.mark.parametrize("streaming, param_dict", TEST_CASES)
def test_reasoning(
    streaming: bool,
    param_dict: dict,
    minimax_m2_tokenizer,
):
    output = minimax_m2_tokenizer.tokenize(param_dict["output"])
    # decode everything to tokens
    output_tokens: list[str] = [
        minimax_m2_tokenizer.convert_tokens_to_string([token]) for token in output
    ]
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        minimax_m2_tokenizer
    )

    reasoning, content = run_reasoning_extraction(
        parser, output_tokens, streaming=streaming
    )

    assert reasoning == param_dict["reasoning"]
    assert content == param_dict["content"]

    # Test is_reasoning_end
    output_ids = minimax_m2_tokenizer.convert_tokens_to_ids(output)
    is_reasoning_end = parser.is_reasoning_end(output_ids)
    assert is_reasoning_end == param_dict["is_reasoning_end"]
