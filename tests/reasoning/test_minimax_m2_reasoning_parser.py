# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer

from tests.reasoning.utils import run_reasoning_extraction
from vllm.reasoning import ReasoningParser, ReasoningParserManager

parser_name = "minimax_m2"
end_token = "</think>"

# MiniMax M2 model path
REASONING_MODEL_NAME = "MiniMaxAI/MiniMax-M2"


@pytest.fixture(scope="module")
def minimax_m2_tokenizer():
    return AutoTokenizer.from_pretrained(REASONING_MODEL_NAME)


# =============================================================================
# MiniMax M2 specific behavior:
# - Model does NOT generate <think> start token
# - Model only generates </think> end token
# - All content before </think> is reasoning
# - All content after </think> is the actual response (content)
# =============================================================================

# Case: reasoning + end token + content (typical case)
SIMPLE_REASONING = {
    "output": "This is a reasoning section</think>This is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
    "is_reasoning_end": True,
}

# Case: reasoning + end token only (no content after)
COMPLETE_REASONING = {
    "output": "This is a reasoning section</think>",
    "reasoning": "This is a reasoning section",
    "content": None,
    "is_reasoning_end": True,
}

# Case: no end token yet (streaming in progress, all is reasoning)
NO_END_TOKEN = {
    "output": "This is reasoning in progress",
    "reasoning": "This is reasoning in progress",
    "content": None,
    "is_reasoning_end": False,
}

# Case: multiple lines of reasoning
MULTIPLE_LINES = {
    "output": "First line\nSecond line</think>Response first line\nResponse second",
    "reasoning": "First line\nSecond line",
    "content": "Response first line\nResponse second",
    "is_reasoning_end": True,
}

# Case: only end token (empty reasoning, immediate response)
SHORTEST_REASONING_NO_STREAMING = {
    "output": "</think>This is the response",
    "reasoning": "",
    "content": "This is the response",
    "is_reasoning_end": True,
}

# Case: only end token streaming (reasoning is None because it's just the token)
SHORTEST_REASONING_STREAMING = {
    "output": "</think>This is the response",
    "reasoning": None,
    "content": "This is the response",
    "is_reasoning_end": True,
}

# Case: empty output
EMPTY = {
    "output": "",
    "reasoning": "",
    "content": None,
    "is_reasoning_end": False,
}

# Case: empty streaming
EMPTY_STREAMING = {
    "output": "",
    "reasoning": None,
    "content": None,
    "is_reasoning_end": False,
}

# Case: long reasoning with special characters
SPECIAL_CHARS = {
    "output": "Let me think... 1+1=2, right?</think>Yes, 1+1=2.",
    "reasoning": "Let me think... 1+1=2, right?",
    "content": "Yes, 1+1=2.",
    "is_reasoning_end": True,
}

# Case: reasoning with code blocks
CODE_IN_REASONING = {
    "output": "```python\nprint('hello')\n```</think>Here is the code.",
    "reasoning": "```python\nprint('hello')\n```",
    "content": "Here is the code.",
    "is_reasoning_end": True,
}

TEST_CASES = [
    # Core cases: no start token (MiniMax M2 actual behavior)
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
        SHORTEST_REASONING_NO_STREAMING,
        id="shortest_reasoning",
    ),
    pytest.param(
        True,
        SHORTEST_REASONING_STREAMING,
        id="shortest_reasoning_streaming",
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
        CODE_IN_REASONING,
        id="code_in_reasoning",
    ),
    pytest.param(
        True,
        CODE_IN_REASONING,
        id="code_in_reasoning_streaming",
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

    # Test extract_content
    if param_dict["content"] is not None:
        content = parser.extract_content_ids(output_ids)
        assert content == minimax_m2_tokenizer.convert_tokens_to_ids(
            minimax_m2_tokenizer.tokenize(param_dict["content"])
        )
    else:
        content = parser.extract_content_ids(output)
        assert content == []
