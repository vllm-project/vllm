# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer

from tests.reasoning.utils import (
    StreamingReasoningReconstructor,
    run_reasoning_extraction,
    run_reasoning_extraction_streaming,
)
from vllm.reasoning import ReasoningParser, ReasoningParserManager

parser_name = "qwen3"
start_token = "<think>"
end_token = "</think>"

REASONING_MODEL_NAMES = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3.5-397B-A17B",
    "Qwen/Qwen3-4B-Thinking-2507",
]


@pytest.fixture(scope="module", params=REASONING_MODEL_NAMES)
def qwen3_tokenizer(request):
    return AutoTokenizer.from_pretrained(request.param)


# --- <think> in prompt, only </think> in output (typical) ---

WITHOUT_START_TOKEN = {
    "output": "This is a reasoning section</think>This is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
}
WITHOUT_START_TOKEN_STREAM = {
    "output": "This is a reasoning section</think>This is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
}
WITHOUT_START_TOKEN_COMPLETE_REASONING = {
    "output": "This is a reasoning section</think>",
    "reasoning": "This is a reasoning section",
    "content": None,
}

# --- <think> present in output (old template / edge case) ---

WITH_THINK = {
    "output": "<think>This is a reasoning section</think>This is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
}
WITH_THINK_STREAM = {
    "output": "<think>This is a reasoning section</think>This is the rest",
    "reasoning": "This is a reasoning section",
    "content": "This is the rest",
}

# --- No think tokens at all (thinking disabled) ---

WITHOUT_THINK = {
    "output": "This is the rest",
    "reasoning": None,
    "content": "This is the rest",
}
# In streaming, the parser cannot distinguish "thinking disabled" from
# "reasoning in progress" when no think tokens have appeared yet.
# It assumes reasoning. The serving layer handles the "thinking disabled"
# case by checking prompt_is_reasoning_end_arr before calling the parser.
WITHOUT_THINK_STREAM = {
    "output": "This is the rest",
    "reasoning": "This is the rest",
    "content": None,
}

# --- Edge cases ---

COMPLETE_REASONING = {
    "output": "<think>This is a reasoning section</think>",
    "reasoning": "This is a reasoning section",
    "content": None,
}
MULTILINE_REASONING = {
    "output": "<think>This is a reasoning\nsection</think>This is the rest\nThat",
    "reasoning": "This is a reasoning\nsection",
    "content": "This is the rest\nThat",
}
ONLY_OPEN_TAG = {
    "output": "<think>This is a reasoning section",
    "reasoning": None,
    "content": "This is a reasoning section",
}

ONLY_OPEN_TAG_STREAM = {
    "output": "<think>This is a reasoning section",
    "reasoning": "This is a reasoning section",
    "content": None,
}

TEST_CASES = [
    pytest.param(
        False,
        WITHOUT_START_TOKEN,
        id="without_start_token",
    ),
    pytest.param(
        True,
        WITHOUT_START_TOKEN_STREAM,
        id="without_start_token_stream",
    ),
    pytest.param(
        False,
        WITHOUT_START_TOKEN_COMPLETE_REASONING,
        id="without_start_token_complete_reasoning",
    ),
    pytest.param(
        True,
        WITHOUT_START_TOKEN_COMPLETE_REASONING,
        id="without_start_token_complete_reasoning_stream",
    ),
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
]


@pytest.mark.parametrize("streaming, param_dict", TEST_CASES)
def test_reasoning(
    streaming: bool,
    param_dict: dict,
    qwen3_tokenizer,
):
    output = qwen3_tokenizer.tokenize(param_dict["output"])
    output_tokens: list[str] = [
        qwen3_tokenizer.convert_tokens_to_string([token]) for token in output
    ]
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        qwen3_tokenizer
    )

    reasoning, content = run_reasoning_extraction(
        parser, output_tokens, streaming=streaming
    )

    assert reasoning == param_dict["reasoning"]
    assert content == param_dict["content"]


# Multi-token delta tests: simulate real-world streaming where a single
# delta can contain multiple tokens (e.g., speculative decoding).
MULTI_TOKEN_DELTA_CASES = [
    pytest.param(
        # <think> grouped with following text in one delta
        ["<think>This is a reasoning section", "</think>", "This is the rest"],
        "This is a reasoning section",
        "This is the rest",
        id="start_token_grouped_with_text",
    ),
    pytest.param(
        # </think> grouped with following content in one delta
        ["reasoning section", "</think>This is the rest"],
        "reasoning section",
        "This is the rest",
        id="end_token_grouped_with_content",
    ),
    pytest.param(
        # <think> and </think> in the same delta, no content after
        ["<think>reasoning</think>"],
        "reasoning",
        None,
        id="start_and_end_in_one_delta_no_content",
    ),
    pytest.param(
        # No start token, end grouped with content (Qwen3.5 style)
        ["reasoning section", "</think>content"],
        "reasoning section",
        "content",
        id="no_start_end_grouped_with_content",
    ),
]


@pytest.mark.parametrize(
    "deltas, expected_reasoning, expected_content", MULTI_TOKEN_DELTA_CASES
)
def test_reasoning_streaming_multi_token_deltas(
    deltas: list[str],
    expected_reasoning: str | None,
    expected_content: str | None,
    qwen3_tokenizer,
):
    """Test that multi-token deltas don't leak <think> into reasoning."""
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        qwen3_tokenizer
    )

    reconstructor: StreamingReasoningReconstructor = run_reasoning_extraction_streaming(
        parser, deltas
    )

    assert reconstructor.reasoning == expected_reasoning
    assert (reconstructor.other_content or None) == expected_content
