# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer

from vllm.reasoning import ReasoningParser
from vllm.reasoning.gptoss_reasoning_parser import GptOssReasoningParser

REASONING_MODEL_NAME = "openai/gpt-oss-120b"


@pytest.fixture(scope="module")
def gpt_oss_tokenizer():
    return AutoTokenizer.from_pretrained(REASONING_MODEL_NAME)


USER_MESSAGE_START = "<|start|>user<|message|>"
REASONING_SECTION_START = "<|end|><|start|>assistant<|channel|>analysis<|message|>"
END = "<|end|>"
ASSISTANT_START = "<|start|>assistant"
ASSISTANT_CONTENT_START_PREFIX = END + ASSISTANT_START + "<|channel|>final"
ASSISTANT_CONTENT_START_SUFFIX = "<|message|>"
ASSISTANT_CONTENT_START = (
    ASSISTANT_CONTENT_START_PREFIX + ASSISTANT_CONTENT_START_SUFFIX
)

BASIC_CONTENT = {
    "output": REASONING_SECTION_START
    + "This is reasoning"
    + ASSISTANT_CONTENT_START
    + "This is the rest",
    "is_reasoning_end": True,
}

BASIC_REASONING_ONLY = {
    "output": REASONING_SECTION_START + "This is reasoning" + "<|end|>",
    "is_reasoning_end": False,
}
BASIC_NO_REASONING_NO_ASSISTANT = {
    "output": USER_MESSAGE_START + "This is a user message",
    "is_reasoning_end": False,
}

# Edge-case where the model omits the assistant tag entirely.
BASIC_NO_REASONING_ASSISTANT = {
    "output": USER_MESSAGE_START + "This is a user message<|end|><|channel|>final",
    "is_reasoning_end": True,
}

COMPLEX_CONTENT_INCOMPLETE_PREFIX_ONLY = {
    "output": REASONING_SECTION_START
    + "This is reasoning"
    + ASSISTANT_CONTENT_START_PREFIX,
    "is_reasoning_end": False,
}

COMPLEX_CONTENT_SUFFIX_ONLY = {
    "output": REASONING_SECTION_START
    + "This is reasoning"
    + ASSISTANT_CONTENT_START_SUFFIX,
    "is_reasoning_end": False,
}

COMPLEX_CONTENT_1_NO_SUFFIX = {
    "output": REASONING_SECTION_START
    + "This is reasoning"
    + ASSISTANT_CONTENT_START_PREFIX
    + "<|constrain|> JSON ",
    "is_reasoning_end": False,
}

COMPLEX_CONTENT_1 = {
    "output": REASONING_SECTION_START
    + "This is reasoning"
    + ASSISTANT_CONTENT_START_PREFIX
    + "<|constrain|> JSON "
    + ASSISTANT_CONTENT_START_SUFFIX,
    "is_reasoning_end": True,
}

COMPLEX_CONTENT_1_WITH_CONTENT = {
    "output": REASONING_SECTION_START
    + "This is reasoning"
    + ASSISTANT_CONTENT_START_PREFIX
    + "<|constrain|> JSON "
    + ASSISTANT_CONTENT_START_SUFFIX
    + "This is the rest",
    "is_reasoning_end": True,
}

COMPLEX_CONTENT_2 = {
    "output": REASONING_SECTION_START
    + "This is reasoning"
    + ASSISTANT_CONTENT_START_PREFIX
    + "<|constrain|>ReplyAction "
    + ASSISTANT_CONTENT_START_SUFFIX
    + "This is the rest",
    "is_reasoning_end": True,
}

MULTI_TURN_CONTENT = {
    "output": USER_MESSAGE_START
    + "1st turn user message"
    + REASONING_SECTION_START
    + "1st turn reasoning"
    + ASSISTANT_CONTENT_START
    + "1st turn response"
    + END
    + USER_MESSAGE_START
    + "2nd turn user message"
    + END
    + ASSISTANT_START,
    "is_reasoning_end": False,
}
TEST_CASES = [
    BASIC_CONTENT,
    BASIC_REASONING_ONLY,
    COMPLEX_CONTENT_INCOMPLETE_PREFIX_ONLY,
    COMPLEX_CONTENT_SUFFIX_ONLY,
    COMPLEX_CONTENT_1_NO_SUFFIX,
    COMPLEX_CONTENT_1,
    COMPLEX_CONTENT_1_WITH_CONTENT,
    COMPLEX_CONTENT_2,
    MULTI_TURN_CONTENT,
]


@pytest.mark.parametrize(
    "output, is_reasoning_end",
    [(t["output"], t["is_reasoning_end"]) for t in TEST_CASES],
)
def test_gptoss_is_reasoning_end(
    output,
    is_reasoning_end,
    gpt_oss_tokenizer,
):
    output = gpt_oss_tokenizer.tokenize(output)
    parser: ReasoningParser = GptOssReasoningParser(gpt_oss_tokenizer)

    # Test is_reasoning_end
    output_ids = gpt_oss_tokenizer.convert_tokens_to_ids(output)
    actual_is_reasoning_end = parser.is_reasoning_end(output_ids)
    assert is_reasoning_end == actual_is_reasoning_end


@pytest.mark.parametrize(
    "output, is_reasoning_end",
    [(t["output"], t["is_reasoning_end"]) for t in TEST_CASES],
)
def test_gptoss_is_reasoning_end_streaming(
    output,
    is_reasoning_end,
    gpt_oss_tokenizer,
):
    """Streaming override must agree with is_reasoning_end for all cases."""
    tokens = gpt_oss_tokenizer.tokenize(output)
    parser: ReasoningParser = GptOssReasoningParser(gpt_oss_tokenizer)
    output_ids = gpt_oss_tokenizer.convert_tokens_to_ids(tokens)
    delta_ids = output_ids[-1:] if output_ids else []
    actual = parser.is_reasoning_end_streaming(output_ids, delta_ids)
    assert is_reasoning_end == actual


@pytest.mark.parametrize(
    "output, is_reasoning_end",
    [(t["output"], t["is_reasoning_end"]) for t in TEST_CASES],
)
def test_gptoss_is_reasoning_end_streaming_long_prefix(
    output,
    is_reasoning_end,
    gpt_oss_tokenizer,
):
    """Windowing must produce correct results even with a long prefix."""
    tokens = gpt_oss_tokenizer.tokenize(output)
    parser: ReasoningParser = GptOssReasoningParser(gpt_oss_tokenizer)
    output_ids = gpt_oss_tokenizer.convert_tokens_to_ids(tokens)
    # Prepend 10k dummy reasoning tokens to simulate a long generation
    long_prefix = [1] * 10_000
    padded_ids = long_prefix + list(output_ids)
    delta_ids = output_ids[-1:] if output_ids else []
    actual = parser.is_reasoning_end_streaming(padded_ids, delta_ids)
    assert is_reasoning_end == actual


@pytest.mark.parametrize(
    "output, is_reasoning_end",
    [(t["output"], t["is_reasoning_end"]) for t in TEST_CASES],
)
def test_gptoss_is_reasoning_end_streaming_large_delta(
    output,
    is_reasoning_end,
    gpt_oss_tokenizer,
):
    """Simulate speculative decoding where the entire test sequence arrives
    as a single large delta appended after a long prefix.  The window must
    expand to cover delta_ids so the end pattern is never missed."""
    tokens = gpt_oss_tokenizer.tokenize(output)
    parser: ReasoningParser = GptOssReasoningParser(gpt_oss_tokenizer)
    output_ids = gpt_oss_tokenizer.convert_tokens_to_ids(tokens)
    long_prefix = [1] * 10_000
    padded_ids = long_prefix + list(output_ids)
    # delta_ids = the entire test sequence (as if accepted in one spec step)
    delta_ids = list(output_ids)
    actual = parser.is_reasoning_end_streaming(padded_ids, delta_ids)
    assert is_reasoning_end == actual


def test_gptoss_is_reasoning_end_streaming_signature(gpt_oss_tokenizer):
    """Verify the method is callable with the expected signature."""
    parser = GptOssReasoningParser(gpt_oss_tokenizer)
    result = parser.is_reasoning_end_streaming([], [])
    assert result is False
