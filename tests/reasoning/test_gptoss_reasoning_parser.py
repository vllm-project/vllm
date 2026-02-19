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
ASSISTANT_CONTENT_START_PREFIX = "<|end|><|start|>assistant<|channel|>final"
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

TEST_CASES = [
    BASIC_CONTENT,
    BASIC_REASONING_ONLY,
    COMPLEX_CONTENT_INCOMPLETE_PREFIX_ONLY,
    COMPLEX_CONTENT_SUFFIX_ONLY,
    COMPLEX_CONTENT_1_NO_SUFFIX,
    COMPLEX_CONTENT_1,
    COMPLEX_CONTENT_1_WITH_CONTENT,
    COMPLEX_CONTENT_2,
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
