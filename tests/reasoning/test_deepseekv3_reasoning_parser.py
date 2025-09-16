# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer

from vllm.reasoning import (
    ReasoningParserManager,
    DeepSeekV3ReasoningParser,
    DeepSeekR1ReasoningParser,
    IdentityReasoningParser,
)
from vllm.entrypoints.openai.protocol import ChatCompletionRequest

REASONING_MODEL_NAME = "deepseek-ai/DeepSeek-V3.1"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(REASONING_MODEL_NAME)


@pytest.mark.parametrize(
    "thinking,separate_reasoning,expected_parser_type",
    [
        (True, True, DeepSeekR1ReasoningParser),
        (False, True, IdentityReasoningParser),
        (True, False, IdentityReasoningParser),
        (False, False, IdentityReasoningParser),
    ],
)
def test_parser_selection(tokenizer, thinking, separate_reasoning, expected_parser_type):
    parser = DeepSeekV3ReasoningParser(
        tokenizer, thinking=thinking, separate_reasoning=separate_reasoning
    )
    assert isinstance(parser._parser, expected_parser_type)


@pytest.mark.parametrize(
    "thinking,separate_reasoning,output,is_reasoning_end,expected_reasoning,expected_content",
    [
        # These cases mirror the logic from R1 parser vs. Identity parser
        # Case 1: DeepSeekR1ReasoningParser logic
        (
            True,
            True,
            "<think>Reasoning text</think>Content text",
            True,
            "Reasoning text",
            "Content text",
        ),
        # Case 2: IdentityReasoningParser logic (output is all reasoning, no separation)
        (
            False,
            True,
            "Just content",
            False,
            "Just content",
            None,
        ),
        # Case 3: IdentityReasoningParser logic with thinking True but separate_reasoning False
        (
            True,
            False,
            "Only reasoning",
            False,
            "Only reasoning",
            None,
        ),
        # Case 4: DeepSeekR1ReasoningParser with only reasoning section
        (
            True,
            True,
            "<think>Reasoning only</think>",
            True,
            "Reasoning only",
            None,
        ),
    ],
)
def test_deepseekv3_reasoning_parser_functionality(
    tokenizer,
    thinking,
    separate_reasoning,
    output,
    is_reasoning_end,
    expected_reasoning,
    expected_content,
):
    parser = DeepSeekV3ReasoningParser(
        tokenizer, thinking=thinking, separate_reasoning=separate_reasoning
    )
    output_tokens = tokenizer.tokenize(output)
    output_ids = tokenizer.convert_tokens_to_ids(output_tokens)

    # is_reasoning_end
    assert parser.is_reasoning_end(output_ids) == is_reasoning_end

    # extract_content_ids
    if expected_content is not None:
        content_tokens = tokenizer.tokenize(expected_content)
        content_ids = tokenizer.convert_tokens_to_ids(content_tokens)
        assert parser.extract_content_ids(output_ids) == content_ids
    else:
        assert parser.extract_content_ids(output_ids) == []

    # extract_reasoning_content
    request = ChatCompletionRequest(
        model="test-model", messages=[], temperature=1.0
    )
    reasoning, content = parser.extract_reasoning_content(output, request)
    assert reasoning == expected_reasoning
    assert content == expected_content
