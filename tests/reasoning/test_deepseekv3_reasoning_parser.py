# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer

from vllm.reasoning import (
    DeepSeekV3ReasoningParser,
    DeepSeekR1ReasoningParser,
    IdentityReasoningParser,
)

REASONING_MODEL_NAME = "deepseek-ai/DeepSeek-V3.1"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(REASONING_MODEL_NAME)


@pytest.mark.parametrize(
    "thinking,expected_parser_type",
    [
        (True, DeepSeekR1ReasoningParser),
        (False, IdentityReasoningParser),
    ],
)
def test_parser_selection(tokenizer, thinking, expected_parser_type):
    parser = DeepSeekV3ReasoningParser(
        tokenizer, thinking=thinking
    )
    assert isinstance(parser._parser, expected_parser_type)


