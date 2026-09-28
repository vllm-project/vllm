# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.reasoning import ReasoningParserManager
from vllm.tokenizers import get_tokenizer

pytestmark = pytest.mark.skip_global_cleanup

REASONING_MODEL_NAME = "stepfun-ai/step3"


@pytest.fixture(scope="module")
def step3_parser():
    tokenizer = get_tokenizer(tokenizer_name=REASONING_MODEL_NAME)
    return ReasoningParserManager.get_reasoning_parser("step3")(tokenizer)


def test_count_reasoning_tokens_stops_at_think_end(step3_parser):
    end = step3_parser.think_end_token_id
    assert step3_parser.count_reasoning_tokens([11, 12, end, 13]) == 2


def test_count_reasoning_tokens_without_think_end(step3_parser):
    """Step3 treats output as reasoning until </think> appears."""
    assert step3_parser.count_reasoning_tokens([11, 12, 13]) == 3
