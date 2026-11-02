# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest

from tests.reasoning.utils import run_reasoning_extraction
from vllm.reasoning.muse_glimmer_reasoning_parser import MuseGlimmerReasoningParser

TURN = "<|start|>assistant "
OPEN = "to=self<|message|>"
EOM = "<|eom|>"

# A single truncated block: unaffected by the multi-block handling, kept as a
# control so the fix cannot regress the common shape.
SINGLE_BLOCK_TRUNCATED = {
    "output": f" {OPEN}Only thought",
    "reasoning": "Only thought",
    "content": None,
}
TWO_BLOCKS_CLOSED = {
    "output": f" {OPEN}First thought{EOM}{TURN}{OPEN}Second thought{EOM}",
    "reasoning": "First thought\nSecond thought",
    "content": None,
}
TWO_BLOCKS_LAST_TRUNCATED = {
    "output": f" {OPEN}First thought{EOM}{TURN}{OPEN}Second thought",
    "reasoning": "First thought\nSecond thought",
    "content": None,
}
THREE_BLOCKS_LAST_TRUNCATED = {
    "output": f" {OPEN}A{EOM}{TURN}{OPEN}B{EOM}{TURN}{OPEN}C",
    "reasoning": "A\nB\nC",
    "content": None,
}

TEST_CASES = [
    SINGLE_BLOCK_TRUNCATED,
    TWO_BLOCKS_CLOSED,
    TWO_BLOCKS_LAST_TRUNCATED,
    THREE_BLOCKS_LAST_TRUNCATED,
]
TEST_IDS = [
    "single_block_truncated",
    "two_blocks_closed",
    "two_blocks_last_truncated",
    "three_blocks_last_truncated",
]


def _parser() -> MuseGlimmerReasoningParser:
    # The parser works on decoded text, so the tokenizer is only needed to
    # satisfy the streaming helper.
    tokenizer = Mock()
    tokenizer.get_vocab.return_value = {}
    tokenizer.tokenize.return_value = []
    return MuseGlimmerReasoningParser(tokenizer)


@pytest.mark.parametrize("streaming", [True, False], ids=["streaming", "nonstreaming"])
@pytest.mark.parametrize("case", TEST_CASES, ids=TEST_IDS)
def test_multiple_reasoning_blocks(case: dict, streaming: bool):
    """Every to=self block survives, in both modes and with either framing.

    Generation that stops inside a later block used to return only that block,
    and the streaming path used to glue consecutive blocks together with no
    separator.
    """
    reasoning, content = run_reasoning_extraction(
        _parser(), [case["output"]], streaming=streaming
    )
    assert reasoning == case["reasoning"]
    assert content == case["content"]


@pytest.mark.parametrize("case", TEST_CASES, ids=TEST_IDS)
def test_multiple_reasoning_blocks_token_by_token(case: dict):
    """The streamed result must not depend on where the deltas land."""
    reasoning, content = run_reasoning_extraction(
        _parser(), list(case["output"]), streaming=True
    )
    assert reasoning == case["reasoning"]
    assert content == case["content"]
