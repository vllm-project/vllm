# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.reasoning.gemma4_utils import parse_thinking_output

TURN_END = "<turn|>"
EOS = "<eos>"


@pytest.mark.parametrize(
    "suffix",
    [
        "",
        TURN_END,
        EOS,
        TURN_END + EOS,
        EOS + TURN_END,
        f" {TURN_END} {EOS}",
        f" {EOS} {TURN_END}",
        TURN_END + TURN_END,
    ],
    ids=[
        "none",
        "turn_only",
        "eos_only",
        "turn_then_eos",
        "eos_then_turn",
        "spaced_turn_then_eos",
        "spaced_eos_then_turn",
        "repeated_turn",
    ],
)
def test_answer_is_cleaned_regardless_of_sentinel_order(suffix: str):
    """Trailing sentinels must be stripped in any arrangement.

    Stripping each sentinel once in a fixed order only cleans one
    arrangement: checking ``<turn|>`` before ``<eos>`` leaves the turn
    marker in the answer for ``...<turn|><eos>``, because at that point the
    text ends with ``<eos>`` and the turn check has already been skipped.
    """
    result = parse_thinking_output(f"final answer{suffix}")

    assert result["answer"] == "final answer"
    assert TURN_END not in result["answer"]
    assert EOS not in result["answer"]


@pytest.mark.parametrize(
    "suffix",
    [TURN_END + EOS, EOS + TURN_END, TURN_END, EOS],
)
def test_thinking_block_answer_is_cleaned(suffix: str):
    """The same cleaning applies to the answer that follows a thinking block."""
    text = f"<|channel>thought\nsome reasoning<channel|>final answer{suffix}"
    result = parse_thinking_output(text)

    assert result["thinking"] == "some reasoning"
    assert result["answer"] == "final answer"


def test_answer_body_containing_sentinel_text_is_preserved():
    """Only trailing sentinels are stripped, not ones inside the answer."""
    result = parse_thinking_output(f"the marker is {TURN_END} in prose{TURN_END}")

    assert result["answer"] == f"the marker is {TURN_END} in prose"
