# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.reasoning.gemma4_utils import (
    _clean_answer,
    _strip_thought_label,
    parse_thinking_output,
)

# ---------------------------------------------------------------------------
# _strip_thought_label
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        pytest.param("thought\nActual content", "Actual content", id="prefix_stripped"),
        pytest.param(
            "Some text thought\nhere",
            "Some text thought\nhere",
            id="prefix_mid_string_not_stripped",
        ),
        pytest.param(
            "Thought\nCapitalised",
            "Thought\nCapitalised",
            id="case_sensitive_capital_not_stripped",
        ),
        pytest.param(
            "thought",
            "thought",
            id="thought_without_newline_not_stripped",
        ),
        pytest.param("", "", id="empty_string"),
    ],
)
def test_strip_thought_label(text: str, expected: str) -> None:
    assert _strip_thought_label(text) == expected


# ---------------------------------------------------------------------------
# _clean_answer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        pytest.param("  answer  ", "answer", id="whitespace_stripped"),
        pytest.param("answer<turn|>", "answer", id="trailing_turn_end_stripped"),
        pytest.param("answer<eos>", "answer", id="trailing_eos_stripped"),
        # <turn|> is NOT the last token here; only <eos> is stripped.
        pytest.param(
            "answer<turn|><eos>",
            "answer<turn|>",
            id="turn_then_eos_only_eos_stripped",
        ),
        pytest.param(
            "answer  <turn|>  ",
            "answer",
            id="turn_end_with_surrounding_whitespace",
        ),
        pytest.param("plain answer", "plain answer", id="no_sentinel_unchanged"),
    ],
)
def test_clean_answer(text: str, expected: str) -> None:
    assert _clean_answer(text) == expected


# ---------------------------------------------------------------------------
# parse_thinking_output — thinking path (end tag present)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected_thinking, expected_answer",
    [
        pytest.param(
            "<|channel>thought\nActual reasoning here<channel|>Final answer",
            "Actual reasoning here",
            "Final answer",
            id="thought_label_stripped",
        ),
        pytest.param(
            "<|channel>thought\nLine1\nLine2<channel|>Answer",
            "Line1\nLine2",
            "Answer",
            id="multiline_thinking",
        ),
        # "thought\n" with no content after: thinking_block = "<|channel>thought\n",
        # strip start tag -> "thought\n", .strip() -> "thought",
        # _strip_thought_label("thought") -> "thought" (no trailing newline),
        # final .strip() -> "thought".
        pytest.param(
            "<|channel>thought\n<channel|>",
            "thought",
            "",
            id="thought_label_only_thinking_equals_thought",
        ),
        pytest.param(
            "<|channel>This is reasoning<channel|>Final answer",
            "This is reasoning",
            "Final answer",
            id="both_tags_present",
        ),
        pytest.param(
            "<|channel>This is reasoning<channel|>Answer<turn|>",
            "This is reasoning",
            "Answer",
            id="answer_with_turn_end",
        ),
        pytest.param(
            "<|channel>This is reasoning<channel|>Answer<eos>",
            "This is reasoning",
            "Answer",
            id="answer_with_eos",
        ),
        # No start tag: thinking_block is not stripped on the left.
        pytest.param(
            "Some prefix<channel|>Final answer",
            "Some prefix",
            "Final answer",
            id="end_tag_only_no_start_tag",
        ),
    ],
)
def test_parse_thinking_output_thinking_path(
    text: str, expected_thinking: str, expected_answer: str
) -> None:
    result = parse_thinking_output(text)
    assert result["thinking"] == expected_thinking
    assert result["answer"] == expected_answer


# ---------------------------------------------------------------------------
# parse_thinking_output — no-thinking path (no end tag)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected_answer",
    [
        pytest.param("Clean answer text", "Clean answer text", id="clean_path"),
        pytest.param(
            "thought\nSpurious label",
            "Spurious label",
            id="spurious_thought_label_stripped",
        ),
        pytest.param(
            "Answer text<turn|>",
            "Answer text",
            id="trailing_turn_end_stripped",
        ),
        pytest.param(
            "Answer text<eos>",
            "Answer text",
            id="trailing_eos_stripped",
        ),
        pytest.param("", "", id="empty_string"),
        pytest.param(
            "<|channel>This is reasoning",
            "<|channel>This is reasoning",
            id="start_tag_only_no_end_tag",
        ),
    ],
)
def test_parse_thinking_output_no_thinking_path(
    text: str, expected_answer: str
) -> None:
    result = parse_thinking_output(text)
    assert result["thinking"] is None
    assert result["answer"] == expected_answer
