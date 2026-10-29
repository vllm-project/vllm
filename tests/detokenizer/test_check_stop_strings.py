# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for check_stop_strings.

These are pure-function tests (no model / GPU). They pin down which stop
string is selected when several stop strings match within the text that was
appended in a single step -- which happens under speculative decoding, where
multiple tokens (and therefore multiple stop strings) can be appended at once.
"""

import pytest

from vllm.v1.engine.detokenizer import check_stop_strings


@pytest.mark.parametrize("stop", [["a", "is"], ["is", "a"]])
def test_earliest_completing_stop_wins_regardless_of_list_order(stop):
    # " The user is a": " is a" (5 chars) was appended in one step. Both "is"
    # (index 10) and " a" (index 13) land in the same window. "is" completes
    # earlier in the text, so it must win over list order.
    text = " The user is a"
    new_char_count = len(" is a")

    assert check_stop_strings(text, new_char_count, stop, include_in_output=False) == (
        "is",
        10,
    )


@pytest.mark.parametrize("stop", [["a", "is"], ["is", "a"]])
def test_earliest_completing_stop_include_in_output(stop):
    text = " The user is a"
    new_char_count = len(" is a")

    # Truncate to the end of "is" (index 12) -> " The user is".
    assert check_stop_strings(text, new_char_count, stop, include_in_output=True) == (
        "is",
        12,
    )


def test_completion_position_not_start_position():
    # "b" starts later than "abc" but completes earlier, so it must win.
    text = "abc"
    assert check_stop_strings(
        text, len(text), ["abc", "b"], include_in_output=False
    ) == ("b", 1)


@pytest.mark.parametrize(
    "stop,expected",
    [
        (["ab", "b"], ("ab", 0)),
        (["b", "ab"], ("b", 1)),
    ],
)
def test_ties_broken_by_list_order(stop, expected):
    # "ab" and "b" both complete at index 2; list order decides the winner.
    text = "ab"
    assert (
        check_stop_strings(text, len(text), stop, include_in_output=False) == expected
    )


def test_single_stop_in_window_unchanged():
    # The common case (one stop in the window) is unaffected by the change.
    text = "hello world."
    assert check_stop_strings(text, 1, ["."], include_in_output=False) == (".", 11)
    # Stop completes at the very end -> no truncation needed (-1).
    assert check_stop_strings(text, 1, ["."], include_in_output=True) == (".", -1)


def test_no_match_and_empty_inputs_return_none():
    assert check_stop_strings("hello", 5, ["zzz"], include_in_output=False) is None
    assert check_stop_strings("hello", 0, ["h"], include_in_output=False) is None
    assert check_stop_strings("hello", 5, [], include_in_output=False) is None
