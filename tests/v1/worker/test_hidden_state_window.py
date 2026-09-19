# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for vllm.v1.worker.hidden_state_window (RFC #56916)."""

import pytest

from vllm.v1.worker.hidden_state_window import (
    HiddenStateWindow,
    window_is_complete,
    window_step_overlap,
)


def test_window_rejects_negative_start():
    with pytest.raises(ValueError):
        HiddenStateWindow(start=-1, end=5)


def test_window_rejects_empty_range():
    with pytest.raises(ValueError):
        HiddenStateWindow(start=5, end=5)


def test_window_rejects_inverted_range():
    with pytest.raises(ValueError):
        HiddenStateWindow(start=5, end=3)


def test_empty_step_contributes_nothing():
    w = HiddenStateWindow(start=1000, end=2000)
    assert window_step_overlap(w, step_start=500, step_end=500) is None


def test_step_fully_before_window():
    w = HiddenStateWindow(start=1000, end=2000)
    assert window_step_overlap(w, step_start=0, step_end=1000) is None


def test_step_fully_after_window():
    w = HiddenStateWindow(start=1000, end=2000)
    assert window_step_overlap(w, step_start=2000, step_end=2001) is None


def test_ordinary_single_token_decode_step_inside_window():
    w = HiddenStateWindow(start=1000, end=2000)
    # one decode step producing exactly position 1500
    assert window_step_overlap(w, step_start=1500, step_end=1501) == (0, 1)


def test_chunked_prefill_step_straddles_window_start():
    w = HiddenStateWindow(start=1000, end=2000)
    # a 200-token prefill chunk covering positions [900, 1100):
    # only positions [1000, 1100) -- local rows 100..199 -- are wanted
    assert window_step_overlap(w, step_start=900, step_end=1100) == (100, 200)


def test_step_straddles_window_end():
    w = HiddenStateWindow(start=1000, end=2000)
    # a 100-token chunk covering [1950, 2050): only [1950, 2000) wanted
    assert window_step_overlap(w, step_start=1950, step_end=2050) == (0, 50)


def test_step_fully_inside_window():
    w = HiddenStateWindow(start=1000, end=2000)
    # a 3-token step (e.g. 1 accepted decode + 2 accepted spec tokens)
    # landing entirely inside the window
    assert window_step_overlap(w, step_start=1500, step_end=1503) == (0, 3)


def test_window_fully_inside_one_large_step():
    w = HiddenStateWindow(start=1000, end=1002)
    # a single big prefill chunk covering [500, 2000) swallows the
    # whole (tiny) window in one step
    assert window_step_overlap(w, step_start=500, step_end=2000) == (500, 502)


def test_boundary_touching_start_is_included():
    w = HiddenStateWindow(start=1000, end=2000)
    # step ending exactly at window start contributes nothing ...
    assert window_step_overlap(w, step_start=990, step_end=1000) is None
    # ... but a step starting exactly at window start does
    assert window_step_overlap(w, step_start=1000, step_end=1010) == (0, 10)


def test_boundary_touching_end_is_excluded():
    w = HiddenStateWindow(start=1000, end=2000)
    # step starting exactly at window end contributes nothing (half-open)
    assert window_step_overlap(w, step_start=2000, step_end=2010) is None
    # a step ending exactly at window end includes everything up to it
    assert window_step_overlap(w, step_start=1990, step_end=2000) == (0, 10)


@pytest.mark.parametrize(
    "num_computed_tokens,expected",
    [(999, False), (2000, True), (2001, True), (1500, False)],
)
def test_window_is_complete(num_computed_tokens, expected):
    w = HiddenStateWindow(start=1000, end=2000)
    assert window_is_complete(w, num_computed_tokens) is expected


def test_multi_step_capture_reconstructs_window_exactly_with_spec_decode():
    """Simulate a realistic step sequence -- ordinary decode, a
    chunked-prefill-sized jump, and variable-size speculative-decode
    steps -- and check that concatenating what each step reports as
    "in window" reconstructs the requested window exactly once, with
    no gaps and no double-counting. This is the property the whole
    feature depends on.
    """
    window = HiddenStateWindow(start=1000, end=1010)

    steps = [
        (0, 998),  # big prefill chunk, entirely before the window
        (998, 999),  # ordinary decode step, still before
        (999, 1002),  # 3 accepted tokens: 999 (outside) + 1000,1001 (inside)
        (1002, 1003),  # inside
        (1003, 1007),  # 4 accepted tokens, all inside
        (1007, 1012),  # 5 accepted tokens: 1007,1008,1009 inside, then overshoot
    ]

    captured_absolute_positions: list[int] = []
    num_computed_tokens = 0
    for step_start, step_end in steps:
        assert step_start == num_computed_tokens  # steps are contiguous
        overlap = window_step_overlap(window, step_start, step_end)
        if overlap is not None:
            lo, hi = overlap
            captured_absolute_positions.extend(
                range(step_start + lo, step_start + hi)
            )
        num_computed_tokens = step_end
        if window_is_complete(window, num_computed_tokens):
            break

    assert captured_absolute_positions == list(range(1000, 1010))
