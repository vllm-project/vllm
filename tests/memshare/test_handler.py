"""Tests for MemShare handler — step detection and Stage 1 similarity."""

import pytest

from vllm.memshare.handler import (
    MIN_STEP_CHARS,
    MAX_BUFFER_CHARS,
    MemShareHandler,
    RequestMemShareState,
    _BOUNDARY_RE,
)


def simulate_stream(full_text, chunk_size=1):
    """Feed text to RequestMemShareState in chunks, return completed steps."""
    state = RequestMemShareState("test-req")
    for i in range(0, len(full_text), chunk_size):
        chunk = full_text[i:i + chunk_size]
        state.on_new_text(chunk)
    state.finalize()
    return state.completed_steps


class TestStuckMatch:
    """Bug: re.search returns the first match. If a boundary pattern appears
    before MIN_STEP_CHARS, it matches every time but fails the length check,
    blocking all subsequent boundaries forever."""

    def test_boundary_at_position_zero(self):
        text = ("Wait, I need to think about this carefully and thoroughly. "
                "Let me verify the calculation is correct.")
        steps = simulate_stream(text)
        assert len(steps) == 2

    def test_two_early_boundaries(self):
        text = ("Wait, hmm, I need to figure out the right approach for "
                "this problem. Let me think about the base case first.")
        steps = simulate_stream(text)
        assert len(steps) == 2


class TestBoundaryPosition:
    """The non-capturing prefix (?:^|\\n|[.!?]\\s) is up to 2 chars.
    pos=MIN_STEP_CHARS-2 ensures we never miss a boundary whose prefix
    straddles the search start."""

    def test_boundary_at_exact_min_step_chars(self):
        # period at 38, space at 39, "Wait," at 40
        text = "x" * 38 + ". Wait, something else happens here that is long."
        steps = simulate_stream(text)
        assert len(steps) == 2

    def test_boundary_one_past_min(self):
        # period at 39, space at 40, "Wait," at 41
        text = "x" * 39 + ". Wait, something else happens here that is long."
        steps = simulate_stream(text)
        assert len(steps) == 2

    def test_boundary_too_short_rejected(self):
        # period at 37, space at 38, "Wait," at 39 → step is < 40 chars
        text = "x" * 37 + ". Wait, something else happens here that is long."
        steps = simulate_stream(text)
        assert len(steps) == 1

    def test_newline_prefix(self):
        text = ("Computing the integral of sin(x) from 0 to pi gives two."
                "\nWait, I should verify this by substitution.")
        steps = simulate_stream(text)
        assert len(steps) == 2


class TestNormalOperation:

    def test_boundary_well_past_min(self):
        text = ("The derivative of x squared is 2x, and integrating "
                "gives us the area. Wait, that might not be right.")
        steps = simulate_stream(text)
        assert len(steps) == 2

    def test_multiple_boundaries(self):
        text = ("First I compute the sum of the series using the formula. "
                "Wait, I should double check the convergence condition. "
                "Let me verify by plugging in n equals one to see if it works. "
                "Actually, I think the answer is different.")
        steps = simulate_stream(text)
        assert len(steps) == 4

    def test_no_boundaries(self):
        text = ("This is a simple response with no reasoning patterns. "
                "Just a plain answer: 42.")
        steps = simulate_stream(text)
        assert len(steps) == 1

    def test_very_short_text(self):
        text = "The answer is 7."
        steps = simulate_stream(text)
        assert len(steps) == 1

    def test_realistic_reasoning_trace(self):
        text = (
            "I need to find how many integers between 1 and 1000 are "
            "divisible by 3 but not by 5. Let me start by counting "
            "multiples of 3. There are floor(1000/3) = 333 multiples of 3. "
            "Wait, I need to subtract those also divisible by 5. Multiples "
            "of both 3 and 5 are multiples of 15. "
            "Let me verify that floor(1000/15) = 66. So the answer is "
            "333 - 66 = 267. "
            "Hmm, let me double check by considering the "
            "inclusion-exclusion principle more carefully. "
            "Actually, I am confident the answer is 267."
        )
        steps = simulate_stream(text)
        assert len(steps) >= 3


class TestStreamingConsistency:
    """Step detection must produce the same results regardless of how
    text arrives — char-by-char, word-sized chunks, or large blocks."""

    def test_chunk_sizes_produce_same_steps(self):
        text = ("The quadratic formula gives x equals negative b plus or "
                "minus the square root of b squared minus 4ac. "
                "Wait, I should check the discriminant first. "
                "Let me verify that b squared minus 4ac is non-negative.")
        steps_1 = simulate_stream(text, chunk_size=1)
        steps_5 = simulate_stream(text, chunk_size=5)
        steps_20 = simulate_stream(text, chunk_size=20)
        assert len(steps_1) == len(steps_5) == len(steps_20)


class TestMemShareHandler:
    """Integration tests for the MemShareHandler wrapper."""

    def test_on_token_creates_state(self):
        handler = MemShareHandler(enabled=True)
        handler.on_token("req-1", "some text")
        assert "req-1" in handler._states

    def test_disabled_handler_skips(self):
        handler = MemShareHandler(enabled=False)
        handler.on_token("req-1", "some text")
        assert "req-1" not in handler._states

    def test_on_request_finished_returns_state(self):
        handler = MemShareHandler(enabled=True)
        text = ("First compute the derivative of the function carefully. "
                "Wait, I think I made an arithmetic error somewhere.")
        for ch in text:
            handler.on_token("req-1", ch)
        state = handler.on_request_finished("req-1")
        assert state is not None
        assert len(state.completed_steps) == 2
        assert "req-1" not in handler._states

    def test_on_request_finished_unknown_id(self):
        handler = MemShareHandler(enabled=True)
        state = handler.on_request_finished("nonexistent")
        assert state is None

    def test_multiple_requests_isolated(self):
        handler = MemShareHandler(enabled=True)
        handler.on_token("req-1", "some text for request one")
        handler.on_token("req-2", "different text for request two")
        assert "req-1" in handler._states
        assert "req-2" in handler._states
        handler.on_request_finished("req-1")
        assert "req-1" not in handler._states
        assert "req-2" in handler._states

    def test_abort_cleans_up_state(self):
        handler = MemShareHandler(enabled=True)
        handler.on_token("req-1", "some text")
        handler.on_token("req-2", "other text")
        assert "req-1" in handler._states
        handler.on_request_aborted("req-1")
        assert "req-1" not in handler._states
        assert "req-2" in handler._states  # other requests unaffected

    def test_abort_unknown_id_no_error(self):
        handler = MemShareHandler(enabled=True)
        handler.on_request_aborted("nonexistent")  # should not raise


class TestSimilarityDetection:
    """Stage 1 cosine similarity should flag near-duplicate steps."""

    def test_identical_steps_detected(self):
        handler = MemShareHandler(enabled=True, threshold=0.8)
        text = ("Let me compute the sum by adding each term one by one. "
                "Wait, let me compute the sum by adding each term one by one. "
                "Actually, the answer is clear now.")
        for ch in text:
            handler.on_token("req-1", ch)
        state = handler.on_request_finished("req-1")
        assert len(state.candidates) >= 1
        # The candidate should pair step 0 with step 1
        pair = state.candidates[0]
        assert pair[2] > 0.8  # similarity score

    def test_dissimilar_steps_not_flagged(self):
        handler = MemShareHandler(enabled=True, threshold=0.8)
        text = ("The integral of cosine from zero to pi equals zero exactly. "
                "Wait, now I need to count the prime numbers below fifty. "
                "Let me check whether the graph is bipartite or not.")
        for ch in text:
            handler.on_token("req-1", ch)
        state = handler.on_request_finished("req-1")
        assert len(state.candidates) == 0


class TestBufferCap:
    """Buffer must not grow unboundedly when no boundaries are found."""

    def test_buffer_capped_at_max(self):
        state = RequestMemShareState("test-req")
        # Feed text with no boundary patterns, exceeding MAX_BUFFER_CHARS
        chunk = "no boundary here. " * 100  # ~1800 chars per call
        for _ in range(5):  # ~9000 chars total, well over 4096
            state.on_new_text(chunk)
        assert len(state.text_buffer) <= MAX_BUFFER_CHARS

    def test_buffer_cap_preserves_tail(self):
        state = RequestMemShareState("test-req")
        # Feed enough text to trigger the cap
        filler = "a" * (MAX_BUFFER_CHARS + 500)
        state.on_new_text(filler)
        # Buffer should contain the tail of the input
        assert state.text_buffer == filler[-MAX_BUFFER_CHARS:]

    def test_boundary_still_found_after_trim(self):
        state = RequestMemShareState("test-req")
        # Fill buffer past the cap with no boundaries
        filler = "x" * (MAX_BUFFER_CHARS + 500)
        state.on_new_text(filler)
        assert len(state.text_buffer) <= MAX_BUFFER_CHARS
        # Now add a boundary — it should still be detected
        state.on_new_text(
            "y" * 50 + ". Wait, this should be found after trimming."
        )
        assert len(state.completed_steps) >= 1
