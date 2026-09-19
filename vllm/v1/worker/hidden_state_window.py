# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure position-interval math for request-scoped hidden-state capture
windows (RFC: vllm-project/vllm#56916 "Windowed Hidden-State Collection
for vLLM Rollouts").

Deliberately kept dependency-free and separate from GPUModelRunner. The
part of this feature that actually needs care is correctly handling a
window that spans multiple forward steps -- including steps where
speculative decoding accepts a variable number of draft tokens per
step, so "one step" does not mean "one token". That logic is pure
interval arithmetic and has nothing to do with tensors or devices, so
it's isolated here where it can be unit tested without a GPU, a loaded
model, or vLLM's full dependency stack.

Convention: `step_start` / `step_end` follow the same meaning as a
request's `num_computed_tokens` before/after a step -- the absolute
position of the first token produced this step, and one-past-the-last.
This holds whether the step advanced by 1 token (ordinary decode), N
tokens (a prefill chunk), or a variable number of accepted draft
tokens (speculative decoding): in all three cases the positions
produced in one step are still contiguous, even though the step size
varies.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class HiddenStateWindow:
    """A half-open, absolute-position window of tokens whose hidden
    states a request wants captured: positions in [start, end)."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"window start must be >= 0, got {self.start}")
        if self.end <= self.start:
            raise ValueError(
                f"window end ({self.end}) must be > start ({self.start})"
            )


def window_step_overlap(
    window: HiddenStateWindow,
    step_start: int,
    step_end: int,
) -> tuple[int, int] | None:
    """Compute the overlap between `window` and one forward step's
    contiguous absolute-position range [step_start, step_end).

    Returns local slice bounds (lo, hi) into that step's own 0-indexed
    hidden-states rows for this request, such that hidden_states[lo:hi]
    are exactly the rows whose absolute position falls inside
    `window`. Returns None if this step contributes nothing to the
    window (fully before it, fully after it, or the step is empty).
    """
    if step_end <= step_start:
        return None

    overlap_start = max(window.start, step_start)
    overlap_end = min(window.end, step_end)
    if overlap_end <= overlap_start:
        return None

    return (overlap_start - step_start, overlap_end - step_start)


def window_is_complete(window: HiddenStateWindow, num_computed_tokens: int) -> bool:
    """True once `num_computed_tokens` (positions produced so far) has
    passed the end of the window -- i.e. nothing still to come for
    this request could fall inside it, so it's safe to stop tracking
    and return what's been captured."""
    return num_computed_tokens >= window.end
