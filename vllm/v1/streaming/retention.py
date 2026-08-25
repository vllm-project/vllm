# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Configuration and bookkeeping for streaming-session retention.

A streaming session runs indefinitely; without eviction its KV cache grows
unbounded. `StreamingRetentionParams` configures how much to keep, and
`HistorySegment` tracks per-segment token ranges so the eviction code (in
`eviction.py`) can drop the oldest segments first. This module just holds the
data structures the scheduler maintains as chunks arrive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from vllm.logger import init_logger

logger = init_logger(__name__)

# Minimum total-token budget: must hold the pinned attention sink plus
# headroom for the next chunk's prefill.
_MIN_SESSION_TOKENS = 1024


@dataclass
class StreamingRetentionParams:
    """How long to keep streaming-session context in the KV cache."""

    max_video_segments: int = 30
    """Hard cap on retained video segments; oldest above the cap are evicted
    at chunk boundaries."""

    max_text_tokens: int | None = None
    """Soft cap on non-pinned text tokens (`user_text` + `assistant_text`);
    None disables the text budget."""

    max_session_tokens: int | None = 7000
    """Hard cap on total prompt length post-eviction; the safety net that
    keeps an unbounded session below `max_model_len`. Required (must not be
    None; see `__post_init__`)."""

    eviction_policy: Literal["sliding_window"] = "sliding_window"
    """Currently only sliding-window is implemented."""

    reprefill_threshold: float = 0.7
    """Fraction of the model's trained position range above which the session
    self-preempts and re-prefills its surviving tokens at fresh positions from
    0 (see `reprefill.py`). Keeps mRoPE positions bounded over very long
    sessions."""

    def __post_init__(self) -> None:
        """Validate the budget on every construction (in-process and
        IPC-deserialized) so a bad config fails loudly at admission."""
        # < 1 would evict every chunk's video before the encoder runs.
        if self.max_video_segments < 1:
            raise ValueError(
                "max_video_segments must be >= 1 "
                f"(got {self.max_video_segments}); a value < 1 evicts "
                "every chunk's video before it is computed."
            )

        # Required: the load-bearing safety net against block-alignment drift.
        if self.max_session_tokens is None:
            raise ValueError(
                "max_session_tokens must be set: it is the hard token "
                "safety net that keeps an unbounded streaming session "
                "below max_model_len."
            )
        if self.max_session_tokens < _MIN_SESSION_TOKENS:
            raise ValueError(
                f"max_session_tokens ({self.max_session_tokens}) must be "
                f">= {_MIN_SESSION_TOKENS} to hold the pinned sink plus a "
                "chunk's prefill headroom."
            )

        # Optional, but if set must be positive and within the session budget.
        if self.max_text_tokens is not None:
            if self.max_text_tokens <= 0:
                raise ValueError(
                    f"max_text_tokens must be > 0 when set (got "
                    f"{self.max_text_tokens})."
                )
            if self.max_text_tokens > self.max_session_tokens:
                raise ValueError(
                    f"max_text_tokens ({self.max_text_tokens}) must not "
                    f"exceed max_session_tokens ({self.max_session_tokens})."
                )

        # A non-positive threshold would re-trigger a full re-prefill at
        # every chunk boundary (unbounded GPU-burn livelock). Written as
        # `not (0 < x)` so NaN — which fails every comparison and would
        # silently disable re-prefill, letting positions grow unbounded —
        # is rejected too.
        if not (self.reprefill_threshold > 0):
            raise ValueError(
                f"reprefill_threshold ({self.reprefill_threshold}) must be "
                "> 0; a non-positive value re-triggers re-prefill at every "
                "chunk boundary."
            )

        # Disabling re-prefill is supported, but nothing else bounds mRoPE
        # position growth, so a long session eventually exceeds the trained
        # range (rotary cache out-of-bounds). Warn once.
        if self.reprefill_threshold >= 1.0:
            logger.warning_once(
                "StreamingRetentionParams.reprefill_threshold=%s disables "
                "re-prefill; mRoPE positions will grow without bound and a "
                "long-running session will eventually exceed the model's "
                "trained position range (rotary-cache out-of-bounds). Use a "
                "value < 1.0 (default 0.7) to keep positions bounded.",
                self.reprefill_threshold,
            )


SegmentType = Literal[
    "system_prompt",
    "video",
    "user_text",
    "assistant_text",
]


@dataclass
class HistorySegment:
    """One contiguous range of tokens in the streaming session prompt.

    Eviction keys purely off `token_range` (see eviction.py). After an
    eviction, surviving segments' `token_range` is shifted down to stay
    consistent with the compacted token arrays, but `_mrope_positions` values
    are not — so the position space has gaps, preserving cached-K/V rotation.
    """

    segment_type: SegmentType
    token_range: tuple[int, int]
    """`[start, end)` into `_all_token_ids`. Shifts down as other segments are
    evicted, so always read it fresh from the segment — never cache it."""

    mm_item_id: str | None = None
    """`MultiModalFeatureSpec` identifier (video segments only), used to free
    the encoder cache entry."""

    pinned: bool = False
    """If True, eviction skips this segment. Set on the chunk-1 anchor."""

    age_chunks: int = 0
    """Chunk boundaries elapsed since this segment was added; 0 means it
    arrived with the current chunk (its KV may not be computed yet, so
    eviction skips it)."""
