# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Retention configuration and chunk bookkeeping for bounded sessions."""

import math
from dataclasses import dataclass
from typing import Literal


@dataclass
class StreamingRetentionParams:
    """Targets for unpinned history and a hard cap on retained prompt tokens.

    Video and text targets are constrained by block and feature boundaries.
    The scheduler closes sessions that cannot fit `max_session_tokens`.
    """

    max_video_segments: int = 30
    max_text_tokens: int | None = None
    max_session_tokens: int = 7000
    reprefill_threshold: float = 0.7

    def __post_init__(self) -> None:
        for name in ("max_video_segments", "max_session_tokens"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_text_tokens is not None and (
            not isinstance(self.max_text_tokens, int)
            or isinstance(self.max_text_tokens, bool)
            or not 0 < self.max_text_tokens <= self.max_session_tokens
        ):
            raise ValueError(
                "max_text_tokens must be positive and <= max_session_tokens"
            )
        if not math.isfinite(self.reprefill_threshold) or not (
            0 < self.reprefill_threshold < 1
        ):
            raise ValueError("reprefill_threshold must be strictly between 0 and 1")


SegmentType = Literal["system_prompt", "video", "user_text", "assistant_text"]


@dataclass
class HistorySegment:
    """A half-open token range in the session's current compacted prompt."""

    segment_type: SegmentType
    token_range: tuple[int, int]
    pinned: bool = False
    age_chunks: int = 0
    complete: bool = True
