# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recovery planning for independently stored draft attention windows."""

from dataclasses import dataclass

TokenRange = tuple[int, int]


def _replay_suffix(
    required: TokenRange, ready_ranges: tuple[TokenRange, ...]
) -> TokenRange | None:
    """Keep a ready prefix, then replay continuously from its first hole."""
    cursor, end = required
    for ready_start, ready_end in sorted(ready_ranges):
        if ready_end <= cursor:
            continue
        if ready_start > cursor or cursor >= end:
            break
        cursor = max(cursor, min(ready_end, end))
    return (cursor, end) if cursor < end else None


@dataclass(frozen=True)
class DraftWindowRecoveryPlan:
    """Target replay needed to restore independently stored draft groups."""

    request_id: str
    main_cached_end: int
    required_ranges: tuple[TokenRange, ...]
    replay_ranges: tuple[TokenRange, ...]


def plan_draft_window_recovery(
    *,
    request_id: str,
    main_cached_end: int,
    retained_history_tokens: tuple[int, ...],
    block_sizes: tuple[int, ...],
    draft_ready_ranges: tuple[tuple[TokenRange, ...], ...],
) -> DraftWindowRecoveryPlan:
    """Plan the bounded historical context needed at a draft boundary."""
    num_groups = len(retained_history_tokens)
    if not (num_groups == len(block_sizes) == len(draft_ready_ranges)):
        raise ValueError("Draft recovery inputs must have one entry per group")
    if main_cached_end < 0:
        raise ValueError("main_cached_end must be non-negative")
    if any(context < 0 for context in retained_history_tokens) or any(
        block <= 0 for block in block_sizes
    ):
        raise ValueError("Retained history must be non-negative and blocks positive")

    required_ranges: list[TokenRange] = []
    group_replays: list[TokenRange] = []
    for retained_context, block, ready in zip(
        retained_history_tokens,
        block_sizes,
        draft_ready_ranges,
        strict=True,
    ):
        required_start = max(0, main_cached_end - retained_context) // block * block
        required = (required_start, main_cached_end)
        required_ranges.append(required)
        replay = _replay_suffix(required, ready)
        if replay is not None:
            group_replays.append(replay)

    replay_ranges = (
        ((min(start for start, _ in group_replays), main_cached_end),)
        if group_replays
        else ()
    )
    return DraftWindowRecoveryPlan(
        request_id=request_id,
        main_cached_end=main_cached_end,
        required_ranges=tuple(required_ranges),
        replay_ranges=replay_ranges,
    )
