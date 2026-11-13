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


def _missing_ranges(
    required: TokenRange, ready_ranges: tuple[TokenRange, ...]
) -> tuple[TokenRange, ...]:
    """Return exact gaps in `required`, preserving non-contiguous holes."""
    cursor, end = required
    missing: list[TokenRange] = []
    for ready_start, ready_end in sorted(ready_ranges):
        if ready_end <= cursor:
            continue
        if ready_start >= end:
            break
        if ready_start > cursor:
            missing.append((cursor, min(ready_start, end)))
        cursor = max(cursor, min(ready_end, end))
        if cursor >= end:
            break
    if cursor < end:
        missing.append((cursor, end))
    return tuple(missing)


def intersect_token_ranges(
    ranges: tuple[TokenRange, ...], selected: TokenRange
) -> tuple[TokenRange, ...]:
    """Intersect and merge token ranges with one scheduled interval."""
    selected_start, selected_end = selected
    intersections = sorted(
        (max(start, selected_start), min(end, selected_end))
        for start, end in ranges
        if max(start, selected_start) < min(end, selected_end)
    )
    merged: list[TokenRange] = []
    for start, end in intersections:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


@dataclass(frozen=True)
class DraftWindowRecoveryPlan:
    """Target replay needed to restore independently stored draft groups."""

    request_id: str
    main_cached_end: int
    required_ranges: tuple[TokenRange, ...]
    ready_ranges: tuple[tuple[TokenRange, ...], ...]
    replay_ranges: tuple[TokenRange, ...]

    @property
    def missing_ranges(self) -> tuple[TokenRange, ...]:
        """Exact union of missing Draft pages across all Draft groups."""
        missing = tuple(
            token_range
            for required, ready in zip(
                self.required_ranges, self.ready_ranges, strict=True
            )
            for token_range in _missing_ranges(required, ready)
        )
        if not missing:
            return ()
        start = min(range_start for range_start, _ in missing)
        end = max(range_end for _, range_end in missing)
        return intersect_token_ranges(missing, (start, end))


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
        ready_ranges=draft_ready_ranges,
        replay_ranges=replay_ranges,
    )
