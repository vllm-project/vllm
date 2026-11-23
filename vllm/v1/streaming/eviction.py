# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Block-aligned eviction at streaming chunk boundaries.

Surviving KV retains its original rotary positions. Multimodal placeholders
are removed whole; text next to an evicted frame can supply alignment slack.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import replace
from typing import TYPE_CHECKING

from vllm.v1.streaming.retention import HistorySegment, StreamingRetentionParams

if TYPE_CHECKING:
    from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.request import Request

_TEXT_TYPES = ("user_text", "assistant_text")


def _eviction_range(
    request: Request, segment: HistorySegment, block_size: int
) -> tuple[int, int] | None:
    if segment.pinned or segment.token_range[1] > request.num_computed_tokens:
        return None
    start, end = segment.token_range
    index = next(
        i for i, existing in enumerate(request.session_history) if existing is segment
    )
    for neighbor in reversed(request.session_history[:index]):
        if (
            neighbor.pinned
            or neighbor.segment_type not in _TEXT_TYPES
            or neighbor.token_range[1] != start
        ):
            break
        start = neighbor.token_range[0]
    for neighbor in request.session_history[index + 1 :]:
        if (
            neighbor.pinned
            or neighbor.segment_type not in _TEXT_TYPES
            or neighbor.token_range[0] != end
            or neighbor.token_range[1] > request.num_computed_tokens
        ):
            break
        end = neighbor.token_range[1]

    victim_start, victim_end = segment.token_range
    start = max(
        (start + block_size - 1) // block_size * block_size,
        victim_start // block_size * block_size,
    )
    end = min(
        end // block_size * block_size,
        (victim_end + block_size - 1) // block_size * block_size,
    )
    for feature in request.mm_features:
        feature_start = feature.mm_position.offset
        feature_end = feature_start + feature.mm_position.length
        if feature_start < start < feature_end:
            start = (feature_end + block_size - 1) // block_size * block_size
    for feature in reversed(request.mm_features):
        feature_start = feature.mm_position.offset
        feature_end = feature_start + feature.mm_position.length
        if feature_start < end < feature_end:
            end = feature_start // block_size * block_size

    if start >= end or start >= victim_end or end <= victim_start:
        return None
    if segment.segment_type == "video" and any(
        feature.mm_position.offset < victim_end
        and feature.mm_position.offset + feature.mm_position.length > victim_start
        and not (
            start <= feature.mm_position.offset
            and feature.mm_position.offset + feature.mm_position.length <= end
        )
        for feature in request.mm_features
    ):
        return None
    return start, end


def evict_segment(
    request: Request,
    segment: HistorySegment,
    kv_cache_manager: KVCacheManager,
    encoder_cache_manager: EncoderCacheManager,
    *,
    pending_encoder_input_ids: set[int] | None = None,
) -> bool:
    """Remove an old segment when a safe whole-block range exists.

    The caller must quiesce the request before calling. A range that would
    split a multimodal feature or touch the new chunk is left intact.
    """
    if segment.pinned:
        raise ValueError("Cannot evict a pinned segment")
    block_size = kv_cache_manager.coordinator.single_type_managers[0].block_size
    span = _eviction_range(request, segment, block_size)
    if span is None:
        return False
    start, end = span
    aligned_start, aligned_end, _ = kv_cache_manager.evict_token_range_for_request(
        request.request_id, start, end
    )
    if (aligned_start, aligned_end) != span:
        # Allocated KV must cover all confirmed tokens at a chunk boundary.
        raise RuntimeError("Streaming eviction range exceeds the allocated KV table")
    _compact_range(
        request,
        start,
        end,
        encoder_cache_manager,
        pending_encoder_input_ids=pending_encoder_input_ids,
    )
    request.position_offset += end - start
    request.pending_evicted_token_ranges.append(span)
    return True


def _compact_range(
    request: Request,
    start: int,
    end: int,
    encoder_cache_manager: EncoderCacheManager,
    *,
    pending_encoder_input_ids: set[int] | None = None,
) -> None:
    width = end - start

    removed = [
        i
        for i, feature in enumerate(request.mm_features)
        if start <= feature.mm_position.offset
        and feature.mm_position.offset + feature.mm_position.length <= end
    ]
    remove_encoder_inputs(
        request, removed, encoder_cache_manager, pending_encoder_input_ids
    )

    surviving_features = []
    for i, feature in enumerate(request.mm_features):
        if i in removed:
            continue
        if feature.mm_position.offset >= end:
            feature = replace(
                feature,
                mm_position=replace(
                    feature.mm_position, offset=feature.mm_position.offset - width
                ),
            )
        surviving_features.append(feature)
    request.mm_features = surviving_features

    def compact(position: int) -> int:
        return position - min(max(position - start, 0), width)

    history = []
    input_complete = True
    for existing in request.session_history:
        old_start, old_end = existing.token_range
        cut = old_start < end and start < old_end
        if existing.segment_type != "assistant_text":
            input_complete = existing.complete and not cut
        elif not input_complete:
            existing.complete = False
        new_start, new_end = compact(old_start), compact(old_end)
        if new_start == new_end:
            continue
        if cut:
            existing.complete = False
        existing.token_range = new_start, new_end
        if existing.segment_type == "video" and not any(
            feature.mm_position.offset < new_end
            and feature.mm_position.offset + feature.mm_position.length > new_start
            for feature in surviving_features
        ):
            existing.segment_type = "user_text"
        history.append(existing)
    request.session_history = history

    del request._all_token_ids[start:end]
    assert request.prompt_token_ids is not None
    del request.prompt_token_ids[start:end]
    request.num_prompt_tokens -= width
    request.num_computed_tokens -= width


def remove_encoder_inputs(
    request: Request,
    removed: list[int],
    encoder_cache_manager: EncoderCacheManager,
    pending_encoder_input_ids: set[int] | None,
) -> None:
    """Release removed inputs before their feature list is compacted."""
    encoder_cache_manager.remove_input_ids(request, removed)
    if pending_encoder_input_ids is not None:
        remaining = {
            i - bisect_left(removed, i)
            for i in pending_encoder_input_ids
            if i not in removed
        }
        pending_encoder_input_ids.clear()
        pending_encoder_input_ids.update(remaining)


def drop_old_segments_for_reprefill(
    request: Request,
    retention: StreamingRetentionParams,
    encoder_cache_manager: EncoderCacheManager,
    *,
    pending_encoder_input_ids: set[int] | None = None,
) -> int:
    """Meet the hard token cap when whole-block eviction cannot make progress.

    This removes exact old segments without modifying KV. The caller must
    immediately free all request KV and re-prefill the surviving prompt.
    """
    dropped = 0
    while request.num_prompt_tokens > retention.max_session_tokens:
        for segment in request.session_history:
            start, end = segment.token_range
            if (
                segment.pinned
                or segment.age_chunks == 0
                or end > request.num_computed_tokens
                or end <= start
                or any(
                    feature.mm_position.offset < end
                    and feature.mm_position.offset + feature.mm_position.length > start
                    and not (
                        start <= feature.mm_position.offset
                        and feature.mm_position.offset + feature.mm_position.length
                        <= end
                    )
                    for feature in request.mm_features
                )
            ):
                continue
            _compact_range(
                request,
                start,
                end,
                encoder_cache_manager,
                pending_encoder_input_ids=pending_encoder_input_ids,
            )
            dropped += 1
            break
        else:
            break
    return dropped


def maybe_evict_old_segments(
    request: Request,
    retention: StreamingRetentionParams,
    kv_cache_manager: KVCacheManager,
    encoder_cache_manager: EncoderCacheManager,
    *,
    pending_encoder_input_ids: set[int] | None = None,
) -> int:
    """Apply video, text, and total-token budgets, oldest segments first.

    Alignment can prevent a budget from being reached. The scheduler must
    reject an over-budget chunk before installing it on a worker.
    """
    evicted = 0
    for phase in ("video", "text", "total"):
        while True:
            unpinned = [s for s in request.session_history if not s.pinned]
            if phase == "video":
                candidates = [s for s in unpinned if s.segment_type == "video"]
                over_budget = len(candidates) > retention.max_video_segments
            elif phase == "text":
                candidates = [s for s in unpinned if s.segment_type in _TEXT_TYPES]
                over_budget = (
                    retention.max_text_tokens is not None
                    and sum(s.token_range[1] - s.token_range[0] for s in candidates)
                    > retention.max_text_tokens
                )
            else:
                candidates = unpinned
                over_budget = request.num_prompt_tokens > retention.max_session_tokens
            if not over_budget:
                break
            for candidate in candidates:
                if candidate.age_chunks > 0 and evict_segment(
                    request,
                    candidate,
                    kv_cache_manager,
                    encoder_cache_manager,
                    pending_encoder_input_ids=pending_encoder_input_ids,
                ):
                    evicted += 1
                    break
            else:
                break
    return evicted
