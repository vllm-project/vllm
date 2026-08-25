# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Intra-session KV eviction for streaming sessions.

A sliding-window policy drops the oldest video segments from a live
session once the segment count exceeds the retention budget, keeping
long-running captioning within its KV budget.

`evict_segment` is the unit step: it frees the segment's KV blocks,
drops its encoder cache entry, and removes the segment's tokens and
mRoPE positions from the request's lockstep arrays. Surviving segments'
`token_range` fields shift down by the evicted length so they stay valid
against the compacted arrays.

mRoPE position VALUES are NOT shifted: surviving tokens keep their
original coordinates, so the position space develops gaps. That is what
keeps cached K/V rotationally consistent — each surviving token's K was
rotated under its original position, which is still what RoPE pairs with
the new chunk's queries.
"""

from __future__ import annotations

import contextlib
from dataclasses import replace
from typing import TYPE_CHECKING

from vllm.v1.streaming.retention import HistorySegment, StreamingRetentionParams

if TYPE_CHECKING:
    from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.request import Request


def evict_segment(
    request: Request,
    segment: HistorySegment,
    kv_cache_manager: KVCacheManager,
    encoder_cache_manager: EncoderCacheManager,
) -> None:
    """Drop one history segment from a live session.

    `segment` may be a single segment OR a packed blob from
    `_pack_oldest_run_to_free_block`; the logic keys off the token range, so
    it handles either. All effects run in lockstep, so the caller must have
    validated the victim already:

      1. Free the segment's KV blocks (back to the free pool).
      2. Release the encoder entry and drop the `MultiModalFeatureSpec` for
         every mm item in the range (one per video, several per blob, none
         for text).
      3. Delete the segment's slice of `_all_token_ids` / `_mrope_positions`.
      4. Shift later features' `mm_position.offset` down.
      5. Shift later segments' `token_range` down.
      6. Decrement `num_prompt_tokens` and `num_computed_tokens`.
      7. Remove the segment from `session_history`.
    """
    if segment.pinned:
        raise ValueError(
            f"refusing to evict pinned segment {segment.segment_type} "
            f"at {segment.token_range}"
        )
    raw_start, raw_end = segment.token_range
    if raw_end <= raw_start:
        # Zombie: neighbour-clipping shrank it to nothing; just retire it.
        with contextlib.suppress(ValueError):
            request.session_history.remove(segment)
        return

    # Clamp to the computed frontier before freeing (see Invariants); must
    # precede `evict_token_range_for_request`, which frees blocks eagerly.
    raw_end = min(raw_end, request.num_computed_tokens)
    if raw_end <= raw_start:
        return  # nothing below the frontier is evictable

    # 1. Free KV blocks; the manager returns the block-aligned range freed.
    start, end, _ = kv_cache_manager.evict_token_range_for_request(
        request.request_id, raw_start, raw_end
    )
    evicted_len = end - start
    if evicted_len <= 0:
        return  # freed no whole block; leave it (the caller pre-filters)

    # Worker replays these ranges to slice mrope_positions, in arrival order.
    request.pending_evicted_token_ranges.append((start, end))
    # 1D-RoPE (text) sessions: survivors shift down by the evicted width but
    # keep their original RoPE positions, so the index->position offset grows.
    request.position_offset += evicted_len

    # 2. Free encoder entries + drop mm_features overlapping the freed range
    # (one per video, several per blob). Reverse + in-place: the worker shares
    # this list, and front-to-back deletion would shift unvisited indices.
    for input_id in range(len(request.mm_features) - 1, -1, -1):
        feature = request.mm_features[input_id]
        f_offset = feature.mm_position.offset
        f_end = f_offset + feature.mm_position.length
        if not (f_offset < end and f_end > start):
            continue
        encoder_cache_manager.free_encoder_input(request, input_id)
        encoder_cache_manager.evict_unreferenced(feature.identifier)
        del request.mm_features[input_id]
        # Engine receiver cache is left alone (streaming forces mm cache off).

    # 3. Compact the lockstep token arrays by the block-aligned range.
    del request._all_token_ids[start:end]
    if len(request._mrope_positions) >= end:
        del request._mrope_positions[start:end]

    # 4. Shift later mm_features' offsets down by the evicted length.
    for feature in request.mm_features:
        if feature.mm_position.offset >= end:
            feature.mm_position = replace(
                feature.mm_position,
                offset=feature.mm_position.offset - evicted_len,
            )

    # 5. Shift later segments down, trim neighbours the freed range cut into,
    # and absorb the edge orphans into a neighbour (forward normally, backward
    # if the victim was the last segment).
    zombies_to_drop: list[HistorySegment] = []
    next_after_evicted: HistorySegment | None = None
    prev_before_evicted: HistorySegment | None = None
    for other in request.session_history:
        if other is segment:
            continue
        o_start, o_end = other.token_range
        if o_start >= end and (
            next_after_evicted is None or o_start < next_after_evicted.token_range[0]
        ):
            next_after_evicted = other
        if (
            not other.pinned
            and o_end <= start
            and (
                prev_before_evicted is None
                or o_end > prev_before_evicted.token_range[1]
            )
        ):
            prev_before_evicted = other

    for other in request.session_history:
        if other is segment:
            continue
        o_start, o_end = other.token_range
        if o_start >= end:
            new_start = o_start - evicted_len
            if other is next_after_evicted:
                new_start = raw_start  # absorb the orphans against the victim
            other.token_range = (new_start, o_end - evicted_len)
        elif o_end > start:
            # Freed range cut into this neighbour; trim it (drop if empty).
            new_start = min(o_start, start)
            new_end_calc = o_end - evicted_len
            if new_end_calc <= new_start:
                zombies_to_drop.append(other)
            else:
                other.token_range = (new_start, new_end_calc)

    # Backward absorption: no following segment, so extend the preceding one
    # over the residue tail (ends at start + (raw_end - end)).
    if next_after_evicted is None and prev_before_evicted is not None:
        residue_end = start + (raw_end - end)
        p_start, p_end = prev_before_evicted.token_range
        if residue_end > p_end:
            prev_before_evicted.token_range = (p_start, residue_end)

    for z in zombies_to_drop:
        with contextlib.suppress(ValueError):
            request.session_history.remove(z)

    # 6. Decrement counters.
    request.num_prompt_tokens = max(0, request.num_prompt_tokens - evicted_len)
    request.num_computed_tokens = max(0, request.num_computed_tokens - evicted_len)
    # `all_token_ids` is a view onto `_all_token_ids`, so it tracks already.
    if request.prompt_token_ids is not None:
        del request.prompt_token_ids[start : min(end, len(request.prompt_token_ids))]

    # Trim block_hashes to match the compacted block table. NOTE: this only
    # splices the list; the surviving hash VALUES still chain over the evicted
    # content. Streaming sessions therefore set cache_salt (async_llm.py) so
    # these blocks are never shareable with other requests.
    bs = kv_cache_manager.coordinator.single_type_managers[0].block_size
    del request.block_hashes[start // bs : end // bs]

    # 7. Remove the segment from history.
    request.session_history.remove(segment)


_TEXT_SEGMENT_TYPES = ("user_text", "assistant_text")


def _block_size(kv_cache_manager: KVCacheManager) -> int:
    """Smallest block size across kv-cache groups. The eviction primitive
    rounds INWARD to whole blocks, so a segment shorter than this can't
    free any KV — phase 2 uses this to filter unevictable text segments."""
    managers = kv_cache_manager.coordinator.single_type_managers
    return min(m.block_size for m in managers)


def _frees_block(token_range: tuple[int, int], block_size: int) -> bool:
    """True if inward block-rounding of `[start, end)` frees at least one
    WHOLE block."""
    start, end = token_range
    return (end // block_size) > ((start + block_size - 1) // block_size)


def _coalesce_adjacent_text(request: Request, block_size: int) -> int:
    """Merge runs of adjacent, contiguous, same-type, unpinned text
    segments so a stack of short captions becomes one segment big enough
    for inward-rounded eviction to free a block. Returns segments removed.
    """

    hist = request.session_history
    n = len(hist)
    new_hist: list[HistorySegment] = []
    merges = 0
    i = 0
    while i < n:
        seg = hist[i]
        if seg.pinned or seg.segment_type not in _TEXT_SEGMENT_TYPES:
            new_hist.append(seg)
            i += 1
            continue
        # Collect a maximal contiguous, same-type, unpinned text run.
        run = [seg]
        j = i + 1
        while j < n:
            nxt = hist[j]
            if (
                nxt.pinned
                or nxt.segment_type != seg.segment_type
                or nxt.token_range[0] != run[-1].token_range[1]
            ):
                break
            run.append(nxt)
            j += 1
        # Group consecutive members until each group's span frees a block.
        group_start = 0
        k = 0
        while k < len(run):
            group = run[group_start : k + 1]
            span = (group[0].token_range[0], group[-1].token_range[1])
            at_run_end = k == len(run) - 1
            if _frees_block(span, block_size) or at_run_end:
                if len(group) == 1:
                    new_hist.append(group[0])
                else:
                    new_hist.append(
                        HistorySegment(
                            segment_type=seg.segment_type,
                            token_range=span,
                            mm_item_id=None,
                            pinned=False,
                            age_chunks=min(g.age_chunks for g in group),
                        )
                    )
                    merges += len(group) - 1
                group_start = k + 1
            k += 1
        i = j
    if merges:
        hist[:] = new_hist
    return merges


def _pack_oldest_run_to_free_block(
    request: Request, block_size: int
) -> HistorySegment | None:
    """Last-resort packing: glue the OLDEST run of adjacent unpinned
    segments — regardless of type — into one block-freeing blob, swap it
    into `session_history`, and return it for the caller to evict. Returns
    None if even the maximally-extended oldest run can't free a block.
    """
    hist = request.session_history
    n = len(hist)
    # Oldest unpinned segment (the run's anchor).
    i = 0
    while i < n and hist[i].pinned:
        i += 1
    if i >= n:
        return None
    # Extend a contiguous, unpinned run from i until its span frees a block.
    j = i
    while True:
        span = (hist[i].token_range[0], hist[j].token_range[1])
        if _frees_block(span, block_size):
            break
        nxt = j + 1
        if (
            nxt >= n
            or hist[nxt].pinned
            or hist[nxt].token_range[0] != hist[j].token_range[1]
            # Never pack past the computed frontier: the just-appended
            # chunk's segments have no KV yet, so `evict_segment` would
            # clamp to `num_computed_tokens` and no-op AFTER the run was
            # already destructively merged in `session_history`.
            or hist[nxt].token_range[1] > request.num_computed_tokens
        ):
            # Maximally-extended oldest run still can't free a block.
            return None
        j = nxt
    if j == i:
        return None  # a single freeable segment; the caller already handles it
    run = hist[i : j + 1]
    merged = HistorySegment(
        segment_type=run[0].segment_type,
        token_range=(run[0].token_range[0], run[-1].token_range[1]),
        mm_item_id=None,
        pinned=False,
        age_chunks=min(s.age_chunks for s in run),
    )
    hist[i : j + 1] = [merged]
    return merged


def maybe_evict_old_segments(
    request: Request,
    retention: StreamingRetentionParams,
    kv_cache_manager: KVCacheManager,
    encoder_cache_manager: EncoderCacheManager,
) -> int:
    """Run the sliding-window retention policy; return segments evicted.

    Called after a chunk is appended, so `session_history` already includes
    the new chunk's segments. Three sequential phases:

      1. Video budget. Drop oldest unpinned `video` segments until
         `len(videos) <= max_video_segments`. Vision tokens cost the most,
         so this fires first.
      2. Text budget (if `max_text_tokens` set). Drop oldest unpinned text
         segments until non-pinned text tokens <= `max_text_tokens`.
      3. Safety net (if `max_session_tokens` set). Drop the oldest unpinned
         segment while `num_prompt_tokens > max_session_tokens` — the hard
         guarantee against drift past `max_model_len`.
    """
    if retention.eviction_policy != "sliding_window":
        raise NotImplementedError(
            f"eviction policy {retention.eviction_policy!r} is not yet supported"
        )

    evicted = 0
    block_size = _block_size(kv_cache_manager)

    # Phase 1: video budget (oldest freeable video first).
    while True:
        video_segments = [
            s
            for s in request.session_history
            if s.segment_type == "video" and not s.pinned
        ]
        if len(video_segments) <= retention.max_video_segments:
            break
        # age_chunks > 0 skips the new chunk (its tokens have no KV yet).
        victim = next(
            (
                s
                for s in video_segments
                if s.age_chunks > 0 and _frees_block(s.token_range, block_size)
            ),
            None,
        )
        if victim is None:
            break
        evict_segment(
            request,
            victim,
            kv_cache_manager,
            encoder_cache_manager,
        )
        evicted += 1

    # Coalesce captions made adjacent by phase-1 eviction (see helper).
    _coalesce_adjacent_text(request, block_size)

    # Phase 2: text budget.
    if retention.max_text_tokens is not None:
        while True:
            non_pinned_text = [
                s
                for s in request.session_history
                if s.segment_type in _TEXT_SEGMENT_TYPES and not s.pinned
            ]
            text_token_count = sum(
                s.token_range[1] - s.token_range[0] for s in non_pinned_text
            )
            if text_token_count <= retention.max_text_tokens:
                break
            # `age_chunks > 0` skips the new chunk's own segment — see Phase 1.
            evictable = [
                s
                for s in non_pinned_text
                if s.age_chunks > 0 and _frees_block(s.token_range, block_size)
            ]
            if not evictable:
                break
            oldest = evictable[0]
            evict_segment(
                request,
                oldest,
                kv_cache_manager,
                encoder_cache_manager,
            )
            evicted += 1

    # Phase 3: total-size backstop.
    if retention.max_session_tokens is not None:
        while request.num_prompt_tokens > retention.max_session_tokens:
            unpinned = [s for s in request.session_history if not s.pinned]
            if not unpinned:
                break
            # `age_chunks > 0` skips the new chunk's own segment — see Phase 1.
            victim = next(
                (
                    s
                    for s in unpinned
                    if s.age_chunks > 0 and _frees_block(s.token_range, block_size)
                ),
                None,
            )
            if victim is None and _coalesce_adjacent_text(request, block_size):
                victim = next(
                    (
                        s
                        for s in request.session_history
                        if not s.pinned
                        and s.age_chunks > 0
                        and _frees_block(s.token_range, block_size)
                    ),
                    None,
                )
            if victim is None:
                victim = _pack_oldest_run_to_free_block(request, block_size)
            if victim is None:
                break
            evict_segment(
                request,
                victim,
                kv_cache_manager,
                encoder_cache_manager,
            )
            evicted += 1

    return evicted
