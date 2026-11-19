# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Rebuild complete session turns before their KV is recomputed."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from vllm.v1.streaming.eviction import remove_encoder_inputs

if TYPE_CHECKING:
    from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
    from vllm.v1.request import Request


def _frame_span(request: Request, start: int, end: int) -> tuple[int, int] | None:
    frames = [
        feature.mm_position
        for feature in request.mm_features
        if start <= feature.mm_position.offset
        and feature.mm_position.offset + feature.mm_position.length <= end
    ]
    if len(frames) != 1:
        return None
    frame = frames[0]
    return frame.offset, frame.offset + frame.length


def note_frame_turn(request: Request, start: int, end: int) -> None:
    """Allow promotion only when later single-image turns have identical text."""
    frame = _frame_span(request, start, end)
    if frame is None:
        request.streaming_frames_uniform = False
        return
    scaffold = (
        tuple(request.all_token_ids[start : frame[0]]),
        tuple(request.all_token_ids[frame[1] : end]),
    )
    if request.streaming_frame_scaffold is None:
        request.streaming_frame_scaffold = scaffold
    elif request.streaming_frame_scaffold != scaffold:
        request.streaming_frames_uniform = False


def rebuild_transcript_for_reprefill(
    request: Request,
    encoder_cache_manager: EncoderCacheManager,
    *,
    pending_encoder_input_ids: set[int] | None = None,
) -> None:
    """Keep complete turns and discard replies to frames outside the window.

    Called after a new chunk is folded into the prompt and immediately before
    freeing KV. Surviving live keys and positions cannot be used afterward.
    The first turn stays pinned; a uniform single-image turn may supply its
    replacement image while its own reply and newer turns keep their order.
    """
    history = request.session_history
    if not history:
        return
    anchor = history[0]
    assert anchor.pinned and anchor.token_range[0] == 0
    assert request.num_tokens == request.num_prompt_tokens
    anchor_end = anchor.token_range[1]
    first_frame = next(
        (s for s in history[1:] if s.complete and s.segment_type == "video"),
        None,
    )
    old_frame = _frame_span(request, 0, anchor_end)
    new_frame = _frame_span(request, *first_frame.token_range) if first_frame else None
    promote = (
        old_frame is not None
        and new_frame is not None
        and request.streaming_frames_uniform
    )
    head = [(0, anchor_end)]
    if promote:
        assert old_frame is not None and new_frame is not None
        head = [(0, old_frame[0]), new_frame, (old_frame[1], anchor_end)]

    body_start = anchor_end
    if first_frame is not None:
        body_start = first_frame.token_range[1 if promote else 0]
    body = [
        segment
        for segment in history[1:]
        if segment.complete
        and (
            segment.token_range[0] >= body_start
            or (
                not promote
                and segment.segment_type == "assistant_text"
                and segment.token_range[0] == anchor_end
            )
        )
    ]
    kept = head + [segment.token_range for segment in body]
    new_tokens = [
        token for start, end in kept for token in request.all_token_ids[start:end]
    ]

    offsets = []
    cursor = 0
    for start, end in kept:
        offsets.append(cursor)
        cursor += end - start
    features = []
    removed = []
    for input_id, feature in enumerate(request.mm_features):
        frame = feature.mm_position
        for (start, end), offset in zip(kept, offsets):
            if start <= frame.offset and frame.offset + frame.length <= end:
                features.append(
                    replace(
                        feature,
                        mm_position=replace(
                            frame, offset=offset + frame.offset - start
                        ),
                    )
                )
                break
        else:
            removed.append(input_id)
    remove_encoder_inputs(
        request, removed, encoder_cache_manager, pending_encoder_input_ids
    )
    request.mm_features = features
    head_len = sum(end - start for start, end in head)
    new_history = [replace(anchor, token_range=(0, head_len))]
    cursor = head_len
    for segment in body:
        width = segment.token_range[1] - segment.token_range[0]
        new_history.append(replace(segment, token_range=(cursor, cursor + width)))
        cursor += width
    request.session_history = new_history
    del request._all_token_ids[:]
    request._all_token_ids.extend(new_tokens)
    assert request.prompt_token_ids is not None
    del request.prompt_token_ids[:]
    request.prompt_token_ids.extend(new_tokens)
    request.num_prompt_tokens = len(new_tokens)
