# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for `evict_segment` and `maybe_evict_old_segments`
operating on a real `KVCacheManager`
plus a synthetic request with pre-populated `session_history`.

These tests exercise the full scheduler-side eviction flow without
running the actual scheduler loop. They validate:

  - `_all_token_ids` and `_mrope_positions` are sliced in lockstep.
  - Subsequent `HistorySegment.token_range` fields shift down.
  - `mm_features` offsets shift down past the evicted range; the
    evicted segment's mm_feature is dropped.
  - `num_prompt_tokens` / `num_computed_tokens` decrement correctly.
  - The KV cache primitive runs (block IDs return to the pool).
  - Pinned segments are NEVER evicted, even when over budget.
  - `max_cached_position` is preserved (gaps are intentional).
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request
from vllm.v1.streaming.eviction import (
    evict_segment,
    maybe_evict_old_segments,
)
from vllm.v1.streaming.retention import (
    HistorySegment,
    StreamingRetentionParams,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _auto_init_hash_fn():
    init_none_hash(sha256)


def _make_managers(block_size: int = 16, num_blocks: int = 32):
    kv = KVCacheManager(
        KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer"],
                    FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float32,
                    ),
                )
            ],
        ),
        max_model_len=1024,
        enable_caching=False,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
    )
    enc = EncoderCacheManager(cache_size=4096)
    return kv, enc


def _make_request(
    request_id: str,
    prompt_token_ids: list[int],
    block_size: int = 16,
    hash_fn: Callable = sha256,
) -> Request:
    sp = SamplingParams(max_tokens=1)
    sp.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sp,
        pooling_params=None,
        block_hasher=get_request_block_hasher(block_size, hash_fn),
    )


def test_evict_segment_compacts_arrays_and_shifts_subsequent():
    """Three segments. Evict the middle one. Verify both arrays compact,
    later segment's token_range shifts down, mm_features get shifted."""
    kv, enc = _make_managers(block_size=16, num_blocks=32)
    req = _make_request("r-evict-mid", [9] * 64)

    # Pretend mRoPE positions were assigned linearly across the prompt.
    req._mrope_positions = [(i, i, i) for i in range(64)]
    req.max_cached_position = 63
    req.num_prompt_tokens = 64
    req.num_computed_tokens = 64

    # Three segments (16, 32, 16 tokens each). Middle is a video.
    seg0 = HistorySegment(
        segment_type="system_prompt", token_range=(0, 16), pinned=True
    )
    seg1 = HistorySegment(
        segment_type="video",
        token_range=(16, 48),
        mm_item_id="vid-1",
    )
    seg2 = HistorySegment(segment_type="user_text", token_range=(48, 64))
    req.session_history = [seg0, seg1, seg2]
    req.mm_features = [
        MultiModalFeatureSpec(
            data=MultiModalKwargsItem.dummy(),
            modality="video",
            identifier="vid-1",
            mm_position=PlaceholderRange(offset=16, length=32),
        )
    ]

    # Allocate real KV blocks for this prompt.
    manager_blocks, _, _ = kv.get_computed_blocks(req)
    kv.allocate_slots(req, 64, 0, manager_blocks)

    evict_segment(req, seg1, kv, enc)

    # Lockstep arrays compacted.
    assert len(req._all_token_ids) == 32, len(req._all_token_ids)
    assert len(req._mrope_positions) == 32, len(req._mrope_positions)
    # Surviving mRoPE values keep their ORIGINAL coordinates (gaps).
    assert req._mrope_positions[0] == (0, 0, 0)
    assert req._mrope_positions[15] == (15, 15, 15)
    assert req._mrope_positions[16] == (48, 48, 48), (
        f"surviving segment's first position should still be (48,48,48); "
        f"got {req._mrope_positions[16]}"
    )
    # max_cached_position preserved.
    assert req.max_cached_position == 63
    # Later segment's token_range shifted down by 32.
    assert seg2.token_range == (16, 32), seg2.token_range
    # Earlier (pinned) segment unchanged.
    assert seg0.token_range == (0, 16), seg0.token_range
    # Evicted segment removed from history.
    assert seg1 not in req.session_history
    # Evicted segment's mm_feature dropped.
    assert all(f.identifier != "vid-1" for f in req.mm_features)
    # Counters decremented.
    assert req.num_prompt_tokens == 32
    assert req.num_computed_tokens == 32


def test_maybe_evict_respects_pinned_segments():
    """Pinned segments stay even when over budget. Only unpinned
    video segments count towards `max_video_segments`."""
    kv, enc = _make_managers()
    req = _make_request("r-pin", [9] * 80)
    req._mrope_positions = [(i, i, i) for i in range(80)]
    req.max_cached_position = 79
    req.num_prompt_tokens = 80
    req.num_computed_tokens = 80
    # 1 pinned system prompt + 4 unpinned video segments. age_chunks > 0
    # on every segment: the eviction phases skip age-0 (just-appended)
    # segments (finding #0), and this fixture models already-aged prior
    # chunks. Ages descend oldest-highest to preserve oldest-first order.
    req.session_history = [
        HistorySegment(
            segment_type="system_prompt",
            token_range=(0, 16),
            pinned=True,
            age_chunks=5,
        ),
        HistorySegment(
            segment_type="video",
            token_range=(16, 32),
            mm_item_id="v0",
            age_chunks=4,
        ),
        HistorySegment(
            segment_type="video",
            token_range=(32, 48),
            mm_item_id="v1",
            age_chunks=3,
        ),
        HistorySegment(
            segment_type="video",
            token_range=(48, 64),
            mm_item_id="v2",
            age_chunks=2,
        ),
        HistorySegment(
            segment_type="video",
            token_range=(64, 80),
            mm_item_id="v3",
            age_chunks=1,
        ),
    ]
    req.mm_features = [
        MultiModalFeatureSpec(
            data=MultiModalKwargsItem.dummy(),
            modality="video",
            identifier=f"v{i}",
            mm_position=PlaceholderRange(offset=16 + i * 16, length=16),
        )
        for i in range(4)
    ]
    manager_blocks, _, _ = kv.get_computed_blocks(req)
    kv.allocate_slots(req, 80, 0, manager_blocks)

    retention = StreamingRetentionParams(max_video_segments=2, max_session_tokens=4000)
    n = maybe_evict_old_segments(req, retention, kv, enc)
    assert n == 2, f"expected 2 evictions to drop 4 video segments to 2; got {n}"

    # Pinned segment present and untouched.
    pinned = [s for s in req.session_history if s.pinned]
    assert len(pinned) == 1 and pinned[0].token_range == (0, 16)
    # Remaining video segments are v2 and v3 (oldest dropped).
    surviving_videos = [s for s in req.session_history if s.segment_type == "video"]
    assert [s.mm_item_id for s in surviving_videos] == ["v2", "v3"]


def test_maybe_evict_noop_under_budget():
    """If video-segment count <= budget, eviction is a no-op."""
    kv, enc = _make_managers()
    req = _make_request("r-under", [9] * 32)
    req._mrope_positions = [(i, i, i) for i in range(32)]
    req.max_cached_position = 31
    req.num_prompt_tokens = 32
    req.num_computed_tokens = 32
    req.session_history = [
        HistorySegment(segment_type="video", token_range=(0, 16), mm_item_id="v0"),
        HistorySegment(segment_type="video", token_range=(16, 32), mm_item_id="v1"),
    ]
    n = maybe_evict_old_segments(
        req,
        StreamingRetentionParams(max_video_segments=4, max_session_tokens=4000),
        kv,
        enc,
    )
    assert n == 0
    assert len(req.session_history) == 2


def test_evict_pinned_segment_raises():
    """Direct `evict_segment` on a pinned segment is a programming error."""
    kv, enc = _make_managers()
    req = _make_request("r-pinned", [9] * 16)
    req.num_prompt_tokens = 16
    req.num_computed_tokens = 16
    pinned_seg = HistorySegment(
        segment_type="system_prompt", token_range=(0, 16), pinned=True
    )
    req.session_history = [pinned_seg]
    with pytest.raises(ValueError, match="refusing to evict pinned"):
        evict_segment(req, pinned_seg, kv, enc)


def test_evict_absorbs_orphans_into_next_segment():
    """When block-aligned eviction leaves orphan tokens at the segment's
    boundaries, the next segment's `token_range` start extends backward
    to absorb them. Without this, orphans pile up over many evictions
    and `num_prompt_tokens` drifts past `max_session_tokens` until the
    worker buffer overflows.

    Set up three segments where the middle one is NOT block-aligned:
      - seg0: pinned anchor (0, 12)
      - seg1: video (12, 44)           ← evicted; size 32
      - seg2: video (44, 76)

    With block_size=16, inward alignment of [12, 44) yields [16, 32),
    freeing 16 tokens but leaving 4 front-orphans at indices [12, 16)
    and 12 back-orphans (originally [32, 44), now at [16, 28) after del).
    seg2's range was [44, 76]; after shift it would be [28, 60]. With
    absorption, seg2 instead becomes (12, 60), so the orphans are now
    part of seg2 and will be reclaimed when seg2 is eventually evicted.
    """
    kv, enc = _make_managers(block_size=16, num_blocks=16)
    req = _make_request("r-orphan-absorb", [9] * 76)
    req._mrope_positions = [(i, i, i) for i in range(76)]
    req.max_cached_position = 75
    req.num_prompt_tokens = 76
    req.num_computed_tokens = 76

    seg0 = HistorySegment(segment_type="user_text", token_range=(0, 12), pinned=True)
    seg1 = HistorySegment(
        segment_type="video", token_range=(12, 44), mm_item_id="v-mid"
    )
    seg2 = HistorySegment(
        segment_type="video", token_range=(44, 76), mm_item_id="v-last"
    )
    req.session_history = [seg0, seg1, seg2]
    req.mm_features = [
        MultiModalFeatureSpec(
            data=MultiModalKwargsItem.dummy(),
            modality="video",
            identifier="v-mid",
            mm_position=PlaceholderRange(offset=12, length=32),
        ),
        MultiModalFeatureSpec(
            data=MultiModalKwargsItem.dummy(),
            modality="video",
            identifier="v-last",
            mm_position=PlaceholderRange(offset=44, length=32),
        ),
    ]
    manager_blocks, _, _ = kv.get_computed_blocks(req)
    kv.allocate_slots(req, 76, 0, manager_blocks)

    evict_segment(req, seg1, kv, enc)

    # Pinned anchor unchanged.
    assert seg0.token_range == (0, 12)
    # seg2 absorbed orphans by extending its start back to seg1's raw
    # start (12). Without absorption it would be (28, 60).
    assert seg2.token_range == (12, 60), seg2.token_range
    # seg1 retired.
    assert seg1 not in req.session_history
    # Arrays compacted by block-aligned 16 tokens (not seg1's raw 32).
    assert req.num_prompt_tokens == 60
    assert len(req._all_token_ids) == 60


def test_evict_token_range_inward_alignment():
    """`evict_token_range_for_request` rounds inward: a 30-token range
    starting at 0 with block_size=16 frees ONE block (covering tokens
    0..15), not two. Tokens 16..29 stay in the partial second block."""
    kv, _ = _make_managers(block_size=16, num_blocks=8)
    req = _make_request("r-align", [9] * 48)
    kv.allocate_slots(req, 48, 0, kv.get_computed_blocks(req)[0])
    free_q = kv.block_pool.free_block_queue
    free_before = free_q.num_free_blocks

    aligned_start, aligned_end, n = kv.evict_token_range_for_request(
        "r-align", token_start=0, token_end=30
    )
    # Inward: only block 0 (tokens 0..15) freed; block 1 (16..31) stays.
    assert n == 1, n
    assert (aligned_start, aligned_end) == (0, 16), (aligned_start, aligned_end)
    assert free_q.num_free_blocks == free_before + 1


# ---------------------------------------------------------------------------
# Tests for asymmetric retention: split kept_output into assistant_text
# segments + separate text-token budget. Added with the change that
# removes the absorb-into-prior-video behavior.
# ---------------------------------------------------------------------------


def _dummy_video_mm_features(n_videos: int) -> list:
    """Build n MultiModalFeatureSpec instances tagged v0..v{n-1}, each
    occupying a 32-token placeholder. Tests that exercise eviction past
    video segments need these so `evict_segment` can find the matching
    mm_feature to drop."""
    return [
        MultiModalFeatureSpec(
            data=MultiModalKwargsItem.dummy(),
            modality="video",
            identifier=f"v{i}",
            mm_position=PlaceholderRange(offset=i * 32, length=32),
        )
        for i in range(n_videos)
    ]


def _record_three_chunks(
    *, anchor_len: int, video_len: int, caption_len: int
) -> list[HistorySegment]:
    """Simulate three calls to `_record_streaming_segment` and return the
    resulting `session_history`. Bypasses the rest of the scheduler — only
    exercises the bookkeeping logic.

    Layout per call:
      call A (chunk 2 arriving): anchor synthesized, ch1_caption +
        ch2_video appended.
      call B (chunk 3 arriving): ch2_caption + ch3_video appended.
      call C (chunk 4 arriving): ch3_caption + ch4_video appended.

    Wait — caller asks for THREE chunks recorded. That maps to two
    `_record_streaming_segment` calls (chunks 2 and 3 arriving), not three.
    We do two calls and end up with: anchor, ch1_caption, ch2_video,
    ch2_caption, ch3_video. Five segments. Matches the design doc's
    "after the third chunk's recording" wording — "the third chunk
    arrived" = chunk 3's call to `_record_streaming_segment`, which is
    the second call.

    NOTE: We don't have a real scheduler instance, so we call
    `Scheduler._record_streaming_segment(None, ...)` as an unbound
    method. The implementation deliberately doesn't touch `self`.
    """
    # Local import to avoid hard scheduler dependency at module import
    # (the existing test file otherwise stays scheduler-free).
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import StreamingUpdate

    sp = SamplingParams(max_tokens=1)
    sp.update_from_generation_config({}, eos_token_id=100)
    sess = Request(
        request_id="r-record",
        prompt_token_ids=[1] * anchor_len,  # chunk 1's prompt only
        sampling_params=sp,
        pooling_params=None,
    )
    sess.mm_features = []

    # Call A (chunk 2 arrives): the scheduler has already extended
    # prompt_token_ids with kept_output (chunk 1's caption) followed by
    # chunk 2's video content. Replicate that final state before calling
    # `_record_streaming_segment`. Lockstep arrays end at
    # anchor_len + caption_len + video_len.
    sess.prompt_token_ids.extend([2] * caption_len)
    sess.prompt_token_ids.extend([3] * video_len)
    sess._all_token_ids = list(sess.prompt_token_ids)
    sess.num_prompt_tokens = len(sess.prompt_token_ids)
    update_a = StreamingUpdate(
        mm_features=[
            MultiModalFeatureSpec(
                data=MultiModalKwargsItem.dummy(),
                modality="video",
                identifier="v2",
                mm_position=PlaceholderRange(
                    offset=anchor_len + caption_len, length=video_len
                ),
            )
        ],
        prompt_token_ids=[3] * video_len,
        max_tokens=16,
        arrival_time=0.0,
        sampling_params=sp,
    )
    sess.mm_features.extend(update_a.mm_features)
    Scheduler._record_streaming_segment(
        None,
        sess,
        update_a,
        prior_response_start=anchor_len,
        segment_start_idx=anchor_len + caption_len,
    )

    # Call B (chunk 3 arrives): prior_response_start = end of chunk 2's
    # video segment (= anchor + caption + video), kept_output is
    # chunk 2's caption, then chunk 3's video tokens.
    chunk2_end = anchor_len + caption_len + video_len
    sess.prompt_token_ids.extend([4] * caption_len)
    sess._all_token_ids.extend([4] * caption_len)
    sess.prompt_token_ids.extend([5] * video_len)
    sess._all_token_ids.extend([5] * video_len)
    sess.num_prompt_tokens = len(sess.prompt_token_ids)
    update_b = StreamingUpdate(
        mm_features=[
            MultiModalFeatureSpec(
                data=MultiModalKwargsItem.dummy(),
                modality="video",
                identifier="v3",
                mm_position=PlaceholderRange(
                    offset=chunk2_end + caption_len, length=video_len
                ),
            )
        ],
        prompt_token_ids=[5] * video_len,
        max_tokens=16,
        arrival_time=0.0,
        sampling_params=sp,
    )
    sess.mm_features.extend(update_b.mm_features)
    Scheduler._record_streaming_segment(
        None,
        sess,
        update_b,
        prior_response_start=chunk2_end,
        segment_start_idx=chunk2_end + caption_len,
    )
    return sess.session_history


def test_record_streaming_segment_splits_kept_output_into_assistant_text():
    """After two `_record_streaming_segment` calls (chunks 2 and 3
    arriving on top of chunk 1's anchor), `session_history` should be:

        [pinned anchor, ch1_caption, ch2_video, ch2_caption, ch3_video]

    Validates the split behavior — captions are NOT absorbed into the
    prior video segment. Anchor is pinned; all others unpinned.
    """
    history = _record_three_chunks(anchor_len=20, video_len=32, caption_len=12)

    assert len(history) == 5, [(s.segment_type, s.token_range) for s in history]
    types = [s.segment_type for s in history]
    assert types == [
        "user_text",  # anchor (chunk 1 prompt)
        "assistant_text",  # chunk 1 caption
        "video",  # chunk 2 video
        "assistant_text",  # chunk 2 caption
        "video",  # chunk 3 video
    ], types

    # Only the anchor is pinned.
    pinned_flags = [s.pinned for s in history]
    assert pinned_flags == [True, False, False, False, False], pinned_flags

    # Token ranges are contiguous and non-overlapping.
    ranges = [s.token_range for s in history]
    assert ranges == [
        (0, 20),  # anchor: chunk 1 prompt
        (20, 32),  # chunk 1 caption (12 tokens)
        (32, 64),  # chunk 2 video (32 tokens)
        (64, 76),  # chunk 2 caption (12 tokens)
        (76, 108),  # chunk 3 video (32 tokens)
    ], ranges

    # Video segments carry mm_item_id; caption + anchor don't.
    assert history[0].mm_item_id is None  # anchor
    assert history[1].mm_item_id is None  # chunk 1 caption
    assert history[2].mm_item_id == "v2"
    assert history[3].mm_item_id is None  # chunk 2 caption
    assert history[4].mm_item_id == "v3"


def _make_request_with_history(
    request_id: str,
    block_size: int,
    segments_spec: list[tuple[str, int, str | None, bool]],
) -> tuple[Request, KVCacheManager, EncoderCacheManager]:
    """Build a request with a synthetic, contiguous session_history.

    `segments_spec` is a list of (segment_type, length, mm_item_id, pinned)
    tuples. The function lays them out back-to-back starting at index 0,
    populates `_all_token_ids` / `_mrope_positions` / `num_prompt_tokens`
    accordingly, allocates real KV blocks for the full range, and returns
    (req, kv_manager, enc_manager) ready for `maybe_evict_old_segments`.

    Video segments get a corresponding `MultiModalFeatureSpec` entry on
    the request so `evict_segment` can drop them.
    """
    total = sum(length for (_, length, _, _) in segments_spec)
    # Allocate enough blocks: ceil(total / block_size) plus generous
    # headroom. KVCacheManager has internal reservations (null block,
    # max_model_len-sized worker buffers) that fail allocate_slots
    # silently if num_blocks is too tight. Use 4× the segment-count
    # estimate so any reasonable test fixture fits.
    n_blocks = max(64, ((total + block_size - 1) // block_size) * 4)
    kv, enc = _make_managers(block_size=block_size, num_blocks=n_blocks)
    req = _make_request(request_id, [9] * total, block_size=block_size)
    req._mrope_positions = [(i, i, i) for i in range(total)]
    req.max_cached_position = total - 1
    req.num_prompt_tokens = total
    req.num_computed_tokens = total

    history: list[HistorySegment] = []
    mm_features: list = []
    cursor = 0
    # Mirror production aging (finding #0): the three eviction phases only
    # pick victims with `age_chunks > 0`, because the just-appended,
    # not-yet-computed segment (age 0 per `_record_streaming_segment`) has
    # no KV blocks yet and must never be evicted. This synthetic history
    # models a session whose ENTIRE content is already-computed, aged prior
    # chunks being trimmed — there is no fresh age-0 append in the fixture
    # — so every segment is given a strictly-positive age. Ages descend
    # oldest-highest so list order still coincides with chronology
    # (oldest-first eviction). Tests that specifically need the age-0 guard
    # build their fixture inline.
    n_segments = len(segments_spec)
    for idx, (seg_type, length, mm_id, pinned) in enumerate(segments_spec):
        # Oldest segment (idx 0) gets the highest age; the newest still
        # gets a positive age (>= 1) so it remains evictable.
        age = n_segments - idx
        history.append(
            HistorySegment(
                segment_type=seg_type,
                token_range=(cursor, cursor + length),
                mm_item_id=mm_id,
                pinned=pinned,
                age_chunks=age,
            )
        )
        if seg_type == "video" and mm_id is not None:
            mm_features.append(
                MultiModalFeatureSpec(
                    data=MultiModalKwargsItem.dummy(),
                    modality="video",
                    identifier=mm_id,
                    mm_position=PlaceholderRange(offset=cursor, length=length),
                )
            )
        cursor += length
    req.session_history = history
    req.mm_features = mm_features

    manager_blocks, _, _ = kv.get_computed_blocks(req)
    kv.allocate_slots(req, total, 0, manager_blocks)
    return req, kv, enc


def test_text_budget_evicts_oldest_assistant_text_first():
    """Phase 2 brings the total non-pinned text tokens at or below
    `max_text_tokens`, evicting oldest assistant_text segments first.
    Video segments stay untouched because phase 1 doesn't trigger.

    Layout (all sizes ≥ block_size=16 so all are evictable by phase 2):
      anchor(32, pinned) | text0(32) | video0(32, v0) | text1(32) |
      video1(32, v1) | text2(32) | video2(32, v2) | text3(32) |
      video3(32, v3)

    Total = 288 tokens, of which 4 text × 32 = 128 are non-pinned text.
    Set max_text_tokens=64, so phase 2 should evict 2 text segments
    (oldest two), leaving 64 text tokens. max_video_segments=10 keeps
    phase 1 quiet; max_session_tokens left at None so phase 3 is quiet.
    """
    req, kv, enc = _make_request_with_history(
        "r-text-budget",
        block_size=16,
        segments_spec=[
            ("user_text", 32, None, True),  # pinned anchor
            ("assistant_text", 32, None, False),  # text0
            ("video", 32, "v0", False),
            ("assistant_text", 32, None, False),  # text1
            ("video", 32, "v1", False),
            ("assistant_text", 32, None, False),  # text2
            ("video", 32, "v2", False),
            ("assistant_text", 32, None, False),  # text3
            ("video", 32, "v3", False),
        ],
    )

    retention = StreamingRetentionParams(
        max_video_segments=10, max_text_tokens=64, max_session_tokens=4000
    )
    n = maybe_evict_old_segments(req, retention, kv, enc)
    # Two text segments evicted to bring 128 → 64.
    assert n == 2, n

    # Remaining text segments are text2 and text3 (youngest two).
    text_segs = [
        s
        for s in req.session_history
        if s.segment_type in ("user_text", "assistant_text") and not s.pinned
    ]
    text_tokens = sum(s.token_range[1] - s.token_range[0] for s in text_segs)
    assert text_tokens <= 64, text_tokens
    assert len(text_segs) == 2, [s.token_range for s in text_segs]

    # All 4 video segments still present.
    video_ids = [s.mm_item_id for s in req.session_history if s.segment_type == "video"]
    assert video_ids == ["v0", "v1", "v2", "v3"], video_ids


def test_text_budget_respects_pinned_segments():
    """Phase 2 skips pinned text segments even if dropping them would
    bring the budget under target. Falls through to the next oldest
    unpinned text segment.

    Layout: anchor(32, pinned) | text0(32, pinned) | text1(32) | text2(32).
    Non-pinned text tokens = 64. Set max_text_tokens=16: phase 2 needs
    to free at least 48. text0 is pinned → skip. text1 is oldest
    unpinned → evict (32 tokens freed). text2 remains; non-pinned text
    tokens now = 32, still over 16. Try again: text2 is now oldest
    unpinned → evict. Non-pinned text = 0, budget met.
    """
    req, kv, enc = _make_request_with_history(
        "r-text-pinned",
        block_size=16,
        segments_spec=[
            ("user_text", 32, None, True),  # anchor (pinned)
            ("assistant_text", 32, None, True),  # pinned mid-session text
            ("assistant_text", 32, None, False),  # text1
            ("assistant_text", 32, None, False),  # text2
        ],
    )
    retention = StreamingRetentionParams(
        max_video_segments=10, max_text_tokens=16, max_session_tokens=4000
    )
    n = maybe_evict_old_segments(req, retention, kv, enc)
    assert n == 2, n
    # Both pinned segments still present.
    pinned = [s for s in req.session_history if s.pinned]
    assert len(pinned) == 2
    # Both unpinned text segments gone.
    non_pinned_text = [
        s
        for s in req.session_history
        if s.segment_type in ("user_text", "assistant_text") and not s.pinned
    ]
    assert non_pinned_text == []


def test_independent_video_and_text_budgets():
    """Phase 1 and phase 2 each hit their own budget independently.

    Layout: anchor(32, pinned) | v0..v4 (each 32) | text0..text4 (each 32).
    Set max_video_segments=2 and max_text_tokens=64. Phase 1 should evict
    3 videos (5→2). Phase 2 should evict 3 texts (160→64). Final: anchor +
    2 videos + 2 texts.
    """
    spec: list[tuple[str, int, str | None, bool]] = [
        ("user_text", 32, None, True),
    ]
    for i in range(5):
        spec.append(("video", 32, f"v{i}", False))
    for i in range(5):
        spec.append(("assistant_text", 32, None, False))
    req, kv, enc = _make_request_with_history(
        "r-indep", block_size=16, segments_spec=spec
    )
    retention = StreamingRetentionParams(
        max_video_segments=2, max_text_tokens=64, max_session_tokens=4000
    )
    n = maybe_evict_old_segments(req, retention, kv, enc)
    assert n == 6, n  # 3 video + 3 text

    surviving_videos = [s for s in req.session_history if s.segment_type == "video"]
    assert [s.mm_item_id for s in surviving_videos] == ["v3", "v4"]

    non_pinned_text = [
        s
        for s in req.session_history
        if s.segment_type in ("user_text", "assistant_text") and not s.pinned
    ]
    text_tokens = sum(s.token_range[1] - s.token_range[0] for s in non_pinned_text)
    assert text_tokens == 64
    assert len(non_pinned_text) == 2


def test_video_eviction_preserves_associated_caption_segment():
    """The point of asymmetric retention: dropping a video segment
    doesn't drop its caption. Layout simulates the post-split
    streaming flow: anchor | v0_caption | v0 | v1_caption | v1 |
    v2_caption | v2 | v3_caption | v3. Set max_video_segments=2 and
    leave text budget unset. Phase 1 evicts v0 and v1; their caption
    segments survive untouched (modulo orphan absorption shifting
    token_ranges). Phase 2 + 3 don't fire.
    """
    spec: list[tuple[str, int, str | None, bool]] = [
        ("user_text", 32, None, True),  # anchor
    ]
    for i in range(4):
        spec.append(("assistant_text", 32, None, False))  # captionN
        spec.append(("video", 32, f"v{i}", False))
    req, kv, enc = _make_request_with_history(
        "r-video-evict-keeps-text", block_size=16, segments_spec=spec
    )

    retention = StreamingRetentionParams(max_video_segments=2, max_session_tokens=4000)
    n = maybe_evict_old_segments(req, retention, kv, enc)
    assert n == 2, n

    # Only 2 videos remain (v2, v3).
    videos = [s for s in req.session_history if s.segment_type == "video"]
    assert [s.mm_item_id for s in videos] == ["v2", "v3"], [
        s.mm_item_id for s in videos
    ]
    # All 4 captions still present — their tokens stayed in
    # _all_token_ids / _mrope_positions (eviction didn't touch them).
    captions = [s for s in req.session_history if s.segment_type == "assistant_text"]
    assert len(captions) == 4, [s.token_range for s in captions]

    # Encoder cache entries for the evicted videos are gone; for the
    # surviving ones they're still referenced.
    remaining_mm_ids = {f.identifier for f in req.mm_features}
    assert remaining_mm_ids == {"v2", "v3"}


def test_sports_preset_steady_state_under_budget():
    """Document the running-cleanly-under-budget case. Sports preset:
    max_video_segments=30, max_text_tokens=4000, max_session_tokens=7000.
    Simulate 10 chunks: anchor(32) + 10 video(64 each) + 10 caption(20
    each). 10 videos < 30 budget. 10 captions × 20 = 200 tokens < 4000.
    Total = 32 + 640 + 200 = 872 < 7000. No phase should fire.
    """
    spec: list[tuple[str, int, str | None, bool]] = [
        ("user_text", 32, None, True),  # anchor
    ]
    for i in range(10):
        spec.append(("video", 64, f"v{i}", False))
        spec.append(("assistant_text", 20, None, False))
    req, kv, enc = _make_request_with_history(
        "r-steady", block_size=16, segments_spec=spec
    )

    history_before = list(req.session_history)
    tokens_before = req.num_prompt_tokens

    retention = StreamingRetentionParams(
        max_video_segments=30,
        max_text_tokens=4000,
        max_session_tokens=7000,
    )
    n = maybe_evict_old_segments(req, retention, kv, enc)
    assert n == 0, n
    assert req.session_history == history_before
    assert req.num_prompt_tokens == tokens_before


# ---------------------------------------------------------------------------
# Unpadded strict eviction (the only mode)
#
# Segments are block-UNALIGNED (nothing is padded). Eviction gates on the
# EXACT "frees a whole block" predicate, falls through past un-freeable
# victims, and coalesces adjacent short captions — so no segment ever hits
# PATH C (silent orphan, no decrement) and short captions can't stack
# unbounded. These tests build unpadded synthetic sessions directly.
# ---------------------------------------------------------------------------


def _owned_tokens(req: Request) -> int:
    """Sum of segment lengths. Equals num_prompt_tokens when there are no
    untracked orphans; a gap means a last-segment eviction left bounded
    forward-absorption residue (<= 2*(block_size-1)). A LARGE gap would
    mean PATH C dropped a segment — the leak the design prevents."""
    return sum(e - s for s, e in (seg.token_range for seg in req.session_history))


def _retention(**kw) -> StreamingRetentionParams:
    """Build a StreamingRetentionParams for the strict-eviction tests.

    Several strict tests deliberately use tiny/illegal budgets (e.g.
    max_session_tokens=1, max_text_tokens=0) to force the phase-3 / coalesce
    / pack code paths. `StreamingRetentionParams.__post_init__` now rejects
    those (max_session_tokens must be >= the minimum; max_text_tokens must be
    > 0 when set). To keep that coverage we build a VALID config and then
    degrade `max_session_tokens` / `max_text_tokens` via direct attribute
    assignment after construction (the dataclass is mutable, so this bypasses
    __post_init__).
    """
    kw.setdefault("max_video_segments", 30)
    requested_mst = kw.pop("max_session_tokens", 4000)
    requested_mtt = kw.pop("max_text_tokens", None)
    # Construct with a valid (large) session budget, then override.
    retention = StreamingRetentionParams(max_session_tokens=4000, **kw)
    retention.max_session_tokens = requested_mst
    retention.max_text_tokens = requested_mtt
    return retention


def test_strict_guard_leaves_straddling_block_sized_segment():
    """A text segment of length == block_size at a MISALIGNED start frees
    ZERO whole blocks (inward rounding: `[4,12)` at bs=8 → ceil(4/8)=1,
    floor(12/8)=1). The exact `_frees_block` guard must DECLINE to evict
    it — leave it in place, no PATH C, no orphan. A length>=block_size
    heuristic would admit it and PATH-C-drop it, leaking. This is the
    central guard-threshold fix."""
    req, kv, enc = _make_request_with_history(
        "r-strict-straddle",
        block_size=8,
        segments_spec=[
            ("user_text", 4, None, True),  # anchor ends at 4 (unaligned)
            ("assistant_text", 8, None, False),  # [4,12) straddles blocks
        ],
    )
    n = maybe_evict_old_segments(req, _retention(max_session_tokens=1), kv, enc)
    assert n == 0, n
    assert len(req.session_history) == 2, "must NOT drop the segment"
    assert req.num_prompt_tokens == _owned_tokens(req), "must not orphan"


def test_strict_coalesce_evicts_stacked_short_captions():
    """Eight 2-token captions stacked after a block-aligned sink. Strict
    mode coalesces the contiguous run into block-freeing segments so phase
    2 can evict them. Without coalescing each is sub-block, skipped
    forever, and the budget can never be met."""
    spec: list[tuple[str, int, str | None, bool]] = [
        ("user_text", 8, None, True)  # block-aligned sink (bs=8)
    ]
    spec += [("assistant_text", 2, None, False) for _ in range(8)]  # [8,24)
    req, kv, enc = _make_request_with_history(
        "r-strict-coalesce", block_size=8, segments_spec=spec
    )
    before = req.num_prompt_tokens  # 24
    n = maybe_evict_old_segments(req, _retention(max_text_tokens=0), kv, enc)
    assert n >= 1, n
    assert req.num_prompt_tokens < before
    assert req.num_prompt_tokens - _owned_tokens(req) <= 2 * (8 - 1)
    text_left = sum(
        e - s
        for s, e in (
            seg.token_range
            for seg in req.session_history
            if seg.segment_type in ("user_text", "assistant_text") and not seg.pinned
        )
    )
    assert text_left <= 2 * (8 - 1), text_left


def test_strict_phase3_fallthrough_skips_front_subblock():
    """Phase 3 must SKIP a front sub-block segment that can't free a block
    (and can't be coalesced because a video separates it from other text)
    and fall through to evict the freeable video — not PATH-C-drop the
    front segment or stall. Layout (bs=8): anchor[0,8) pinned |
    small[8,10) | video[10,42) | tail[42,44)."""
    req, kv, enc = _make_request_with_history(
        "r-strict-fallthrough",
        block_size=8,
        segments_spec=[
            ("user_text", 8, None, True),
            ("assistant_text", 2, None, False),  # sub-block front caption
            ("video", 32, "v0", False),  # freeable, not coalescable w/ text
            ("assistant_text", 2, None, False),  # tail
        ],
    )
    before = req.num_prompt_tokens  # 44
    n = maybe_evict_old_segments(
        req, _retention(max_video_segments=30, max_session_tokens=20), kv, enc
    )
    assert n >= 1
    assert req.num_prompt_tokens < before
    assert req.num_prompt_tokens - _owned_tokens(req) <= 2 * (8 - 1)
    # The freeable video was evicted; the front sub-block caption survived
    # (skipped by the fall-through, NOT PATH-C-dropped).
    assert "v0" not in {f.identifier for f in req.mm_features}
    assert any(
        (s.token_range[1] - s.token_range[0]) == 2
        and s.segment_type == "assistant_text"
        for s in req.session_history
    ), "front sub-block caption must survive"


def test_strict_phase1_skips_subblock_video():
    """Phase 1 (video budget) evicts the oldest video that can free a
    block, skipping a sub-block video that would PATH-C. The sub-block
    video survives (bounded); the loop terminates."""
    req, kv, enc = _make_request_with_history(
        "r-strict-subvideo",
        block_size=8,
        segments_spec=[
            ("user_text", 8, None, True),
            ("video", 3, "v-small", False),  # sub-block video [8,11)
            ("video", 32, "v-big", False),  # [11,43) freeable
        ],
    )
    maybe_evict_old_segments(req, _retention(max_video_segments=1), kv, enc)
    ids = {f.identifier for f in req.mm_features}
    assert "v-big" not in ids, "freeable video should be evicted"
    assert "v-small" in ids, "sub-block video should survive (not PATH-C-dropped)"
    videos = [s for s in req.session_history if s.segment_type == "video"]
    assert len(videos) == 1
    assert req.num_prompt_tokens - _owned_tokens(req) <= 2 * (8 - 1)


def test_strict_does_not_coalesce_across_text_subtypes():
    """user_text and assistant_text both count as text but carry different
    retention intent; coalesce must NOT merge across the subtype boundary,
    and neither tiny segment is PATH-C-dropped."""
    req, kv, enc = _make_request_with_history(
        "r-strict-no-cross",
        block_size=8,
        segments_spec=[
            ("user_text", 8, None, True),
            ("user_text", 2, None, False),  # [8,10)
            ("assistant_text", 2, None, False),  # [10,12)
        ],
    )
    n = maybe_evict_old_segments(req, _retention(max_session_tokens=8), kv, enc)
    assert n == 0, n
    types = [s.segment_type for s in req.session_history]
    assert "user_text" in types and "assistant_text" in types
    assert req.num_prompt_tokens == _owned_tokens(req), "no leak / no drop"


def test_strict_evs_packs_alternating_tiny_segments():
    """EVS regime: frames pruned so small that captions AND videos are all
    sub-block and ALTERNATE (cap, vid, cap, vid). No single segment frees
    a block and text-coalesce can't merge across the videos — so Phase 3
    packs the oldest run ACROSS types into a block-freeing blob, evicting
    whole captions+frames together and releasing the frames' encoder
    entries. Without this the session would drift unbounded (the original
    crash, now in the EVS regime)."""
    spec: list[tuple[str, int, str | None, bool]] = [
        ("user_text", 8, None, True)  # block-aligned sink (bs=8)
    ]
    for k in range(3):
        spec.append(("assistant_text", 2, None, False))  # tiny caption
        spec.append(("video", 3, f"v{k}", False))  # EVS-pruned tiny frame
    spec.append(("assistant_text", 2, None, False))
    req, kv, enc = _make_request_with_history(
        "r-strict-evs", block_size=8, segments_spec=spec
    )
    before = req.num_prompt_tokens  # 25; every non-sink segment sub-block
    n = maybe_evict_old_segments(req, _retention(max_session_tokens=8), kv, enc)

    # Packing fired and made progress toward budget (no stall).
    assert n >= 1, n
    assert req.num_prompt_tokens < before
    # No unbounded orphan leak — only bounded last-blob edge residue.
    assert req.num_prompt_tokens - _owned_tokens(req) <= 2 * (8 - 1)
    # Drained to ~budget (bounded overshoot only), proving it didn't stall
    # on the all-sub-block alternating layout.
    assert req.num_prompt_tokens <= 8 + 2 * (8 - 1)
    # Cross-type packing released evicted frames' encoder entries /
    # mm_features (range-based free over the blob): at least one frame
    # gone, so the multi-frame blob's release path ran.
    assert len(req.mm_features) < 3, [f.identifier for f in req.mm_features]
