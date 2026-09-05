# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Eviction edge-case tests from the streaming deep review (C15, C16, C26,
C31).

Scenarios that the main `test_eviction.py` suite never constructs:

  - C26 (4): `evict_segment`'s computed-frontier clamp on a PARTIALLY
    computed session (every existing fixture is fully computed).
  - C31 / C15 (2): `_pack_oldest_run_to_free_block` must stop at the
    computed frontier instead of destructively merging the just-appended,
    uncomputed segment into a blob whose eviction then no-ops.
  - C15 (3): evicting the LAST segment when the only predecessor is pinned
    leaves bounded, non-compounding untracked residue.
  - C15 (1): an mm placeholder straddling the block-aligned freed range is
    dropped whole (feature + encoder entry) while its edge tokens survive
    via absorption.
  - C15 (4): duplicate mm identifiers within one session collapse to a
    single encoder-cache reference (set semantics) — the eviction of one
    occurrence physically frees the embedding out from under the survivor.
    The test pins today's semantics (incl. no double-free) and documents
    the residual hazard.
  - C16: `_neutralize_orphan_mm_placeholders` must zero orphan IMAGE
    placeholders too (the streaming REST endpoint pushes image frames), not
    only video ones, and must not touch live/mm-backed or non-streaming
    rows.

Fixtures mirror tests/v1/streaming/test_eviction.py (real Request +
KVCacheManager + EncoderCacheManager).
"""

from __future__ import annotations

from types import SimpleNamespace

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
    _pack_oldest_run_to_free_block,
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


def _make_managers(block_size: int = 16, num_blocks: int = 64):
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


def _make_request_with_history(
    request_id: str,
    block_size: int,
    segments_spec: list[tuple[str, int, str | None, bool, int]],
    num_computed_tokens: int | None = None,
):
    """Like test_eviction.py's helper, but with an EXPLICIT per-segment
    `age_chunks` (last tuple field) and an overridable computed frontier —
    the two knobs the edge cases here need. Duplicate mm_item_ids are
    allowed (C15 (4))."""
    total = sum(length for (_, length, _, _, _) in segments_spec)
    kv, enc = _make_managers(block_size=block_size, num_blocks=64)
    sp = SamplingParams(max_tokens=1)
    sp.update_from_generation_config({}, eos_token_id=100)
    req = Request(
        request_id=request_id,
        prompt_token_ids=[9] * total,
        sampling_params=sp,
        pooling_params=None,
        block_hasher=get_request_block_hasher(block_size, sha256),
    )
    req._mrope_positions = [(i, i, i) for i in range(total)]
    req.max_cached_position = total - 1
    req.num_prompt_tokens = total
    req.num_computed_tokens = (
        total if num_computed_tokens is None else num_computed_tokens
    )

    history: list[HistorySegment] = []
    mm_features: list[MultiModalFeatureSpec] = []
    cursor = 0
    for seg_type, length, mm_id, pinned, age in segments_spec:
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


def _degraded_retention(**kw) -> StreamingRetentionParams:
    """Valid construction, then degrade below the constructor floors via
    attribute assignment (same pattern as test_eviction.py) so tiny budgets
    can force phase 3 / packing."""
    requested_mst = kw.pop("max_session_tokens", 4000)
    retention = StreamingRetentionParams(max_session_tokens=4000, **kw)
    retention.max_session_tokens = requested_mst
    return retention


def _assert_session_invariants(
    req: Request, block_size: int, check_features: bool = True
) -> int:
    """Shared post-eviction invariant helper (C15 fix (a)). Returns the
    untracked-residue gap so callers can pin its exact value."""
    # Lockstep arrays.
    assert len(req._all_token_ids) == req.num_prompt_tokens
    if req._mrope_positions:
        assert len(req._mrope_positions) == len(req._all_token_ids)
    # Segments sorted, disjoint, in-range.
    segs = sorted(req.session_history, key=lambda s: s.token_range[0])
    prev_end = 0
    owned = 0
    for seg in segs:
        start, end = seg.token_range
        assert 0 <= start <= end <= req.num_prompt_tokens, seg.token_range
        assert start >= prev_end, f"overlapping segments at {seg.token_range}"
        prev_end = end
        owned += end - start
    gap = req.num_prompt_tokens - owned
    assert 0 <= gap <= 2 * (block_size - 1), (
        f"untracked residue {gap} exceeds the bounded-orphan guarantee"
    )
    if check_features:
        ranges = [s.token_range for s in segs]
        for feature in req.mm_features:
            f_start = feature.mm_position.offset
            f_end = f_start + feature.mm_position.length
            assert any(s <= f_start and f_end <= e for s, e in ranges), (
                f"feature {feature.identifier} placeholder [{f_start},"
                f"{f_end}) not fully inside any surviving segment"
            )
    return gap


# ---------------------------------------------------------------------------
# C26 (4): computed-frontier clamp on a partially-computed session
# ---------------------------------------------------------------------------


def test_evict_segment_clamps_to_computed_frontier():
    """Guards C26 gap (4): with the computed frontier mid-victim,
    `evict_segment` frees only whole blocks BELOW the frontier and keeps
    arrays / segments / counters in lockstep."""
    req, kv, enc = _make_request_with_history(
        "r-frontier-clamp",
        block_size=16,
        segments_spec=[
            ("user_text", 16, None, True, 3),
            ("video", 32, "v0", False, 1),  # victim [16, 48)
            ("user_text", 16, None, False, 0),
        ],
        num_computed_tokens=40,  # frontier mid-victim
    )
    victim = req.session_history[1]
    tail = req.session_history[2]

    evict_segment(req, victim, kv, enc)

    # Clamped to min(48, 40) = 40, then inward-aligned to [16, 32).
    assert req.pending_evicted_token_ranges == [(16, 32)]
    assert req.num_prompt_tokens == 48
    assert req.num_computed_tokens == 24  # 40 - 16
    assert victim not in req.session_history
    # Tail absorbed the orphans back to the victim's raw start (16).
    assert tail.token_range == (16, 48), tail.token_range
    # Victim's feature dropped even though only part of its range was freed.
    assert all(f.identifier != "v0" for f in req.mm_features)
    _assert_session_invariants(req, block_size=16)


def test_evict_segment_noops_when_victim_above_frontier():
    """Guards C26 gap (4): a victim entirely (or all but a sub-block part)
    beyond `num_computed_tokens` must be left completely untouched — no
    pending ranges, no history mutation, no counter drift."""
    for frontier in (16, 20):  # fully above / sub-block below
        req, kv, enc = _make_request_with_history(
            f"r-frontier-noop-{frontier}",
            block_size=16,
            segments_spec=[
                ("user_text", 16, None, True, 1),
                ("video", 32, "v0", False, 0),  # [16, 48), uncomputed
            ],
            num_computed_tokens=frontier,
        )
        victim = req.session_history[1]
        history_before = list(req.session_history)
        tokens_before = list(req._all_token_ids)

        evict_segment(req, victim, kv, enc)

        assert req.pending_evicted_token_ranges == []
        assert req.session_history == history_before
        assert req._all_token_ids == tokens_before
        assert req.num_prompt_tokens == 48
        assert req.num_computed_tokens == frontier
        assert [f.identifier for f in req.mm_features] == ["v0"]


# ---------------------------------------------------------------------------
# C31 / C15 (2): packing must stop at the computed frontier
# ---------------------------------------------------------------------------


def _pack_fixture(num_computed_tokens: int):
    """anchor(8, pinned) | text[8,12) computed | video[12,17) age-0.
    With bs=8, the text alone frees no block; only extending over the
    age-0 video would — exactly the run the frontier guard must refuse
    while the video is uncomputed."""
    return _make_request_with_history(
        "r-pack-frontier",
        block_size=8,
        segments_spec=[
            ("user_text", 8, None, True, 3),
            ("assistant_text", 4, None, False, 1),
            ("video", 5, "v-new", False, 0),
        ],
        num_computed_tokens=num_computed_tokens,
    )


def test_pack_refuses_to_cross_computed_frontier():
    """Guards C31: `_pack_oldest_run_to_free_block` must NOT extend the run
    over the just-appended uncomputed segment. Before the fix it merged
    text+video into one blob (destroying types/mm_item_id in
    session_history) and the subsequent eviction no-op'd on the clamp."""
    req, kv, enc = _pack_fixture(num_computed_tokens=12)

    assert _pack_oldest_run_to_free_block(req, 8) is None
    # History untouched: still three distinct segments.
    assert [s.segment_type for s in req.session_history] == [
        "user_text",
        "assistant_text",
        "video",
    ]
    assert req.session_history[1].token_range == (8, 12)
    assert req.session_history[2].mm_item_id == "v-new"


def test_phase3_with_uncomputed_tail_stalls_cleanly_without_blob():
    """Guards C31 via the full phase-3 path: over budget but with nothing
    evictable below the frontier, `maybe_evict_old_segments` must return 0
    and leave `session_history` unmangled (no merged blob, no orphaned
    tokens) instead of packing across the frontier."""
    req, kv, enc = _pack_fixture(num_computed_tokens=12)
    history_before = [
        (s.segment_type, s.token_range, s.mm_item_id) for s in req.session_history
    ]

    n = maybe_evict_old_segments(
        req, _degraded_retention(max_session_tokens=8), kv, enc
    )

    assert n == 0
    assert [
        (s.segment_type, s.token_range, s.mm_item_id) for s in req.session_history
    ] == history_before
    assert req.num_prompt_tokens == 17
    assert req.pending_evicted_token_ranges == []
    _assert_session_invariants(req, block_size=8)


def test_pack_proceeds_once_run_is_computed():
    """Positive control for the C31 guard: the identical layout with the
    frontier past the video packs and evicts normally (the frontier — not
    age — is the invariant; a computed age-0 segment is safe to pack)."""
    req, kv, enc = _pack_fixture(num_computed_tokens=17)

    n = maybe_evict_old_segments(
        req, _degraded_retention(max_session_tokens=8), kv, enc
    )

    assert n == 1, n
    # The packed blob [8, 17) evicted its aligned span [8, 16).
    assert req.pending_evicted_token_ranges == [(8, 16)]
    assert req.num_prompt_tokens == 9
    assert all(f.identifier != "v-new" for f in req.mm_features)
    _assert_session_invariants(req, block_size=8)


# ---------------------------------------------------------------------------
# C15 (3): last-segment eviction with only a pinned predecessor
# ---------------------------------------------------------------------------


def test_last_segment_eviction_with_pinned_predecessor_bounded_residue():
    """Guards C15 (3): with no following segment and only a PINNED
    predecessor, backward absorption cannot run, so the sub-block tail
    residue becomes untracked. It must stay bounded (< block_size) and must
    NOT compound across later eviction passes."""
    req, kv, enc = _make_request_with_history(
        "r-pinned-prev-residue",
        block_size=8,
        segments_spec=[
            ("user_text", 8, None, True, 2),
            ("video", 34, "v-last", False, 1),  # [8, 42), unaligned end
        ],
    )
    victim = req.session_history[1]

    evict_segment(req, victim, kv, enc)

    # Inward alignment freed [8, 40); tokens [40, 42) survive at [8, 10)
    # owned by NO segment (anchor is pinned, nothing follows).
    assert req.pending_evicted_token_ranges == [(8, 40)]
    assert len(req.session_history) == 1
    assert req.session_history[0].pinned
    assert req.num_prompt_tokens == 10
    gap = _assert_session_invariants(req, block_size=8)
    assert gap == 2
    assert gap < 8, "residue must stay below one block"

    # Later eviction passes find no unpinned victim; the gap must not grow.
    for _ in range(3):
        n = maybe_evict_old_segments(
            req, _degraded_retention(max_session_tokens=8), kv, enc
        )
        assert n == 0
        assert _assert_session_invariants(req, block_size=8) == 2


# ---------------------------------------------------------------------------
# C15 (1): placeholder straddling the aligned freed boundary
# ---------------------------------------------------------------------------


def test_straddling_placeholder_dropped_whole_and_orphans_absorbed():
    """Guards C15 (1): a feature whose placeholder extends past the inward
    block-aligned freed range is dropped ENTIRELY (feature + physical
    encoder eviction) while its out-of-range pad tokens survive via
    absorption into the next segment. Surviving features must remain fully
    inside surviving segments; the freed mm_hash must reach `freed` so the
    worker drops the GPU tensor (the orphan pad tokens are then zeroed at
    re-prefill by the C16 neutralization — tested below)."""
    req, kv, enc = _make_request_with_history(
        "r-straddle",
        block_size=16,
        segments_spec=[
            ("user_text", 12, None, True, 3),
            ("video", 32, "v-straddle", False, 2),  # [12, 44), unaligned
            ("video", 32, "v-last", False, 1),  # [44, 76)
        ],
    )
    enc.allocate(req, input_id=0)
    enc.allocate(req, input_id=1)
    cache_size = enc.cache_size
    victim = req.session_history[1]
    tail = req.session_history[2]

    evict_segment(req, victim, kv, enc)

    # Aligned free of [16, 32): the placeholder [12, 44) straddles BOTH
    # edges, yet the feature is dropped whole.
    assert req.pending_evicted_token_ranges == [(16, 32)]
    assert [f.identifier for f in req.mm_features] == ["v-last"]
    # Physically evicted, exactly once, and drained.
    assert "v-straddle" not in enc.cached
    assert "v-straddle" not in enc.freeable
    assert enc.get_freed_mm_hashes() == ["v-straddle"]
    assert enc.get_freed_mm_hashes() == []
    # Slot accounting: only v-last's 32 embeds remain charged, and the
    # gating invariant num_freeable_slots == num_free_slots + sum(freeable)
    # holds with freeable empty.
    assert enc.num_free_slots == cache_size - 32
    assert enc.num_freeable_slots == enc.num_free_slots
    # The orphan pad tokens ([12,16) and the survivors of [32,44)) were
    # absorbed into the next segment, which extends back to the raw start.
    assert tail.token_range == (12, 60)
    # v-last's placeholder shifted down and still sits inside `tail`.
    assert req.mm_features[0].mm_position.offset == 28
    _assert_session_invariants(req, block_size=16)


# ---------------------------------------------------------------------------
# C15 (4): duplicate mm identifier within one session
# ---------------------------------------------------------------------------


def test_duplicate_identifier_evicts_physically_under_live_reference():
    """Guards C15 (4) by PINNING today's semantics: EncoderCacheManager
    refcounts are a per-mm_hash SET of request ids, so two features of one
    request sharing an identifier (repeated/static frames) collapse to one
    reference. Evicting the first occurrence therefore physically frees the
    embedding while the surviving feature still references it — the
    residual hazard the review documented (a re-prefill would need the
    freed embedding). The test also pins that the second eviction is
    idempotent: no double slot credit, no crash. If a guard lands (skip
    `evict_unreferenced` while another live feature shares the identifier),
    update the physical-evict asserts here."""
    req, kv, enc = _make_request_with_history(
        "r-dup-id",
        block_size=16,
        segments_spec=[
            ("user_text", 16, None, True, 3),
            ("video", 32, "dup", False, 2),  # [16, 48)
            ("video", 32, "dup", False, 1),  # [48, 80), same content hash
        ],
    )
    cache_size = enc.cache_size
    enc.allocate(req, input_id=0)
    enc.allocate(req, input_id=1)
    # Set semantics: two allocations, ONE reference; both charged.
    assert enc.cached["dup"] == {req.request_id}
    assert enc.num_free_slots == cache_size - 64

    seg_a, seg_b = req.session_history[1], req.session_history[2]
    evict_segment(req, seg_a, kv, enc)

    # Physically evicted despite the live duplicate feature (the hazard).
    assert "dup" not in enc.cached
    assert enc.get_freed_mm_hashes() == ["dup"]
    surviving = [f for f in req.mm_features if f.identifier == "dup"]
    assert len(surviving) == 1, (
        "the second occurrence must still be live on the request"
    )
    # Only ONE allocation's worth of slots was credited back; the
    # survivor's charge remains (bounded leak, documented).
    assert enc.num_free_slots == cache_size - 32
    assert enc.num_freeable_slots == enc.num_free_slots

    # Evicting the surviving occurrence must be idempotent on the encoder
    # cache: no double credit, no KeyError, nothing new in `freed`.
    evict_segment(req, seg_b, kv, enc)
    assert enc.get_freed_mm_hashes() == []
    assert enc.num_free_slots == cache_size - 32
    assert enc.num_freeable_slots == enc.num_free_slots
    assert req.mm_features == []
    _assert_session_invariants(req, block_size=16)


# ---------------------------------------------------------------------------
# C16: orphan-placeholder neutralization covers IMAGE (and video) markers
# ---------------------------------------------------------------------------

IMAGE_TOKEN_ID = 151655  # Qwen3-VL <|image_pad|>
VIDEO_TOKEN_ID = 151656  # Qwen3-VL <|video_pad|>


def _run_neutralize(
    token_ids: list[int],
    req_specs: list[tuple[str, int, object]],
    is_mm_flags: list[bool],
    placeholder_ids: tuple[int, ...] = (IMAGE_TOKEN_ID, VIDEO_TOKEN_ID),
) -> torch.Tensor:
    """Drive `GPUModelRunner._neutralize_orphan_mm_placeholders` on a stub
    runner (same unbound-method pattern as test_worker_slice.py).
    `req_specs` is [(req_id, num_scheduled_tokens, sampling_params)]."""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    num_tokens = len(token_ids)
    assert sum(n for _, n, _ in req_specs) == num_tokens
    stub = SimpleNamespace(
        mm_placeholder_ids_cpu=(
            torch.tensor(sorted(placeholder_ids), dtype=torch.int32)
            if placeholder_ids
            else None
        ),
        input_batch=SimpleNamespace(req_ids=[rid for rid, _, _ in req_specs]),
        requests={rid: SimpleNamespace(sampling_params=sp) for rid, _, sp in req_specs},
        input_ids=SimpleNamespace(gpu=torch.tensor(token_ids, dtype=torch.int32)),
        _req_uses_streaming_retention=(GPUModelRunner._req_uses_streaming_retention),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={rid: n for rid, n, _ in req_specs}
    )
    inputs_embeds = torch.ones((num_tokens, 4))
    is_mm_embed = torch.tensor(is_mm_flags, dtype=torch.bool)
    GPUModelRunner._neutralize_orphan_mm_placeholders(
        stub, scheduler_output, inputs_embeds, is_mm_embed, num_tokens
    )
    return inputs_embeds


def _streaming_sp():
    return SimpleNamespace(
        extra_args={"streaming_retention": {"reprefill_threshold": 0.7}}
    )


def _plain_sp():
    return SimpleNamespace(extra_args=None)


def test_neutralize_zeros_orphan_image_and_video_placeholders():
    """Guards C16: orphan <|image_pad|> markers (the deployed streaming
    workload pushes IMAGE frames) must be zeroed alongside <|video_pad|>;
    live (mm-backed) placeholder rows, plain text rows, and non-streaming
    requests' rows must be untouched."""
    embeds = _run_neutralize(
        token_ids=[
            IMAGE_TOKEN_ID,  # streaming, orphan image pad  -> zeroed
            7,  #               streaming, text             -> kept
            IMAGE_TOKEN_ID,  # streaming, LIVE (is_mm_embed)-> kept
            VIDEO_TOKEN_ID,  # streaming, orphan video pad  -> zeroed
            IMAGE_TOKEN_ID,  # NON-streaming request        -> kept
        ],
        req_specs=[
            ("sess-stream", 4, _streaming_sp()),
            ("req-plain", 1, _plain_sp()),
        ],
        is_mm_flags=[False, False, True, False, False],
    )
    zeroed = (embeds == 0).all(dim=1).tolist()
    assert zeroed == [True, False, False, True, False], zeroed


def test_neutralize_noop_without_placeholder_ids():
    """C16 companion: a model defining neither image_token_id nor
    video_token_id short-circuits (mm_placeholder_ids_cpu is None)."""
    embeds = _run_neutralize(
        token_ids=[IMAGE_TOKEN_ID, 7],
        req_specs=[("sess-stream", 2, _streaming_sp())],
        is_mm_flags=[False, False],
        placeholder_ids=(),
    )
    assert (embeds == 1).all()


def test_neutralize_noop_without_streaming_requests():
    """C16 companion: non-streaming batches are never touched (the
    `eligible` mask gates the zeroing to retention sessions)."""
    embeds = _run_neutralize(
        token_ids=[IMAGE_TOKEN_ID, VIDEO_TOKEN_ID],
        req_specs=[("req-plain", 2, _plain_sp())],
        is_mm_flags=[False, False],
    )
    assert (embeds == 1).all()
