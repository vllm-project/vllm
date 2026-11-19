# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Live range eviction must preserve KV, token, and encoder-cache ownership."""

import pytest
import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheGroup,
)
from vllm.v1.request import Request
from vllm.v1.streaming.eviction import (
    drop_old_segments_for_reprefill,
    evict_segment,
    maybe_evict_old_segments,
)
from vllm.v1.streaming.reprefill import (
    note_frame_turn,
    rebuild_transcript_for_reprefill,
)
from vllm.v1.streaming.retention import HistorySegment, StreamingRetentionParams

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def init_hash():
    init_none_hash(sha256)


def make_manager(caching=False):
    return KVCacheManager(
        KVCacheConfig(
            num_blocks=64,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroup.from_layers(
                    {
                        "layer": FullAttentionManager.layer_config(
                            AttentionSpec(
                                block_size=16,
                                num_kv_heads=1,
                                head_size=1,
                                dtype=torch.float32,
                            )
                        )
                    }
                )
            ],
        ),
        max_model_len=1024,
        scheduler_block_size=16,
        hash_block_size=16,
        enable_caching=caching,
    )


def make_request(request_id, tokens, *, retention=None, mm_features=None):
    params = SamplingParams(
        max_tokens=8,
        extra_args={"streaming_retention": retention} if retention else None,
    )
    return Request(
        request_id=request_id,
        prompt_token_ids=tokens,
        sampling_params=params,
        pooling_params=None,
        mm_features=mm_features,
        resumable=retention is not None,
        block_hasher=get_request_block_hasher(16, sha256),
    )


def allocate(manager, request):
    computed, count, _ = manager.get_computed_blocks(request)
    assert (
        manager.allocate_slots(request, request.num_tokens - count, count, computed)
        is not None
    )
    request.num_computed_tokens = request.num_tokens


@pytest.mark.parametrize("caching", [False, True])
def test_live_range_reports_actual_span_and_allows_more_allocation(caching):
    manager = make_manager(caching)
    request = make_request("r", list(range(64)))
    allocate(manager, request)
    before = manager.get_block_ids("r")[0]

    # The returned span excludes the partial leading block and unavailable tail.
    assert manager.evict_token_range_for_request("r", 17, 1000) == (32, 64, 2)
    assert manager.get_block_ids("r")[0] == before[:2]
    request.num_computed_tokens = 32
    assert manager.allocate_slots(request, 16) is not None
    assert len(manager.get_block_ids("r")[0]) == 3
    manager.free(request)
    assert manager.block_pool.get_num_free_blocks() == 63


def test_shared_evicted_blocks_are_not_counted_as_reclaimed_memory():
    manager = make_manager(caching=True)
    first = make_request("first", [1] * 64)
    allocate(manager, first)
    second = make_request("second", [1] * 64 + [2] * 16)
    allocate(manager, second)
    before = manager.get_block_ids("first")[0]
    removed = [manager.block_pool.blocks[i] for i in before[1:3]]
    assert all(block.ref_cnt == 2 for block in removed)

    assert manager.evict_token_range_for_request("first", 16, 48) == (16, 48, 0)
    assert manager.get_block_ids("first")[0] == [before[0], before[3]]
    assert all(block.ref_cnt == 1 and block.block_hash is None for block in removed)
    assert manager.coordinator.single_type_managers[0].num_cached_block["first"] == 2
    assert manager.get_block_ids("second")[0][:4] == before
    manager.free(first)
    manager.free(second)
    assert manager.block_pool.get_num_free_blocks() == 63


def test_compacted_sessions_do_not_publish_prefixes_with_stale_history():
    manager = make_manager(caching=True)
    request = make_request("bounded", [1] * 64, retention=StreamingRetentionParams())
    allocate(manager, request)
    assert request.block_hashes == []
    assert request.prefix_cache_hit_limit == 0
    assert all(
        manager.block_pool.blocks[i].block_hash is None
        for i in manager.get_block_ids("bounded")[0]
    )
    ordinary = make_request("ordinary", [1] * 80)
    assert manager.get_computed_blocks(ordinary)[1] == 0
    allocate(manager, ordinary)
    assert ordinary.block_hashes
    assert any(
        manager.block_pool.blocks[i].block_hash is not None
        for i in manager.get_block_ids("ordinary")[0]
    )


def test_empty_unknown_and_null_ranges_leave_the_table_unchanged():
    manager = make_manager()
    request = make_request("r", list(range(48)))
    allocate(manager, request)
    before = manager.get_block_ids("r")[0]
    for start, end in [(1, 15), (48, 1000), (32, 16)]:
        assert manager.evict_token_range_for_request("r", start, end) == (
            start,
            start,
            0,
        )
        assert manager.get_block_ids("r")[0] == before
    assert manager.evict_token_range_for_request("unknown", 0, 64) == (0, 0, 0)
    with pytest.raises(ValueError, match="nonnegative"):
        manager.evict_token_range_for_request("r", -1, 32)

    owner = manager.coordinator.single_type_managers[0]
    removed = owner.req_to_blocks["r"][1]
    manager.block_pool.free_blocks([removed])
    owner.req_to_blocks["r"][1] = manager.block_pool.null_block
    with pytest.raises(ValueError, match="null"):
        manager.evict_token_range_for_request("r", 16, 32)
    assert len(owner.req_to_blocks["r"]) == 3
    manager.free(request)


def make_session(specs, *, frontier=None, retention=None):
    retention = retention or StreamingRetentionParams(max_video_segments=1)
    total = sum(length for _, length, _, _, _ in specs)
    request = make_request("session", list(range(total)), retention=retention)
    cursor = 0
    for kind, length, pinned, age, identifier in specs:
        request.session_history.append(
            HistorySegment(
                kind, (cursor, cursor + length), pinned=pinned, age_chunks=age
            )
        )
        if identifier:
            request.mm_features.append(
                MultiModalFeatureSpec(
                    data=None,
                    modality="image",
                    identifier=identifier,
                    mm_position=PlaceholderRange(offset=cursor, length=length),
                )
            )
        cursor += length
    manager = make_manager()
    allocate(manager, request)
    if frontier is not None:
        request.num_computed_tokens = frontier
    return request, manager, EncoderCacheManager(cache_size=128)


def test_eviction_reindexes_duplicate_encoder_references_without_freeing_survivor():
    request, kv, encoder = make_session(
        [
            ("system_prompt", 16, True, 3, None),
            ("video", 32, False, 2, "duplicate"),
            ("assistant_text", 16, False, 1, None),
            ("video", 32, False, 1, "duplicate"),
            ("user_text", 16, False, 0, None),
        ],
        frontier=96,
    )
    encoder.allocate(request, 0)
    assert encoder.check_and_update_cache(request, 1)
    pending = {0, 1}
    old_tokens = list(request.all_token_ids)

    assert (
        maybe_evict_old_segments(
            request,
            request.streaming_retention,
            kv,
            encoder,
            pending_encoder_input_ids=pending,
        )
        == 1
    )
    assert request.pending_evicted_token_ranges == [(16, 48)]
    assert list(request.all_token_ids) == old_tokens[:16] + old_tokens[48:]
    assert request.prompt_token_ids == list(request.all_token_ids)
    assert request.num_prompt_tokens == 80
    assert request.num_computed_tokens == 64
    assert request.position_offset == 32
    assert request.mm_features[0].mm_position.offset == 32
    assert pending == {0}
    assert encoder.get_cached_input_ids(request) == {0}
    assert encoder.request_hash_refs[request.request_id] == {"duplicate": 1}
    assert encoder.cached["duplicate"] == {request.request_id}
    assert "duplicate" not in encoder.freeable
    encoder.free(request)
    assert not encoder.get_cached_input_ids(request)
    assert encoder.num_freeable_slots == encoder.cache_size


def test_alignment_uses_neighbor_text_without_splitting_multimodal_placeholders():
    request, kv, encoder = make_session(
        [
            ("system_prompt", 16, True, 3, None),
            ("assistant_text", 8, False, 2, None),
            ("video", 32, False, 2, "old"),
            ("assistant_text", 24, False, 1, None),
            ("video", 32, False, 0, "new"),
        ],
        frontier=80,
    )
    assert evict_segment(request, request.session_history[2], kv, encoder)
    assert request.pending_evicted_token_ranges == [(16, 64)]
    assert [(f.identifier, f.mm_position.offset) for f in request.mm_features] == [
        ("new", 32)
    ]
    assert [s.token_range for s in request.session_history] == [
        (0, 16),
        (16, 32),
        (32, 64),
    ]
    assert sum(e - s for s, e in (x.token_range for x in request.session_history)) == 64


@pytest.mark.parametrize("mode", ["uniform", "annotated", "untrusted", "text"])
def test_reprefill_rebuilds_complete_turns_and_keeps_buffer_and_encoder_ownership(mode):
    frames = [[10] * 16 + [900] * 32 + [20] * 32]
    for index in range(1, 4):
        lead = 30 + index if mode == "annotated" else 30
        frames.append([lead] * 8 + [900 + index] * 32 + [40] * 24)
    frame_kind = "user_text" if mode == "text" else "video"
    parts = [
        ("system_prompt", frames[0]),
        ("assistant_text", [] if mode == "untrusted" else [100] * 24),
        (frame_kind, frames[1]),
        ("assistant_text", [101] * 24),
        (frame_kind, frames[2]),
        ("assistant_text", [102] * 16),
        (frame_kind, frames[3]),
    ]
    history, features = [], []
    cursor = frame_index = 0
    for index, (kind, tokens) in enumerate(parts):
        if tokens:
            history.append(
                HistorySegment(kind, (cursor, cursor + len(tokens)), pinned=index == 0)
            )
        if index == 2:
            oldest_turn = history[-1]
        if index in (0, 2, 4, 6) and mode != "text":
            features.append(
                MultiModalFeatureSpec(
                    data=None,
                    modality="image",
                    identifier=f"frame{frame_index}",
                    mm_position=PlaceholderRange(
                        offset=cursor + (16 if index == 0 else 8), length=32
                    ),
                )
            )
            frame_index += 1
        cursor += len(tokens)
    request = make_request(
        "session",
        [token for _, tokens in parts for token in tokens],
        retention=StreamingRetentionParams(max_session_tokens=512),
        mm_features=features,
    )
    request.session_history = history
    # Admission certifies the application's image-only turns. Identical text
    # from a generic SDK caller is not proof that its annotations are disposable.
    request.streaming_frames_uniform = mode in ("uniform", "annotated")
    kv, encoder = make_manager(), EncoderCacheManager(cache_size=128)
    allocate(kv, request)
    for input_id in range(len(features)):
        encoder.allocate(request, input_id)
    for segment in history:
        if segment.segment_type == "video":
            note_frame_turn(request, *segment.token_range)
    pending = set(range(len(features)))
    all_tokens, prompt_tokens, token_view = (
        request._all_token_ids,
        request.prompt_token_ids,
        request.all_token_ids,
    )

    # Eviction can cut replies or leave a whole answer to a removed input.
    # Neither belongs in the newly computed transcript.
    assert evict_segment(
        request, oldest_turn, kv, encoder, pending_encoder_input_ids=pending
    )
    rebuild_transcript_for_reprefill(
        request, encoder, pending_encoder_input_ids=pending
    )

    expected = frames[0] + frames[2] + parts[5][1] + frames[3]
    identifiers = ["frame0", "frame2", "frame3"] if mode != "text" else []
    if mode == "uniform":
        expected = (
            frames[0][:16] + [902] * 32 + frames[0][48:] + parts[5][1] + frames[3]
        )
        identifiers = ["frame2", "frame3"]
    assert list(token_view) == expected
    assert request._all_token_ids is all_tokens
    assert request.prompt_token_ids is prompt_tokens
    assert list(prompt_tokens) == expected
    assert request.num_prompt_tokens == len(expected)
    assert [(f.identifier, f.mm_position.offset) for f in request.mm_features] == [
        (identifier, expected.index(900 + int(identifier[-1])))
        for identifier in identifiers
    ]
    assert request.session_history[0].pinned
    assert all(s.complete for s in request.session_history)
    assert [s.token_range[0] for s in request.session_history] == [
        0,
        *[s.token_range[1] for s in request.session_history[:-1]],
    ]
    assert request.session_history[-1].token_range[1] == len(expected)
    assert (
        pending == encoder.get_cached_input_ids(request) == set(range(len(identifiers)))
    )
    assert encoder.request_hash_refs.get(request.request_id, {}) == {
        identifier: 1 for identifier in identifiers
    }
    assert request.block_hashes == [] and request.prefix_cache_hit_limit == 0
    kv.free(request)
    encoder.free(request)
    assert kv.block_pool.get_num_free_blocks() == 63
    assert encoder.num_freeable_slots == encoder.cache_size


def test_alignment_cannot_drop_pinned_or_uncomputed_multimodal_content():
    request, kv, encoder = make_session(
        [
            ("system_prompt", 12, True, 2, None),
            ("video", 32, False, 1, "old"),
            ("video", 32, False, 0, "new"),
        ],
        frontier=44,
    )
    original_tokens = list(request.all_token_ids)
    original_blocks = kv.get_block_ids(request.request_id)
    assert not evict_segment(request, request.session_history[1], kv, encoder)
    assert not evict_segment(request, request.session_history[2], kv, encoder)
    assert list(request.all_token_ids) == original_tokens
    assert kv.get_block_ids(request.request_id) == original_blocks
    assert request.pending_evicted_token_ranges == []
    assert len(request.mm_features) == 2


def test_text_and_total_budgets_compact_short_adjacent_segments():
    retention = StreamingRetentionParams(max_text_tokens=16, max_session_tokens=48)
    request, kv, encoder = make_session(
        [("system_prompt", 16, True, 3, None)]
        + [("assistant_text", 4, False, 1, None)] * 12
        + [("user_text", 16, False, 0, None)],
        frontier=64,
        retention=retention,
    )
    assert maybe_evict_old_segments(request, retention, kv, encoder) > 0
    assert request.num_prompt_tokens <= retention.max_session_tokens
    assert request.session_history[0].token_range == (0, 16)
    assert request.session_history[-1].token_range[1] == request.num_prompt_tokens
    assert request.num_computed_tokens == request.num_prompt_tokens - 16


@pytest.mark.parametrize("image_tokens", [196, 256])
def test_replay_fallback_bounds_unaligned_frames_without_orphan_placeholders(
    image_tokens,
):
    """Short replies leave no block boundary between some complete images."""
    chunk_tokens = image_tokens + 12
    retention = StreamingRetentionParams(
        max_video_segments=1, max_session_tokens=2 * chunk_tokens + 128
    )
    request, kv, _ = make_session(
        [
            ("system_prompt", 65, True, 3, None),
            ("assistant_text", 1, False, 2, None),
            ("video", chunk_tokens, False, 2, "old-a"),
            ("assistant_text", 1, False, 1, None),
            ("video", chunk_tokens, False, 1, "old-b"),
            ("assistant_text", 1, False, 0, None),
            ("video", chunk_tokens, False, 0, "new"),
        ],
        frontier=68 + 2 * chunk_tokens,
        retention=retention,
    )
    for feature in request.mm_features:
        feature.mm_position = PlaceholderRange(
            offset=feature.mm_position.offset + 6, length=image_tokens
        )
    encoder = EncoderCacheManager(cache_size=2 * image_tokens)
    encoder.allocate(request, 0)
    encoder.allocate(request, 1)
    pending = {0, 1, 2}
    original_tokens = list(request.all_token_ids)
    original_blocks = kv.get_block_ids(request.request_id)
    assert maybe_evict_old_segments(request, retention, kv, encoder) == 0
    assert request.num_prompt_tokens > retention.max_session_tokens

    assert (
        drop_old_segments_for_reprefill(
            request,
            retention,
            encoder,
            pending_encoder_input_ids=pending,
        )
        == 2
    )
    assert request.num_prompt_tokens <= retention.max_session_tokens
    assert list(request.all_token_ids) == (
        original_tokens[:65] + original_tokens[66 + chunk_tokens :]
    )
    assert request.prompt_token_ids == list(request.all_token_ids)
    assert [feature.identifier for feature in request.mm_features] == ["old-b", "new"]
    assert [feature.mm_position.length for feature in request.mm_features] == [
        image_tokens,
        image_tokens,
    ]
    assert request.mm_features[0].mm_position.offset == 72
    assert pending == {0, 1}
    assert encoder.get_cached_input_ids(request) == {0}
    assert not encoder.cached["old-a"]
    assert encoder.cached["old-b"] == {request.request_id}
    assert request.pending_evicted_token_ranges == []
    assert kv.get_block_ids(request.request_id) == original_blocks
    assert request.session_history[0].token_range == (0, 65)
    assert (
        sum(
            end - start
            for start, end in (s.token_range for s in request.session_history)
        )
        == request.num_prompt_tokens
    )
    # The caller invalidates the old KV table before scheduling replay.
    kv.free(request)
    encoder.free(request)
    assert kv.block_pool.get_num_free_blocks() == 63
    assert encoder.num_freeable_slots == encoder.cache_size


def test_replay_fallback_preserves_pinned_and_newly_arrived_segments():
    retention = StreamingRetentionParams(max_session_tokens=32)
    request, kv, encoder = make_session(
        [
            ("system_prompt", 16, True, 3, None),
            ("video", 32, False, 0, "new"),
        ],
        retention=retention,
    )
    original_tokens = list(request.all_token_ids)
    assert drop_old_segments_for_reprefill(request, retention, encoder) == 0
    assert list(request.all_token_ids) == original_tokens
    assert request.num_prompt_tokens == 48
    kv.free(request)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_session_tokens": None},
        {"max_session_tokens": 0},
        {"max_video_segments": 0},
        {"max_text_tokens": 0},
        {"reprefill_threshold": float("nan")},
        {"reprefill_threshold": 1.0},
    ],
)
def test_invalid_retention_cannot_disable_the_session_safety_limits(kwargs):
    with pytest.raises(ValueError):
        StreamingRetentionParams(**kwargs)
