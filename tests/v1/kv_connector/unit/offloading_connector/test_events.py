# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock

import pytest
import torch

from tests.v1.kv_connector.unit.utils import create_vllm_config
from vllm.config import KVEventsConfig, KVTransferConfig
from vllm.distributed.kv_events import (
    MEDIUM_CPU,
    MEDIUM_FS,
    MEDIUM_OBJ,
    BlockRemoved,
    BlockStored,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    build_offloading_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.events import (
    OffloadingEventGroupSpec,
    OffloadingEventsTracker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    GroupOffloadConfig,
)
from vllm.v1.core.kv_cache_utils import BlockHash, maybe_convert_block_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpecKind,
)
from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    OffloadingEvent,
    OffloadingKVEventsConfig,
    OffloadKey,
    make_offload_key,
)
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

_CPU_MEDIUM = MEDIUM_CPU
_FULL_ATTENTION_EVENT_SPEC = OffloadingEventGroupSpec(
    kv_cache_spec_kind=KVCacheSpecKind.FULL_ATTENTION.value,
    kv_cache_spec_sliding_window=None,
)


def _tracker(
    *,
    enable_kv_cache_events: bool = True,
    self_describing_kv_events: bool = True,
) -> OffloadingEventsTracker:
    return OffloadingEventsTracker(
        OffloadingKVEventsConfig(
            enable_kv_cache_events=enable_kv_cache_events,
            self_describing_kv_events=self_describing_kv_events,
        )
    )


def _hash(i: int) -> BlockHash:
    return BlockHash(str(i).encode())


def _wire_hash(block_hash: BlockHash):
    return maybe_convert_block_hash(block_hash)


def _request(*, block_hashes: list[BlockHash], token_count: int, req_id: str = "req"):
    req = MagicMock()
    req.request_id = req_id
    req.block_hashes = block_hashes
    req.all_token_ids = list(range(1, token_count + 1))
    req.lora_request = None
    return req


def _group_config(
    *,
    group_idx: int = 0,
    block_size: int = 4,
    blocks_per_chunk: int = 1,
    sliding_window_size_in_chunks: int | None = None,
) -> GroupOffloadConfig:
    return GroupOffloadConfig(
        group_idx=group_idx,
        tokens_per_block=block_size,
        tokens_per_chunk=block_size * blocks_per_chunk,
        hashes_per_chunk=blocks_per_chunk,
        sliding_window_size_in_chunks=sliding_window_size_in_chunks,
        kv_event_group_spec=_FULL_ATTENTION_EVENT_SPEC,
    )


def _record_chunks(
    tracker: OffloadingEventsTracker,
    req,
    group_config: GroupOffloadConfig,
    num_chunks: int,
) -> list[OffloadKey]:
    keys: list[OffloadKey] = []
    hbf = group_config.hashes_per_chunk
    for chunk_idx in range(num_chunks):
        tail_hash = req.block_hashes[(chunk_idx + 1) * hbf - 1]
        assert tail_hash is not None
        key = make_offload_key(tail_hash, group_config.group_idx)
        tracker.record_store(req, group_config, chunk_idx, key)
        keys.append(key)
    return keys


def _record_speculative_chunks(
    tracker: OffloadingEventsTracker,
    req,
    group_config: GroupOffloadConfig,
    num_chunks: int,
    *,
    result: LookupResult = LookupResult.RETRY,
) -> list[OffloadKey]:
    keys: list[OffloadKey] = []
    hbf = group_config.hashes_per_chunk
    for chunk_idx in range(num_chunks):
        tail_hash = req.block_hashes[(chunk_idx + 1) * hbf - 1]
        assert tail_hash is not None
        key = make_offload_key(tail_hash, group_config.group_idx)
        tracker.record_speculative(
            req,
            group_config,
            chunk_idx,
            key,
            result,
        )
        keys.append(key)
    return keys


def _stored_event(
    keys: list[OffloadKey],
    medium: str = _CPU_MEDIUM,
    locality: Locality | None = None,
) -> OffloadingEvent:
    return OffloadingEvent(
        keys=keys,
        medium=medium,
        removed=False,
        locality=locality,
    )


def _removed_event(
    keys: list[OffloadKey],
    medium: str = _CPU_MEDIUM,
    locality: Locality | None = None,
) -> OffloadingEvent:
    return OffloadingEvent(
        keys=keys,
        medium=medium,
        removed=True,
        locality=locality,
    )


def _speculative_chunk(
    *,
    req_id: str = "req",
    result: LookupResult = LookupResult.RETRY,
) -> tuple[OffloadingEventsTracker, MagicMock, GroupOffloadConfig, OffloadKey]:
    tracker = _tracker()
    req = _request(block_hashes=[_hash(0)], token_count=4, req_id=req_id)
    group_config = _group_config()
    key = _record_speculative_chunks(
        tracker,
        req,
        group_config,
        num_chunks=1,
        result=result,
    )[0]
    return tracker, req, group_config, key


def test_take_events_forwards_locality_to_rich_store():
    tracker = _tracker()
    req = _request(block_hashes=[_hash(0)], token_count=4)
    key = _record_chunks(tracker, req, _group_config(), num_chunks=1)[0]

    events = list(
        tracker.take_events(
            [_stored_event([key], locality=Locality.LOCAL, medium=MEDIUM_FS)]
        )
    )

    assert len(events) == 1
    assert isinstance(events[0], BlockStored)
    assert events[0].token_ids == [1, 2, 3, 4]
    assert events[0].block_size == 4
    assert events[0].locality == "LOCAL"


def test_take_events_forwards_locality_to_placeholder_store():
    tracker = _tracker(self_describing_kv_events=False)
    req = _request(block_hashes=[_hash(0)], token_count=4)
    key = _record_chunks(tracker, req, _group_config(), num_chunks=1)[0]

    events = list(
        tracker.take_events(
            [_stored_event([key], locality=Locality.REMOTE, medium=MEDIUM_FS)]
        )
    )

    assert len(events) == 1
    assert isinstance(events[0], BlockStored)
    assert events[0].block_size == 0
    assert events[0].locality == "REMOTE"


def test_take_events_forwards_locality_to_remove():
    tracker = _tracker()
    req = _request(block_hashes=[_hash(0)], token_count=4)
    key = _record_chunks(tracker, req, _group_config(), num_chunks=1)[0]

    events = list(
        tracker.take_events(
            [_removed_event([key], locality=Locality.LOCAL, medium=MEDIUM_FS)]
        )
    )

    assert len(events) == 1
    assert isinstance(events[0], BlockRemoved)
    assert events[0].locality == "LOCAL"


def test_take_events_publishes_routable_block_stored():
    block_size = 4
    tracker = _tracker()
    group_config = _group_config(block_size=block_size)
    req = _request(
        block_hashes=[_hash(i) for i in range(6)],
        token_count=block_size * 6,
    )
    keys = _record_chunks(tracker, req, group_config, num_chunks=6)

    batch1 = list(tracker.take_events([_stored_event(keys[:3])]))
    assert len(batch1) == 3

    for i, event in enumerate(batch1):
        assert isinstance(event, BlockStored)
        assert event.medium == _CPU_MEDIUM
        assert event.block_hashes == [_wire_hash(_hash(i))]
        assert event.block_size == block_size
        assert event.token_ids == list(
            range(i * block_size + 1, (i + 1) * block_size + 1)
        )
        if i == 0:
            assert event.parent_block_hash is None
        else:
            assert event.parent_block_hash == _wire_hash(_hash(i - 1))
        assert event.lora_id is None
        assert event.lora_name is None
        assert event.extra_keys is None
        assert event.group_idx == 0
        assert event.kv_cache_spec_kind == KVCacheSpecKind.FULL_ATTENTION.value
        assert event.kv_cache_spec_sliding_window is None

    batch2 = list(tracker.take_events([_stored_event(keys[3:])]))
    assert len(batch2) == 3
    assert batch2[0].parent_block_hash == batch1[-1].block_hashes[-1]

    assert len(tracker._pending_event_metadata) == 6


def test_promotion_emits_full_cpu_stored_event():
    tracker, _, _, key = _speculative_chunk()

    [event] = tracker.take_events([_stored_event([key])])

    assert isinstance(event, BlockStored)
    assert event.medium == MEDIUM_CPU
    assert event.block_hashes == [_wire_hash(_hash(0))]
    assert event.parent_block_hash is None
    assert event.token_ids == [1, 2, 3, 4]
    assert event.block_size == 4
    assert event.lora_id is None
    assert event.lora_name is None
    assert event.extra_keys is None
    assert event.group_idx == 0
    assert event.kv_cache_spec_kind == KVCacheSpecKind.FULL_ATTENTION.value
    assert event.kv_cache_spec_sliding_window is None


def test_speculative_promotion_factor_gt_1_store_and_remove():
    block_size = 4
    blocks_per_chunk = 3
    tracker = _tracker()
    group_config = _group_config(
        block_size=block_size, blocks_per_chunk=blocks_per_chunk
    )
    req = _request(
        block_hashes=[_hash(i) for i in range(6)],
        token_count=block_size * blocks_per_chunk * 2,
    )
    keys = _record_speculative_chunks(tracker, req, group_config, num_chunks=2)

    assert set(tracker._speculative_owners) == set(keys)

    stored = list(tracker.take_events([_stored_event(keys)]))
    assert len(stored) == 2

    expected_hashes = []
    for chunk_idx, event in enumerate(stored):
        assert isinstance(event, BlockStored)
        expected_chunk_hashes = [
            _wire_hash(_hash(i))
            for i in range(
                chunk_idx * blocks_per_chunk,
                (chunk_idx + 1) * blocks_per_chunk,
            )
        ]
        assert event.block_hashes == expected_chunk_hashes
        assert event.block_size == block_size
        assert len(event.token_ids) == block_size * blocks_per_chunk
        if chunk_idx == 0:
            assert event.parent_block_hash is None
        else:
            assert event.parent_block_hash == _wire_hash(_hash(blocks_per_chunk - 1))
        expected_hashes.extend(expected_chunk_hashes)

    assert len(tracker._pending_event_metadata) == 2
    assert not tracker._speculative_owners

    tracker.on_request_finished("req")
    assert len(tracker._pending_event_metadata) == 2

    removed = list(tracker.take_events([_removed_event(keys)]))
    assert len(removed) == 1
    assert isinstance(removed[0], BlockRemoved)
    assert removed[0].block_hashes == expected_hashes
    assert removed[0].medium == _CPU_MEDIUM
    assert removed[0].group_idx == 0
    assert not tracker._pending_event_metadata
    assert not tracker._speculative_keys


def test_take_events_factor_gt_1_store_is_order_independent():
    blocks_per_chunk = 3
    tracker = _tracker()
    group_config = _group_config(blocks_per_chunk=blocks_per_chunk)
    req = _request(
        block_hashes=[_hash(i) for i in range(6)],
        token_count=4 * blocks_per_chunk * 2,
    )
    keys = _record_chunks(tracker, req, group_config, num_chunks=2)
    unknown_key = make_offload_key(_hash(12345), 0)

    events = list(tracker.take_events([_stored_event([keys[1], unknown_key, keys[0]])]))

    assert len(events) == 3
    chunk1, placeholder, chunk0 = events
    assert [len(event.block_hashes) for event in events] == [3, 1, 3]
    assert placeholder.block_size == 0
    assert placeholder.token_ids == []
    assert chunk0.parent_block_hash is None
    assert chunk1.parent_block_hash == chunk0.block_hashes[-1]


def test_take_events_opt_out_keeps_placeholders():
    tracker = _tracker(self_describing_kv_events=False)
    group_config = _group_config()
    req = _request(block_hashes=[_hash(i) for i in range(3)], token_count=12)
    keys = _record_chunks(tracker, req, group_config, num_chunks=3)
    _record_speculative_chunks(tracker, req, group_config, num_chunks=3)
    tracker.on_request_finished("req")

    assert not tracker.self_describing_enabled
    assert not tracker._pending_event_metadata
    assert not tracker._speculative_owners
    assert not tracker._speculative_keys

    events = list(
        tracker.take_events(
            [
                _stored_event(keys),
                _removed_event(keys),
            ]
        )
    )
    assert len(events) == 4
    for event in events[:3]:
        assert isinstance(event, BlockStored)
        assert event.block_size == 0
        assert event.token_ids == []
        assert event.parent_block_hash is None
    assert isinstance(events[3], BlockRemoved)
    assert len(events[3].block_hashes) == 3


@pytest.mark.parametrize(
    "sliding_window_size_in_chunks",
    [1, 2],
    ids=["ssm", "sliding-window"],
)
def test_event_metadata_skips_non_full_attention_group(
    sliding_window_size_in_chunks: int,
):
    tracker = _tracker()
    group_config = _group_config(
        sliding_window_size_in_chunks=sliding_window_size_in_chunks
    )
    req = _request(block_hashes=[_hash(i) for i in range(3)], token_count=12)
    keys = _record_chunks(tracker, req, group_config, num_chunks=3)
    _record_speculative_chunks(tracker, req, group_config, num_chunks=3)

    assert not tracker._pending_event_metadata
    assert not tracker._speculative_owners
    assert not tracker._speculative_keys

    events = list(tracker.take_events([_stored_event(keys[:1])]))
    assert len(events) == 1
    assert isinstance(events[0], BlockStored)
    assert events[0].block_size == 0


@pytest.mark.parametrize("lookup_result", [LookupResult.MISS, LookupResult.RETRY])
def test_pending_cpu_removal_survives_interleaved_lookup(
    lookup_result: LookupResult,
):
    tracker = _tracker()
    block_hashes = [_hash(0), _hash(1)]
    req = _request(block_hashes=block_hashes, token_count=8)
    group_config = _group_config(blocks_per_chunk=2)
    key = _record_chunks(tracker, req, group_config, num_chunks=1)[0]
    confirmed_meta = tracker._pending_event_metadata[key]
    lookup_req = _request(
        block_hashes=block_hashes,
        token_count=8,
        req_id="new-request",
    )

    tracker.record_speculative(
        lookup_req,
        group_config,
        0,
        key,
        lookup_result,
    )
    assert tracker._pending_event_metadata == {key: confirmed_meta}
    assert not tracker._speculative_owners
    assert not tracker._speculative_keys

    tracker.on_request_finished("new-request")
    assert tracker._pending_event_metadata == {key: confirmed_meta}

    removed = list(tracker.take_events([_removed_event([key])]))
    assert len(removed) == 1
    assert removed[0].block_hashes == [
        _wire_hash(_hash(0)),
        _wire_hash(_hash(1)),
    ]

    if lookup_result is LookupResult.RETRY:
        stored = list(tracker.take_events([_stored_event([key])]))
        assert len(stored) == 1
        assert stored[0].block_size == 0
        assert stored[0].token_ids == []


def test_two_speculators_survive_first_request_finish():
    tracker, req, group_config, key = _speculative_chunk(req_id="req-a")
    req_b = _request(
        block_hashes=req.block_hashes,
        token_count=len(req.all_token_ids),
        req_id="req-b",
    )
    tracker.record_speculative(
        req_b,
        group_config,
        0,
        key,
        LookupResult.HIT_PENDING,
    )

    tracker.on_request_finished("req-a")
    assert tracker._speculative_owners[key] == {"req-b"}

    stored = list(tracker.take_events([_stored_event([key])]))
    assert stored[0].token_ids == [1, 2, 3, 4]
    assert key not in tracker._speculative_owners

    tracker.on_request_finished("req-b")
    assert key in tracker._pending_event_metadata
    removed = list(tracker.take_events([_removed_event([key])]))
    assert removed[0].block_hashes == [_wire_hash(_hash(0))]
    assert not tracker._pending_event_metadata
    assert not tracker._speculative_keys


def test_store_path_confirms_speculative_metadata():
    tracker, req, group_config, key = _speculative_chunk()

    tracker.record_store(req, group_config, 0, key)
    assert key not in tracker._speculative_owners

    tracker.on_request_finished("req")
    assert key in tracker._pending_event_metadata
    removed = list(tracker.take_events([_removed_event([key])]))
    assert removed[0].block_hashes == [_wire_hash(_hash(0))]


def test_primary_hit_confirms_before_store_event_translation():
    tracker, req, group_config, key = _speculative_chunk()

    tracker.record_speculative(
        req,
        group_config,
        0,
        key,
        LookupResult.HIT,
    )
    tracker.on_request_finished("req")

    assert key in tracker._pending_event_metadata
    stored = list(tracker.take_events([_stored_event([key])]))
    assert stored[0].token_ids == [1, 2, 3, 4]


def test_hit_pending_remains_speculative_until_request_finish():
    tracker, _, _, key = _speculative_chunk(result=LookupResult.HIT_PENDING)

    tracker.on_request_finished("req")

    assert not tracker._pending_event_metadata
    assert not tracker._speculative_owners
    assert not tracker._speculative_keys
    stored = list(tracker.take_events([_stored_event([key])]))
    assert stored[0].block_size == 0


@pytest.mark.parametrize("medium", [MEDIUM_FS, MEDIUM_OBJ])
def test_secondary_events_do_not_mutate_cpu_metadata(medium: str):
    tracker, req, group_config, key = _speculative_chunk()

    stored = list(tracker.take_events([_stored_event([key], medium)]))
    assert stored[0].token_ids == [1, 2, 3, 4]
    assert tracker._speculative_owners[key] == {"req"}

    removed = list(tracker.take_events([_removed_event([key], medium)]))
    assert removed[0].block_hashes == [_wire_hash(_hash(0))]
    assert key in tracker._pending_event_metadata
    assert tracker._speculative_owners[key] == {"req"}

    tracker.record_store(req, group_config, 0, key)
    cpu_removed = list(tracker.take_events([_removed_event([key])]))
    assert cpu_removed[0].block_hashes == [_wire_hash(_hash(0))]
    assert not tracker._pending_event_metadata


def test_take_events_groups_removed_hashes_by_kv_group():
    tracker = _tracker()
    group0_config = _group_config(group_idx=0, blocks_per_chunk=2)
    group1_config = _group_config(group_idx=1, blocks_per_chunk=2)
    req0 = _request(block_hashes=[_hash(0), _hash(1)], token_count=8)
    req1 = _request(block_hashes=[_hash(10), _hash(11)], token_count=8)
    key0 = _record_chunks(tracker, req0, group0_config, num_chunks=1)[0]
    key1 = _record_chunks(tracker, req1, group1_config, num_chunks=1)[0]

    removed = list(tracker.take_events([_removed_event([key0, key1])]))

    assert len(removed) == 2
    by_group = {event.group_idx: event.block_hashes for event in removed}
    assert by_group == {
        0: [_wire_hash(_hash(0)), _wire_hash(_hash(1))],
        1: [_wire_hash(_hash(10)), _wire_hash(_hash(11))],
    }


def test_take_events_supports_restore_after_eviction():
    block_size = 4
    tracker = _tracker()
    group_config = _group_config(block_size=block_size)
    req = _request(block_hashes=[_hash(0)], token_count=block_size)
    key = _record_chunks(tracker, req, group_config, num_chunks=1)[0]

    first_store = list(tracker.take_events([_stored_event([key])]))
    assert len(first_store) == 1
    assert isinstance(first_store[0], BlockStored)
    assert first_store[0].token_ids == [1, 2, 3, 4]

    removed = list(tracker.take_events([_removed_event([key])]))
    assert len(removed) == 1
    assert isinstance(removed[0], BlockRemoved)
    assert not tracker._pending_event_metadata

    req.all_token_ids = [5, 6, 7, 8]
    tracker.record_store(req, group_config, chunk_idx=0, offload_key=key)

    second_store = list(tracker.take_events([_stored_event([key])]))
    assert len(second_store) == 1
    assert isinstance(second_store[0], BlockStored)
    assert second_store[0].token_ids == [5, 6, 7, 8]


def test_reset_cache_clears_side_table():
    tracker = _tracker()
    group_config = _group_config()
    req = _request(block_hashes=[_hash(i) for i in range(3)], token_count=12)
    _record_speculative_chunks(tracker, req, group_config, num_chunks=3)

    assert tracker._pending_event_metadata
    assert tracker._speculative_owners
    assert tracker._speculative_keys

    tracker.reset()

    assert not tracker._pending_event_metadata
    assert not tracker._speculative_owners
    assert not tracker._speculative_keys


def test_tiering_accepts_self_describing_kv_events():
    vllm_config = create_vllm_config(
        block_size=4,
        max_num_batched_tokens=16,
        disable_hybrid_kv_cache_manager=False,
    )
    vllm_config.kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "spec_name": "TieringOffloadingSpec",
            "cpu_bytes_to_use": 1 << 20,
            "self_describing_kv_events": True,
            "secondary_tiers": [{"type": "example"}],
        },
    )
    vllm_config.kv_events_config = KVEventsConfig(
        enable_kv_cache_events=True,
        publisher="null",
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=4,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )

    spec = TieringOffloadingSpec(build_offloading_config(vllm_config, kv_cache_config))
    tracker = OffloadingEventsTracker(spec.kv_events_config)

    assert spec.kv_events_config.enable_kv_cache_events
    assert spec.kv_events_config.self_describing_kv_events
    assert tracker.self_describing_enabled
