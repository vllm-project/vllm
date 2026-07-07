# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock

import pytest
import torch

from tests.v1.kv_connector.unit.utils import create_vllm_config
from vllm.config import KVEventsConfig, KVTransferConfig
from vllm.distributed.kv_events import MEDIUM_CPU, BlockRemoved, BlockStored
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


def _request(*, block_hashes: list[BlockHash], token_count: int):
    req = MagicMock()
    req.block_hashes = block_hashes
    req.all_token_ids = list(range(1, token_count + 1))
    req.lora_request = None
    return req


def _group_config(
    *,
    group_idx: int = 0,
    block_size: int = 4,
    block_size_factor: int = 1,
    sliding_window_size_in_blocks: int | None = None,
) -> GroupOffloadConfig:
    return GroupOffloadConfig(
        group_idx=group_idx,
        gpu_block_size=block_size,
        offloaded_block_size=block_size * block_size_factor,
        hash_block_size_factor=block_size_factor,
        sliding_window_size_in_blocks=sliding_window_size_in_blocks,
        kv_event_group_spec=_FULL_ATTENTION_EVENT_SPEC,
    )


def _record_chunks(
    tracker: OffloadingEventsTracker,
    req,
    group_config: GroupOffloadConfig,
    num_chunks: int,
) -> list[OffloadKey]:
    keys: list[OffloadKey] = []
    hbf = group_config.hash_block_size_factor
    for chunk_idx in range(num_chunks):
        tail_hash = req.block_hashes[(chunk_idx + 1) * hbf - 1]
        assert tail_hash is not None
        key = make_offload_key(tail_hash, group_config.group_idx)
        tracker.record_store(req, group_config, chunk_idx, key)
        keys.append(key)
    return keys


def _stored_event(keys: list[OffloadKey]) -> OffloadingEvent:
    return OffloadingEvent(keys=keys, medium=_CPU_MEDIUM, removed=False)


def _removed_event(keys: list[OffloadKey]) -> OffloadingEvent:
    return OffloadingEvent(keys=keys, medium=_CPU_MEDIUM, removed=True)


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


def test_take_events_factor_gt_1_chunk_store_and_remove():
    block_size = 4
    block_size_factor = 3
    tracker = _tracker()
    group_config = _group_config(
        block_size=block_size, block_size_factor=block_size_factor
    )
    req = _request(
        block_hashes=[_hash(i) for i in range(6)],
        token_count=block_size * block_size_factor * 2,
    )
    keys = _record_chunks(tracker, req, group_config, num_chunks=2)

    stored = list(tracker.take_events([_stored_event(keys)]))
    assert len(stored) == 2

    expected_hashes = []
    for chunk_idx, event in enumerate(stored):
        assert isinstance(event, BlockStored)
        expected_chunk_hashes = [
            _wire_hash(_hash(i))
            for i in range(
                chunk_idx * block_size_factor,
                (chunk_idx + 1) * block_size_factor,
            )
        ]
        assert event.block_hashes == expected_chunk_hashes
        assert event.block_size == block_size
        assert len(event.token_ids) == block_size * block_size_factor
        if chunk_idx == 0:
            assert event.parent_block_hash is None
        else:
            assert event.parent_block_hash == _wire_hash(_hash(block_size_factor - 1))
        expected_hashes.extend(expected_chunk_hashes)

    assert len(tracker._pending_event_metadata) == 2

    removed = list(tracker.take_events([_removed_event(keys)]))
    assert len(removed) == 1
    assert isinstance(removed[0], BlockRemoved)
    assert removed[0].block_hashes == expected_hashes
    assert removed[0].medium == _CPU_MEDIUM
    assert removed[0].group_idx == 0
    assert not tracker._pending_event_metadata


def test_take_events_factor_gt_1_store_is_order_independent():
    block_size_factor = 3
    tracker = _tracker()
    group_config = _group_config(block_size_factor=block_size_factor)
    req = _request(
        block_hashes=[_hash(i) for i in range(6)],
        token_count=4 * block_size_factor * 2,
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

    assert not tracker.self_describing_enabled
    assert not tracker._pending_event_metadata

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


def test_record_store_skips_sliding_window_group():
    tracker = _tracker()
    group_config = _group_config(sliding_window_size_in_blocks=2)
    req = _request(block_hashes=[_hash(i) for i in range(3)], token_count=12)
    keys = _record_chunks(tracker, req, group_config, num_chunks=3)

    assert not tracker._pending_event_metadata

    events = list(tracker.take_events([_stored_event(keys[:1])]))
    assert len(events) == 1
    assert isinstance(events[0], BlockStored)
    assert events[0].block_size == 0


def test_take_events_groups_removed_hashes_by_kv_group():
    tracker = _tracker()
    group0_config = _group_config(group_idx=0, block_size_factor=2)
    group1_config = _group_config(group_idx=1, block_size_factor=2)
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
    tracker.record_store(req, group_config, offload_block_idx=0, offload_key=key)

    second_store = list(tracker.take_events([_stored_event([key])]))
    assert len(second_store) == 1
    assert isinstance(second_store[0], BlockStored)
    assert second_store[0].token_ids == [5, 6, 7, 8]


def test_reset_cache_clears_side_table():
    tracker = _tracker()
    group_config = _group_config()
    req = _request(block_hashes=[_hash(i) for i in range(3)], token_count=12)
    _record_chunks(tracker, req, group_config, num_chunks=3)

    assert tracker._pending_event_metadata

    tracker.reset()

    assert not tracker._pending_event_metadata


def test_tiering_rejects_self_describing_kv_events():
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

    with pytest.raises(ValueError, match="TieringOffloadingSpec"):
        TieringOffloadingSpec(vllm_config, kv_cache_config)
