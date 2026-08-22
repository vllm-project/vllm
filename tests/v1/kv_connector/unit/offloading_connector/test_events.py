# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tests.v1.kv_connector.unit.utils import create_vllm_config
from vllm.config import KVEventsConfig, KVTransferConfig
from vllm.distributed.kv_events import (
    MEDIUM_CPU,
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
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    generate_block_hash_extra_keys,
    hash_block_tokens,
    init_none_hash,
    maybe_convert_block_hash,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpecKind,
)
from vllm.v1.kv_offload.base import (
    Locality,
    Medium,
    OffloadingEvent,
    OffloadingKVEventsConfig,
    OffloadKey,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

_CPU_MEDIUM = Medium.CPU
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


def _rehash_request(req, block_size: int) -> None:
    init_none_hash(sha256)
    block_hashes: list[BlockHash] = []
    parent: BlockHash | None = None
    mm_idx = 0
    for start in range(0, len(req.all_token_ids), block_size):
        end = start + block_size
        assert end <= len(req.all_token_ids)
        extra_keys, mm_idx = generate_block_hash_extra_keys(req, start, end, mm_idx)
        parent = hash_block_tokens(
            sha256,
            parent,
            req.all_token_ids[start:end],
            extra_keys,
        )
        block_hashes.append(parent)
    req.block_hashes = block_hashes


def _request(*, block_hashes: list[BlockHash], token_count: int, req_id: str = "req"):
    req = MagicMock()
    req.request_id = req_id
    req.block_hashes = block_hashes
    req.all_token_ids = list(range(1, token_count + 1))
    req.lora_request = None
    req.mm_features = []
    req.cache_salt = None
    req.prompt_embeds = None
    req._prompt_embeds_per_block_hashes = {}
    req.req_context = ReqContext(req_id)
    req.resumable = False
    return req


def _group_config(
    *,
    group_idx: int = 0,
    block_size: int = 4,
    blocks_per_chunk: int = 1,
    tokens_per_hash: int | None = None,
    sliding_window_size_in_chunks: int | None = None,
) -> GroupOffloadConfig:
    if tokens_per_hash is None:
        tokens_per_hash = block_size
    tokens_per_chunk = block_size * blocks_per_chunk
    assert tokens_per_chunk % tokens_per_hash == 0
    return GroupOffloadConfig(
        group_idx=group_idx,
        tokens_per_block=block_size,
        tokens_per_chunk=tokens_per_chunk,
        hashes_per_chunk=tokens_per_chunk // tokens_per_hash,
        sliding_window_size_in_chunks=sliding_window_size_in_chunks,
        kv_event_group_spec=_FULL_ATTENTION_EVENT_SPEC,
    )


def _record_chunks(
    tracker: OffloadingEventsTracker,
    req,
    group_config: GroupOffloadConfig,
    num_chunks: int,
) -> list[OffloadKey]:
    tracker.on_new_request(req.req_context, req, (group_config,))
    keys: list[OffloadKey] = []
    hbf = group_config.hashes_per_chunk
    for chunk_idx in range(num_chunks):
        tail_hash = req.block_hashes[(chunk_idx + 1) * hbf - 1]
        assert tail_hash is not None
        key = make_offload_key(tail_hash, group_config.group_idx)
        keys.append(key)
    return keys


def _record_lookup_chunks(
    tracker: OffloadingEventsTracker,
    req,
    group_config: GroupOffloadConfig,
    num_chunks: int,
) -> list[OffloadKey]:
    return _record_chunks(tracker, req, group_config, num_chunks)


def _stored_event(
    keys: list[OffloadKey],
    medium: Medium = _CPU_MEDIUM,
    locality: Locality | None = None,
    req_context: ReqContext | None = None,
) -> OffloadingEvent:
    return OffloadingEvent(
        keys=keys,
        medium=medium,
        removed=False,
        locality=locality,
        req_context=req_context,
    )


def _removed_event(
    keys: list[OffloadKey],
    medium: Medium = _CPU_MEDIUM,
    locality: Locality | None = None,
) -> OffloadingEvent:
    return OffloadingEvent(
        keys=keys,
        medium=medium,
        removed=True,
        locality=locality,
    )


def _lookup_chunk() -> tuple[
    OffloadingEventsTracker, MagicMock, GroupOffloadConfig, OffloadKey
]:
    tracker = _tracker()
    req = _request(block_hashes=[_hash(0)], token_count=4)
    group_config = _group_config()
    key = _record_lookup_chunks(
        tracker,
        req,
        group_config,
        num_chunks=1,
    )[0]
    return tracker, req, group_config, key


def test_take_events_forwards_locality_to_rich_store():
    tracker = _tracker()
    req = _request(block_hashes=[_hash(0)], token_count=4)
    key = _record_chunks(tracker, req, _group_config(), num_chunks=1)[0]

    events = list(
        tracker.take_events(
            [
                _stored_event(
                    [key],
                    locality=Locality.LOCAL,
                    medium=Medium.STORAGE,
                    req_context=req.req_context,
                )
            ]
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
            [
                _stored_event(
                    [key],
                    locality=Locality.REMOTE,
                    medium=Medium.STORAGE,
                    req_context=req.req_context,
                )
            ]
        )
    )

    assert len(events) == 1
    assert isinstance(events[0], BlockStored)
    assert events[0].block_size == 0
    assert events[0].locality == "REMOTE"


def test_partial_tail_event_describes_hash_aligned_physical_block_prefix():
    tracker = _tracker()
    group_config = _group_config(block_size=16, blocks_per_chunk=1)._replace(
        hashes_per_chunk=4
    )
    req = _request(block_hashes=[_hash(i) for i in range(8)], token_count=32)
    key = make_offload_key(req.block_hashes[6], group_config.group_idx)

    tracker.on_new_request(
        req.req_context,
        req,
        (group_config,),
        supports_partial_tail=True,
    )
    [event] = tracker.take_events([_stored_event([key], req_context=req.req_context)])

    assert isinstance(event, BlockStored)
    assert event.block_hashes == [_wire_hash(_hash(i)) for i in range(4, 7)]
    assert event.parent_block_hash == _wire_hash(_hash(3))
    assert event.token_ids == list(range(17, 29))
    assert event.block_size == 4


def test_partial_tail_event_uses_its_exact_request_context():
    tracker = _tracker()
    group_config = _group_config()
    stored_req = _request(block_hashes=[_hash(0)], token_count=4)
    lookup_req = _request(block_hashes=[_hash(0)], token_count=4)
    lookup_req.all_token_ids = [9, 9, 9, 9]
    key = make_offload_key(stored_req.block_hashes[0], group_config.group_idx)

    tracker.on_new_request(
        stored_req.req_context,
        stored_req,
        (group_config,),
        supports_partial_tail=True,
    )
    tracker.on_new_request(
        lookup_req.req_context,
        lookup_req,
        (group_config,),
        supports_partial_tail=True,
    )
    [event] = tracker.take_events(
        [_stored_event([key], req_context=stored_req.req_context)]
    )

    assert isinstance(event, BlockStored)
    assert event.token_ids == [1, 2, 3, 4]


def test_partial_tail_sliding_window_event_uses_placeholder():
    tracker = _tracker()
    group_config = _group_config(sliding_window_size_in_chunks=1)
    req = _request(block_hashes=[_hash(0)], token_count=4)
    key = make_offload_key(req.block_hashes[0], group_config.group_idx)

    tracker.on_new_request(req.req_context, req, (group_config,))
    [event] = tracker.take_events([_stored_event([key], req_context=req.req_context)])

    assert isinstance(event, BlockStored)
    assert event.block_hashes == [_wire_hash(_hash(0))]
    assert event.token_ids == []
    assert event.block_size == 0


def test_take_events_forwards_locality_to_remove():
    tracker = _tracker()
    req = _request(block_hashes=[_hash(0)], token_count=4)
    key = _record_chunks(tracker, req, _group_config(), num_chunks=1)[0]

    events = list(
        tracker.take_events(
            [_removed_event([key], locality=Locality.LOCAL, medium=Medium.STORAGE)]
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

    batch1 = list(
        tracker.take_events([_stored_event(keys[:3], req_context=req.req_context)])
    )
    assert len(batch1) == 3

    for i, event in enumerate(batch1):
        assert isinstance(event, BlockStored)
        assert event.medium == _CPU_MEDIUM.value
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
        assert event.extra_keys == [None]
        assert event.group_idx == 0
        assert event.kv_cache_spec_kind == KVCacheSpecKind.FULL_ATTENTION.value
        assert event.kv_cache_spec_sliding_window is None

    batch2 = list(
        tracker.take_events([_stored_event(keys[3:], req_context=req.req_context)])
    )
    assert len(batch2) == 3
    assert batch2[0].parent_block_hash == batch1[-1].block_hashes[-1]

    assert len(tracker._removal_metadata) == 6


def test_promotion_emits_full_cpu_stored_event():
    tracker, req, _, key = _lookup_chunk()

    [event] = tracker.take_events([_stored_event([key], req_context=req.req_context)])

    assert isinstance(event, BlockStored)
    assert event.medium == MEDIUM_CPU
    assert event.block_hashes == [_wire_hash(_hash(0))]
    assert event.parent_block_hash is None
    assert event.token_ids == [1, 2, 3, 4]
    assert event.block_size == 4
    assert event.lora_id is None
    assert event.lora_name is None
    assert event.extra_keys == [None]
    assert event.group_idx == 0
    assert event.kv_cache_spec_kind == KVCacheSpecKind.FULL_ATTENTION.value
    assert event.kv_cache_spec_sliding_window is None


def test_request_event_context_builds_payload_lazily():
    tracker = _tracker()
    req = _request(block_hashes=[_hash(0)], token_count=4)
    [key] = _record_chunks(tracker, req, _group_config(), num_chunks=1)
    state = tracker._request_event_context(req.req_context)

    assert state is not None
    assert not state.locators

    list(tracker.take_events([_stored_event([key], req_context=req.req_context)]))

    assert state.locators[key] == 4


def test_request_event_context_indexes_only_valid_group_chunk_tails():
    tracker = _tracker()
    group0 = _group_config(group_idx=0, block_size=4, blocks_per_chunk=2)
    group1 = _group_config(
        group_idx=1,
        block_size=8,
        blocks_per_chunk=2,
        tokens_per_hash=4,
    )
    req = _request(block_hashes=[_hash(i) for i in range(4)], token_count=16)
    tracker.on_new_request(req.req_context, req, (group0, group1))
    state = tracker._request_event_context(req.req_context)
    assert state is not None
    assert not state.locators

    key0 = make_offload_key(req.block_hashes[1], group0.group_idx)
    key1 = make_offload_key(req.block_hashes[3], group1.group_idx)
    events = list(
        tracker.take_events([_stored_event([key0, key1], req_context=req.req_context)])
    )

    assert [(event.group_idx, len(event.token_ids)) for event in events] == [
        (0, 8),
        (1, 16),
    ]
    assert set(state.locators) == {
        key0,
        make_offload_key(req.block_hashes[3], group0.group_idx),
        key1,
    }


def test_stored_event_uses_context_carried_by_raw_event():
    tracker = _tracker()
    group_config = _group_config()
    first_req = _request(block_hashes=[_hash(0)], token_count=4)
    second_req = _request(block_hashes=[_hash(1)], token_count=4)
    second_req.all_token_ids = [9, 9, 9, 9]
    [first_key] = _record_chunks(tracker, first_req, group_config, num_chunks=1)
    [second_key] = _record_chunks(tracker, second_req, group_config, num_chunks=1)

    [first_event] = tracker.take_events(
        [_stored_event([first_key], req_context=first_req.req_context)]
    )
    [second_event] = tracker.take_events(
        [_stored_event([second_key], req_context=second_req.req_context)]
    )

    assert first_req.request_id == second_req.request_id
    assert first_req.req_context is not second_req.req_context
    assert first_event.token_ids == [1, 2, 3, 4]
    assert second_event.token_ids == [9, 9, 9, 9]


@pytest.mark.parametrize(
    "event_context",
    [None, ReqContext("external")],
    ids=["missing", "external"],
)
def test_stored_event_without_matching_context_uses_placeholder(event_context):
    tracker = _tracker()
    req = _request(block_hashes=[_hash(0)], token_count=4)
    [key] = _record_chunks(tracker, req, _group_config(), num_chunks=1)

    [event] = tracker.take_events([_stored_event([key], req_context=event_context)])

    assert isinstance(event, BlockStored)
    assert event.block_hashes == [_wire_hash(_hash(0))]
    assert event.parent_block_hash is None
    assert event.token_ids == []
    assert event.block_size == 0


@pytest.mark.parametrize("unsafe_path", ["resumable", "mamba_truncation"])
def test_token_mutating_request_uses_placeholder(unsafe_path):
    tracker = _tracker()
    req = _request(block_hashes=[_hash(0)], token_count=4)
    [key] = _record_chunks(tracker, req, _group_config(), num_chunks=1)
    if unsafe_path == "resumable":
        req.resumable = True
    else:
        req.kv_transfer_params = {"_p_side_truncated": True}

    [event] = tracker.take_events([_stored_event([key], req_context=req.req_context)])

    assert event.block_hashes == [_wire_hash(_hash(0))]
    assert event.token_ids == []
    assert event.block_size == 0


def test_full_event_extra_keys_match_gpu_block_granularity():
    tracker = _tracker()
    req = _request(block_hashes=[_hash(0), _hash(1)], token_count=8)
    req.cache_salt = "salt"
    req.lora_request = SimpleNamespace(
        adapter_id=7,
        name="adapter",
        lora_name="adapter",
    )
    [key] = _record_chunks(
        tracker,
        req,
        _group_config(blocks_per_chunk=2),
        num_chunks=1,
    )

    [event] = tracker.take_events([_stored_event([key], req_context=req.req_context)])

    assert event.extra_keys == [("adapter", "salt"), ("adapter",)]
    assert event.lora_id == 7
    assert event.lora_name == "adapter"


def test_full_event_extra_keys_include_prompt_embeddings():
    tracker = _tracker()
    group_config = _group_config(blocks_per_chunk=2)
    req = _request(block_hashes=[_hash(0), _hash(1)], token_count=8)
    req.prompt_embeds = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    _rehash_request(req, block_size=4)
    [key] = _record_chunks(tracker, req, group_config, num_chunks=1)

    [event] = tracker.take_events([_stored_event([key], req_context=req.req_context)])

    expected = []
    mm_idx = 0
    for start in (0, 4):
        extra_keys, mm_idx = generate_block_hash_extra_keys(
            req, start, start + 4, mm_idx
        )
        expected.append(extra_keys)
    assert event.block_hashes == [_wire_hash(value) for value in req.block_hashes]
    assert event.token_ids == req.all_token_ids
    assert event.extra_keys == expected


def test_partial_event_extra_keys_include_cache_salt_and_lora():
    tracker = _tracker()
    group_config = _group_config(
        block_size=16,
        tokens_per_hash=4,
    )
    req = _request(block_hashes=[_hash(i) for i in range(4)], token_count=16)
    req.cache_salt = "salt"
    req.lora_request = SimpleNamespace(
        adapter_id=7,
        name="adapter",
        lora_name="adapter",
    )
    key = make_offload_key(req.block_hashes[2], group_config.group_idx)

    tracker.on_new_request(
        req.req_context,
        req,
        (group_config,),
        supports_partial_tail=True,
    )
    [event] = tracker.take_events([_stored_event([key], req_context=req.req_context)])

    assert event.extra_keys == [
        ("adapter", "salt"),
        ("adapter",),
        ("adapter",),
    ]
    assert event.lora_id == 7
    assert event.lora_name == "adapter"


def test_partial_event_extra_keys_replay_mm_features_from_request_start():
    tracker = _tracker()
    group_config = _group_config(
        block_size=16,
        tokens_per_hash=4,
    )
    req = _request(block_hashes=[_hash(i) for i in range(8)], token_count=32)
    req.mm_features = [
        SimpleNamespace(
            identifier="A",
            mm_position=SimpleNamespace(offset=18, length=4),
        ),
        SimpleNamespace(
            identifier="B",
            mm_position=SimpleNamespace(offset=40, length=4),
        ),
    ]
    _rehash_request(req, block_size=4)
    key = make_offload_key(req.block_hashes[6], group_config.group_idx)
    tracker.on_new_request(
        req.req_context,
        req,
        (group_config,),
        supports_partial_tail=True,
    )
    [event] = tracker.take_events([_stored_event([key], req_context=req.req_context)])

    assert event.block_hashes == [_wire_hash(value) for value in req.block_hashes[4:7]]
    assert event.token_ids == req.all_token_ids[16:28]
    assert event.extra_keys == [(("A", 2),), (("A", -2),), None]


@pytest.mark.parametrize(
    ("blocks_per_chunk", "expected_hash_indices"),
    [(1, [63]), (2, [63, 127])],
)
def test_event_hashes_use_group_block_size(
    blocks_per_chunk: int, expected_hash_indices: list[int]
):
    tokens_per_hash = 4
    block_size = 256
    hashes_per_block = block_size // tokens_per_hash
    tracker = _tracker()
    group_config = _group_config(
        block_size=block_size,
        blocks_per_chunk=blocks_per_chunk,
        tokens_per_hash=tokens_per_hash,
    )
    req = _request(
        block_hashes=[_hash(i) for i in range(hashes_per_block * blocks_per_chunk)],
        token_count=block_size * blocks_per_chunk,
    )
    [key] = _record_chunks(tracker, req, group_config, num_chunks=1)

    [event] = tracker.take_events([_stored_event([key], req_context=req.req_context)])

    assert isinstance(event, BlockStored)
    assert event.block_hashes == [_wire_hash(_hash(i)) for i in expected_hash_indices]
    assert event.block_size == block_size
    assert len(event.token_ids) == block_size * blocks_per_chunk


def test_lookup_promotion_factor_gt_1_store_and_remove():
    block_size = 4
    blocks_per_chunk = 2
    tracker = _tracker()
    group_config = _group_config(
        block_size=block_size, blocks_per_chunk=blocks_per_chunk
    )
    req = _request(
        block_hashes=[_hash(i) for i in range(4)],
        token_count=block_size * blocks_per_chunk * 2,
    )
    keys = _record_lookup_chunks(tracker, req, group_config, num_chunks=2)

    stored = list(
        tracker.take_events([_stored_event(keys, req_context=req.req_context)])
    )
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

    assert len(tracker._removal_metadata) == 2

    removed = list(tracker.take_events([_removed_event(keys)]))
    assert len(removed) == 1
    assert isinstance(removed[0], BlockRemoved)
    assert removed[0].block_hashes == expected_hashes
    assert removed[0].medium == _CPU_MEDIUM.value
    assert removed[0].group_idx == 0
    assert not tracker._removal_metadata


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

    events = list(
        tracker.take_events(
            [
                _stored_event(
                    [keys[1], unknown_key, keys[0]],
                    req_context=req.req_context,
                )
            ]
        )
    )

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
    _record_lookup_chunks(tracker, req, group_config, num_chunks=3)

    assert not tracker.self_describing_enabled
    assert tracker._request_event_context(req.req_context) is None

    events = list(
        tracker.take_events(
            [
                _stored_event(keys, req_context=req.req_context),
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
    _record_lookup_chunks(tracker, req, group_config, num_chunks=3)

    state = tracker._request_event_context(req.req_context)
    assert state is not None
    assert not state.locators

    events = list(
        tracker.take_events([_stored_event(keys[:1], req_context=req.req_context)])
    )
    assert len(events) == 1
    assert isinstance(events[0], BlockStored)
    assert events[0].block_size == 0


def test_pending_cpu_removal_consumes_hit_backfill_until_next_hit():
    tracker = _tracker()
    block_hashes = [_hash(0), _hash(1)]
    req = _request(block_hashes=block_hashes, token_count=8)
    group_config = _group_config(blocks_per_chunk=2)
    key = _record_chunks(tracker, req, group_config, num_chunks=1)[0]
    tracker.record_hit(req.req_context, key)
    lookup_req = _request(
        block_hashes=block_hashes,
        token_count=8,
        req_id="new-request",
    )

    removed = list(tracker.take_events([_removed_event([key])]))
    assert len(removed) == 1
    assert removed[0].block_hashes == [
        _wire_hash(_hash(0)),
        _wire_hash(_hash(1)),
    ]

    tracker.on_new_request(lookup_req.req_context, lookup_req, (group_config,))
    tracker.record_hit(lookup_req.req_context, key)
    removed = list(tracker.take_events([_removed_event([key])]))
    assert removed[0].block_hashes == [
        _wire_hash(_hash(0)),
        _wire_hash(_hash(1)),
    ]


def test_storage_stored_event_does_not_create_legacy_removal_metadata():
    tracker, req, _, key = _lookup_chunk()

    stored = list(
        tracker.take_events(
            [
                _stored_event(
                    [key],
                    Medium.STORAGE,
                    req_context=req.req_context,
                )
            ]
        )
    )
    assert stored[0].token_ids == [1, 2, 3, 4]
    assert not tracker._removal_metadata


def test_take_events_groups_removed_hashes_by_kv_group():
    tracker = _tracker()
    group0_config = _group_config(group_idx=0, blocks_per_chunk=2)
    group1_config = _group_config(group_idx=1, blocks_per_chunk=2)
    req0 = _request(block_hashes=[_hash(0), _hash(1)], token_count=8)
    req1 = _request(block_hashes=[_hash(10), _hash(11)], token_count=8)
    key0 = _record_chunks(tracker, req0, group0_config, num_chunks=1)[0]
    key1 = _record_chunks(tracker, req1, group1_config, num_chunks=1)[0]
    list(tracker.take_events([_stored_event([key0], req_context=req0.req_context)]))
    list(tracker.take_events([_stored_event([key1], req_context=req1.req_context)]))

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

    first_store = list(
        tracker.take_events([_stored_event([key], req_context=req.req_context)])
    )
    assert len(first_store) == 1
    assert isinstance(first_store[0], BlockStored)
    assert first_store[0].token_ids == [1, 2, 3, 4]

    removed = list(tracker.take_events([_removed_event([key])]))
    assert len(removed) == 1
    assert isinstance(removed[0], BlockRemoved)
    assert not tracker._removal_metadata

    replacement_req = _request(
        block_hashes=[_hash(0)],
        token_count=block_size,
        req_id=req.request_id,
    )
    tracker.on_new_request(
        replacement_req.req_context, replacement_req, (group_config,)
    )

    second_store = list(
        tracker.take_events(
            [_stored_event([key], req_context=replacement_req.req_context)]
        )
    )
    assert len(second_store) == 1
    assert isinstance(second_store[0], BlockStored)
    assert second_store[0].token_ids == [1, 2, 3, 4]


def test_reset_keeps_request_context_for_secondary_events():
    tracker = _tracker()
    group_config = _group_config()
    req = _request(block_hashes=[_hash(i) for i in range(3)], token_count=12)
    _record_lookup_chunks(tracker, req, group_config, num_chunks=3)

    key = make_offload_key(req.block_hashes[0], group_config.group_idx)
    tracker.record_hit(req.req_context, key)
    assert tracker._request_event_context(req.req_context) is not None
    assert tracker._removal_metadata

    tracker.reset()

    assert tracker._request_event_context(req.req_context) is not None
    assert not tracker._removal_metadata

    key = make_offload_key(req.block_hashes[0], group_config.group_idx)
    [event] = tracker.take_events(
        [_stored_event([key], medium=Medium.STORAGE, req_context=req.req_context)]
    )
    assert event.token_ids == [1, 2, 3, 4]


def test_tiering_accepts_self_describing_kv_events():
    vllm_config = create_vllm_config(
        block_size=4,
        max_num_batched_tokens=16,
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
