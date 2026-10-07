# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pins the KV cache event wire contract documented in
docs/features/kv_events.md.

External routers and cache indexes decode these events. A failure here means
a change breaks them, or leaves that page out of date.
"""

from collections import Counter
from typing import Any

import msgspec
import pytest

from tests.v1.core.utils import create_requests
from vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
    KVEventBatch,
)
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import ExternalBlockHash

pytestmark = pytest.mark.cpu_test

EVENT_TYPES = (BlockStored, BlockRemoved, AllBlocksCleared)

# Fields on the wire in every event, possibly nil. Consumers rely on them, so
# they never change.
REQUIRED_FIELDS: dict[type[KVCacheEvent], dict[str, Any]] = {
    BlockStored: {
        "block_hashes": list[ExternalBlockHash],
        "parent_block_hash": ExternalBlockHash | None,
        "token_ids": list[int],
        "block_size": int,
        "lora_id": int | None,
        "medium": str | None,
        "lora_name": str | None,
    },
    BlockRemoved: {
        "block_hashes": list[ExternalBlockHash],
        "medium": str | None,
    },
    AllBlocksCleared: {},
}

# Fields omitted from the wire when unset. A new field is added here and to
# docs/features/kv_events.md.
OPTIONAL_FIELDS: dict[type[KVCacheEvent], dict[str, Any]] = {
    BlockStored: {
        "extra_keys": list[tuple[Any, ...] | None] | None,
        "group_idx": int | None,
        "kv_cache_spec_kind": str | None,
        "kv_cache_spec_sliding_window": int | None,
        "locality": str | None,
        "ownership": str | None,
        "session_id": str | None,
    },
    BlockRemoved: {
        "group_idx": int | None,
        "locality": str | None,
        "ownership": str | None,
    },
    AllBlocksCleared: {},
}


def _fields(event_type: type[KVCacheEvent], required: bool) -> dict[str, Any]:
    return {
        field.name: field.type
        for field in msgspec.structs.fields(event_type)
        if (field.default is msgspec.NODEFAULT) == required
    }


@pytest.mark.parametrize("event_type", EVENT_TYPES)
def test_required_fields_are_frozen(event_type: type[KVCacheEvent]):
    assert _fields(event_type, required=True) == REQUIRED_FIELDS[event_type], (
        "Required KV event fields are part of the wire contract and cannot be "
        "added, removed or retyped. New fields must default to None."
    )


@pytest.mark.parametrize("event_type", EVENT_TYPES)
def test_optional_fields_default_to_none(event_type: type[KVCacheEvent]):
    assert _fields(event_type, required=False) == OPTIONAL_FIELDS[event_type], (
        "Optional KV event fields must be listed here and documented in "
        "docs/features/kv_events.md; existing ones cannot be removed or retyped."
    )
    for field in msgspec.structs.fields(event_type):
        if field.default is not msgspec.NODEFAULT:
            assert field.default is None, field.name


def test_batch_wire_shape():
    batch = KVEventBatch(
        ts=1.5,
        events=[
            BlockStored(
                block_hashes=[11, 2**64 - 1],
                parent_block_hash=10,
                token_ids=[1, 2, 3, 4],
                block_size=2,
                lora_id=None,
                medium="GPU",
                lora_name="adapter",
                extra_keys=[("adapter",), ("adapter", ("mm-id", 0))],
                group_idx=0,
                kv_cache_spec_kind="full_attention",
                locality="LOCAL",
                session_id="session",
            ),
            BlockRemoved(
                block_hashes=[11], medium="CPU", group_idx=0, ownership="kvcr"
            ),
            AllBlocksCleared(),
        ],
        data_parallel_rank=1,
    )

    assert msgspec.msgpack.decode(msgspec.msgpack.encode(batch)) == [
        1.5,
        [
            {
                "type": "BlockStored",
                "block_hashes": [11, 2**64 - 1],
                "parent_block_hash": 10,
                "token_ids": [1, 2, 3, 4],
                "block_size": 2,
                "lora_id": None,
                "medium": "GPU",
                "lora_name": "adapter",
                "extra_keys": [["adapter"], ["adapter", ["mm-id", 0]]],
                "group_idx": 0,
                "kv_cache_spec_kind": "full_attention",
                "locality": "LOCAL",
                "session_id": "session",
            },
            {
                "type": "BlockRemoved",
                "block_hashes": [11],
                "medium": "CPU",
                "group_idx": 0,
                "ownership": "kvcr",
            },
            {"type": "AllBlocksCleared"},
        ],
        1,
    ]


def test_decoding_ignores_unknown_fields_and_trailing_elements():
    payload = msgspec.msgpack.encode(
        [
            1.5,
            [{"type": "BlockRemoved", "block_hashes": [11], "medium": "GPU", "new": 1}],
            0,
            "new trailing element",
        ]
    )

    assert msgspec.msgpack.decode(payload, type=KVEventBatch) == KVEventBatch(
        ts=1.5,
        events=[BlockRemoved(block_hashes=[11], medium="GPU")],
        data_parallel_rank=0,
    )


def test_each_copy_of_a_hash_is_stored_and_removed():
    block_size = 4
    pool = BlockPool(
        num_gpu_blocks=5,
        enable_caching=True,
        hash_block_size=block_size,
        enable_kv_cache_events=True,
    )
    (request,) = create_requests(1, num_tokens=2 * block_size, block_size=block_size)

    # Cache two physical copies of the same two hashes.
    copies = [pool.get_new_blocks(2), pool.get_new_blocks(2)]
    for blocks in copies:
        pool.cache_full_blocks(
            request=request,
            blocks=blocks,
            num_cached_blocks=0,
            num_full_blocks=2,
            block_size=block_size,
            kv_cache_group_id=0,
        )
        pool.free_blocks(blocks)
    stored = Counter(
        block_hash
        for event in pool.take_events()
        if isinstance(event, BlockStored)
        for block_hash in event.block_hashes
    )
    assert len(stored) == 2
    assert set(stored.values()) == {2}

    # Freeing emits nothing; reallocating evicts the first copy only.
    assert pool.take_events() == []
    pool.get_new_blocks(2)
    first_removed = Counter(
        block_hash
        for event in pool.take_events()
        if isinstance(event, BlockRemoved)
        for block_hash in event.block_hashes
    )
    assert first_removed == Counter(dict.fromkeys(stored, 1))
    assert len(pool.cached_block_hash_to_block) == 2

    # Evicting the second copy balances every store.
    pool.get_new_blocks(2)
    second_removed = Counter(
        block_hash
        for event in pool.take_events()
        if isinstance(event, BlockRemoved)
        for block_hash in event.block_hashes
    )
    assert first_removed + second_removed == stored
    assert len(pool.cached_block_hash_to_block) == 0
