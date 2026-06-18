# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import pytest

import vllm.v1.core.kv_cache_utils as kv_cache_utils
from vllm.distributed.kv_events import BlockRemoved, BlockStored
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
    get_request_block_hasher,
    hash_block_tokens,
    init_none_hash,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _auto_init_hash_fn():
    init_none_hash(sha256)


def make_request(
    request_id: str,
    prompt_token_ids: list[int],
    partial_cache_unit: int,
    hash_fn: Callable,
) -> Request:
    sampling_params = SamplingParams(max_tokens=17)
    sampling_params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(partial_cache_unit, hash_fn),
    )


def cache_full_block_and_tail_alias(
    token_ids: list[int],
    *,
    enable_kv_cache_events: bool = False,
) -> tuple[BlockPool, Request, list[KVCacheBlock], BlockHash]:
    partial_cache_unit = 2
    block_size = 6
    kv_cache_group_id = 0
    req = make_request("0", token_ids, partial_cache_unit, sha256)
    pool = BlockPool(
        num_gpu_blocks=3,
        enable_caching=True,
        hash_block_size=block_size,
        partial_cache_unit=partial_cache_unit,
        enable_kv_cache_events=enable_kv_cache_events,
    )
    blocks = pool.get_new_blocks(2)

    pool.cache_full_blocks(
        request=req,
        blocks=blocks,
        num_cached_blocks=0,
        num_full_blocks=1,
        block_size=block_size,
        kv_cache_group_id=kv_cache_group_id,
    )
    partial_hash = req.block_hashes.get_partial_block_hash(block_size, len(token_ids))
    assert pool.cache_block_alias(
        request=req,
        block=blocks[1],
        num_tokens=len(token_ids),
        kv_cache_group_id=kv_cache_group_id,
        block_size=block_size,
    )
    return pool, req, blocks, partial_hash


def test_partial_tail_hash_uses_previous_full_block_parent():
    partial_cache_unit = 2
    block_size = 6
    token_ids = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    req = make_request("0", token_ids, partial_cache_unit, sha256)

    full_hashes = req.block_hashes.get_block_hashes(block_size)
    tail_hash = req.block_hashes.get_partial_block_hash(block_size, 10)
    assert tail_hash == hash_block_tokens(
        sha256,
        full_hashes[0],
        token_ids[block_size : block_size + 2 * partial_cache_unit],
    )
    shorter_tail_hash = hash_block_tokens(
        sha256,
        full_hashes[0],
        token_ids[block_size : block_size + partial_cache_unit],
    )
    assert tail_hash != hash_block_tokens(
        sha256,
        shorter_tail_hash,
        token_ids[
            block_size + partial_cache_unit : block_size + 2 * partial_cache_unit
        ],
    )


def test_cache_block_alias_kv_cache_events():
    partial_cache_unit = 4
    block_size = partial_cache_unit
    kv_cache_group_id = 2

    pool = BlockPool(
        num_gpu_blocks=2,
        enable_caching=True,
        hash_block_size=block_size,
        partial_cache_unit=partial_cache_unit,
        enable_kv_cache_events=True,
    )
    req = make_request(
        "req_alias_events",
        prompt_token_ids=list(range(partial_cache_unit * 2)),
        partial_cache_unit=partial_cache_unit,
        hash_fn=sha256,
    )

    block = pool.get_new_blocks(1)[0]
    alias_hash = pool.cache_block_alias(
        request=req,
        block=block,
        num_tokens=partial_cache_unit * 2,
        kv_cache_group_id=kv_cache_group_id,
    )

    events = pool.take_events()
    assert len(events) == 1
    stored_event = events[0]
    assert isinstance(stored_event, BlockStored)
    assert alias_hash is not None
    assert stored_event.block_hashes == [
        kv_cache_utils.maybe_convert_block_hash(req.block_hashes[1])
    ]
    assert stored_event.parent_block_hash == kv_cache_utils.maybe_convert_block_hash(
        req.block_hashes[0]
    )
    assert stored_event.token_ids == req.all_token_ids[partial_cache_unit:]
    assert stored_event.block_size == 4
    assert stored_event.group_idx == kv_cache_group_id

    duplicate_alias_hash = pool.cache_block_alias(
        request=req,
        block=block,
        num_tokens=partial_cache_unit * 2,
        kv_cache_group_id=kv_cache_group_id,
    )
    assert duplicate_alias_hash == alias_hash
    assert pool.take_events() == []

    pool.free_blocks([block])
    pool.get_new_blocks(1)
    events = pool.take_events()
    assert len(events) == 1
    removed_event = events[0]
    assert isinstance(removed_event, BlockRemoved)
    assert removed_event.block_hashes == stored_event.block_hashes
    assert removed_event.group_idx == kv_cache_group_id


def test_cache_block_alias_disabled_without_partial_cache_unit():
    partial_cache_unit = 4
    block_size = partial_cache_unit
    req = make_request(
        "req_alias_disabled",
        prompt_token_ids=list(range(partial_cache_unit * 2)),
        partial_cache_unit=partial_cache_unit,
        hash_fn=sha256,
    )
    pool = BlockPool(
        num_gpu_blocks=2,
        enable_caching=True,
        hash_block_size=block_size,
    )
    block = pool.get_new_blocks(1)[0]

    alias_hash = pool.cache_block_alias(
        request=req,
        block=block,
        num_tokens=partial_cache_unit * 2,
        kv_cache_group_id=0,
    )

    assert alias_hash is None
    assert pool.take_events() == []


def test_partial_cache_unit_must_not_exceed_hash_block_size():
    with pytest.raises(AssertionError):
        BlockPool(
            num_gpu_blocks=2,
            enable_caching=True,
            hash_block_size=4,
            partial_cache_unit=8,
        )


def test_partial_block_replacement_emits_remove_then_store_events():
    partial_cache_unit = 2
    block_size = 6
    kv_cache_group_id = 0
    req = make_request("0", [0, 0, 1, 1, 2, 2, 3, 3], partial_cache_unit, sha256)
    pool = BlockPool(
        num_gpu_blocks=3,
        enable_caching=True,
        hash_block_size=block_size,
        partial_cache_unit=partial_cache_unit,
        enable_kv_cache_events=True,
    )
    blocks = pool.get_new_blocks(2)

    pool.cache_full_blocks(
        request=req,
        blocks=blocks,
        num_cached_blocks=0,
        num_full_blocks=1,
        block_size=block_size,
        kv_cache_group_id=kv_cache_group_id,
    )
    partial_hash_8 = req.block_hashes.get_partial_block_hash(block_size, 8)
    assert pool.cache_block_alias(
        request=req,
        block=blocks[1],
        num_tokens=8,
        kv_cache_group_id=kv_cache_group_id,
        block_size=block_size,
    )
    assert pool.get_cached_block(partial_hash_8, [kv_cache_group_id]) == [blocks[1]]
    pool.take_events()

    req.append_output_token_ids([4, 4])
    partial_hash_10 = req.block_hashes.get_partial_block_hash(block_size, 10)
    assert pool.cache_block_alias(
        request=req,
        block=blocks[1],
        num_tokens=10,
        kv_cache_group_id=kv_cache_group_id,
        block_size=block_size,
    )
    events = pool.take_events()

    assert len(events) == 2
    removed_event, stored_event = events
    assert isinstance(removed_event, BlockRemoved)
    assert removed_event.block_hashes == [
        kv_cache_utils.maybe_convert_block_hash(partial_hash_8)
    ]
    assert removed_event.group_idx == kv_cache_group_id
    assert isinstance(stored_event, BlockStored)
    assert stored_event.block_hashes == [
        kv_cache_utils.maybe_convert_block_hash(partial_hash_10)
    ]
    assert stored_event.token_ids == req.all_token_ids[block_size:10]
    assert stored_event.block_size == 4
    assert stored_event.group_idx == kv_cache_group_id
    assert pool.get_cached_block(partial_hash_8, [kv_cache_group_id]) is None
    assert pool.get_cached_block(partial_hash_10, [kv_cache_group_id]) == [blocks[1]]


def test_later_request_hits_cached_partial_tail_alias():
    partial_cache_unit = 2
    block_size = 6
    kv_cache_group_id = 0
    cached_token_ids = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    req = make_request("0", cached_token_ids, partial_cache_unit, sha256)
    pool = BlockPool(
        num_gpu_blocks=3,
        enable_caching=True,
        hash_block_size=block_size,
        partial_cache_unit=partial_cache_unit,
    )
    blocks = pool.get_new_blocks(2)

    pool.cache_full_blocks(
        request=req,
        blocks=blocks,
        num_cached_blocks=0,
        num_full_blocks=1,
        block_size=block_size,
        kv_cache_group_id=kv_cache_group_id,
    )
    partial_hash_10 = req.block_hashes.get_partial_block_hash(block_size, 10)
    assert pool.cache_block_alias(
        request=req,
        block=blocks[1],
        num_tokens=10,
        kv_cache_group_id=kv_cache_group_id,
        block_size=block_size,
    )

    replay = make_request("1", cached_token_ids, partial_cache_unit, sha256)
    replay_hash_10 = replay.block_hashes.get_partial_block_hash(block_size, 10)
    assert replay_hash_10 == partial_hash_10
    assert pool.get_cached_block(replay_hash_10, [kv_cache_group_id]) == [blocks[1]]

    extended = make_request("2", cached_token_ids + [10], partial_cache_unit, sha256)
    extended_hash_10 = extended.block_hashes.get_partial_block_hash(block_size, 10)
    assert extended_hash_10 == partial_hash_10
    assert pool.get_cached_block(extended_hash_10, [kv_cache_group_id]) == [blocks[1]]


def test_cache_block_alias_only_computes_requested_partial_hash(monkeypatch):
    partial_cache_unit = 2
    block_size = 6
    kv_cache_group_id = 0
    token_ids = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    req = make_request("0", token_ids, partial_cache_unit, sha256)
    pool = BlockPool(
        num_gpu_blocks=3,
        enable_caching=True,
        hash_block_size=block_size,
        partial_cache_unit=partial_cache_unit,
    )
    blocks = pool.get_new_blocks(2)

    pool.cache_full_blocks(
        request=req,
        blocks=blocks,
        num_cached_blocks=0,
        num_full_blocks=1,
        block_size=block_size,
        kv_cache_group_id=kv_cache_group_id,
    )

    original_get_partial_block_hash = req._block_hasher.get_partial_block_hash
    requested_num_tokens: list[int] = []

    def get_requested_partial_hash(
        request: Request, block_size: int, num_tokens: int
    ) -> BlockHash:
        requested_num_tokens.append(num_tokens)
        return original_get_partial_block_hash(request, block_size, num_tokens)

    monkeypatch.setattr(
        req._block_hasher,
        "get_partial_block_hash",
        get_requested_partial_hash,
    )
    assert pool.cache_block_alias(
        request=req,
        block=blocks[1],
        num_tokens=10,
        kv_cache_group_id=kv_cache_group_id,
        block_size=block_size,
    )
    assert requested_num_tokens == [10]


def test_reset_prefix_cache_clears_partial_alias_metadata():
    pool, req, blocks, partial_hash_10 = cache_full_block_and_tail_alias(
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    )

    assert pool.get_cached_block(req.block_hashes.get_block_hashes(6)[0], [0]) == [
        blocks[0]
    ]
    assert pool.get_cached_block(partial_hash_10, [0]) == [blocks[1]]

    pool.free_blocks(blocks)
    assert pool.reset_prefix_cache()

    assert pool.get_cached_block(req.block_hashes.get_block_hashes(6)[0], [0]) is None
    assert pool.get_cached_block(partial_hash_10, [0]) is None
    assert pool.cached_block_hashes_by_block == {}


def test_evict_cached_block_removes_full_hash_and_partial_alias():
    pool, req, blocks, partial_hash_10 = cache_full_block_and_tail_alias(
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    )
    full_hash = req.block_hashes.get_block_hashes(6)[0]

    assert pool.get_cached_block(full_hash, [0]) == [blocks[0]]
    assert pool.get_cached_block(partial_hash_10, [0]) == [blocks[1]]

    pool.evict_blocks({blocks[0].block_id, blocks[1].block_id})

    assert pool.get_cached_block(full_hash, [0]) is None
    assert pool.get_cached_block(partial_hash_10, [0]) is None
    assert pool.cached_block_hashes_by_block == {}


def test_partial_block_promotes_to_direct_full_block_hash():
    partial_cache_unit = 2
    block_size = 6
    kv_cache_group_id = 0
    token_ids = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    req = make_request("0", token_ids, partial_cache_unit, sha256)
    pool = BlockPool(
        num_gpu_blocks=3,
        enable_caching=True,
        hash_block_size=block_size,
        partial_cache_unit=partial_cache_unit,
    )
    blocks = pool.get_new_blocks(2)

    pool.cache_full_blocks(
        request=req,
        blocks=blocks,
        num_cached_blocks=0,
        num_full_blocks=1,
        block_size=block_size,
        kv_cache_group_id=kv_cache_group_id,
    )
    partial_hash_10 = req.block_hashes.get_partial_block_hash(block_size, 10)
    assert pool.cache_block_alias(
        request=req,
        block=blocks[1],
        num_tokens=10,
        kv_cache_group_id=kv_cache_group_id,
        block_size=block_size,
    )
    assert pool.get_cached_block(partial_hash_10, [kv_cache_group_id]) == [blocks[1]]

    req.append_output_token_ids([5, 5])
    full_hashes = req.block_hashes.get_block_hashes(block_size)
    promoted_full_hash = full_hashes[1]
    concat_hash = BlockHash(
        req.block_hashes[3] + req.block_hashes[4] + req.block_hashes[5]
    )
    assert promoted_full_hash != concat_hash
    assert promoted_full_hash == hash_block_tokens(
        sha256,
        full_hashes[0],
        req.all_token_ids[block_size : 2 * block_size],
    )

    pool.cache_full_blocks(
        request=req,
        blocks=blocks,
        num_cached_blocks=1,
        num_full_blocks=2,
        block_size=block_size,
        kv_cache_group_id=kv_cache_group_id,
    )
    assert pool.get_cached_block(promoted_full_hash, [kv_cache_group_id]) == [blocks[1]]
    assert pool.get_cached_block(partial_hash_10, [kv_cache_group_id]) is None
