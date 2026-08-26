# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import pytest
import torch

import vllm.v1.core.kv_cache_utils as kv_cache_utils
from vllm.distributed.kv_events import BlockRemoved, BlockStored
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashListWithBlockSize,
    KVCacheBlock,
    get_request_block_hasher,
    hash_block_tokens,
    init_none_hash,
)
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, MambaManager
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _auto_init_hash_fn():
    init_none_hash(sha256)


def make_request(
    request_id: str,
    prompt_token_ids: list[int],
    hash_block_size: int,
    hash_fn: Callable,
    session_id: str | None = None,
) -> Request:
    sampling_params = SamplingParams(max_tokens=17)
    sampling_params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(hash_block_size, hash_fn),
        session_id=session_id,
    )


def boundary_hash(req: Request, hash_block_size: int, num_tokens: int) -> BlockHash:
    # Every boundary at a hash_block_size multiple is just the fine-grained
    # chain hash ending there.
    return req.block_hashes[num_tokens // hash_block_size - 1]


def cache_full_block_and_partial_tail(
    token_ids: list[int],
    *,
    enable_kv_cache_events: bool = False,
    block_size: int = 6,
) -> tuple[BlockPool, Request, list[KVCacheBlock], BlockHash]:
    hash_block_size = 2
    kv_cache_group_id = 0
    req = make_request("0", token_ids, hash_block_size, sha256)
    pool = BlockPool(
        num_gpu_blocks=3,
        enable_caching=True,
        hash_block_size=hash_block_size,
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
    partial_hash = boundary_hash(req, hash_block_size, len(token_ids))
    assert pool.cache_partial_block(
        request=req,
        block=blocks[1],
        num_tokens=len(token_ids),
        kv_cache_group_id=kv_cache_group_id,
        block_size=block_size,
    )
    return pool, req, blocks, partial_hash


def test_boundary_hashes_reuse_fine_grained_chain():
    hash_block_size = 2
    block_size = 6
    token_ids = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    req = make_request("0", token_ids, hash_block_size, sha256)

    coarse = BlockHashListWithBlockSize(req.block_hashes, hash_block_size, block_size)
    # The block_size=6 full-block hash is the fine hash at the 6-token boundary,
    # not a concatenation of the three fine hashes inside the block.
    assert coarse[0] == req.block_hashes[6 // hash_block_size - 1]
    assert coarse[0] != BlockHash(
        req.block_hashes[0] + req.block_hashes[1] + req.block_hashes[2]
    )
    # A partial tail at 10 tokens is the fine hash at the 10-token boundary,
    # which chains over the entire prefix.
    tail_hash = boundary_hash(req, hash_block_size, 10)
    assert tail_hash == req.block_hashes[4]
    assert tail_hash == hash_block_tokens(sha256, req.block_hashes[3], token_ids[8:10])


def test_cache_partial_block_kv_cache_events():
    hash_block_size = 4
    block_size = 12
    kv_cache_group_id = 2

    pool = BlockPool(
        num_gpu_blocks=2,
        enable_caching=True,
        hash_block_size=hash_block_size,
        enable_kv_cache_events=True,
    )
    req = make_request(
        "req_partial_events",
        prompt_token_ids=list(range(hash_block_size * 2)),
        hash_block_size=hash_block_size,
        hash_fn=sha256,
        session_id="agent-session-partial",
    )

    block = pool.get_new_blocks(1)[0]
    partial_entry_hash = pool.cache_partial_block(
        request=req,
        block=block,
        num_tokens=hash_block_size * 2,
        kv_cache_group_id=kv_cache_group_id,
        block_size=block_size,
    )

    events = pool.take_events()
    assert len(events) == 1
    stored_event = events[0]
    assert isinstance(stored_event, BlockStored)
    assert partial_entry_hash is not None
    assert stored_event.block_hashes == [
        kv_cache_utils.maybe_convert_block_hash(req.block_hashes[1])
    ]
    assert stored_event.parent_block_hash == kv_cache_utils.maybe_convert_block_hash(
        req.block_hashes[0]
    )
    assert stored_event.token_ids == req.all_token_ids[hash_block_size:]
    assert stored_event.block_size == 4
    assert stored_event.group_idx == kv_cache_group_id
    assert stored_event.session_id == "agent-session-partial"

    duplicate_entry_hash = pool.cache_partial_block(
        request=req,
        block=block,
        num_tokens=hash_block_size * 2,
        kv_cache_group_id=kv_cache_group_id,
        block_size=block_size,
    )
    assert duplicate_entry_hash == partial_entry_hash
    assert pool.take_events() == []

    pool.free_blocks([block])
    pool.get_new_blocks(1)
    events = pool.take_events()
    assert len(events) == 1
    removed_event = events[0]
    assert isinstance(removed_event, BlockRemoved)
    assert removed_event.block_hashes == stored_event.block_hashes
    assert removed_event.group_idx == kv_cache_group_id


def test_partial_block_replacement_emits_remove_then_store_events():
    hash_block_size = 2
    block_size = 6
    kv_cache_group_id = 0
    req = make_request("0", [0, 0, 1, 1, 2, 2, 3, 3], hash_block_size, sha256)
    pool = BlockPool(
        num_gpu_blocks=3,
        enable_caching=True,
        hash_block_size=hash_block_size,
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
    partial_hash_8 = boundary_hash(req, hash_block_size, 8)
    assert pool.cache_partial_block(
        request=req,
        block=blocks[1],
        num_tokens=8,
        kv_cache_group_id=kv_cache_group_id,
        block_size=block_size,
    )
    assert pool.get_cached_block(partial_hash_8, [kv_cache_group_id]) == [blocks[1]]
    pool.take_events()

    req.append_output_token_ids([4, 4])
    partial_hash_10 = boundary_hash(req, hash_block_size, 10)
    assert pool.cache_partial_block(
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
    assert stored_event.parent_block_hash == kv_cache_utils.maybe_convert_block_hash(
        boundary_hash(req, hash_block_size, 8)
    )
    assert stored_event.token_ids == req.all_token_ids[8:10]
    assert stored_event.block_size == hash_block_size
    assert stored_event.group_idx == kv_cache_group_id
    assert pool.get_cached_block(partial_hash_8, [kv_cache_group_id]) is None
    assert pool.get_cached_block(partial_hash_10, [kv_cache_group_id]) == [blocks[1]]


def test_later_request_hits_cached_partial_tail():
    hash_block_size = 2
    block_size = 6
    kv_cache_group_id = 0
    cached_token_ids = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    req = make_request("0", cached_token_ids, hash_block_size, sha256)
    pool = BlockPool(
        num_gpu_blocks=3,
        enable_caching=True,
        hash_block_size=hash_block_size,
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
    partial_hash_10 = boundary_hash(req, hash_block_size, 10)
    assert pool.cache_partial_block(
        request=req,
        block=blocks[1],
        num_tokens=10,
        kv_cache_group_id=kv_cache_group_id,
        block_size=block_size,
    )

    replay = make_request("1", cached_token_ids, hash_block_size, sha256)
    replay_hash_10 = boundary_hash(replay, hash_block_size, 10)
    assert replay_hash_10 == partial_hash_10
    assert pool.get_cached_block(replay_hash_10, [kv_cache_group_id]) == [blocks[1]]

    extended = make_request("2", cached_token_ids + [10], hash_block_size, sha256)
    extended_hash_10 = boundary_hash(extended, hash_block_size, 10)
    assert extended_hash_10 == partial_hash_10
    assert pool.get_cached_block(extended_hash_10, [kv_cache_group_id]) == [blocks[1]]


def test_cache_partial_block_uses_fine_grained_boundary_hash():
    hash_block_size = 2
    block_size = 6
    kv_cache_group_id = 0
    token_ids = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    req = make_request("0", token_ids, hash_block_size, sha256)
    pool = BlockPool(
        num_gpu_blocks=3,
        enable_caching=True,
        hash_block_size=hash_block_size,
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

    partial_entry_hash = pool.cache_partial_block(
        request=req,
        block=blocks[1],
        num_tokens=10,
        kv_cache_group_id=kv_cache_group_id,
        block_size=block_size,
    )
    # The partial entry is keyed by the fine-grained hash at the 10-token
    # boundary, regardless of the owning group's block_size.
    expected = boundary_hash(req, hash_block_size, 10)
    assert partial_entry_hash == kv_cache_utils.make_block_hash_with_group_id(
        expected, kv_cache_group_id
    )
    assert pool.get_cached_block(expected, [kv_cache_group_id]) == [blocks[1]]


def test_cache_partial_block_requires_hash_boundary():
    hash_block_size = 2
    block_size = 4
    req = make_request("0", [0, 0, 1, 1], hash_block_size, sha256)
    pool = BlockPool(
        num_gpu_blocks=2,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )
    block = pool.get_new_blocks(1)[0]

    with pytest.raises(AssertionError):
        pool.cache_partial_block(
            request=req,
            block=block,
            num_tokens=3,
            kv_cache_group_id=0,
            block_size=block_size,
        )


def test_cache_partial_block_duplicate_checks_all_blocks_for_hash():
    hash_block_size = 2
    block_size = 4
    kv_cache_group_id = 0
    req = make_request("0", [0, 0, 1, 1], hash_block_size, sha256)
    pool = BlockPool(
        num_gpu_blocks=4,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )
    blocks = pool.get_new_blocks(2)

    first_entry_hash = pool.cache_partial_block(
        request=req,
        block=blocks[0],
        num_tokens=2,
        kv_cache_group_id=kv_cache_group_id,
        block_size=block_size,
    )
    second_entry_hash = pool.cache_partial_block(
        request=req,
        block=blocks[1],
        num_tokens=2,
        kv_cache_group_id=kv_cache_group_id,
        block_size=block_size,
    )
    assert first_entry_hash == second_entry_hash

    duplicate_entry_hash = pool.cache_partial_block(
        request=req,
        block=blocks[1],
        num_tokens=2,
        kv_cache_group_id=kv_cache_group_id,
        block_size=block_size,
    )
    assert duplicate_entry_hash == second_entry_hash
    assert pool.cached_block_hashes_by_block == {}


def test_reset_prefix_cache_clears_partial_entry_metadata():
    pool, req, blocks, partial_hash_10 = cache_full_block_and_partial_tail(
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    )
    full_hash = BlockHashListWithBlockSize(req.block_hashes, 2, 6)[0]

    assert pool.get_cached_block(full_hash, [0]) == [blocks[0]]
    assert pool.get_cached_block(partial_hash_10, [0]) == [blocks[1]]

    pool.free_blocks(blocks)
    assert pool.reset_prefix_cache()

    assert pool.get_cached_block(full_hash, [0]) is None
    assert pool.get_cached_block(partial_hash_10, [0]) is None
    assert pool.cached_block_hashes_by_block == {}


@pytest.mark.parametrize("dcp_world_size", [1, 2, 4])
def test_evict_cached_block_removes_full_hash_and_partial_entry(
    dcp_world_size: int,
):
    block_size = 6 * dcp_world_size
    partial_num_tokens = 2 * block_size - 2
    pool, req, blocks, partial_hash = cache_full_block_and_partial_tail(
        list(range(partial_num_tokens)), block_size=block_size
    )
    full_hash = BlockHashListWithBlockSize(req.block_hashes, 2, block_size)[0]

    assert pool.get_cached_block(full_hash, [0]) == [blocks[0]]
    assert pool.get_cached_block(partial_hash, [0]) == [blocks[1]]

    pool.evict_blocks({blocks[0].block_id, blocks[1].block_id})

    assert pool.get_cached_block(full_hash, [0]) is None
    assert pool.get_cached_block(partial_hash, [0]) is None
    assert pool.cached_block_hashes_by_block == {}


@pytest.mark.parametrize("dcp_world_size", [1, 2, 4])
def test_partial_block_promotes_to_direct_full_block_hash(dcp_world_size: int):
    hash_block_size = 2
    block_size = 6 * dcp_world_size
    kv_cache_group_id = 0
    partial_num_tokens = 2 * block_size - hash_block_size
    token_ids = list(range(partial_num_tokens))
    req = make_request("0", token_ids, hash_block_size, sha256)
    pool = BlockPool(
        num_gpu_blocks=3,
        enable_caching=True,
        hash_block_size=hash_block_size,
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
    partial_hash = boundary_hash(req, hash_block_size, partial_num_tokens)
    assert pool.cache_partial_block(
        request=req,
        block=blocks[1],
        num_tokens=partial_num_tokens,
        kv_cache_group_id=kv_cache_group_id,
        block_size=block_size,
    )
    assert pool.get_cached_block(partial_hash, [kv_cache_group_id]) == [blocks[1]]

    req.append_output_token_ids(list(range(partial_num_tokens, 2 * block_size)))
    full_hashes = BlockHashListWithBlockSize(
        req.block_hashes, hash_block_size, block_size
    )
    promoted_full_hash = full_hashes[1]
    assert promoted_full_hash == req.block_hashes[2 * block_size // hash_block_size - 1]

    pool.cache_full_blocks(
        request=req,
        blocks=blocks,
        num_cached_blocks=1,
        num_full_blocks=2,
        block_size=block_size,
        kv_cache_group_id=kv_cache_group_id,
    )
    assert pool.get_cached_block(promoted_full_hash, [kv_cache_group_id]) == [blocks[1]]
    assert pool.get_cached_block(partial_hash, [kv_cache_group_id]) is None


@pytest.mark.parametrize("dcp_world_size", [1, 2, 4])
def test_cache_blocks_does_not_resurrect_stale_partial_hash_after_promotion(
    dcp_world_size: int,
):
    """Regression test for partial-hash resurrection after full-block promotion.

    When a request's prompt ends inside a block, _cache_partial_tail_block
    registers a partial hash for that boundary.  Once decode fills the block
    completely, cache_full_blocks promotes it to a full-block hash and removes
    the partial entry.  _cache_partial_tail_block must then skip re-inserting
    the now-superseded partial hash on every subsequent cache_blocks call,
    because boundary_tokens is derived from the fixed num_prompt_tokens and
    does not change across decode steps.

    The cache map and the KV-cache event stream are both checked: a fix that
    keeps get_cached_block clean but still emits BlockStored for the retired
    partial hash would leave external KV-aware routers with the contradictory
    view this bug is actually reported for.
    """
    hash_block_size = 2
    block_size = 6 * dcp_world_size
    kv_cache_group_id = 0

    # Prompt: 10 tokens, ends 4 tokens into block 1 (partial tail).
    prompt_token_ids = list(range(10 * dcp_world_size))
    req = make_request("R", prompt_token_ids, hash_block_size, sha256)

    pool = BlockPool(
        num_gpu_blocks=4,
        enable_caching=True,
        hash_block_size=hash_block_size,
        enable_kv_cache_events=True,
    )
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    manager = FullAttentionManager(
        kv_cache_spec=spec,
        block_pool=pool,
        enable_caching=True,
        kv_cache_group_id=kv_cache_group_id,
        scheduler_block_size=block_size,
    )
    manager.req_to_blocks[req.request_id] = pool.get_new_blocks(2)

    # Prefill: registers the partial hash for the prompt boundary and
    # announces it to KV-cache-event subscribers.
    manager.cache_blocks(req, num_tokens=len(prompt_token_ids))
    partial_hash = boundary_hash(req, hash_block_size, len(prompt_token_ids))
    stale_event_hash = kv_cache_utils.maybe_convert_block_hash(partial_hash)
    block1 = manager.req_to_blocks[req.request_id][1]
    assert pool.get_cached_block(partial_hash, [kv_cache_group_id]) == [block1]
    assert any(
        isinstance(e, BlockStored) and stale_event_hash in e.block_hashes
        for e in pool.take_events()
    ), "prefill must announce the partial boundary hash"

    # Decode: fill block 1 to completion, triggering promotion.
    tokens_to_fill = block_size - (len(prompt_token_ids) % block_size)
    for i in range(tokens_to_fill):
        req.append_output_token_ids([100 + i])
        manager.cache_blocks(req, num_tokens=len(prompt_token_ids) + i + 1)
    promotion_events = pool.take_events()

    # After promotion the stale partial hash must not be live in the cache.
    full_hashes = BlockHashListWithBlockSize(
        req.block_hashes, hash_block_size, block_size
    )
    promoted_full_hash = full_hashes[1]
    assert pool.get_cached_block(promoted_full_hash, [kv_cache_group_id]) == [block1], (
        "promoted full-block hash must be cached"
    )
    assert pool.get_cached_block(partial_hash, [kv_cache_group_id]) is None, (
        "stale partial hash must not be resurrected after promotion"
    )

    # ...and the event stream must agree with the cache map. Promotion
    # announces the partial hash as removed, so nothing may re-announce it as
    # stored afterwards.
    removed_at = [
        i
        for i, e in enumerate(promotion_events)
        if isinstance(e, BlockRemoved) and stale_event_hash in e.block_hashes
    ]
    assert removed_at, "promotion must announce the partial hash as removed"
    assert not any(
        isinstance(e, BlockStored) and stale_event_hash in e.block_hashes
        for e in promotion_events[removed_at[-1] :]
    ), "stale partial hash must not be re-announced as stored after promotion"

    # Additional decode steps must not re-insert the stale entry either, in
    # the cache map or on the event stream.
    for i in range(tokens_to_fill, tokens_to_fill + 3):
        req.append_output_token_ids([200 + i])
        manager.cache_blocks(req, num_tokens=len(prompt_token_ids) + i + 1)
    assert pool.get_cached_block(partial_hash, [kv_cache_group_id]) is None, (
        "stale partial hash must stay absent on subsequent decode steps"
    )
    assert pool.take_events() == [], (
        "decode inside an already-promoted block must emit no further events"
    )


@pytest.mark.parametrize("dcp_world_size", [1, 2, 4])
def test_cache_partial_block_refuses_boundary_the_block_has_outgrown(
    dcp_world_size: int,
):
    """The pool itself must refuse a partial entry a promoted block outgrew.

    test_cache_blocks_does_not_resurrect_stale_partial_hash_after_promotion
    covers the caller that actually makes this mistake in production.  This
    test pins the invariant at the insert site instead, so it holds for any
    future caller of cache_partial_block: once a block carries a hash covering
    at least as many tokens as the incoming boundary, that boundary has been
    superseded and re-registering it is never correct.

    Refusal is reported by returning None -- the same contract as a null block
    -- because callers such as MambaManager._cache_partial_tail_block key
    follow-up bookkeeping off the returned hash and must not record a partial
    entry that was never inserted.
    """
    hash_block_size = 2
    block_size = 6 * dcp_world_size
    kv_cache_group_id = 0
    partial_num_tokens = 2 * block_size - hash_block_size
    token_ids = list(range(partial_num_tokens))
    req = make_request("0", token_ids, hash_block_size, sha256)
    pool = BlockPool(
        num_gpu_blocks=3,
        enable_caching=True,
        hash_block_size=hash_block_size,
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
    partial_hash = boundary_hash(req, hash_block_size, partial_num_tokens)
    assert pool.cache_partial_block(
        request=req,
        block=blocks[1],
        num_tokens=partial_num_tokens,
        kv_cache_group_id=kv_cache_group_id,
        block_size=block_size,
    )
    assert pool.get_cached_block(partial_hash, [kv_cache_group_id]) == [blocks[1]]

    # Fill the block and promote it past the partial boundary.
    req.append_output_token_ids(list(range(partial_num_tokens, 2 * block_size)))
    full_hashes = BlockHashListWithBlockSize(
        req.block_hashes, hash_block_size, block_size
    )
    promoted_full_hash = full_hashes[1]
    pool.cache_full_blocks(
        request=req,
        blocks=blocks,
        num_cached_blocks=1,
        num_full_blocks=2,
        block_size=block_size,
        kv_cache_group_id=kv_cache_group_id,
    )
    pool.take_events()

    # Re-registering the outgrown boundary must be refused outright.
    assert (
        pool.cache_partial_block(
            request=req,
            block=blocks[1],
            num_tokens=partial_num_tokens,
            kv_cache_group_id=kv_cache_group_id,
            block_size=block_size,
        )
        is None
    ), "a boundary the block has outgrown must not be registered"
    assert pool.get_cached_block(partial_hash, [kv_cache_group_id]) is None, (
        "the outgrown boundary must not become reachable again"
    )
    assert pool.get_cached_block(promoted_full_hash, [kv_cache_group_id]) == [
        blocks[1]
    ], "the promoted full-block hash must be left intact"
    assert blocks[1].block_hash_num_tokens == 2 * block_size, (
        "the refused call must not shorten the block's recorded coverage"
    )
    assert pool.take_events() == [], "a refused registration must emit no events"


@pytest.mark.parametrize("dcp_world_size", [1, 2, 4])
def test_cache_partial_block_still_grows_a_partial_entry_forward(
    dcp_world_size: int,
):
    """The refusal must not catch a partial entry legitimately growing.

    The guard keys on ">= incoming", so the 8 -> 10 token growth path inside a
    single block -- where the block's existing hash covers *fewer* tokens --
    must still retire the shorter key and register the longer one.
    """
    hash_block_size = 2
    block_size = 6 * dcp_world_size
    kv_cache_group_id = 0
    shorter_num_tokens = block_size + hash_block_size
    longer_num_tokens = block_size + 2 * hash_block_size
    token_ids = list(range(longer_num_tokens))
    req = make_request("0", token_ids, hash_block_size, sha256)
    pool = BlockPool(
        num_gpu_blocks=3,
        enable_caching=True,
        hash_block_size=hash_block_size,
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
    shorter_hash = boundary_hash(req, hash_block_size, shorter_num_tokens)
    longer_hash = boundary_hash(req, hash_block_size, longer_num_tokens)

    assert pool.cache_partial_block(
        request=req,
        block=blocks[1],
        num_tokens=shorter_num_tokens,
        kv_cache_group_id=kv_cache_group_id,
        block_size=block_size,
    )
    assert pool.get_cached_block(shorter_hash, [kv_cache_group_id]) == [blocks[1]]

    assert pool.cache_partial_block(
        request=req,
        block=blocks[1],
        num_tokens=longer_num_tokens,
        kv_cache_group_id=kv_cache_group_id,
        block_size=block_size,
    ), "growing a partial entry forward inside its block must still be allowed"
    assert pool.get_cached_block(longer_hash, [kv_cache_group_id]) == [blocks[1]]
    assert pool.get_cached_block(shorter_hash, [kv_cache_group_id]) is None, (
        "the superseded shorter boundary must be retired"
    )


@pytest.mark.parametrize("dcp_world_size", [1, 2, 4])
def test_promoted_tail_stops_calling_into_the_pool_each_decode_step(
    dcp_world_size: int,
):
    """The caller-side guard is an optimisation; pin what it actually buys.

    BlockPool.cache_partial_block already refuses an outgrown boundary, so
    correctness does not depend on the check in _cache_partial_tail_block --
    dropping it leaves every other test in this file green.  What it buys is
    that steady-state decode inside a promoted block stops re-deriving the
    partial hash and calling into the pool once per step, forever, only to
    have the call refused.  Without an assertion on the call count that
    property is invisible and a future refactor would silently drop it.
    """
    hash_block_size = 2
    block_size = 6 * dcp_world_size
    kv_cache_group_id = 0

    prompt_token_ids = list(range(10 * dcp_world_size))
    req = make_request("R", prompt_token_ids, hash_block_size, sha256)

    pool = BlockPool(
        num_gpu_blocks=4,
        enable_caching=True,
        hash_block_size=hash_block_size,
    )
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    manager = FullAttentionManager(
        kv_cache_spec=spec,
        block_pool=pool,
        enable_caching=True,
        kv_cache_group_id=kv_cache_group_id,
        scheduler_block_size=block_size,
    )
    manager.req_to_blocks[req.request_id] = pool.get_new_blocks(2)

    calls = 0
    original = pool.cache_partial_block

    def counting_cache_partial_block(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    pool.cache_partial_block = counting_cache_partial_block

    manager.cache_blocks(req, num_tokens=len(prompt_token_ids))
    assert calls == 1, "the prompt boundary itself must be registered once"

    # Fill the tail block to completion, promoting it.
    tokens_to_fill = block_size - (len(prompt_token_ids) % block_size)
    for i in range(tokens_to_fill):
        req.append_output_token_ids([100 + i])
        manager.cache_blocks(req, num_tokens=len(prompt_token_ids) + i + 1)

    # Steady-state decode inside the promoted block: no further pool calls.
    calls_after_promotion = calls
    for i in range(tokens_to_fill, tokens_to_fill + 5):
        req.append_output_token_ids([200 + i])
        manager.cache_blocks(req, num_tokens=len(prompt_token_ids) + i + 1)
    assert calls == calls_after_promotion, (
        "decode inside an already-promoted block must not call into the pool"
    )


def _mamba_align_manager(
    hash_block_size: int, block_size: int
) -> tuple[BlockPool, MambaManager]:
    spec = MambaSpec(
        shapes=((8,),),
        dtypes=(torch.float32,),
        block_size=block_size,
        mamba_cache_mode="align",
        num_speculative_blocks=0,
    )
    pool = BlockPool(
        num_gpu_blocks=8,
        enable_caching=True,
        hash_block_size=hash_block_size,
        enable_kv_cache_events=True,
    )
    manager = MambaManager(
        kv_cache_spec=spec,
        block_pool=pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=block_size,
    )
    return pool, manager


def test_mamba_partial_tail_registers_the_boundary_when_it_is_not_superseded():
    """Positive control for the refusal test below.

    On an un-promoted tail block MambaManager._cache_partial_tail_block still
    registers the boundary and records all three pieces of follow-up
    bookkeeping, the last of which hands the CoW block to a KV connector for
    partial-tail offload.  Without this the refusal test could pass vacuously.
    """
    hash_block_size, block_size = 2, 6
    kv_cache_group_id = 0
    pool, manager = _mamba_align_manager(hash_block_size, block_size)

    # 10 prompt tokens: block 0 is full, block 1 holds tokens 6..9, and the
    # last prompt hash boundary is 10 -- inside block 1.
    req = make_request("0", list(range(10)), hash_block_size, sha256)
    blocks = pool.get_new_blocks(2)
    manager.req_to_blocks[req.request_id] = blocks

    partial_hash = manager._cache_partial_tail_block(req, num_tokens=10)

    assert partial_hash is not None
    assert pool.get_cached_block(
        boundary_hash(req, hash_block_size, 10), [kv_cache_group_id]
    ) == [blocks[1]]
    assert manager._partial_hit_reqs[req.request_id] == (1, blocks[1])
    assert manager.num_cached_block[req.request_id] == 1
    assert manager._producer_partial_tail_reqs[req.request_id] == (blocks[1], 10)


def test_mamba_partial_tail_skips_all_bookkeeping_when_the_boundary_is_superseded():
    """The Mamba caller must treat a superseded boundary as "nothing happened".

    cache_partial_block now has two reasons to return None -- null block, and a
    boundary the block has outgrown.  MambaManager._cache_partial_tail_block
    keys three updates off that value, and the third,
    _producer_partial_tail_reqs, is what makes allocate_new_blocks hand the CoW
    block to a KV connector for offload under the *boundary* sub-hash.

    Skipping it is the only correct behaviour: the pool has just refused to
    publish that key locally, and the block records coverage of a later
    boundary, so offloading it would advertise a superseded key to an external
    store backed by the wrong state.  That is the same corruption this module's
    other tests pin, one surface further out.

    This state is not reachable from MambaManager on its own -- its
    ``num_tokens != latest_prompt_hash_boundary`` guard fires the boundary
    exactly once, so a block it targets is never already promoted past it.  The
    promotion is therefore forced out of band here, to pin the contract for any
    future caller rather than to reproduce a live bug.
    """
    hash_block_size, block_size = 2, 6
    kv_cache_group_id = 0
    pool, manager = _mamba_align_manager(hash_block_size, block_size)

    req = make_request("0", list(range(10)), hash_block_size, sha256)
    blocks = pool.get_new_blocks(2)
    manager.req_to_blocks[req.request_id] = blocks

    # Force the tail block past the boundary: a 12-token request promotes both
    # blocks, so block 1 carries a full-block hash covering 12 >= 10 tokens.
    promoter = make_request("1", list(range(12)), hash_block_size, sha256)
    pool.cache_full_blocks(
        request=promoter,
        blocks=blocks,
        num_cached_blocks=0,
        num_full_blocks=2,
        block_size=block_size,
        kv_cache_group_id=kv_cache_group_id,
    )
    assert blocks[1].block_hash_num_tokens == 12
    pool.take_events()

    assert manager._cache_partial_tail_block(req, num_tokens=10) is None

    # None of the three updates may happen, and nothing may be published --
    # not in the cache map, not on the event stream.
    assert req.request_id not in manager._partial_hit_reqs
    assert req.request_id not in manager.num_cached_block
    assert req.request_id not in manager._producer_partial_tail_reqs
    assert (
        pool.get_cached_block(
            boundary_hash(req, hash_block_size, 10), [kv_cache_group_id]
        )
        is None
    )
    assert pool.take_events() == []
    assert blocks[1].block_hash_num_tokens == 12, (
        "the refusal must not shorten the block's recorded coverage"
    )
