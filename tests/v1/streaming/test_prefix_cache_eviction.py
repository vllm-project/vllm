# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Eviction x prefix-caching cross-request tests from the streaming deep
review (C17, C18). Prefix caching is ON in production, but until now every
streaming-eviction test ran with `enable_caching=False`.

  - C18 (a1): splice-miss regression — after a mid-stream eviction, an
    identical-stream request must MISS at the spliced range (hash strip +
    first-miss break), never chain across it.
  - C18 (b):  num_cached_block accounting when an evicted block is SHARED
    (ref_cnt > 1): memory not reclaimed, sharer untouched, hash stripped,
    exact per-request counters.
  - C18 (d):  the re-prefill KV sequence (free + evict_blocks, as
    `_reprefill_streaming_session` does) makes a same-chain admission miss
    completely.
  - C17:      session block-hash chains are salted (`cache_salt` = the
    session's internal request id, set by AsyncLLM.handle_inputs), so a
    session's KV — whose content/positions diverge from the content-chain
    hashes after eviction/re-prefill — is unreachable by any other request
    while preempt-resume self-hits still work.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
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

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16


@pytest.fixture(autouse=True)
def _auto_init_hash_fn():
    init_none_hash(sha256)


def _make_manager(num_blocks: int = 64) -> KVCacheManager:
    return KVCacheManager(
        KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer"],
                    FullAttentionSpec(
                        block_size=BLOCK_SIZE,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float32,
                    ),
                )
            ],
        ),
        max_model_len=1024,
        enable_caching=True,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
    )


def _make_request(
    request_id: str,
    prompt_token_ids: list[int],
    cache_salt: str | None = None,
    hash_fn: Callable = sha256,
) -> Request:
    sp = SamplingParams(max_tokens=1)
    sp.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sp,
        pooling_params=None,
        cache_salt=cache_salt,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, hash_fn),
    )


def _allocate_and_cache(
    manager: KVCacheManager, req: Request, num_tokens: int
) -> list[int]:
    """Admit `req`, allocate its whole prompt, and return the first-group
    block ids. With enable_caching=True, allocate_slots also publishes the
    full blocks into the prefix cache."""
    computed, num_computed, _ = manager.get_computed_blocks(req)
    manager.allocate_slots(req, num_tokens - num_computed, num_computed, computed)
    return [
        b.block_id
        for b in manager.coordinator.single_type_managers[0].req_to_blocks[
            req.request_id
        ]
        if not b.is_null
    ]


# The identical "original stream" every replay request uses. Varied tokens
# per block so each block hash is distinct.
STREAM = [i % 100 for i in range(96)]  # 6 blocks


# ---------------------------------------------------------------------------
# C18 (a1): splice-miss regression
# ---------------------------------------------------------------------------


def test_identical_stream_misses_at_evicted_splice():
    """Guards C18 (a1): after session A evicts a middle token range, a
    fresh request replaying the identical original stream must have its
    cache hit stop strictly BEFORE the spliced range — the evicted blocks'
    hashes were stripped and find_longest_cache_hit breaks at the first
    miss, so it can never chain into A's post-splice blocks."""
    manager = _make_manager()
    req_a = _make_request("sess-a", STREAM)
    _allocate_and_cache(manager, req_a, 96)
    # All 6 full blocks are now published in the prefix cache.
    assert manager.coordinator.single_type_managers[0].num_cached_block["sess-a"] == 6

    # Session A evicts the middle two blocks' worth ([32, 64)).
    aligned_start, aligned_end, n_freed = manager.evict_token_range_for_request(
        "sess-a", token_start=32, token_end=64
    )
    assert (aligned_start, aligned_end, n_freed) == (32, 64, 2)

    # Identical-stream replay request: the hit must stop at the splice.
    req_b = _make_request("replay-b", STREAM)
    _, num_hit_tokens, _ = manager.get_computed_blocks(req_b)
    assert num_hit_tokens == 32, (
        f"cache hit must stop at the evicted range (32), got {num_hit_tokens}"
    )


# ---------------------------------------------------------------------------
# C18 (b): shared-block eviction accounting
# ---------------------------------------------------------------------------


def test_evicting_shared_block_reclaims_nothing_and_keeps_sharer_intact():
    """Guards C18 (b): evicting a block still shared with another request
    (ref_cnt > 1) must return 0 (no memory reclaimed), strip the hash from
    the prefix cache, leave the sharer's block table untouched, and keep
    each request's num_cached_block exactly right (no reliance on the
    max(0, ...) clamp)."""
    manager = _make_manager()
    single = manager.coordinator.single_type_managers[0]
    tokens = [i % 100 for i in range(48)]  # 3 blocks

    req_a = _make_request("sess-a", tokens)
    a_blocks = _allocate_and_cache(manager, req_a, 48)
    assert len(a_blocks) == 3

    # B replays the same stream and shares A's cached prefix. The hit is
    # capped at num_tokens - 1, so B shares the first TWO blocks.
    req_b = _make_request("sess-b", tokens)
    computed, num_hit, _ = manager.get_computed_blocks(req_b)
    assert num_hit == 32
    manager.allocate_slots(req_b, 48 - num_hit, num_hit, computed)
    b_blocks = [b.block_id for b in single.req_to_blocks["sess-b"]]
    shared_id = a_blocks[1]
    assert shared_id in b_blocks
    assert manager.block_pool.blocks[shared_id].ref_cnt == 2

    cached_a_before = single.num_cached_block["sess-a"]
    cached_b_before = single.num_cached_block["sess-b"]
    free_before = manager.block_pool.free_block_queue.num_free_blocks

    n_freed = manager.coordinator.evict_blocks_for_request("sess-a", [shared_id])

    # Memory NOT reclaimed: the sharer still holds a reference.
    assert n_freed == 0
    assert manager.block_pool.free_block_queue.num_free_blocks == free_before
    assert manager.block_pool.blocks[shared_id].ref_cnt == 1
    # Dropped from A's table only; B's table untouched.
    assert shared_id not in [b.block_id for b in single.req_to_blocks["sess-a"]]
    assert [b.block_id for b in single.req_to_blocks["sess-b"]] == b_blocks
    # Hash stripped so no third request can hit the stale entry.
    assert manager.block_pool.blocks[shared_id].block_hash is None
    # Exact accounting: A lost exactly the one cached block; B unchanged.
    assert single.num_cached_block["sess-a"] == cached_a_before - 1
    assert single.num_cached_block["sess-b"] == cached_b_before

    # A third identical-stream request must now stop before the stripped
    # block (only block 0 remains hittable).
    req_c = _make_request("replay-c", tokens)
    _, num_hit_c, _ = manager.get_computed_blocks(req_c)
    assert num_hit_c == 16


# ---------------------------------------------------------------------------
# C18 (d): re-prefill's evict_blocks makes a same-chain admission miss
# ---------------------------------------------------------------------------


def test_reprefill_block_eviction_prevents_same_chain_hit():
    """Guards C18 (d): `_reprefill_streaming_session` frees the session's
    blocks and then evicts their prefix-cache entries
    (kv_cache_manager.evict_blocks) — a request with the identical hash
    chain admitted afterwards must get ZERO computed blocks; the session's
    old KV (about to be recomputed at fresh dense positions) must not be
    hittable."""
    manager = _make_manager()
    req_a = _make_request("sess-a", STREAM)
    a_block_ids = set(_allocate_and_cache(manager, req_a, 96))

    # Sanity: before the re-prefill sequence the chain IS hittable.
    probe = _make_request("probe", STREAM)
    _, num_hit_before, _ = manager.get_computed_blocks(probe)
    assert num_hit_before > 0

    # The exact KV sequence of _reprefill_streaming_session
    # (scheduler.py: gather ids -> free -> evict_blocks).
    gathered: set[int] = set()
    for per_group_ids in manager.get_block_ids("sess-a"):
        gathered.update(per_group_ids)
    assert gathered == a_block_ids
    manager.free(req_a)
    manager.evict_blocks(gathered)

    req_b = _make_request("replay-b", STREAM)
    _, num_hit_after, _ = manager.get_computed_blocks(req_b)
    assert num_hit_after == 0, (
        "same-chain admission after re-prefill eviction must miss entirely"
    )


# ---------------------------------------------------------------------------
# C17: session hash chains are salted and therefore globally unique
# ---------------------------------------------------------------------------


def test_cache_salt_isolates_session_hash_chain():
    """Guards C17: with the fix, every streaming session gets
    `cache_salt = <internal session request id>` before its first chunk, so
    its block-hash chain can never collide with (a) an unsalted request
    replaying the same stream or (b) another session (different salt) —
    while the session itself (same salt, e.g. preempt-resume) still
    self-hits. This is what keeps post-eviction / post-re-prefill KV from
    poisoning other requests despite the spliced content chain."""
    manager = _make_manager()

    salted = _make_request("sess-a", STREAM, cache_salt="sess-a-internal-id")
    unsalted = _make_request("plain-b", STREAM)
    other_session = _make_request("sess-c", STREAM, cache_salt="sess-c-internal-id")
    same_salt = _make_request("sess-a-resume", STREAM, cache_salt="sess-a-internal-id")

    # The salt feeds the FIRST block's extra keys and chains into every
    # later hash: the salted chain shares NO hash with the unsalted or
    # differently-salted chains.
    assert len(salted.block_hashes) == 6
    assert set(salted.block_hashes).isdisjoint(unsalted.block_hashes)
    assert set(salted.block_hashes).isdisjoint(other_session.block_hashes)
    assert salted.block_hashes == same_salt.block_hashes

    _allocate_and_cache(manager, salted, 96)

    # No cross-request read of the session's blocks...
    _, hit_unsalted, _ = manager.get_computed_blocks(unsalted)
    assert hit_unsalted == 0
    _, hit_other, _ = manager.get_computed_blocks(other_session)
    assert hit_other == 0
    # ...but the session still self-hits (preempt-resume path), capped at
    # num_tokens - 1 -> 5 of 6 blocks.
    _, hit_self, _ = manager.get_computed_blocks(same_salt)
    assert hit_self == 80
