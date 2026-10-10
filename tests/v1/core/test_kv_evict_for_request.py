# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for `KVCacheCoordinator.evict_blocks_for_request`.

The primitive frees specific physical blocks from a still-alive request,
compacting the request's block table on the scheduler side. It is the
block-id-level free that `KVCacheManager.evict_token_range_for_request`
builds on.
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


@pytest.fixture(autouse=True)
def _auto_init_hash_fn():
    init_none_hash(sha256)


def _make_kv_cache_config(block_size: int, num_blocks: int) -> KVCacheConfig:
    return KVCacheConfig(
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
    )


def _make_request(
    request_id: str,
    prompt_token_ids: list[int],
    block_size: int,
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


def _allocated_block_ids(manager: KVCacheManager, request_id: str) -> list[int]:
    """First-group block IDs allocated for a request, in logical order,
    excluding null_block sentinels."""
    blocks = manager.coordinator.single_type_managers[0].req_to_blocks.get(
        request_id, []
    )
    return [b.block_id for b in blocks if not b.is_null]


def test_evict_blocks_compacts_block_list_no_caching():
    """Caller frees a subset; surviving blocks shift down. Freed blocks
    return to the free pool (their ref_cnt drops to 0)."""
    block_size = 16
    manager = KVCacheManager(
        _make_kv_cache_config(block_size, num_blocks=8),
        max_model_len=1024,
        scheduler_block_size=block_size,
        enable_caching=False,
        hash_block_size=block_size,
    )

    # 4 full blocks worth of tokens (64 tokens).
    req = _make_request("r0", [1] * 64, block_size)
    computed, num_computed, _ = manager.get_computed_blocks(req)
    assert num_computed == 0
    blocks_owned = manager.allocate_slots(req, 64, 0, computed)
    assert blocks_owned is not None

    allocated = _allocated_block_ids(manager, "r0")
    assert len(allocated) == 4, f"expected 4 allocated, got {allocated}"

    # Free the middle two.
    to_evict = [allocated[1], allocated[2]]
    free_q = manager.block_pool.free_block_queue
    free_before = free_q.num_free_blocks

    n_freed = manager.coordinator.evict_blocks_for_request("r0", to_evict)
    assert n_freed == 2

    surviving = _allocated_block_ids(manager, "r0")
    assert surviving == [allocated[0], allocated[3]], (
        f"block list should be compacted to [{allocated[0]}, {allocated[3]}], "
        f"got {surviving}"
    )

    # Freed blocks must be back in the free pool. With caching disabled
    # each block's ref_cnt drops to 0 immediately.
    assert free_q.num_free_blocks == free_before + 2

    # Request must remain alive.
    manager.free(req)


def test_evict_blocks_idempotent_on_unknown_ids():
    """Block IDs not in the request's table are silently skipped."""
    block_size = 16
    manager = KVCacheManager(
        _make_kv_cache_config(block_size, num_blocks=8),
        max_model_len=1024,
        scheduler_block_size=block_size,
        enable_caching=False,
        hash_block_size=block_size,
    )
    req = _make_request("r1", [1] * 32, block_size)
    computed, _, _ = manager.get_computed_blocks(req)
    manager.allocate_slots(req, 32, 0, computed)
    allocated = _allocated_block_ids(manager, "r1")

    # Block ID 999 is not in this request's table.
    n_freed = manager.coordinator.evict_blocks_for_request("r1", [999, 999, 999])
    assert n_freed == 0
    assert _allocated_block_ids(manager, "r1") == allocated
    manager.free(req)


def test_evict_blocks_empty_inputs_are_noop():
    block_size = 16
    manager = KVCacheManager(
        _make_kv_cache_config(block_size, num_blocks=8),
        max_model_len=1024,
        scheduler_block_size=block_size,
        enable_caching=False,
        hash_block_size=block_size,
    )
    req = _make_request("r2", [1] * 32, block_size)
    manager.allocate_slots(req, 32, 0, manager.get_computed_blocks(req)[0])
    allocated = _allocated_block_ids(manager, "r2")

    assert manager.coordinator.evict_blocks_for_request("r2", []) == 0
    missing = manager.coordinator.evict_blocks_for_request("nonexistent-req", [1, 2, 3])
    assert missing == 0
    assert _allocated_block_ids(manager, "r2") == allocated
    manager.free(req)


def test_evict_blocks_with_caching_removes_from_prefix_cache():
    """When caching is on, evicted blocks must be removed from the
    prefix-cache hash map so future requests don't hit stale K/V."""
    block_size = 16
    manager = KVCacheManager(
        _make_kv_cache_config(block_size, num_blocks=8),
        max_model_len=1024,
        scheduler_block_size=block_size,
        enable_caching=True,
        hash_block_size=block_size,
    )

    # Allocate 3 full blocks of identical tokens — guaranteed hashable.
    req = _make_request("r3", [1] * 48, block_size)
    computed, _, _ = manager.get_computed_blocks(req)
    manager.allocate_slots(req, 48, 0, computed)
    allocated = _allocated_block_ids(manager, "r3")
    assert len(allocated) == 3

    # Free the middle block. After eviction it must be neither in
    # `cached_block_hash_to_block` nor have a block_hash.
    middle_id = allocated[1]
    pre_hash_count = len(manager.block_pool.cached_block_hash_to_block)

    n_freed = manager.coordinator.evict_blocks_for_request("r3", [middle_id])
    assert n_freed == 1
    assert manager.block_pool.blocks[middle_id].block_hash is None
    assert len(manager.block_pool.cached_block_hash_to_block) < pre_hash_count, (
        "evicted block should have left the prefix-cache hash map"
    )

    # Surviving list is compacted.
    assert _allocated_block_ids(manager, "r3") == [allocated[0], allocated[2]]
    manager.free(req)


def test_evict_blocks_then_continue_allocating():
    """After eviction the request must still be allocatable (still alive)."""
    block_size = 16
    manager = KVCacheManager(
        _make_kv_cache_config(block_size, num_blocks=8),
        max_model_len=1024,
        scheduler_block_size=block_size,
        enable_caching=False,
        hash_block_size=block_size,
    )
    req = _make_request("r4", [1] * 48, block_size)
    manager.allocate_slots(req, 48, 0, manager.get_computed_blocks(req)[0])
    allocated_before = _allocated_block_ids(manager, "r4")
    assert len(allocated_before) == 3

    # Drop the oldest.
    manager.coordinator.evict_blocks_for_request("r4", [allocated_before[0]])
    after_evict = _allocated_block_ids(manager, "r4")
    assert len(after_evict) == 2

    # Append more tokens; allocator should hand us a fresh block on top
    # of the surviving two.
    req.num_computed_tokens = 32  # 2 surviving blocks worth.
    req._all_token_ids.extend([2] * 16)
    req.prompt_token_ids = list(req._all_token_ids)
    manager.allocate_slots(req, 16, 0)
    after_grow = _allocated_block_ids(manager, "r4")
    assert len(after_grow) == 3, f"expected 3 blocks after re-growth, got {after_grow}"
    manager.free(req)
