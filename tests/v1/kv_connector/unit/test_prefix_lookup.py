# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""External residency must use the same replay boundaries as local caching."""

import pytest
import torch

from tests.v1.core.test_prefix_caching import make_request
from tests.v1.core.utils import new_kv_cache_group
from vllm.distributed.kv_transfer.kv_connector.lookup import ExternalKVCacheLookup
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.core.kv_cache_utils import get_block_hash, get_group_id, init_none_hash
from vllm.v1.core.single_type_kv_cache_manager import (
    CircularBufferManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("flash,lookahead", [(False, 2), (True, 0)])
def test_external_hit_matches_core_at_sparse_replay_boundaries(flash, lookahead):
    """Sparse replay tails may exist below the full-attention prefix upper bound."""
    init_none_hash(sha256)
    groups = []
    sizes = [32, 32, 32, 32, 128] if flash else [64, 64, 256, 8, 4]
    full_group = 4 if flash else 2
    hash_size = 128 if flash else 4
    for i, size in enumerate(sizes):
        spec = AttentionSpec(
            block_size=size, num_kv_heads=1, head_size=8, dtype=torch.bfloat16
        )
        config = (
            spec
            if i == full_group
            else SlidingWindowManager.layer_config(
                spec,
                sliding_window=8 if i == 4 else 128,
                bounded_replay=flash,
                extra_retained_tokens=max(0, lookahead - 1),
            )
        )
        group = new_kv_cache_group([f"layer{i}"], config)
        groups.append(group)
    if flash:
        spec = AttentionSpec(
            block_size=8, num_kv_heads=1, head_size=8, dtype=torch.bfloat16
        )
        groups.append(
            new_kv_cache_group(["circular"], CircularBufferManager.layer_config(spec))
        )
    kv_config = KVCacheConfig(4096, [], groups, prefix_cache_retention_interval=0)
    options = dict(
        max_model_len=4096,
        max_in_flight_tokens=4096,
        scheduler_block_size=128 if flash else 256,
        hash_block_size=hash_size,
        num_hash_lookahead_tokens=lookahead,
    )
    core = get_kv_cache_coordinator(
        kv_config,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        **options,
    )
    request = make_request(
        "owner",
        list(range(1540)),
        hash_size,
        sha256,
        num_hash_lookahead_tokens=lookahead,
    )
    # Keep replay tails at 1024; retain full attention beyond that boundary.
    for group_id, manager in enumerate(core.single_type_managers):
        if not groups[group_id].prefix_cacheable:
            continue
        count = 1536 // manager.block_size
        blocks = core.block_pool.get_new_blocks(count)
        core.block_pool.cache_full_blocks(
            request, blocks, 0, count, manager.block_size, group_id
        )
    pool = core.block_pool
    if not flash:
        to_remove = {
            b.block_id
            for b in pool.blocks
            if b.block_hash is not None
            and get_group_id(b.block_hash) != 2
            and not 768 < (b.block_hash_num_tokens or 0) <= 1088
        }
        pool.evict_blocks(to_remove)
    present = {
        (get_block_hash(b.block_hash), get_group_id(b.block_hash))
        for b in pool.blocks
        if b.block_hash is not None
    }
    external = ExternalKVCacheLookup(kv_config, **options)
    for limit in [256, 768, 1024, 1280, 1536]:
        matched_blocks, expected_length, _ = core.find_longest_cache_hit(
            request.block_hashes, limit, request.num_tokens
        )
        expected = tuple(
            tuple(
                None
                if b.is_null or b.block_hash is None
                else get_block_hash(b.block_hash)
                for b in group
            )
            for group in matched_blocks
        )
        result = external.find_longest_cache_hit(
            request.block_hashes,
            limit,
            lambda h, g: (h, g) in present,
            num_request_tokens=request.num_tokens,
        )
        assert result == (expected_length, expected)
