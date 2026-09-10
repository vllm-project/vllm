# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.request import Request


def get_computed_blocks_for_group_completion(
    cache_manager: KVCacheManager,
    request: Request,
    completion_group_ids: frozenset[int],
) -> tuple[KVCacheBlocks, int, int, bool, int | None]:
    """Preserve deeper local groups while a connector restores lagging ones."""
    if not cache_manager.prefix_cache_lookup_enabled(request):
        return cache_manager.empty_kv_cache_blocks, 0, 0, False, None

    persistent_group_ids = {
        group_id
        for group_id, group in enumerate(cache_manager.kv_cache_config.kv_cache_groups)
        if group.kv_cache_spec.prefix_cacheable
    }
    completion_group_ids = completion_group_ids.intersection(persistent_group_ids)
    fixed_group_ids = persistent_group_ids - completion_group_ids
    if not completion_group_ids or not fixed_group_ids:
        return *cache_manager.get_computed_blocks(request), False, None

    coordinator = cache_manager.coordinator
    assert isinstance(coordinator, HybridKVCacheCoordinator)
    computed, per_group_hits = coordinator.find_longest_cache_hit_per_group(
        request.block_hashes, request.num_tokens - 1
    )
    local_hit = min(per_group_hits[group_id] for group_id in persistent_group_ids)
    completion_boundary = min(per_group_hits[group_id] for group_id in fixed_group_ids)
    if completion_boundary <= local_hit:
        return *cache_manager.get_computed_blocks(request), False, 0

    blocks = truncate_group_completion_blocks(
        cache_manager,
        cache_manager.create_kv_cache_blocks(computed),
        local_hit,
        completion_boundary,
        completion_group_ids,
    )
    return blocks, local_hit, 0, True, completion_boundary - local_hit


def truncate_group_completion_blocks(
    cache_manager: KVCacheManager,
    blocks: KVCacheBlocks,
    num_local_computed_tokens: int,
    num_completed_tokens: int,
    completion_group_ids: frozenset[int],
) -> KVCacheBlocks:
    """Keep connector-completed groups at the pre-transfer local boundary."""
    truncated: list[list[KVCacheBlock]] = []
    for group_id, (group_blocks, manager, group) in enumerate(
        zip(
            blocks.blocks,
            cache_manager.coordinator.single_type_managers,
            cache_manager.kv_cache_config.kv_cache_groups,
            strict=True,
        )
    ):
        if not group.kv_cache_spec.prefix_cacheable:
            truncated.append(list(group_blocks))
            continue
        endpoint = (
            num_local_computed_tokens
            if group_id in completion_group_ids
            else num_completed_tokens
        )
        assert endpoint % manager.block_size == 0
        num_blocks = endpoint // manager.block_size
        assert num_blocks <= len(group_blocks)
        truncated.append(list(group_blocks[:num_blocks]))
    return cache_manager.create_kv_cache_blocks(tuple(truncated))
