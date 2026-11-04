# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.utils import KVImportLandingPlan
from vllm.v1.hisparse.coordinator import get_hisparse_coordinator
from vllm.v1.kv_cache_interface import HiSparseResidentSpec, KVCacheGroupRole

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager


class HiSparseImportLanding:
    """Land a connector import directly on resident GPU pages."""

    def __init__(self, kv_cache_manager: "KVCacheManager") -> None:
        self.coordinator = get_hisparse_coordinator(kv_cache_manager)
        groups = kv_cache_manager.kv_cache_config.kv_cache_groups
        (self.indexer_group_id,) = (
            group_id
            for group_id, group in enumerate(groups)
            if group.role is KVCacheGroupRole.HISPARSE_INDEXER
        )
        (self.source_group_id,) = (
            group_id
            for group_id, group in enumerate(groups)
            if group.role is KVCacheGroupRole.HISPARSE_SOURCE
        )
        self.resident_group_ids = [
            group_id
            for group_id, group in enumerate(groups)
            if isinstance(group.kv_cache_spec, HiSparseResidentSpec)
        ]

    def prepare(self, request_id: str) -> None:
        self.coordinator.prepare_gpu_import(request_id)

    def plan(
        self,
        request_id: str,
        blocks: "KVCacheBlocks",
        num_computed_blocks: tuple[int, ...],
    ) -> KVImportLandingPlan | None:
        """Pull host source pages into resident GPU pages past the local prefix.

        Resident pages are never hashed. The indexer's hashed pages give the
        local hit length; host source pages can outlast it.
        """
        if self.coordinator.imports_to_host(request_id):
            return None
        num_prefix_pages = num_computed_blocks[self.indexer_group_id]
        alias_block_ids = {
            group_id: [
                block.block_id
                for block in blocks.blocks[group_id][num_prefix_pages:]
                if not block.is_null
            ]
            for group_id in self.resident_group_ids
        }
        self.coordinator.record_gpu_import(
            request_id,
            num_prefix_pages + max(map(len, alias_block_ids.values()), default=0),
        )
        computed = list(num_computed_blocks)
        computed[self.source_group_id] = num_prefix_pages
        return KVImportLandingPlan(alias_block_ids, tuple(computed))
