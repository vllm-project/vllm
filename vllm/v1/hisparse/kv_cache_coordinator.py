# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HiSparse allocation ordering on top of the existing KV cache coordinators."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.v1.core.kv_cache_coordinator import (
    HybridKVCacheCoordinator,
    KVCacheCoordinator,
    KVCacheCoordinatorNoPrefixCache,
)
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_interface import KVCacheConfig

if TYPE_CHECKING:
    from vllm.v1.hisparse.coordinator import HiSparseCoordinator


class _HiSparseKVCacheCoordinator(KVCacheCoordinator):
    hisparse_coordinator: "HiSparseCoordinator"

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
        prefill_end: int = 0,
    ) -> int:
        required = super().get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            new_computed_blocks,
            num_encoder_tokens,
            total_computed_tokens,
            num_local_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap=apply_admission_cap,
            prefill_end=prefill_end,
        )
        if required > self.block_pool.num_gpu_blocks:
            return required
        return max(
            0,
            required - self.hisparse_coordinator.get_num_reclaimable_blocks(request_id),
        )

    def allocate_new_blocks(
        self,
        request_id: str,
        num_tokens: int,
        num_tokens_main_model: int,
        num_encoder_tokens: int = 0,
    ) -> tuple[list[KVCacheBlock], ...]:
        hot_blocks = self.hisparse_coordinator.prepare_allocation(
            request_id, num_tokens, num_tokens_main_model
        )
        blocks = super().allocate_new_blocks(
            request_id, num_tokens, num_tokens_main_model, num_encoder_tokens
        )
        # Already prepared hot groups need no further blocks in the base call.
        return tuple(hot_blocks.get(i, group) for i, group in enumerate(blocks))


class _HiSparseNoPrefixCacheCoordinator(
    _HiSparseKVCacheCoordinator, KVCacheCoordinatorNoPrefixCache
):
    pass


class _HiSparseHybridKVCacheCoordinator(
    _HiSparseKVCacheCoordinator, HybridKVCacheCoordinator
):
    pass


def get_hisparse_kv_cache_coordinator(
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    max_in_flight_tokens: int,
    use_eagle: bool,
    enable_caching: bool,
    enable_kv_cache_events: bool,
    dcp_world_size: int,
    pcp_world_size: int,
    scheduler_block_size: int,
    hash_block_size: int,
    metrics_collector: KVCacheMetricsCollector | None = None,
    num_hash_lookahead_tokens: int = 0,
    num_prefill_lookahead: int = 0,
    allow_partial_hash_hits: bool = True,
) -> KVCacheCoordinator:
    if not enable_caching:
        return _HiSparseNoPrefixCacheCoordinator(
            kv_cache_config,
            max_model_len,
            max_in_flight_tokens,
            use_eagle,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
            num_hash_lookahead_tokens=num_hash_lookahead_tokens,
            num_prefill_lookahead=num_prefill_lookahead,
        )
    return _HiSparseHybridKVCacheCoordinator(
        kv_cache_config,
        max_model_len,
        max_in_flight_tokens,
        use_eagle,
        enable_caching,
        enable_kv_cache_events,
        dcp_world_size=dcp_world_size,
        pcp_world_size=pcp_world_size,
        scheduler_block_size=scheduler_block_size,
        hash_block_size=hash_block_size,
        metrics_collector=metrics_collector,
        num_hash_lookahead_tokens=num_hash_lookahead_tokens,
        num_prefill_lookahead=num_prefill_lookahead,
        allow_partial_hash_hits=allow_partial_hash_hits,
    )
