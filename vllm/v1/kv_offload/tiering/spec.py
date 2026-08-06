# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TieringOffloadingSpec: Spec for multi-tier KV cache offloading.

This spec creates a TieringOffloadingManager with a CPU primary tier
and configurable secondary tiers (e.g., Storage, Network).

Configuration via kv_connector_extra_config:
  - cpu_bytes_to_use: (required) Bytes to allocate for CPU primary tier
  - block_size: (optional) Block size for offloaded blocks (default: GPU block size)
  - eviction_policy: (optional) Primary tier eviction policy: built-in "lru"/
    "arc", or the name of a policy registered via CachePolicyFactory, or an
    out-of-tree CachePolicy class name paired with cache_policy_module_path
    (default: "lru")
  - cache_policy_module_path: (optional) Python import path to load
    eviction_policy from when it names an out-of-tree CachePolicy not
    registered via CachePolicyFactory
  - secondary_tiers: (optional) List of secondary tier configurations
    Each secondary tier config is a dict with:
      - type: (required) Type of secondary tier (e.g., "example", "fs",
        "p2p", "obj"), or the class name of an out-of-tree
        SecondaryTierManager paired with module_path.
      - module_path: (optional) Python import path to load 'type' from
        when it names an out-of-tree SecondaryTierManager not registered
        via SecondaryTierFactory.register_tier()
      - Additional tier-specific parameters are passed directly to the tier
        constructor. See each tier's documentation for supported parameters.

Example configuration:
{
    "cpu_bytes_to_use": 10737418240,  # 10 GB
    "block_size": 16,
    "eviction_policy": "lru",
    "secondary_tiers": [
        {
            "type": "example",
            "custom_param": 67
        }
    ]
}

Example out-of-tree tier configuration:
{
    "cpu_bytes_to_use": 10737418240,
    "secondary_tiers": [
        {
            "type": "MyCustomTier",
            "module_path": "my_package.my_module",
            "custom_param": "value"
        }
    ]
}
"""

from typing import Any

import torch
from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingHistogramMetadata,
    OffloadingManager,
    OffloadingMetricMetadata,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.tiering.base import TieringOffloadingMetrics
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)

logger = init_logger(__name__)


class TieringOffloadingSpec(CPUOffloadingSpec):
    """
    Spec for multi-tier KV cache offloading.

    Creates a TieringOffloadingManager with:
    - Primary tier: CPU (LRU or ARC eviction policy)
    - Secondary tiers: Configurable via extra_config

    The CPU primary tier has direct GPU access and serves as the gateway for
    all GPU↔offload operations. Secondary tiers cannot directly access GPU
    memory and must transfer data through the primary tier.
    """

    BLOCK_SIZE_ALIGNMENT = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT

    @classmethod
    @override
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        metrics = super().build_metric_definitions(extra_config)
        metrics[TieringOffloadingMetrics.LOOKUP_SYNC_DELAY] = (
            OffloadingHistogramMetadata(
                documentation=(
                    "Histogram of total blocking time spent querying secondary "
                    "tiers for a request, accumulated from first lookup until "
                    "the request is allocated or finishes, in seconds."
                ),
                buckets=(
                    0.00001,
                    0.00005,
                    0.0001,
                    0.0005,
                    0.001,
                    0.005,
                    0.01,
                    0.05,
                    0.1,
                    0.5,
                    1,
                ),
            )
        )
        metrics[TieringOffloadingMetrics.LOOKUP_ASYNC_DELAY] = (
            OffloadingHistogramMetadata(
                documentation=(
                    "Histogram of wall-clock time from a request's first deferred "
                    "secondary-tier lookup until the request is allocated or "
                    "finishes, in seconds."
                ),
                buckets=(
                    0.0001,
                    0.0005,
                    0.001,
                    0.005,
                    0.01,
                    0.05,
                    0.1,
                    0.5,
                    1,
                    5,
                    10,
                ),
            )
        )
        secondary_tier_configs = extra_config.get("secondary_tiers", [])
        if not isinstance(secondary_tier_configs, list):
            raise ValueError("secondary_tiers must be a list of tier configurations")

        for tier_config in secondary_tier_configs:
            assert isinstance(tier_config, dict)
            tier_cls = SecondaryTierFactory.get_tier_class(tier_config)
            metrics.update(tier_cls.build_metric_definitions(tier_config))
        return metrics

    def __init__(self, config: OffloadingConfig):
        super().__init__(config)
        # Redeclare for mypy: parent sets this but `--follow-imports skip` hides it
        self._manager: OffloadingManager | None = None

        # Parse secondary tier configurations
        self.secondary_tier_configs = self.extra_config.get("secondary_tiers", [])
        if not isinstance(self.secondary_tier_configs, list):
            raise ValueError("secondary_tiers must be a list of tier configurations")

        # Scheduler-side mmap (rank=None); kept for cleanup
        self._scheduler_mmap: SharedOffloadRegion | None = None

        # engine_id is unique per DP replica (suffixed with _dp{rank} in both
        # the Ray and multiprocessing paths), so it names a per-replica offload
        # region.
        self._engine_id = config.engine_id

    @override
    def get_manager(self) -> OffloadingManager:
        """
        Get the TieringOffloadingManager.

        Creates a TieringOffloadingManager with:
        - Primary tier: CPU (LRU or ARC)
        - Secondary tiers: As configured in extra_config

        Returns:
            TieringOffloadingManager instance
        """
        if not self._manager:
            if int(self.extra_config.get("store_threshold", 0)) >= 2:
                raise ValueError(
                    "store_threshold is not supported for TieringOffloadingSpec"
                )

            scheduler_mmap: SharedOffloadRegion | None = None
            primary_tier: CPUPrimaryTierOffloadingManager | None = None
            secondary_tiers = []
            try:
                # Create scheduler-side SharedOffloadRegion (rank=None) so the
                # primary tier can eagerly create a memoryview over _base.
                scheduler_mmap = SharedOffloadRegion(
                    engine_id=self._engine_id,
                    num_blocks=self.num_blocks,
                    rank=None,
                    kv_bytes_per_block=self.kv_bytes_per_chunk,
                    cpu_page_size=self.cpu_page_size_per_worker,
                )
                self._scheduler_mmap = scheduler_mmap

                # Create primary tier (CPU-based)
                primary_tier = CPUPrimaryTierOffloadingManager(
                    num_blocks=self.num_blocks,
                    cache_policy=self.eviction_policy,
                    cache_policy_module_path=self.cache_policy_module_path,
                    enable_events=self.kv_events_config.enable_kv_cache_events,
                    mmap_region=scheduler_mmap,
                )

                # Create secondary tiers
                primary_kv_view = primary_tier.get_kv_memoryview()
                for i, tier_config in enumerate(self.secondary_tier_configs):
                    tier = SecondaryTierFactory.create_secondary_tier(
                        tier_config, primary_kv_view, self
                    )
                    secondary_tiers.append(tier)
                    logger.info(
                        "Created secondary tier #%d (%s)",
                        i,
                        tier.tier_type,
                    )

                # Create TieringOffloadingManager. GPU↔CPU transfers use the inherited
                # get_worker(). Secondary tier transfers are handled by the
                # secondary tier managers and need no additional workers here.
                tiering_manager = TieringOffloadingManager(
                    primary_tier=primary_tier,
                    secondary_tiers=secondary_tiers,
                )
                self._manager = tiering_manager
            except Exception:
                for tier in reversed(secondary_tiers):
                    try:
                        tier.shutdown()
                    except Exception:
                        logger.exception(
                            "Failed to shut down secondary tier during "
                            "initialization cleanup"
                        )
                if primary_tier is not None:
                    try:
                        primary_tier.shutdown()
                    except Exception:
                        logger.exception(
                            "Failed to shut down primary tier during "
                            "initialization cleanup"
                        )
                elif scheduler_mmap is not None:
                    try:
                        scheduler_mmap.cleanup()
                    except Exception:
                        logger.exception(
                            "Failed to clean up scheduler mmap during "
                            "initialization cleanup"
                        )
                self._scheduler_mmap = None
                raise

            logger.info(
                "Created TieringOffloadingManager with primary tier "
                "(%s, %s blocks) and %s secondary tier(s)",
                self.eviction_policy,
                self.num_blocks,
                len(secondary_tiers),
            )

        return self._manager

    @override
    def _uses_shared_region(self) -> bool:
        # Tiering always allocates on the shared region (every platform), so the
        # replicated-layout gate must not be narrowed by the CPU spec's
        # CUDA-alike check.
        return True

    @override
    def create_worker(self, kv_caches: CanonicalKVCaches) -> CPUOffloadingWorker:
        world_size = self.config.parallel.world_size
        if self.replicated_layout:
            rank = 0
        else:
            # Fold the global physical device index into the replica-local
            # [0, world_size) slot range.
            rank = torch.accelerator.current_device_index() % world_size
        worker_mmap = SharedOffloadRegion(
            engine_id=self._engine_id,
            num_blocks=self.num_blocks,
            rank=rank,
            kv_bytes_per_block=self.kv_bytes_per_chunk,
            cpu_page_size=self.cpu_page_size_per_worker,
        )
        try:
            return CPUOffloadingWorker(
                kv_caches=kv_caches,
                blocks_per_chunk=self.blocks_per_chunk,
                num_cpu_blocks=self.num_blocks,
                mmap_region=worker_mmap,
            )
        except Exception:
            worker_mmap.cleanup()
            raise
