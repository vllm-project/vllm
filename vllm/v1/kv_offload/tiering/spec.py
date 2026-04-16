# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TieringOffloadingSpec: Spec for multi-tier KV cache offloading.

This spec creates a TieringOffloadingManager with a CPU-based primary tier
and configurable secondary tiers (e.g., Storage, Network).

Configuration via kv_connector_extra_config:
  - cpu_bytes_to_use: (required) Bytes to allocate for CPU primary tier
  - block_size: (optional) Block size for offloaded blocks (default: GPU block size)
  - eviction_policy: (optional) Primary tier eviction policy: "lru" or
    "arc" (default: "lru")
  - store_threshold: (optional) How many times a block must appear in lookup()
    before it is eligible for CPU offloading. Values < 2 disable filtering
    (default: 0)
  - max_tracker_size: (optional) Maximum number of blocks tracked for
    store_threshold filtering (default: 64000)
  - secondary_tiers: (optional) List of secondary tier configurations
    Each secondary tier config is a dict with:
      - type: (required) Type of secondary tier (e.g., "dummy", "storage", "network")
      - tier_name: (required) Name for this tier (used for logging and identification)
      - Additional tier-specific parameters are passed directly to the tier
        constructor. See each tier's documentation for supported parameters.

Example configuration:
{
    "cpu_bytes_to_use": 10737418240,  # 10 GB
    "block_size": 16,
    "eviction_policy": "lru",
    "secondary_tiers": [
        {
            "type": "dummy",
            "tier_name": "TestStorage",
            # Tier-specific parameters (for DummySecondaryTier):
            "max_blocks": 10000,
            "simulate_async": False
        }
    ]
}
"""

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.abstract import OffloadingManager
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.secondary_tiers.dummy import DummySecondaryTier
from vllm.v1.kv_offload.spec import CanonicalKVCaches
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)
from vllm.v1.kv_offload.worker.cpu_gpu import CpuGpuOffloadingHandlers

logger = init_logger(__name__)


class TieringOffloadingSpec(CPUOffloadingSpec):
    """
    Spec for multi-tier KV cache offloading.

    Creates a TieringOffloadingManager with:
    - Primary tier: CPU-based (LRU or ARC eviction policy)
    - Secondary tiers: Configurable via extra_config

    The primary tier has direct GPU access and serves as the gateway for all
    GPU↔offload operations. Secondary tiers cannot directly access GPU memory
    and must coordinate with the primary tier for data transfers.
    """

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)

        # Parse secondary tier configurations
        self.secondary_tier_configs = self.extra_config.get("secondary_tiers", [])
        if not isinstance(self.secondary_tier_configs, list):
            raise ValueError("secondary_tiers must be a list of tier configurations")

        # Scheduler-side mmap (rank=None); kept for cleanup
        self._scheduler_mmap: SharedOffloadRegion | None = None

    def _create_secondary_tier(self, tier_config: dict):
        """
        Create a secondary tier from configuration.

        Args:
            tier_config: Dictionary with tier configuration containing:
                - type (required): Type of secondary tier (e.g., "dummy")
                - tier_name (required): Name for this tier
                - Additional tier-specific parameters are passed directly
                  to the tier constructor

        Returns:
            SecondaryTierManager instance

        Raises:
            ValueError: If tier type is unknown or configuration is invalid
        """
        # Make a copy to avoid modifying the original config
        config = tier_config.copy()

        # Extract common parameters
        tier_type = config.pop("type", None)
        if not tier_type:
            raise ValueError("Secondary tier configuration must include 'type'")

        tier_name = config.pop("tier_name", None)
        if not tier_name:
            raise ValueError("Secondary tier configuration must include 'tier_name'")

        # Remaining parameters in config are tier-specific
        if tier_type == "dummy":
            # DummySecondaryTier for testing
            # Pass tier_name and tier-specific params to constructor
            return DummySecondaryTier(tier_name=tier_name, **config)
        else:
            raise ValueError(
                f"Unknown secondary tier type: {tier_type}. Supported types: dummy"
            )

    def get_manager(self) -> OffloadingManager:
        """
        Get the TieringOffloadingManager.

        Creates a TieringOffloadingManager with:
        - Primary tier: CPU-based (LRU or ARC)
        - Secondary tiers: As configured in extra_config

        Returns:
            TieringOffloadingManager instance
        """
        if not self._manager:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None and kv_events_config.enable_kv_cache_events
            )

            # Create scheduler-side SharedOffloadRegion (rank=None) first so
            # CPUPrimaryTierOffloadingManager can return _base from
            # get_primary_kv_tensor(), which TieringOffloadingManager.__init__
            # calls immediately to wire secondary tier memoryviews.
            world_size = self.vllm_config.parallel_config.world_size
            scheduler_mmap = SharedOffloadRegion(
                instance_id=self.vllm_config.instance_id,
                total_size_bytes=self.cpu_page_size_per_worker
                * world_size
                * self.num_blocks,
                num_blocks=self.num_blocks,
                rank=None,
                num_workers=world_size,
                cpu_page_size=self.cpu_page_size_per_worker,
            )
            self._scheduler_mmap = scheduler_mmap

            # Create primary tier (CPU-based)
            assert len(self.gpu_block_size) == 1
            offloaded_block_size = self.gpu_block_size[0] * self.block_size_factor
            primary_tier = CPUPrimaryTierOffloadingManager(
                block_size=offloaded_block_size,
                num_blocks=self.num_blocks,
                cache_policy=self.eviction_policy,  # type: ignore[arg-type]
                enable_events=enable_events,
                mmap_region=scheduler_mmap,
            )

            # Create secondary tiers
            secondary_tiers = []
            for tier_config in self.secondary_tier_configs:
                try:
                    tier = self._create_secondary_tier(tier_config)
                    secondary_tiers.append(tier)
                    logger.info(
                        "Created secondary tier: %s (type: %s)",
                        tier.get_tier_name(),
                        tier_config.get("type"),
                    )
                except Exception as e:
                    logger.error(
                        "Failed to create secondary tier from config %s: %s",
                        tier_config,
                        e,
                    )
                    raise

            # Create TieringOffloadingManager. GPU↔CPU transfers use the inherited
            # get_handlers(); secondary tier transfers are handled by the
            # secondary tier managers and need no additional handlers here.
            tiering_manager = TieringOffloadingManager(
                primary_tier=primary_tier,
                secondary_tiers=secondary_tiers,
                enable_events=enable_events,
            )
            # PRNOTE: should the store_filter apply to the TieringOffloadingManager or
            # to the primary CPU manager?
            self._manager = self._maybe_apply_store_filter(tiering_manager)

            logger.info(
                "Created TieringOffloadingManager with primary tier "
                "(%s, %s blocks) and %s secondary tier(s)",
                self.eviction_policy,
                self.num_blocks,
                len(secondary_tiers),
            )

        return self._manager

    def _create_handlers(
        self, kv_caches: CanonicalKVCaches
    ) -> CpuGpuOffloadingHandlers:
        world_size = self.vllm_config.parallel_config.world_size
        rank = torch.accelerator.current_device_index()
        worker_mmap = SharedOffloadRegion(
            instance_id=self.vllm_config.instance_id,
            total_size_bytes=self.cpu_page_size_per_worker
            * world_size
            * self.num_blocks,
            num_blocks=self.num_blocks,
            rank=rank,
            num_workers=world_size,
            cpu_page_size=self.cpu_page_size_per_worker,
        )
        return CpuGpuOffloadingHandlers(
            kv_caches=kv_caches,
            block_size_factor=self.block_size_factor,
            num_cpu_blocks=self.num_blocks,
            mmap_region=worker_mmap,
        )
