# SPDX-License-Identifier: Apache-2.0
"""Tests for ``l1_exposes_single_memory_region`` (P2P L1 compatibility)."""

# First Party
from lmcache.v1.distributed.config import (
    EvictionConfig,
    GdsL1Config,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
    l1_exposes_single_memory_region,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig

_SIZE = 128 * 1024 * 1024


def _config(
    memory_config: L1MemoryManagerConfig,
    gds_l1_config: GdsL1Config | None = None,
) -> StorageManagerConfig:
    return StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=memory_config,
            gds_l1_config=gds_l1_config,
            write_ttl_seconds=600,
            read_ttl_seconds=300,
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        l2_adapter_config=L2AdaptersConfig(adapters=[]),
    )


def test_plain_dram_is_single_region():
    config = _config(L1MemoryManagerConfig(size_in_bytes=_SIZE, use_lazy=False))
    assert l1_exposes_single_memory_region(config) is True


def test_gds_l1_is_not_single_region():
    config = _config(
        L1MemoryManagerConfig(size_in_bytes=_SIZE, use_lazy=False),
        gds_l1_config=GdsL1Config(file_location="/tmp/gds_slab", size_in_bytes=_SIZE),
    )
    assert l1_exposes_single_memory_region(config) is False


def test_devdax_l1_is_not_single_region():
    config = _config(
        L1MemoryManagerConfig(
            size_in_bytes=_SIZE,
            use_lazy=False,
            devdax_path="/dev/dax0.0",
            shm_name="",
        )
    )
    assert l1_exposes_single_memory_region(config) is False
