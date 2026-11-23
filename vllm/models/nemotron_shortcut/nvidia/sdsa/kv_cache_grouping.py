# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SDSA's KV cache grouping requirement.

An SDSA layer has two caches: its core's K/V cache (`<layer>.core`) and its
indexer's key cache (`<layer>.indexer.cache`). The indexer selects keys by
position and maps them to core cache slots through its own block table, so the
two must share one KV cache group (one block table, one block size). A model
with SDSA layers inherits SdsaKVCacheGrouping, which adds each indexer cache to
its core's group.
"""

from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import SupportsKVCacheGrouping
from vllm.v1.kv_cache_interface import KVCacheGroup, KVCacheLayerConfig

# Suffixes of an SDSA layer's cache names (SdsaAttention's prefix + suffix).
SDSA_CORE_SUFFIX = ".core"
SDSA_INDEXER_SUFFIX = ".indexer"
SDSA_INDEXER_CACHE_SUFFIX = f"{SDSA_INDEXER_SUFFIX}.cache"


class SdsaKVCacheGrouping(SupportsKVCacheGrouping):
    """vLLM's default KV cache grouping, with each SDSA indexer cache in its
    core's group."""

    @classmethod
    def get_kv_cache_groups(
        cls,
        vllm_config: VllmConfig,
        layer_configs: dict[str, KVCacheLayerConfig],
    ) -> list[KVCacheGroup]:
        from vllm.v1.core.kv_cache_planning import get_kv_cache_groups

        others = {
            name: config
            for name, config in layer_configs.items()
            if not name.endswith(SDSA_INDEXER_CACHE_SUFFIX)
        }
        groups = get_kv_cache_groups(vllm_config, others)
        for group in groups:
            indexers = [
                name.removesuffix(SDSA_CORE_SUFFIX) + SDSA_INDEXER_CACHE_SUFFIX
                for name in group.layer_names
                if name.endswith(SDSA_CORE_SUFFIX)
            ]
            # After the cores, so that the cores keep the regions that they
            # share with the layers of the other groups.
            for name in indexers:
                config = layer_configs[name]
                assert (
                    config.manager_config == group.manager_config
                    and config.spec.block_size == group.block_size
                    and config.spec.dcp_sharded == group.dcp_sharded
                ), f"{name} cannot share its core's block table"
                group.kv_cache_specs[name] = config.spec
        return groups
