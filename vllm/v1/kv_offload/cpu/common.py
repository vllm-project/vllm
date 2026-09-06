# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, fields

from vllm.v1.kv_offload.base import BlockIDsLoadStoreSpec


class CPUOffloadingMetrics:
    STORES_SKIPPED = "vllm:kv_offload_stores_skipped"
    CPU_CACHE_USAGE_PERC = "vllm:kv_offload_cpu_cache_usage_perc"
    CPU_ALLOCATION_SIZE = "vllm:kv_offload_cpu_allocation_size"
    CPU_CACHE_WRITE_USAGE_PERC = "vllm:kv_offload_cpu_cache_write_usage_perc"
    CPU_CACHE_READ_USAGE_PERC = "vllm:kv_offload_cpu_cache_read_usage_perc"
    CPU_CONFIG_INFO = "vllm:kv_offload_cpu_config_info"


@dataclass(frozen=True)
class CPUCacheTierInfo:
    """
    Static, per-engine facts about the CPU offload tier.
    """

    # Chunk count, not GPU blocks; see blocks_per_chunk.
    num_chunks: int
    # GPU blocks per chunk; the CPU-slot to GPU-block conversion factor.
    blocks_per_chunk: int
    # Page-aligned bytes per chunk. With num_chunks this is the tier's exact
    # size in bytes, the only capacity valid for every model shape.
    kv_bytes_per_chunk: int
    # KV tokens the tier can serve when it is full and every request reaches
    # max_model_len. The largest value the capacity takes, because a longer
    # request spreads the fixed per-request chunks (one Mamba state, one sliding
    # window) over more tokens. None when max_model_len is not known.
    # See _capacity_tokens_at_max_len.
    capacity_tokens_at_max_len: int | None

    def as_labelvalues(self) -> tuple[str, ...]:
        """Render label values in CPU_TIER_INFO_LABELS order."""
        return tuple(str(getattr(self, name)) for name in CPU_TIER_INFO_LABELS)


# Derived from CPUCacheTierInfo so the declaration and the emission cannot drift;
# the offloading metrics path binds label values positionally.
CPU_TIER_INFO_LABELS: tuple[str, ...] = tuple(f.name for f in fields(CPUCacheTierInfo))


class CPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """
    Spec for loading/storing a KV block to CPU memory.
    """
