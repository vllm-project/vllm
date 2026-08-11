# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.kv_offload.base import BlockIDsLoadStoreSpec


class CPUOffloadingMetrics:
    STORES_SKIPPED = "vllm:kv_offload_stores_skipped"
    CPU_CACHE_USAGE_PERC = "vllm:kv_offload_cpu_cache_usage_perc"
    CPU_ALLOCATION_SIZE = "vllm:kv_offload_cpu_allocation_size"
    CPU_CACHE_WRITE_USAGE_PERC = "vllm:kv_offload_cpu_cache_write_usage_perc"
    CPU_CACHE_READ_USAGE_PERC = "vllm:kv_offload_cpu_cache_read_usage_perc"


class CPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """
    Spec for loading/storing KV chunks to/from CPU memory.

    The inherited block_ids field holds chunk indices into the
    CPU cache (not GPU block IDs).
    """
