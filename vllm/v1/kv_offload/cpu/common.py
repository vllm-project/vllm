# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.kv_offload.base import BlockIDsLoadStoreSpec


class CPUOffloadingMetrics:
    STORES_SKIPPED = "vllm:kv_offload_stores_skipped"
    CPU_CACHE_USAGE_PERC = "vllm:kv_offload_cpu_cache_usage_perc"
    CPU_ALLOCATION_SIZE = "vllm:kv_offload_cpu_allocation_size"
    CPU_CACHE_WRITE_USAGE_PERC = "vllm:kv_offload_cpu_cache_write_usage_perc"
    CPU_CACHE_READ_USAGE_PERC = "vllm:kv_offload_cpu_cache_read_usage_perc"
    CPU_CONFIG_INFO = "vllm:kv_offload_cpu_config_info"


# Label names for the CPU_CONFIG_INFO metric. build_metric_definitions()
# declares these as the metric's labels, and CPUOffloadingManager.get_config_info()
# returns a matching {label: value} dict that record_config_info() emits once at
# startup, reading the values back out in this label order.
CPU_CONFIG_INFO_LABELS = (
    "num_blocks",
    "blocks_per_chunk",
    "kv_bytes_per_chunk",
    "cpu_page_size_per_worker",
    "eviction_policy",
)


class CPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """
    Spec for loading/storing a KV block to CPU memory.
    """
