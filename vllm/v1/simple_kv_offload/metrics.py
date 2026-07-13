# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side gauge metric definitions for SimpleCPUOffloadConnector."""

from vllm.v1.kv_offload.base import OffloadingGaugeMetadata, OffloadingMetricMetadata


class SimpleCPUMetricName:
    """Flat gauge metric names for the SimpleCPU offload block pool."""

    TOTAL_BLOCKS = "vllm:simple_cpu_offload_total_blocks"
    FREE_BLOCKS = "vllm:simple_cpu_offload_free_blocks"
    USED_BLOCKS = "vllm:simple_cpu_offload_used_blocks"
    USAGE_PERC = "vllm:simple_cpu_offload_usage_perc"
    PENDING_LOADS = "vllm:simple_cpu_offload_pending_loads"
    PENDING_STORES = "vllm:simple_cpu_offload_pending_stores"


def get_simple_cpu_metric_definitions() -> dict[str, OffloadingMetricMetadata]:
    return {
        SimpleCPUMetricName.TOTAL_BLOCKS: OffloadingGaugeMetadata(
            documentation="Total number of usable CPU KV cache blocks "
            "(excludes the null block).",
        ),
        SimpleCPUMetricName.FREE_BLOCKS: OffloadingGaugeMetadata(
            documentation="Number of free CPU KV cache blocks.",
        ),
        SimpleCPUMetricName.USED_BLOCKS: OffloadingGaugeMetadata(
            documentation="Number of used CPU KV cache blocks.",
        ),
        SimpleCPUMetricName.USAGE_PERC: OffloadingGaugeMetadata(
            documentation="Fraction of CPU KV cache blocks in use, between "
            "0.0 and 1.0 (1.0 = 100% used).",
        ),
        SimpleCPUMetricName.PENDING_LOADS: OffloadingGaugeMetadata(
            documentation="Number of requests with an outstanding CPU-to-GPU "
            "load, including abandoned loads still draining after a cache "
            "reset.",
        ),
        SimpleCPUMetricName.PENDING_STORES: OffloadingGaugeMetadata(
            documentation="Number of in-flight GPU-to-CPU store events, "
            "including abandoned ones still draining.",
        ),
    }
