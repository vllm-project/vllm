# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metric declarations for the HiSparse connector."""

from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    CounterMetadata,
    MetricDefinitions,
)


class HiSparseMetricName:
    CACHE_HITS = "vllm:hisparse_cache_hits"
    CACHE_MISSES = "vllm:hisparse_cache_misses"
    HOST_TO_DEVICE_BYTES = "vllm:hisparse_host_to_device_bytes"


HISPARSE_METRIC_DEFINITIONS: MetricDefinitions = {
    HiSparseMetricName.CACHE_HITS: CounterMetadata(
        documentation="Number of HiSparse device hot-buffer hits.",
    ),
    HiSparseMetricName.CACHE_MISSES: CounterMetadata(
        documentation="Number of HiSparse device hot-buffer misses.",
    ),
    HiSparseMetricName.HOST_TO_DEVICE_BYTES: CounterMetadata(
        documentation=(
            "Bytes transferred from host KV storage to HiSparse hot buffers."
        ),
    ),
}
