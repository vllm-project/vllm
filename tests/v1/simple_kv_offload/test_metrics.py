# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SimpleCPUOffloadConnector metrics."""

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
    OffloadingOperationMetrics,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    SimpleCPUOffloadConnector,
    SimpleCPUOffloadConnectorStats,
)
from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker

pytestmark = pytest.mark.skip_global_cleanup


def test_build_kv_connector_stats_with_none():
    stats = SimpleCPUOffloadConnector.build_kv_connector_stats(data=None)

    assert stats is not None
    assert isinstance(stats, SimpleCPUOffloadConnectorStats)
    assert stats.is_empty()


def test_build_kv_connector_stats_reconstructs_serialized_payload():
    serialized_data = {
        "CPU_to_GPU": [
            {"op_size": 16, "op_time": 1.0},
            {"op_size": 8, "op_time": 0.5},
        ],
        "GPU_to_CPU": [{"op_size": 4, "op_time": 0.25}],
    }

    stats = SimpleCPUOffloadConnector.build_kv_connector_stats(data=serialized_data)

    assert isinstance(stats, SimpleCPUOffloadConnectorStats)
    assert stats.data == serialized_data


def test_simple_cpu_offload_stats_aggregate_and_reduce_transfers_and_pool():
    stats1 = SimpleCPUOffloadConnectorStats(
        data={
            "CPU_to_GPU": [{"op_size": 16, "op_time": 1.0}],
            "GPU_to_CPU": [{"op_size": 4, "op_time": 0.25}],
            "cpu_pool_total_blocks": 8,
            "cpu_pool_free_blocks": 6,
            "cpu_pool_used_blocks": 2,
            "cpu_pool_usage_perc": 0.25,
        }
    )
    stats2 = SimpleCPUOffloadConnectorStats(
        data={
            "CPU_to_GPU": [{"op_size": 8, "op_time": 0.5}],
            "GPU_to_CPU": [{"op_size": 12, "op_time": 0.75}],
            "cpu_pool_total_blocks": 8,
            "cpu_pool_free_blocks": 4,
            "cpu_pool_used_blocks": 4,
            "cpu_pool_usage_perc": 0.5,
        }
    )

    stats1.aggregate(stats2)
    reduced = stats1.reduce()

    assert reduced["CPU_to_GPU_total_bytes"] == 24
    assert reduced["CPU_to_GPU_total_time"] == 1.5
    assert reduced["GPU_to_CPU_total_bytes"] == 16
    assert reduced["GPU_to_CPU_total_time"] == 1.0
    assert reduced["cpu_pool_total_blocks"] == 8
    assert reduced["cpu_pool_free_blocks"] == 4
    assert reduced["cpu_pool_used_blocks"] == 4
    assert reduced["cpu_pool_usage_perc"] == 0.5


def test_worker_get_kv_connector_stats_returns_once_then_resets():
    worker = SimpleCPUOffloadWorker(
        vllm_config=None,  # type: ignore[arg-type]
        kv_cache_config=None,
        cpu_capacity_bytes=1024,
    )
    worker.kv_connector_stats.record_transfer(16, 1.0, ("CPU", "GPU"))

    stats = worker.get_kv_connector_stats()
    assert isinstance(stats, OffloadingConnectorStats)
    assert stats.data == {
        "CPU_to_GPU": [OffloadingOperationMetrics(op_size=16, op_time=1.0)]
    }

    assert worker.get_kv_connector_stats() is None
