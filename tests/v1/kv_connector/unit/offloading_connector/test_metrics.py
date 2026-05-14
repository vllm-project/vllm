# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OFFLOAD_CACHE_USAGE_KEY,
    OffloadingConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)


def test_build_kv_connector_stats_with_none():
    """Test that build_kv_connector_stats returns empty stats when given None."""
    stats = OffloadingConnector.build_kv_connector_stats(data=None)

    assert stats is not None
    assert isinstance(stats, OffloadingConnectorStats)
    assert len(stats.data) == 0
    assert stats.is_empty()


def test_build_kv_connector_stats_with_empty_dict():
    """Test that build_kv_connector_stats returns empty stats with empty dict."""
    stats = OffloadingConnector.build_kv_connector_stats(data={})

    assert stats is not None
    assert isinstance(stats, OffloadingConnectorStats)
    assert len(stats.data) == 0
    assert stats.is_empty()


def test_build_kv_connector_stats_reconstructs_offload_stats():
    """Test that OffloadingConnector stats are properly reconstructed with
    correct data."""
    serialized_data = {
        "CPU_to_GPU": [
            {"op_size": 16, "op_time": 1.0},
            {"op_size": 8, "op_time": 0.5},
        ],
        "GPU_to_CPU": [
            {"op_size": 1, "op_time": 0.1},
            {"op_size": 2, "op_time": 0.2},
        ],
    }

    stats = OffloadingConnector.build_kv_connector_stats(data=serialized_data)

    offload_connector_stats = stats
    assert isinstance(offload_connector_stats, OffloadingConnectorStats)
    assert offload_connector_stats.data["CPU_to_GPU"] == [
        {"op_size": 16, "op_time": 1.0},
        {"op_size": 8, "op_time": 0.5},
    ]
    assert offload_connector_stats.data["GPU_to_CPU"] == [
        {"op_size": 1, "op_time": 0.1},
        {"op_size": 2, "op_time": 0.2},
    ]


def test_aggregate_same_connector():
    """Test aggregating stats from the same connector type."""
    stats1 = OffloadingConnectorStats(
        data={
            "CPU_to_GPU": [
                {"op_size": 16, "op_time": 1.0},
                {"op_size": 8, "op_time": 0.5},
            ],
            "GPU_to_CPU": [
                {"op_size": 1, "op_time": 0.1},
                {"op_size": 2, "op_time": 0.2},
            ],
        }
    )

    stats2 = OffloadingConnectorStats(
        data={
            "CPU_to_GPU": [
                {"op_size": 3, "op_time": 0.2},
                {"op_size": 7, "op_time": 0.9},
            ],
            "GPU_to_CPU": [{"op_size": 16, "op_time": 2}],
        }
    )

    result = stats1.aggregate(stats2)

    assert result is stats1  # Should return self
    offload_connector_stats = result
    assert offload_connector_stats.data["CPU_to_GPU"] == [
        {"op_size": 16, "op_time": 1.0},
        {"op_size": 8, "op_time": 0.5},
        {"op_size": 3, "op_time": 0.2},
        {"op_size": 7, "op_time": 0.9},
    ]
    assert offload_connector_stats.data["GPU_to_CPU"] == [
        {"op_size": 1, "op_time": 0.1},
        {"op_size": 2, "op_time": 0.2},
        {"op_size": 16, "op_time": 2},
    ]


def test_reduce():
    """Test that reduce() correctly reduces all nested connector stats."""
    stats = OffloadingConnectorStats(
        data={
            "CPU_to_GPU": [
                {"op_size": 16, "op_time": 1.0},
                {"op_size": 8, "op_time": 0.5},
                {"op_size": 3, "op_time": 0.2},
                {"op_size": 7, "op_time": 0.9},
            ],
            "GPU_to_CPU": [
                {"op_size": 1, "op_time": 0.1},
                {"op_size": 2, "op_time": 0.2},
                {"op_size": 16, "op_time": 2},
            ],
        }
    )

    reduced = stats.reduce()

    assert isinstance(reduced, dict)
    # Check that the stats were reduced (should have aggregated values)
    assert "CPU_to_GPU_total_bytes" in reduced
    assert "CPU_to_GPU_total_time" in reduced
    assert "GPU_to_CPU_total_bytes" in reduced
    assert "GPU_to_CPU_total_time" in reduced
    assert reduced["CPU_to_GPU_total_bytes"] == 34
    assert reduced["CPU_to_GPU_total_time"] == 2.6
    assert reduced["GPU_to_CPU_total_time"] == 2.3
    assert reduced["GPU_to_CPU_total_bytes"] == 19


def test_reset():
    """Test that reset() resets all nested connector stats."""
    offload_connector_stats = OffloadingConnectorStats(
        data={
            "CPU_to_GPU": [
                {"op_size": 3, "op_time": 0.2},
                {"op_size": 7, "op_time": 0.9},
            ],
            "GPU_to_CPU": [{"op_size": 16, "op_time": 2}],
        }
    )

    assert not offload_connector_stats.is_empty()

    offload_connector_stats.reset()

    # After reset, stats should be empty
    assert offload_connector_stats.is_empty()
    assert len(offload_connector_stats.data) == 0


def test_reduce_offload_usage_uses_max():
    stats = OffloadingConnectorStats()
    stats.record_offload_usage(0.2)
    stats.record_offload_usage(0.7)
    stats.record_offload_usage(0.4)

    reduced = stats.reduce()

    assert reduced[OFFLOAD_CACHE_USAGE_KEY] == 0.7


def test_aggregate_merges_offload_usage_samples():
    stats1 = OffloadingConnectorStats()
    stats1.record_offload_usage(0.1)
    stats1.record_offload_usage(0.3)

    stats2 = OffloadingConnectorStats()
    stats2.record_offload_usage(0.8)

    merged = stats1.aggregate(stats2)

    assert merged is stats1
    assert stats1.data[OFFLOAD_CACHE_USAGE_KEY] == [0.1, 0.3, 0.8]


def test_scheduler_get_kv_connector_stats_includes_offload_usage():
    connector = OffloadingConnector.__new__(OffloadingConnector)
    connector.connector_worker = None
    connector.connector_scheduler = SimpleNamespace(
        manager=SimpleNamespace(offload_cache_usage_fraction=lambda: 0.42)
    )

    stats = connector.get_kv_connector_stats()

    assert isinstance(stats, OffloadingConnectorStats)
    assert stats.data[OFFLOAD_CACHE_USAGE_KEY] == [0.42]
