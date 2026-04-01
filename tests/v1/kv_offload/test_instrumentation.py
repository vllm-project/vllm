# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for AccessTracer and OffloadingMetrics.
"""
import json
import tempfile
from pathlib import Path

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.instrumentation import AccessTracer


def to_hash(i: int) -> BlockHash:
    return BlockHash(str(i).encode())


def test_basic_recording():
    """Record various events and verify metrics."""
    tracer = AccessTracer(max_records=1000)

    tracer.record_lookup(to_hash(1), hit=True)
    tracer.record_lookup(to_hash(2), hit=True)
    tracer.record_lookup(to_hash(3), hit=False)

    metrics = tracer.get_metrics()
    assert metrics.total_lookups == 3
    assert metrics.total_hits == 2
    assert metrics.total_misses == 1
    assert abs(metrics.hit_rate - 2 / 3) < 1e-6


def test_store_and_load_tracking():
    """Track stores, loads, and evictions."""
    tracer = AccessTracer()

    tracer.record_store(to_hash(1))
    tracer.record_store(to_hash(2))
    tracer.record_load(to_hash(1))
    tracer.record_eviction(to_hash(2))

    metrics = tracer.get_metrics()
    assert metrics.total_stores == 2
    assert metrics.total_loads == 1
    assert metrics.total_evictions == 1


def test_prefetch_tracking():
    """Track prefetch events and accuracy."""
    tracer = AccessTracer()

    tracer.record_prefetch(to_hash(1), hit=True)
    tracer.record_prefetch(to_hash(2), hit=False)
    tracer.record_prefetch(to_hash(3), hit=True)

    metrics = tracer.get_metrics()
    assert metrics.total_prefetches == 3
    assert metrics.prefetch_hits == 2
    assert abs(metrics.prefetch_accuracy - 2 / 3) < 1e-6


def test_transfer_recording():
    """Record transfer timing and bandwidth."""
    tracer = AccessTracer()

    tracer.record_transfer("gpu_to_cpu", size_bytes=1024, transfer_time=0.001)
    tracer.record_transfer("gpu_to_cpu", size_bytes=2048, transfer_time=0.002)
    tracer.record_transfer("cpu_to_gpu", size_bytes=512, transfer_time=0.0005)

    metrics = tracer.get_metrics()
    assert metrics.bytes_transferred_gpu_to_cpu == 3072
    assert metrics.bytes_transferred_cpu_to_gpu == 512
    assert abs(metrics.avg_transfer_time_gpu_to_cpu - 0.0015) < 1e-6


def test_reuse_distance():
    """Reuse distance is computed correctly."""
    tracer = AccessTracer()

    # Access pattern: 1, 2, 3, 1, 2, 1
    tracer.record_lookup(to_hash(1), hit=True)
    tracer.record_lookup(to_hash(2), hit=True)
    tracer.record_lookup(to_hash(3), hit=True)
    tracer.record_lookup(to_hash(1), hit=True)  # reuse dist = 3
    tracer.record_lookup(to_hash(2), hit=True)  # reuse dist = 3
    tracer.record_lookup(to_hash(1), hit=True)  # reuse dist = 2

    stats = tracer.compute_workload_stats()
    assert stats["reuse_distance"]["count"] == 3
    # distances: [3, 3, 2], mean = 8/3
    assert abs(stats["reuse_distance"]["mean"] - 8 / 3) < 1e-6


def test_access_frequency():
    """Access frequency distribution is tracked correctly."""
    tracer = AccessTracer()

    for _ in range(5):
        tracer.record_lookup(to_hash(1), hit=True)
    for _ in range(3):
        tracer.record_lookup(to_hash(2), hit=True)
    tracer.record_lookup(to_hash(3), hit=True)

    stats = tracer.compute_workload_stats()
    assert stats["access_frequency"]["max"] == 5
    assert stats["access_frequency"]["num_unique_blocks"] == 3

    # Hot blocks should list block 1 first
    hot = stats["hot_blocks"]
    assert len(hot) == 3
    assert hot[0]["access_count"] == 5


def test_export_traces():
    """Export traces to JSONL file."""
    tracer = AccessTracer(max_records=100)

    tracer.record_lookup(to_hash(1), hit=True, attention_score=0.5)
    tracer.record_store(to_hash(2))
    tracer.record_eviction(to_hash(1))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "traces.jsonl")
        count = tracer.export_traces(path)
        assert count == 3

        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 3

        record = json.loads(lines[0])
        assert record["event_type"] == "lookup"
        assert record["hit"] is True
        assert record["attention_score"] == 0.5


def test_max_records_limit():
    """Records are capped at max_records."""
    tracer = AccessTracer(max_records=5)

    for i in range(10):
        tracer.record_lookup(to_hash(i), hit=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "traces.jsonl")
        count = tracer.export_traces(path)
        assert count == 5


def test_metrics_window_reset():
    """get_metrics resets the current window."""
    tracer = AccessTracer()

    tracer.record_lookup(to_hash(1), hit=True)
    metrics1 = tracer.get_metrics()
    assert metrics1.total_lookups == 1

    # Second window should start fresh
    tracer.record_lookup(to_hash(2), hit=False)
    metrics2 = tracer.get_metrics()
    assert metrics2.total_lookups == 1
    assert metrics2.total_misses == 1


def test_clear():
    """Clear resets all state."""
    tracer = AccessTracer()

    tracer.record_lookup(to_hash(1), hit=True)
    tracer.record_store(to_hash(2))
    tracer.clear()

    metrics = tracer.get_metrics()
    assert metrics.total_lookups == 0
    assert metrics.total_stores == 0

    stats = tracer.compute_workload_stats()
    assert stats["reuse_distance"] == {}
    assert stats["access_frequency"] == {}
