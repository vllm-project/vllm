# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest

from vllm.v1.core.kv_cache_metrics import (
    BlockMetricsState,
    KVCacheMetricsCollector,
)
from vllm.v1.core.kv_cache_utils import KVCacheBlock


class TestBlockMetricsState:
    def test_init(self):
        with patch("time.monotonic_ns", return_value=1000000000):
            state = BlockMetricsState()
            assert state.birth_time_ns == 1000000000
            assert state.last_access_ns == 1000000000
            assert len(state.access_history) == 0

    def test_access_tracking(self):
        with patch("time.monotonic_ns", return_value=1000000000):
            state = BlockMetricsState()

        with patch("time.monotonic_ns", return_value=2000000000):
            state.record_access()

        assert state.last_access_ns == 2000000000
        assert list(state.access_history) == [2000000000]

    def test_ring_buffer_wraps_at_4(self):
        with patch("time.monotonic_ns", return_value=1000000000):
            state = BlockMetricsState()

        for i in range(5):
            t = 1000000000 + (i + 1) * 1000000000
            with patch("time.monotonic_ns", return_value=t):
                state.record_access()

        assert len(state.access_history) == 4
        assert list(state.access_history) == [
            3000000000,
            4000000000,
            5000000000,
            6000000000,
        ]

    def test_lifetime(self):
        with patch("time.monotonic_ns", return_value=1000000000):
            state = BlockMetricsState()
        with patch("time.monotonic_ns", return_value=6500000000):
            assert abs(state.get_lifetime_seconds() - 5.5) < 0.001

    def test_idle_time(self):
        with patch("time.monotonic_ns", return_value=1000000000):
            state = BlockMetricsState()
        state.last_access_ns = 2000000000
        with patch("time.monotonic_ns", return_value=5200000000):
            assert abs(state.get_idle_time_seconds() - 3.2) < 0.001

    def test_reuse_gaps(self):
        with patch("time.monotonic_ns", return_value=1000000000):
            state = BlockMetricsState()

        base = 1000000000
        for offset in [0, 1.5, 3.0, 5.5]:
            state.access_history.append(base + int(offset * 1e9))

        gaps = state.get_reuse_gaps_seconds()
        assert len(gaps) == 3
        assert gaps[0] == 1.5 and gaps[1] == 1.5 and gaps[2] == 2.5

    def test_ring_wrap_only_gives_3_gaps(self):
        # 5 accesses in size-4 buffer = 3 gaps
        with patch("time.monotonic_ns", return_value=1000000000):
            state = BlockMetricsState()

        for i in range(5):
            state.access_history.append(1000000000 + i * 1000000000)

        assert len(state.get_reuse_gaps_seconds()) == 3


class TestKVCacheMetricsCollector:
    def test_sample_rate_validation(self):
        with pytest.raises(AssertionError):
            KVCacheMetricsCollector(sample_rate=-0.1)
        with pytest.raises(AssertionError):
            KVCacheMetricsCollector(sample_rate=1.5)
        with pytest.raises(AssertionError):
            KVCacheMetricsCollector(sample_rate=0.0)

    def test_sampling(self):
        c = KVCacheMetricsCollector(sample_rate=1.0)
        assert sum(1 for _ in range(100) if c.should_sample_block()) == 100

        c = KVCacheMetricsCollector(sample_rate=0.5)
        samples = sum(1 for _ in range(1000) if c.should_sample_block())
        assert 400 < samples < 600

    def test_alloc(self):
        c = KVCacheMetricsCollector(sample_rate=1.0)

        blocks = [KVCacheBlock(block_id=i) for i in range(5)]
        with patch("time.monotonic_ns", return_value=1000000000):
            for block in blocks:
                c.on_block_allocated(block)

        assert len(c.block_metrics) == 5

    def test_access(self):
        c = KVCacheMetricsCollector(sample_rate=1.0)
        block = KVCacheBlock(block_id=0)

        with patch("time.monotonic_ns", return_value=1000000000):
            c.on_block_allocated(block)

        for i in range(3):
            t = 1000000000 + (i + 1) * 1000000000
            with patch("time.monotonic_ns", return_value=t):
                c.on_block_accessed(block)

        assert len(c.block_metrics[0].access_history) == 3

    def test_evict_no_accesses(self):
        # lifetime should equal idle if never accessed
        c = KVCacheMetricsCollector(sample_rate=1.0)

        block = KVCacheBlock(block_id=0)
        with patch("time.monotonic_ns", return_value=1000000000):
            c.on_block_allocated(block)

        with patch("time.monotonic_ns", return_value=6000000000):
            c.on_block_evicted(block)

        events = c.drain_events()
        assert len(events) == 1
        assert abs(events[0].lifetime_seconds - 5.0) < 0.001
        assert abs(events[0].idle_seconds - 5.0) < 0.001

    def test_evict(self):
        c = KVCacheMetricsCollector(sample_rate=1.0)

        block = KVCacheBlock(block_id=0)
        with patch("time.monotonic_ns", return_value=1000000000):
            c.on_block_allocated(block)

        with patch("time.monotonic_ns", return_value=2000000000):
            c.on_block_accessed(block)
        with patch("time.monotonic_ns", return_value=3000000000):
            c.on_block_accessed(block)

        with patch("time.monotonic_ns", return_value=4000000000):
            c.on_block_evicted(block)

        events = c.drain_events()
        assert len(events) == 1
        sample = events[0]
        assert abs(sample.lifetime_seconds - 3.0) < 0.001
        assert abs(sample.idle_seconds - 1.0) < 0.001
        assert sample.reuse_gaps_seconds == (1.0,)
        assert 0 not in c.block_metrics

    def test_reset(self):
        c = KVCacheMetricsCollector(sample_rate=1.0)

        with patch("time.monotonic_ns", return_value=1000000000):
            for i in range(5):
                c.on_block_allocated(KVCacheBlock(block_id=i))

        assert len(c.block_metrics) == 5
        c.reset()
        assert len(c.block_metrics) == 0

        with patch("time.monotonic_ns", return_value=2000000000):
            c.on_block_allocated(KVCacheBlock(block_id=10))
        assert 10 in c.block_metrics

    def test_huge_time_jump(self):
        c = KVCacheMetricsCollector(sample_rate=1.0)

        block = KVCacheBlock(block_id=0)
        with patch("time.monotonic_ns", return_value=1000000000):
            c.on_block_allocated(block)

        with patch("time.monotonic_ns", return_value=9999999999999999):
            c.on_block_evicted(block)

        events = c.drain_events()
        assert len(events) == 1
        assert events[0].lifetime_seconds > 0


def test_kv_cache_metrics_collector_smoke() -> None:
    """Simple smoke test for KVCacheMetricsCollector on CPU."""
    collector = KVCacheMetricsCollector(sample_rate=1.0)
    block = KVCacheBlock(block_id=123)

    # Allocate at t = 1.0s.
    with patch("time.monotonic_ns", return_value=1_000_000_000):
        collector.on_block_allocated(block)

    # Access at t = 2.0s and t = 3.0s.
    with patch("time.monotonic_ns", return_value=2_000_000_000):
        collector.on_block_accessed(block)
    with patch("time.monotonic_ns", return_value=3_000_000_000):
        collector.on_block_accessed(block)

    # Evict at t = 4.0s.
    with patch("time.monotonic_ns", return_value=4_000_000_000):
        collector.on_block_evicted(block)

    events = collector.drain_events()
    assert len(events) == 1

    event = events[0]
    # Lifetime: 1.0s → 4.0s.
    assert abs(event.lifetime_seconds - 3.0) < 1e-6
    # Idle: last access at 3.0s, evicted at 4.0s.
    assert abs(event.idle_seconds - 1.0) < 1e-6
    # One reuse gap between the two accesses.
    assert event.reuse_gaps_seconds == (1.0,)
