# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from unittest.mock import MagicMock, patch

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
            assert state.max_request_end_ns == 0

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
            2000000000,
            3000000000,
            4000000000,
            5000000000,
        ]

    def test_lifetime(self):
        with patch("time.monotonic_ns", return_value=1000000000):
            state = BlockMetricsState()

        assert abs(state.get_lifetime_seconds(6500000000) - 5.5) < 0.001

    def test_idle_time(self):
        with patch("time.monotonic_ns", return_value=1000000000):
            state = BlockMetricsState()
        state.last_access_ns = 2000000000

        assert abs(state.get_idle_time_seconds(5200000000) - 3.2) < 0.001

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
    def test_disabled(self):
        c = KVCacheMetricsCollector(enabled=False)
        assert not c.enabled

        block = KVCacheBlock(block_id=0)
        c.on_block_allocated(block)
        c.on_block_accessed(block)
        c.on_block_evicted(block)

        assert len(c.block_metrics) == 0

    def test_sample_rate_validation(self):
        c = KVCacheMetricsCollector(enabled=True, sample_rate=-0.1)
        assert not c.enabled

        c = KVCacheMetricsCollector(enabled=True, sample_rate=1.5)
        assert c.sample_rate == 1.0

    def test_sampling(self):
        c = KVCacheMetricsCollector(enabled=True, sample_rate=1.0)
        assert sum(1 for _ in range(100) if c.should_sample_block()) == 100

        c = KVCacheMetricsCollector(enabled=True, sample_rate=0.5)
        samples = sum(1 for _ in range(1000) if c.should_sample_block())
        assert 400 < samples < 600

    def test_alloc(self):
        c = KVCacheMetricsCollector(enabled=True, sample_rate=1.0)

        blocks = [KVCacheBlock(block_id=i) for i in range(5)]
        with patch("time.monotonic_ns", return_value=1000000000):
            for block in blocks:
                c.on_block_allocated(block)

        assert len(c.block_metrics) == 5

    def test_access(self):
        c = KVCacheMetricsCollector(enabled=True, sample_rate=1.0)
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
        c = KVCacheMetricsCollector(enabled=True, sample_rate=1.0)
        c.histogram_block_lifetime = MagicMock()
        c.histogram_idle_before_evict = MagicMock()

        block = KVCacheBlock(block_id=0)
        with patch("time.monotonic_ns", return_value=1000000000):
            c.on_block_allocated(block)

        with patch("time.monotonic_ns", return_value=6000000000):
            c.on_block_evicted(block)

        lifetime = c.histogram_block_lifetime.observe.call_args[0][0]
        idle = c.histogram_idle_before_evict.observe.call_args[0][0]
        assert abs(lifetime - 5.0) < 0.001
        assert abs(idle - 5.0) < 0.001

    def test_evict(self):
        c = KVCacheMetricsCollector(enabled=True, sample_rate=1.0)
        c.histogram_block_lifetime = MagicMock()
        c.histogram_idle_before_evict = MagicMock()
        c.histogram_reuse_gap = MagicMock()

        block = KVCacheBlock(block_id=0)
        with patch("time.monotonic_ns", return_value=1000000000):
            c.on_block_allocated(block)

        with patch("time.monotonic_ns", return_value=2000000000):
            c.on_block_accessed(block)
        with patch("time.monotonic_ns", return_value=3000000000):
            c.on_block_accessed(block)

        with patch("time.monotonic_ns", return_value=4000000000):
            c.on_block_evicted(block)

        assert c.histogram_block_lifetime.observe.called
        assert c.histogram_idle_before_evict.observe.called
        assert c.histogram_reuse_gap.observe.called
        assert 0 not in c.block_metrics

    def test_prefix_residency(self):
        c = KVCacheMetricsCollector(enabled=True, sample_rate=1.0)
        c.histogram_prefix_residency = MagicMock()

        block = KVCacheBlock(block_id=0)
        with patch("time.monotonic_ns", return_value=1000000000):
            c.on_block_allocated(block)

        with patch("time.monotonic_ns", return_value=2000000000):
            c.on_request_prefill_complete("req1", {0})
        with patch("time.monotonic_ns", return_value=5000000000):
            c.on_request_prefill_complete("req2", {0})

        # tracks latest request
        assert c.block_metrics[0].max_request_end_ns == 5000000000

        with patch("time.monotonic_ns", return_value=10000000000):
            c.on_block_evicted(block)

        residency = c.histogram_prefix_residency.observe.call_args[0][0]
        assert abs(residency - 5.0) < 0.001

    def test_reset(self):
        c = KVCacheMetricsCollector(enabled=True, sample_rate=1.0)

        with patch("time.monotonic_ns", return_value=1000000000):
            for i in range(5):
                c.on_block_allocated(KVCacheBlock(block_id=i))

        assert len(c.block_metrics) == 5
        c.reset()
        assert len(c.block_metrics) == 0

        with patch("time.monotonic_ns", return_value=2000000000):
            c.on_block_allocated(KVCacheBlock(block_id=10))
        assert 10 in c.block_metrics

    def test_concurrent_access(self):
        # shouldn't crash with concurrent calls
        c = KVCacheMetricsCollector(enabled=True, sample_rate=1.0)
        c.histogram_block_lifetime = MagicMock()
        c.histogram_idle_before_evict = MagicMock()

        def worker(op, start, n):
            for i in range(n):
                block = KVCacheBlock(block_id=start + i)
                if op == "alloc":
                    c.on_block_allocated(block)
                elif op == "access":
                    c.on_block_accessed(block)
                else:
                    c.on_block_evicted(block)

        threads = [
            threading.Thread(target=worker, args=("alloc", 0, 50)),
            threading.Thread(target=worker, args=("alloc", 50, 50)),
            threading.Thread(target=worker, args=("access", 0, 50)),
            threading.Thread(target=worker, args=("evict", 0, 25)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def test_huge_time_jump(self):
        c = KVCacheMetricsCollector(enabled=True, sample_rate=1.0)
        c.histogram_block_lifetime = MagicMock()

        block = KVCacheBlock(block_id=0)
        with patch("time.monotonic_ns", return_value=1000000000):
            c.on_block_allocated(block)

        with patch("time.monotonic_ns", return_value=9999999999999999):
            c.on_block_evicted(block)

        assert c.histogram_block_lifetime.observe.called
