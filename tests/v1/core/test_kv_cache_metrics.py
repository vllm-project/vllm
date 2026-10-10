# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_metrics import (
    BlockMetricsState,
    KVCacheMetricsCollector,
)
from vllm.v1.core.kv_cache_utils import (
    KVCacheBlock,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.request import Request


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
        state.idle_since_ns = 2000000000
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
        # A never-released block is still busy at eviction.
        c = KVCacheMetricsCollector(sample_rate=1.0)

        block = KVCacheBlock(block_id=0)
        with patch("time.monotonic_ns", return_value=1000000000):
            c.on_block_allocated(block)

        with patch("time.monotonic_ns", return_value=6000000000):
            c.on_block_evicted(block)

        events = c.drain_events()
        assert len(events) == 1
        assert abs(events[0].lifetime_seconds - 5.0) < 0.001
        assert events[0].idle_seconds == 0.0

    def test_evict(self):
        c = KVCacheMetricsCollector(sample_rate=1.0)

        block = KVCacheBlock(block_id=0)
        with patch("time.monotonic_ns", return_value=1000000000):
            c.on_block_allocated(block)

        with patch("time.monotonic_ns", return_value=2000000000):
            c.on_block_accessed(block)
        with patch("time.monotonic_ns", return_value=3000000000):
            c.on_block_accessed(block)
            c.on_block_freed(block)

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
        collector.on_block_freed(block)

    # Evict at t = 4.0s.
    with patch("time.monotonic_ns", return_value=4_000_000_000):
        collector.on_block_evicted(block)

    events = collector.drain_events()
    assert len(events) == 1

    event = events[0]
    # Lifetime: 1.0s → 4.0s.
    assert abs(event.lifetime_seconds - 3.0) < 1e-6
    # Idle: released at 3.0s, evicted at 4.0s.
    assert abs(event.idle_seconds - 1.0) < 1e-6
    # One reuse gap between the two accesses.
    assert event.reuse_gaps_seconds == (1.0,)


@pytest.fixture
def cached_pool():
    init_none_hash(sha256)
    collector = KVCacheMetricsCollector(sample_rate=1.0)
    pool = BlockPool(2, True, 16, metrics_collector=collector)
    with patch("time.monotonic_ns", return_value=0):
        block = pool.get_new_blocks(1)[0]
    request = Request(
        request_id="test",
        prompt_token_ids=list(range(16)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_hasher=get_request_block_hasher(16, sha256),
    )
    pool.cache_full_blocks(request, [block], 0, 1, 16, 7)
    return pool, collector, block


@pytest.mark.parametrize("release", ["free_blocks", "unpin_blocks"])
def test_idle_starts_when_last_reference_is_released(cached_pool, release):
    pool, collector, block = cached_pool
    with patch("time.monotonic_ns", return_value=99_000_000_000):
        if release == "free_blocks":
            pool.free_blocks([block])
        else:
            pool.unpin_blocks([block], on_reuse=lambda _: None)
    with patch("time.monotonic_ns", return_value=100_000_000_000):
        pool.get_new_blocks(1)
    (event,) = collector.drain_events()
    assert event.lifetime_seconds == 100.0
    assert event.idle_seconds == 1.0


@pytest.mark.parametrize("evict_while_pinned", [False, True])
def test_transfer_pins_do_not_count_as_prefix_reuses(cached_pool, evict_while_pinned):
    pool, collector, block = cached_pool
    with patch("time.monotonic_ns", return_value=3_000_000_000):
        pool.touch([block])
        pool.free_blocks([block, block])
    with patch("time.monotonic_ns", return_value=5_000_000_000):
        pool.touch([block], record_access=False)
    with patch("time.monotonic_ns", return_value=7_000_000_000):
        pool.touch([block])
        pool.free_blocks([block])
    if not evict_while_pinned:
        with patch("time.monotonic_ns", return_value=9_000_000_000):
            pool.free_blocks([block])
    with patch("time.monotonic_ns", return_value=10_000_000_000):
        pool.evict_blocks({block.block_id})
    (event,) = collector.drain_events()
    assert event.lifetime_seconds == 10.0
    assert event.idle_seconds == (0.0 if evict_while_pinned else 1.0)
    assert event.reuse_gaps_seconds == (4.0,)


@pytest.mark.parametrize("eviction_path", ["reallocate", "connector"])
def test_uncached_blocks_do_not_emit_eviction_samples(eviction_path):
    collector = KVCacheMetricsCollector(sample_rate=1.0)
    pool = BlockPool(2, True, 16, metrics_collector=collector)
    (block,) = pool.get_new_blocks(1)
    if eviction_path == "reallocate":
        pool.free_blocks([block])
        pool.get_new_blocks(1)
    else:
        pool.evict_blocks({block.block_id})
        # A no-op invalidation must preserve tracking if this allocation is
        # subsequently published to the prefix cache.
        assert block.block_id in collector.block_metrics
    assert collector.drain_events() == []


@pytest.mark.parametrize("enable_caching", [False, True])
def test_unsampled_reallocation_discards_previous_lifetime(enable_caching):
    collector = KVCacheMetricsCollector(sample_rate=1.0)
    pool = BlockPool(2, enable_caching, 16, metrics_collector=collector)
    with patch.object(collector, "should_sample_block", side_effect=[True, False]):
        (block,) = pool.get_new_blocks(1)
        pool.free_blocks([block])
        (reused,) = pool.get_new_blocks(1)
    assert reused is block
    assert collector.block_metrics == {}
    assert collector.drain_events() == []
