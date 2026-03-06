# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for BlockReuseTracker (Strategy A / P0)."""
import pytest

from vllm.v1.kv_offload.reuse_tracker import BlockReuseTracker


# Use simple integer hashes for testing (BlockHash is a hashable type)
def h(n: int) -> int:
    """Fake block hash — just an integer."""
    return n


class TestBlockReuseTrackerThreshold:
    """Threshold behaviour: store_threshold defaults to 2."""

    def test_first_occurrence_skip(self):
        t = BlockReuseTracker(store_threshold=2)
        assert t.record_and_check(h(1)) is False

    def test_second_occurrence_store(self):
        t = BlockReuseTracker(store_threshold=2)
        t.record_and_check(h(1))
        assert t.record_and_check(h(1)) is True

    def test_third_occurrence_still_stores(self):
        t = BlockReuseTracker(store_threshold=2)
        t.record_and_check(h(1))
        t.record_and_check(h(1))
        assert t.record_and_check(h(1)) is True

    def test_threshold_one_always_stores(self):
        t = BlockReuseTracker(store_threshold=1)
        assert t.record_and_check(h(99)) is True

    def test_threshold_three(self):
        t = BlockReuseTracker(store_threshold=3)
        assert t.record_and_check(h(5)) is False  # count=1
        assert t.record_and_check(h(5)) is False  # count=2
        assert t.record_and_check(h(5)) is True   # count=3

    def test_independent_hashes(self):
        t = BlockReuseTracker(store_threshold=2)
        t.record_and_check(h(1))
        # h(2) is tracked independently from h(1)
        assert t.record_and_check(h(2)) is False


class TestBlockReuseTrackerLRUEviction:
    """LRU eviction when max_size is exceeded."""

    def test_eviction_drops_lru_entry(self):
        t = BlockReuseTracker(max_size=3, store_threshold=2)
        t.record_and_check(h(1))  # h(1) count=1
        t.record_and_check(h(2))  # h(2) count=1
        t.record_and_check(h(3))  # h(3) count=1 — full
        # Adding h(4) should evict h(1) (LRU)
        t.record_and_check(h(4))
        # h(1) was evicted; re-adding starts its count at 1 again
        assert t.record_and_check(h(1)) is False  # count=1 again (was evicted)

    def test_recently_accessed_survives_eviction(self):
        t = BlockReuseTracker(max_size=3, store_threshold=2)
        t.record_and_check(h(1))
        t.record_and_check(h(2))
        t.record_and_check(h(3))
        # Access h(1) again — moves it to MRU position
        t.record_and_check(h(1))  # count=2, now MRU
        # Adding h(4) should evict h(2) (now the LRU)
        t.record_and_check(h(4))
        # h(1) should still be tracked with count=2
        assert h(1) in t
        assert h(2) not in t

    def test_size_stays_bounded(self):
        t = BlockReuseTracker(max_size=10, store_threshold=2)
        for i in range(50):
            t.record_and_check(h(i))
        assert len(t) <= 10

    def test_eviction_on_max_size_zero(self):
        # edge case: max_size=0 means every entry is immediately evicted
        # (initial entry fills it, then gets evicted on next insert)
        t = BlockReuseTracker(max_size=1, store_threshold=2)
        t.record_and_check(h(1))  # fills the single slot
        t.record_and_check(h(2))  # evicts h(1), adds h(2)
        assert h(1) not in t
        assert h(2) in t


class TestBlockReuseTrackerBurstyInput:
    """Simulate bursty traffic with many distinct hashes."""

    def test_bursty_unique_hashes_never_store(self):
        t = BlockReuseTracker(max_size=100, store_threshold=2)
        results = [t.record_and_check(h(i)) for i in range(200)]
        # All unique → all should be False (first occurrences only)
        assert not any(results)

    def test_repeated_hash_within_burst_stores(self):
        t = BlockReuseTracker(max_size=100, store_threshold=2)
        for i in range(50):
            t.record_and_check(h(i))  # first pass: all False
        # Second pass over same hashes: should all return True
        for i in range(50):
            if h(i) in t:  # only check if not already evicted
                assert t.record_and_check(h(i)) is True


class TestBlockReuseTrackerContainsDunder:
    def test_contains(self):
        t = BlockReuseTracker()
        t.record_and_check(h(42))
        assert h(42) in t
        assert h(99) not in t

    def test_len(self):
        t = BlockReuseTracker(max_size=5)
        for i in range(5):
            t.record_and_check(h(i))
        assert len(t) == 5


class TestTransferTimingStats:
    """Unit tests for TransferTimingStats (lives here as it's a small module)."""

    def test_default_before_min_observations(self):
        from vllm.v1.kv_offload.transfer_timing import TransferTimingStats
        s = TransferTimingStats(min_observations=10, default_ms_per_token=0.003)
        for _ in range(9):
            s.record(tokens=100, elapsed_ms=1.0)
        # Not enough observations yet
        assert s.ms_per_token == pytest.approx(0.003)

    def test_rolling_average_after_enough_samples(self):
        from vllm.v1.kv_offload.transfer_timing import TransferTimingStats
        s = TransferTimingStats(min_observations=5, default_ms_per_token=0.003)
        for _ in range(5):
            # 100 tokens in 0.5 ms -> 0.005 ms/token
            s.record(tokens=100, elapsed_ms=0.5)
        assert s.ms_per_token == pytest.approx(0.005)

    def test_zero_token_guard_ignored(self):
        from vllm.v1.kv_offload.transfer_timing import TransferTimingStats
        s = TransferTimingStats(min_observations=1)
        s.record(tokens=0, elapsed_ms=1.0)  # should be ignored
        assert len(s) == 0

    def test_estimate_ms(self):
        from vllm.v1.kv_offload.transfer_timing import TransferTimingStats
        s = TransferTimingStats(min_observations=1, default_ms_per_token=0.01)
        assert s.estimate_ms(100) == pytest.approx(1.0)

    def test_window_eviction(self):
        from vllm.v1.kv_offload.transfer_timing import TransferTimingStats
        s = TransferTimingStats(window_size=3, min_observations=1)
        s.record(100, 10.0)  # 0.1 ms/tok — will be evicted
        s.record(100, 10.0)
        s.record(100, 10.0)
        s.record(100, 1.0)   # 0.01 ms/tok — replaces first
        # window now has [10.0, 10.0, 1.0] -> avg = 21/300 ms/token
        # but the oldest (10.0) was evicted first
        assert len(s) == 3
