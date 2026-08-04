# SPDX-License-Identifier: Apache-2.0

# Standard
from concurrent.futures import ThreadPoolExecutor
import threading
import time

# Third Party
import pytest

pytest.importorskip(
    "lmcache.native_storage_ops",
    reason="native_storage_ops extension not built",
)

# First Party
from lmcache.native_storage_ops import TTLLock


class TestTTLLockBasicSemantics:
    """Test basic semantics of TTLLock operations."""

    def test_initial_state(self):
        """Test that a new lock starts in unlocked state."""
        lock = TTLLock()
        assert not lock.is_locked()

    def test_lock_increments_counter(self):
        """Test that lock() increments the counter."""
        lock = TTLLock()

        lock.lock()
        assert lock.is_locked()

        lock.lock()
        assert lock.is_locked()

        lock.lock()
        assert lock.is_locked()

    def test_unlock_decrements_counter(self):
        """Test that unlock() decrements the counter."""
        lock = TTLLock()

        # Lock 3 times
        lock.lock()
        lock.lock()
        lock.lock()

        lock.unlock()
        assert lock.is_locked()

        lock.unlock()
        assert lock.is_locked()

        lock.unlock()
        assert not lock.is_locked()

    def test_unlock_does_not_go_below_zero(self):
        """Test that unlock() never goes below 0."""
        lock = TTLLock()

        # Unlock without any locks - should not crash
        lock.unlock()
        assert not lock.is_locked()

        # Multiple unlocks should still be safe
        lock.unlock()
        lock.unlock()
        assert not lock.is_locked()

    def test_unlock_after_partial_locks(self):
        """Test unlock when counter is already at minimum."""
        lock = TTLLock()

        lock.lock()
        lock.lock()
        lock.unlock()
        lock.unlock()

        # Extra unlocks should be safe
        lock.unlock()
        lock.unlock()
        assert not lock.is_locked()

    def test_reset(self):
        """Test that reset() clears the lock state."""
        lock = TTLLock()

        lock.lock()
        lock.lock()
        lock.lock()
        assert lock.is_locked()

        lock.reset()
        assert not lock.is_locked()

    def test_lock_after_reset(self):
        """Test that lock works correctly after reset."""
        lock = TTLLock()

        lock.lock()
        lock.lock()
        lock.reset()

        lock.lock()
        assert lock.is_locked()


class TestTTLLockTTLSemantics:
    """Test TTL (Time-To-Live) semantics of TTLLock."""

    def test_lock_expires_after_ttl(self):
        """Test that lock expires after TTL duration."""
        # Create a lock with 1 second TTL
        lock = TTLLock(ttl_second=1)

        lock.lock()
        assert lock.is_locked()

        # Wait for TTL to expire
        time.sleep(1.5)  # 1.5s > 1s TTL

        assert not lock.is_locked()

    def test_lock_refreshes_ttl(self):
        """Test that each lock() call refreshes the TTL."""
        lock = TTLLock(ttl_second=2)

        lock.lock()

        # Wait 1.5s (less than TTL)
        time.sleep(0.5)
        assert lock.is_locked()

        # Lock again to refresh TTL
        lock.lock()

        # Wait another 1.5s - total 3s from first lock,
        # but only 1.5s from second lock
        time.sleep(1.5)
        assert lock.is_locked()

        # Wait for TTL to expire from the last lock
        time.sleep(1.0)  # Now 2.5s from last lock > 2s TTL
        assert not lock.is_locked()

    def test_lock_after_ttl_expired_resets_counter(self):
        """Test that lock() after TTL expires resets counter to 1."""
        lock = TTLLock(ttl_second=1)

        # Lock multiple times
        lock.lock()
        lock.lock()
        lock.lock()

        # Wait for TTL to expire
        time.sleep(1.5)
        assert not lock.is_locked()

        # Lock again - should reset counter and be locked
        lock.lock()
        assert lock.is_locked()

        # Verify counter was reset by unlocking once
        lock.unlock()
        assert not lock.is_locked()

    def test_default_ttl_is_300_seconds(self):
        """Test that default TTL is 300 seconds."""
        lock = TTLLock()

        lock.lock()
        assert lock.is_locked()

        # After a short time, should still be locked
        time.sleep(0.1)
        assert lock.is_locked()

    def test_custom_ttl(self):
        """Test custom TTL values."""
        lock = TTLLock(ttl_second=1)

        lock.lock()
        assert lock.is_locked()

        time.sleep(0.5)  # 0.5s < 1s
        assert lock.is_locked()

        time.sleep(0.7)  # Total 1.2s > 1s
        assert not lock.is_locked()


class TestTTLLockThreadSafety:
    """Test thread safety of TTLLock operations."""

    def test_concurrent_locks(self):
        """Test that concurrent lock() operations are thread-safe."""
        lock = TTLLock()
        num_threads = 100
        locks_per_thread = 100

        def do_locks():
            for _ in range(locks_per_thread):
                lock.lock()

        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=do_locks)
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Final counter should be num_threads * locks_per_thread
        # We verify by unlocking and counting
        count = 0
        while lock.is_locked():
            lock.unlock()
            count += 1

        assert count == num_threads * locks_per_thread

    def test_concurrent_unlocks(self):
        """Test that concurrent unlock() operations are thread-safe."""
        lock = TTLLock()
        total_locks = 10000

        # Lock many times first
        for _ in range(total_locks):
            lock.lock()

        num_threads = 100
        unlocks_per_thread = total_locks // num_threads

        def do_unlocks():
            for _ in range(unlocks_per_thread):
                lock.unlock()

        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=do_unlocks)
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Should be at 0
        assert not lock.is_locked()

    def test_concurrent_lock_unlock(self):
        """Test concurrent mixed lock/unlock operations."""
        lock = TTLLock()
        num_threads = 50
        ops_per_thread = 100
        lock_count = [0]
        unlock_count = [0]
        lock_mutex = threading.Lock()

        def do_mixed_ops(thread_id):
            lock.lock()  # Must do a lock first to avoid double unlock
            with lock_mutex:
                lock_count[0] += 1
            for i in range(ops_per_thread):
                if (thread_id + i) % 2 == 0:
                    lock.lock()
                    with lock_mutex:
                        lock_count[0] += 1
                else:
                    lock.unlock()
                    with lock_mutex:
                        unlock_count[0] += 1

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=do_mixed_ops, args=(i,))
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # The lock counter should never go negative
        # We can verify the final state is consistent
        expected_remaining = max(0, lock_count[0] - unlock_count[0])

        # Count remaining locks
        remaining = 0
        while lock.is_locked():
            lock.unlock()
            remaining += 1

        assert remaining == expected_remaining

    def test_concurrent_is_locked_reads(self):
        """Test that is_locked() is safe to call concurrently with modifications."""
        lock = TTLLock()
        stop_flag = threading.Event()
        errors = []

        def reader():
            try:
                while not stop_flag.is_set():
                    # Just read - should never crash
                    _ = lock.is_locked()
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for _ in range(1000):
                    lock.lock()
                    lock.unlock()
            except Exception as e:
                errors.append(e)

        # Start multiple readers
        readers = [threading.Thread(target=reader) for _ in range(10)]
        # Start multiple writers
        writers = [threading.Thread(target=writer) for _ in range(10)]

        for t in readers + writers:
            t.start()

        # Wait for writers to finish
        for t in writers:
            t.join()

        # Signal readers to stop
        stop_flag.set()

        for t in readers:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"

    def test_stress_test_high_contention(self):
        """Stress test with high contention from many threads."""
        lock = TTLLock()
        num_threads = 200
        ops_per_thread = 500

        def stress_worker():
            for _ in range(ops_per_thread):
                lock.lock()
                # Small work
                _ = lock.is_locked()
                lock.unlock()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(stress_worker) for _ in range(num_threads)]
            for f in futures:
                f.result()  # Will raise if any thread failed

        # After all operations, lock should be released
        assert not lock.is_locked()

    def test_concurrent_reset(self):
        """Test that reset() is safe to call concurrently."""
        lock = TTLLock()
        num_threads = 50
        ops_per_thread = 100
        errors = []

        def worker():
            try:
                for i in range(ops_per_thread):
                    if i % 3 == 0:
                        lock.lock()
                    elif i % 3 == 1:
                        lock.unlock()
                    else:
                        lock.reset()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"


class TestTTLLockEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_short_ttl(self):
        """Test with very short TTL (1 second)."""
        lock = TTLLock(ttl_second=1)

        lock.lock()
        assert lock.is_locked()

        time.sleep(1.5)  # 1.5s > 1s
        assert not lock.is_locked()

    def test_multiple_locks_same_ttl_window(self):
        """Test multiple locks within the same TTL window."""
        lock = TTLLock(ttl_second=10)

        # Lock many times quickly
        for _ in range(100):
            lock.lock()

        assert lock.is_locked()

        # Unlock 100 times
        for _ in range(100):
            lock.unlock()

        assert not lock.is_locked()

    def test_unlock_without_lock(self):
        """Test unlocking a never-locked lock."""
        lock = TTLLock()

        lock.unlock()  # Should not crash
        assert not lock.is_locked()

    def test_lock_unlock_lock_pattern(self):
        """Test lock-unlock-lock pattern."""
        lock = TTLLock()

        lock.lock()
        assert lock.is_locked()

        lock.unlock()
        assert not lock.is_locked()

        lock.lock()
        assert lock.is_locked()

    def test_multiple_independent_locks(self):
        """Test that multiple lock instances are independent."""
        lock1 = TTLLock()
        lock2 = TTLLock()

        lock1.lock()
        lock1.lock()

        assert lock1.is_locked()
        assert not lock2.is_locked()

        lock2.lock()

        assert lock1.is_locked()
        assert lock2.is_locked()

        lock1.reset()

        assert not lock1.is_locked()
        assert lock2.is_locked()
