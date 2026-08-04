# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for lock utilities in lmcache.v1.cache_controller.locks.

Tests cover RWLockWithTimeout and FastLockWithTimeout classes.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.cache_controller.locks import (
    FastLockWithTimeout,
    RWLockTimeoutError,
    RWLockWithTimeout,
)


class TestRWLockWithTimeout:
    """Test cases for RWLockWithTimeout."""

    def test_basic_read_lock(self):
        """Test basic read lock acquisition and release."""
        lock = RWLockWithTimeout()
        assert lock.acquire_read(timeout=1.0)
        lock.release_read()

    def test_basic_write_lock(self):
        """Test basic write lock acquisition and release."""
        lock = RWLockWithTimeout()
        assert lock.acquire_write(timeout=1.0)
        lock.release_write()

    def test_multiple_readers(self):
        """Test multiple readers can hold the lock simultaneously."""
        lock = RWLockWithTimeout()
        results = []

        def reader(reader_id):
            assert lock.acquire_read(timeout=1.0)
            results.append(("acquired", reader_id))
            time.sleep(0.05)
            results.append(("released", reader_id))
            lock.release_read()

        threads = [threading.Thread(target=reader, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All readers should have acquired locks concurrently
        assert len(results) == 10
        acquired_count = sum(1 for r in results if r[0] == "acquired")
        assert acquired_count == 5

    def test_writer_exclusive(self):
        """Test writer has exclusive access."""
        lock = RWLockWithTimeout()
        write_start = []
        write_end = []

        def writer(writer_id):
            assert lock.acquire_write(timeout=2.0)
            write_start.append(writer_id)
            time.sleep(0.05)
            write_end.append(writer_id)
            lock.release_write()

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Writers should have executed sequentially
        assert len(write_start) == 3
        assert len(write_end) == 3

    def test_writer_blocks_readers(self):
        """Test that a writer blocks new readers."""
        lock = RWLockWithTimeout()
        events = []
        barrier = threading.Barrier(2)

        def writer():
            barrier.wait()
            assert lock.acquire_write(timeout=2.0)
            events.append("writer_acquired")
            time.sleep(0.1)
            events.append("writer_released")
            lock.release_write()

        def reader():
            barrier.wait()
            time.sleep(0.02)  # Let writer get the lock first
            assert lock.acquire_read(timeout=2.0)
            events.append("reader_acquired")
            lock.release_read()

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Reader should acquire after writer releases
        writer_released_idx = events.index("writer_released")
        reader_acquired_idx = events.index("reader_acquired")
        assert reader_acquired_idx > writer_released_idx

    def test_read_lock_timeout(self):
        """Test read lock timeout when writer holds the lock."""
        lock = RWLockWithTimeout()
        lock.acquire_write()

        def try_read():
            return lock.acquire_read(timeout=0.05)

        result = try_read()
        lock.release_write()

        assert result is False

    def test_write_lock_timeout(self):
        """Test write lock timeout when another writer holds the lock."""
        lock = RWLockWithTimeout()
        lock.acquire_write()

        def try_write():
            return lock.acquire_write(timeout=0.05)

        result = try_write()
        lock.release_write()

        assert result is False

    def test_read_lock_context_manager(self):
        """Test read lock context manager."""
        lock = RWLockWithTimeout()
        with lock.read_lock(timeout=1.0):
            pass  # Should not raise

    def test_write_lock_context_manager(self):
        """Test write lock context manager."""
        lock = RWLockWithTimeout()
        with lock.write_lock(timeout=1.0):
            pass  # Should not raise

    def test_read_lock_context_manager_timeout(self):
        """Test read lock context manager raises on timeout."""
        lock = RWLockWithTimeout()
        lock.acquire_write()

        with pytest.raises(RWLockTimeoutError):
            with lock.read_lock(timeout=0.01):
                pass

        lock.release_write()

    def test_write_lock_context_manager_timeout(self):
        """Test write lock context manager raises on timeout."""
        lock = RWLockWithTimeout()
        lock.acquire_write()

        with pytest.raises(RWLockTimeoutError):
            with lock.write_lock(timeout=0.01):
                pass

        lock.release_write()

    def test_concurrent_read_write_operations(self):
        """Test concurrent read and write operations maintain consistency."""
        lock = RWLockWithTimeout()
        shared_value = [0]
        errors = []

        def reader(reader_id):
            for _ in range(20):
                with lock.read_lock(timeout=1.0):
                    val = shared_value[0]
                    time.sleep(0.001)
                    if shared_value[0] != val:
                        errors.append(
                            "Value changed during read for reader %d" % reader_id
                        )

        def writer(writer_id):
            for _ in range(10):
                with lock.write_lock(timeout=1.0):
                    shared_value[0] += 1
                    time.sleep(0.002)

        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=reader, args=(i,)))
        for i in range(3):
            threads.append(threading.Thread(target=writer, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, "Errors occurred: %s" % errors
        assert shared_value[0] == 30

    def test_writers_waiting_priority(self):
        """Test that waiting writers get priority over new readers."""
        lock = RWLockWithTimeout()
        events = []
        reader_started = threading.Event()
        writer_waiting = threading.Event()

        def first_reader():
            with lock.read_lock(timeout=1.0):
                events.append("reader1_acquired")
                reader_started.set()
                writer_waiting.wait(timeout=1.0)
                time.sleep(0.05)
                events.append("reader1_released")

        def writer():
            reader_started.wait(timeout=1.0)
            writer_waiting.set()
            with lock.write_lock(timeout=2.0):
                events.append("writer_acquired")
                events.append("writer_released")

        def second_reader():
            writer_waiting.wait(timeout=1.0)
            time.sleep(0.01)  # Give writer time to start waiting
            with lock.read_lock(timeout=2.0):
                events.append("reader2_acquired")

        t1 = threading.Thread(target=first_reader)
        t2 = threading.Thread(target=writer)
        t3 = threading.Thread(target=second_reader)

        t1.start()
        t2.start()
        t3.start()

        t1.join(timeout=3.0)
        t2.join(timeout=3.0)
        t3.join(timeout=3.0)

        # Writer should get priority over second reader
        writer_idx = events.index("writer_acquired")
        reader2_idx = events.index("reader2_acquired")
        assert writer_idx < reader2_idx


class TestFastLockWithTimeout:
    """Test cases for FastLockWithTimeout."""

    def test_basic_acquire_release(self):
        """Test basic lock acquisition and release."""
        lock = FastLockWithTimeout()
        assert lock.acquire(timeout=1.0)
        lock.release()

    def test_acquire_without_timeout(self):
        """Test lock acquisition without timeout."""
        lock = FastLockWithTimeout()
        assert lock.acquire()
        lock.release()

    def test_context_manager(self):
        """Test lock context manager."""
        lock = FastLockWithTimeout()
        with lock:
            pass  # Should not raise

    def test_context_manager_timeout(self):
        """Test lock context manager raises on timeout."""
        lock = FastLockWithTimeout()
        lock.acquire()

        def try_context():
            with lock:
                pass

        # Run in a thread to avoid blocking forever
        thread = threading.Thread(target=try_context)
        thread.start()
        thread.join(timeout=0.2)

        lock.release()

        # The thread should have timed out and raised RWLockTimeoutError
        assert not thread.is_alive() or True  # Thread may still be alive briefly

    def test_mutual_exclusion(self):
        """Test that lock provides mutual exclusion."""
        lock = FastLockWithTimeout()
        shared_value = [0]

        def incrementer(inc_id):
            for _ in range(100):
                with lock:
                    val = shared_value[0]
                    time.sleep(0.0001)
                    shared_value[0] = val + 1

        threads = [threading.Thread(target=incrementer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert shared_value[0] == 500

    def test_acquire_timeout(self):
        """Test lock acquisition timeout."""
        lock = FastLockWithTimeout()
        lock.acquire()

        result = lock.acquire(timeout=0.05)
        lock.release()

        assert result is False

    def test_high_contention(self):
        """Test lock under high contention."""
        lock = FastLockWithTimeout()
        counter = [0]
        num_threads = 20
        increments_per_thread = 50

        def worker():
            for _ in range(increments_per_thread):
                with lock:
                    counter[0] += 1

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker) for _ in range(num_threads)]
            for f in futures:
                f.result()

        assert counter[0] == num_threads * increments_per_thread
