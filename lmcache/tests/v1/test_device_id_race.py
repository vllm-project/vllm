# SPDX-License-Identifier: Apache-2.0
"""Unit tests for NixlDynamicStorageBackend._alloc_device_ids.

These tests verify the fix for the NIXL OBJ devIdToObjKey_ race condition
(https://github.com/LMCache/LMCache/issues/2983).  When nixl_async_put is
enabled, async PUT cleanup and sync GET run concurrently.  Both previously
used device_id = 0, 1, 2, ... for every call, causing the PUT's
deregister to erase the GET registration in NIXL's flat map.

The fix uses a monotonically increasing counter so each register/deregister
cycle gets globally unique device_ids.  These tests validate correctness
and thread safety of that counter without requiring NIXL or CUDA hardware.
"""

# Standard
import threading

# Third Party
import pytest


# ---------------------------------------------------------------------------
# Helper: build a minimal NixlDynamicStorageBackend with only the fields
# needed by _alloc_device_ids, bypassing __init__'s heavy dependencies.
# ---------------------------------------------------------------------------
def _make_stub_backend():
    """Return a NixlDynamicStorageBackend instance with __init__ bypassed."""
    # Import inside function so the test file can be collected even when
    # nixl is not installed (the importorskip below handles the skip).
    # First Party
    from lmcache.v1.storage_backend.nixl_storage_backend import (
        NixlDynamicStorageBackend,
    )

    obj = object.__new__(NixlDynamicStorageBackend)
    obj._device_id_counter = 0
    obj._device_id_lock = threading.Lock()
    return obj


# Skip the entire module if nixl is not importable (mirrors existing tests)
pytest.importorskip("nixl", reason="nixl package is required for nixl tests")


# ---- Basic correctness ----------------------------------------------------


class TestAllocDeviceIds:
    """Tests for _alloc_device_ids correctness."""

    def test_single_alloc_returns_sequential_range(self):
        backend = _make_stub_backend()
        ids = backend._alloc_device_ids(5)
        assert ids == [0, 1, 2, 3, 4]

    def test_successive_allocs_are_non_overlapping(self):
        backend = _make_stub_backend()
        first = backend._alloc_device_ids(3)
        second = backend._alloc_device_ids(4)
        third = backend._alloc_device_ids(2)
        assert first == [0, 1, 2]
        assert second == [3, 4, 5, 6]
        assert third == [7, 8]

    def test_alloc_zero_returns_empty(self):
        backend = _make_stub_backend()
        ids = backend._alloc_device_ids(0)
        assert ids == []
        # Counter should not advance
        assert backend._device_id_counter == 0

    def test_alloc_one_returns_single_element(self):
        backend = _make_stub_backend()
        ids = backend._alloc_device_ids(1)
        assert ids == [0]
        assert backend._device_id_counter == 1

    def test_counter_advances_correctly(self):
        backend = _make_stub_backend()
        backend._alloc_device_ids(10)
        assert backend._device_id_counter == 10
        backend._alloc_device_ids(5)
        assert backend._device_id_counter == 15

    def test_all_ids_globally_unique_across_many_allocs(self):
        backend = _make_stub_backend()
        all_ids = []
        for n in [1, 3, 7, 2, 10, 5]:
            all_ids.extend(backend._alloc_device_ids(n))
        assert len(all_ids) == len(set(all_ids)), "duplicate device_ids found"
        assert all_ids == list(range(28))


# ---- Thread safety ---------------------------------------------------------


class TestAllocDeviceIdsThreadSafety:
    """Verify no duplicate IDs under concurrent access."""

    def test_concurrent_allocs_produce_unique_ids(self):
        """Simulate the race: multiple threads calling _alloc_device_ids
        concurrently, as happens when async PUT cleanup and sync GET
        overlap."""
        backend = _make_stub_backend()
        num_threads = 16
        allocs_per_thread = 200
        batch_size = 5  # typical: one device_id per key in a batch
        results: list[list[int]] = [[] for _ in range(num_threads)]

        barrier = threading.Barrier(num_threads)

        def worker(thread_idx):
            barrier.wait()  # maximize contention
            for _ in range(allocs_per_thread):
                ids = backend._alloc_device_ids(batch_size)
                results[thread_idx].extend(ids)

        threads = [
            threading.Thread(target=worker, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        all_ids = []
        for r in results:
            all_ids.extend(r)

        expected_total = num_threads * allocs_per_thread * batch_size
        assert len(all_ids) == expected_total
        assert len(set(all_ids)) == expected_total, (
            f"found {expected_total - len(set(all_ids))} duplicate device_ids "
            f"under concurrent access"
        )

    def test_interleaved_put_get_simulation(self):
        """Simulate interleaved PUT (async cleanup) and GET (sync worker)
        operations to verify the fix prevents the exact race from #44."""
        backend = _make_stub_backend()
        put_ids: list[list[int]] = []
        get_ids: list[list[int]] = []
        barrier = threading.Barrier(2)

        def put_worker():
            """Simulates async PUT: register → transfer → deregister."""
            barrier.wait()
            for _ in range(100):
                ids = backend._alloc_device_ids(3)
                put_ids.append(ids)

        def get_worker():
            """Simulates sync GET: register → prepXfer → postXfer → deregister."""
            barrier.wait()
            for _ in range(100):
                ids = backend._alloc_device_ids(3)
                get_ids.append(ids)

        t_put = threading.Thread(target=put_worker)
        t_get = threading.Thread(target=get_worker)
        t_put.start()
        t_get.start()
        t_put.join()
        t_get.join()

        # Flatten and check uniqueness
        all_put = [id_ for batch in put_ids for id_ in batch]
        all_get = [id_ for batch in get_ids for id_ in batch]

        # No overlap between PUT and GET id ranges
        overlap = set(all_put) & set(all_get)
        assert len(overlap) == 0, (
            f"PUT and GET device_ids overlap: {sorted(list(overlap))[:10]}..."
        )

        # All IDs globally unique
        combined = all_put + all_get
        assert len(combined) == len(set(combined))
