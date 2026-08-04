# SPDX-License-Identifier: Apache-2.0
"""Unit tests for AddressManager class."""

# Standard
from typing import List, Tuple
import concurrent.futures
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.memory_management import AddressManager


class TestAddressManagerBasic:
    """Test basic functionality of AddressManager."""

    def test_initialization(self):
        """Test that AddressManager initializes correctly."""
        size = 4096 * 10  # 10 pages
        manager = AddressManager(size)

        assert manager.total_allocated_size == 0
        assert manager.get_free_size() == size
        assert manager.check_consistency()

    def test_initialization_custom_alignment(self):
        """Test initialization with custom alignment."""
        size = 1024 * 100
        align_bytes = 1024
        manager = AddressManager(size, align_bytes=align_bytes)

        assert manager.total_allocated_size == 0
        assert manager.get_free_size() == size
        assert manager.check_consistency()

    def test_compute_aligned_size(self):
        """Test that compute_aligned_size works correctly."""
        manager = AddressManager(4096 * 10, align_bytes=4096)

        # Size already aligned
        assert manager.compute_aligned_size(4096) == 4096
        assert manager.compute_aligned_size(8192) == 8192

        # Size needs alignment
        assert manager.compute_aligned_size(1) == 4096
        assert manager.compute_aligned_size(4097) == 8192
        assert manager.compute_aligned_size(100) == 4096

    def test_compute_aligned_size_custom_alignment(self):
        """Test compute_aligned_size with different alignment values."""
        manager = AddressManager(10000, align_bytes=256)

        assert manager.compute_aligned_size(1) == 256
        assert manager.compute_aligned_size(256) == 256
        assert manager.compute_aligned_size(257) == 512
        assert manager.compute_aligned_size(512) == 512


class TestAddressManagerAllocation:
    """Test allocation functionality."""

    def test_single_allocation(self):
        """Test a single allocation."""
        size = 4096 * 10
        manager = AddressManager(size)

        address, allocated_size = manager.allocate(100)

        # Verify allocation returns valid values
        assert address >= 0
        assert allocated_size == 4096  # Aligned to 4096
        assert manager.total_allocated_size == 4096
        assert manager.get_free_size() == size - 4096
        assert manager.check_consistency()

    def test_multiple_allocations(self):
        """Test multiple sequential allocations."""
        size = 4096 * 10
        manager = AddressManager(size)

        # First allocation
        addr1, size1 = manager.allocate(1000)
        assert addr1 >= 0
        assert size1 == 4096

        # Second allocation
        addr2, size2 = manager.allocate(2000)
        assert addr2 >= 0
        assert size2 == 4096

        # Third allocation
        addr3, size3 = manager.allocate(5000)
        assert addr3 >= 0
        assert size3 == 8192

        # Verify allocations don't overlap
        allocations = [(addr1, size1), (addr2, size2), (addr3, size3)]
        assert _no_overlap(allocations)

        assert manager.total_allocated_size == 4096 + 4096 + 8192
        assert manager.get_free_size() == size - (4096 + 4096 + 8192)
        assert manager.check_consistency()

    def test_allocation_exact_size(self):
        """Test allocation with exact aligned size."""
        manager = AddressManager(4096 * 5)

        addr, alloc_size = manager.allocate(4096)
        assert addr >= 0
        assert alloc_size == 4096
        assert manager.check_consistency()

    def test_allocation_failure_out_of_memory(self):
        """Test that allocation fails when out of memory."""
        size = 4096 * 2
        manager = AddressManager(size)

        # First allocation succeeds
        manager.allocate(4096)

        # Second allocation succeeds
        manager.allocate(4096)

        # Third allocation should fail
        with pytest.raises(RuntimeError, match="no memory is available"):
            manager.allocate(1)

        assert manager.check_consistency()

    def test_allocation_failure_too_large(self):
        """Test allocation failure when requested size is too large."""
        size = 4096 * 2
        manager = AddressManager(size)

        with pytest.raises(RuntimeError, match="no memory is available"):
            manager.allocate(size + 1)

        assert manager.check_consistency()


class TestAddressManagerBatchedAllocation:
    """Test batched_allocate functionality."""

    def test_single_batch(self):
        """Test batched_allocate with batch_size=1 behaves like allocate."""
        size = 4096 * 10
        manager = AddressManager(size)

        results = manager.batched_allocate(100, 1)

        assert len(results) == 1
        addr, allocated_size = results[0]
        assert addr >= 0
        assert allocated_size == 4096  # Aligned to 4096
        assert manager.total_allocated_size == 4096
        assert manager.get_free_size() == size - 4096
        assert manager.check_consistency()

    def test_multiple_batch(self):
        """Test batched_allocate with multiple blocks."""
        size = 4096 * 10
        manager = AddressManager(size)

        results = manager.batched_allocate(1000, 5)

        assert len(results) == 5
        for addr, allocated_size in results:
            assert addr >= 0
            assert allocated_size == 4096

        assert _no_overlap(results)
        assert manager.total_allocated_size == 4096 * 5
        assert manager.get_free_size() == size - 4096 * 5
        assert manager.check_consistency()

    def test_batch_exact_aligned_size(self):
        """Test batched_allocate with exact aligned size."""
        manager = AddressManager(4096 * 5)

        results = manager.batched_allocate(4096, 5)

        assert len(results) == 5
        for addr, allocated_size in results:
            assert addr >= 0
            assert allocated_size == 4096

        assert _no_overlap(results)
        assert manager.get_free_size() == 0
        assert manager.check_consistency()

    def test_batch_allocate_all_memory(self):
        """Test batched_allocate that exhausts all memory."""
        size = 4096 * 8
        manager = AddressManager(size)

        results = manager.batched_allocate(4096, 8)

        assert len(results) == 8
        assert _no_overlap(results)
        assert manager.get_free_size() == 0
        assert manager.total_allocated_size == size
        assert manager.check_consistency()

    def test_batch_failure_out_of_memory(self):
        """Test that batched_allocate fails when out of memory."""
        size = 4096 * 2
        manager = AddressManager(size)

        # First batch succeeds
        manager.batched_allocate(4096, 2)

        # Second batch should fail - no memory left
        with pytest.raises(RuntimeError):
            manager.batched_allocate(4096, 1)

        assert manager.check_consistency()

    def test_batch_failure_insufficient_memory(self):
        """Test batched_allocate fails when not enough memory for all blocks."""
        size = 4096 * 3
        manager = AddressManager(size)

        # Request 5 blocks but only 3 pages available
        with pytest.raises(RuntimeError):
            manager.batched_allocate(4096, 5)

        # Verify no partial allocation occurred (atomicity)
        assert manager.total_allocated_size == 0
        assert manager.get_free_size() == size
        assert manager.check_consistency()

    def test_batch_failure_too_large_single_block(self):
        """Test batched_allocate failure when single block size is too large."""
        size = 4096 * 2
        manager = AddressManager(size)

        with pytest.raises(RuntimeError):
            manager.batched_allocate(size + 1, 1)

        assert manager.total_allocated_size == 0
        assert manager.check_consistency()

    def test_batch_atomicity_on_failure(self):
        """Test that failed batched_allocate does not modify state."""
        size = 4096 * 5
        manager = AddressManager(size)

        # Pre-allocate some memory
        manager.allocate(4096 * 2)
        allocated_before = manager.total_allocated_size
        free_before = manager.get_free_size()

        # This should fail - requesting more than available
        with pytest.raises(RuntimeError):
            manager.batched_allocate(4096, 10)

        # State should be unchanged
        assert manager.total_allocated_size == allocated_before
        assert manager.get_free_size() == free_before
        assert manager.check_consistency()

    def test_batch_after_free(self):
        """Test batched_allocate after freeing memory."""
        size = 4096 * 5
        manager = AddressManager(size)

        # Allocate all
        addrs = []
        for _ in range(5):
            addr, alloc_size = manager.allocate(4096)
            addrs.append((addr, alloc_size))

        # Free 3 blocks
        for addr, alloc_size in addrs[:3]:
            manager.free(addr, alloc_size)

        # Batched allocate should succeed with 3 blocks
        results = manager.batched_allocate(4096, 3)
        assert len(results) == 3
        assert _no_overlap(results + addrs[3:])
        assert manager.check_consistency()

    def test_batch_from_fragmented_memory(self):
        """Test batched_allocate from fragmented free list."""
        size = 4096 * 6
        manager = AddressManager(size)

        # Allocate all 6 blocks
        addrs = []
        for _ in range(6):
            addr, alloc_size = manager.allocate(4096)
            addrs.append((addr, alloc_size))

        # Free alternating blocks to create fragmentation
        # Free blocks 0, 2, 4 => 3 free non-contiguous blocks
        manager.free(addrs[0][0], addrs[0][1])
        manager.free(addrs[2][0], addrs[2][1])
        manager.free(addrs[4][0], addrs[4][1])

        assert manager.get_free_size() == 4096 * 3

        # Batched allocate 3 blocks from fragmented free list
        results = manager.batched_allocate(4096, 3)
        assert len(results) == 3
        # All allocated + still-occupied blocks should not overlap
        all_allocs = results + [addrs[1], addrs[3], addrs[5]]
        assert _no_overlap(all_allocs)
        assert manager.get_free_size() == 0
        assert manager.check_consistency()

    def test_batch_carves_from_large_block(self):
        """
        Test batched_allocate carving multiple chunks from a single large free block.
        """
        size = 4096 * 10
        manager = AddressManager(size)

        # All memory is one contiguous free block
        results = manager.batched_allocate(4096, 5)
        assert len(results) == 5
        assert _no_overlap(results)
        assert manager.total_allocated_size == 4096 * 5
        assert manager.get_free_size() == 4096 * 5
        assert manager.check_consistency()

    def test_batch_with_non_aligned_size(self):
        """Test batched_allocate with non-aligned size request."""
        size = 4096 * 10
        manager = AddressManager(size)

        results = manager.batched_allocate(100, 3)

        assert len(results) == 3
        for addr, allocated_size in results:
            assert addr >= 0
            assert allocated_size == 4096  # Aligned up to 4096
        assert _no_overlap(results)
        assert manager.total_allocated_size == 4096 * 3
        assert manager.check_consistency()

    def test_batch_mixed_with_single_allocate(self):
        """Test batched_allocate interleaved with single allocate."""
        size = 4096 * 10
        manager = AddressManager(size)

        # Single allocate
        addr1, size1 = manager.allocate(4096)

        # Batch allocate
        batch_results = manager.batched_allocate(4096, 3)

        # Single allocate again
        addr2, size2 = manager.allocate(4096)

        all_allocs = [(addr1, size1), (addr2, size2)] + batch_results
        assert _no_overlap(all_allocs)
        assert manager.total_allocated_size == 4096 * 5
        assert manager.check_consistency()

    def test_batch_after_sbrk(self):
        """Test batched_allocate after sbrk expansion."""
        size = 4096 * 2
        manager = AddressManager(size)

        # Allocate all initial memory
        manager.allocate(4096)
        manager.allocate(4096)
        assert manager.get_free_size() == 0

        # Expand
        manager.sbrk(4096 * 5)

        # Now batched_allocate should succeed
        results = manager.batched_allocate(4096, 3)
        assert len(results) == 3
        assert _no_overlap(results)
        assert manager.check_consistency()

    def test_batch_free_and_rebatch(self):
        """Test free after batched_allocate and re-allocate."""
        size = 4096 * 10
        manager = AddressManager(size)

        # Batched allocate
        results = manager.batched_allocate(4096, 5)
        assert len(results) == 5

        # Free all batched blocks
        for addr, alloc_size in results:
            manager.free(addr, alloc_size)

        assert manager.total_allocated_size == 0
        assert manager.get_free_size() == size
        assert manager.check_consistency()

        # Re-allocate the same batch
        results2 = manager.batched_allocate(4096, 5)
        assert len(results2) == 5
        assert _no_overlap(results2)
        assert manager.check_consistency()

    def test_batch_size_zero(self):
        """Test batched_allocate with batch_size=0."""
        size = 4096 * 10
        manager = AddressManager(size)

        results = manager.batched_allocate(4096, 0)
        assert len(results) == 0
        assert manager.total_allocated_size == 0
        assert manager.get_free_size() == size
        assert manager.check_consistency()


class TestAddressManagerFree:
    """Test free functionality."""

    def test_simple_free(self):
        """Test simple allocation and free."""
        size = 4096 * 10
        manager = AddressManager(size)

        addr, alloc_size = manager.allocate(1000)
        assert manager.total_allocated_size == alloc_size

        manager.free(addr, alloc_size)
        assert manager.total_allocated_size == 0
        assert manager.get_free_size() == size
        assert manager.check_consistency()

    def test_free_and_realloc(self):
        """Test that freed memory can be reallocated."""
        size = 4096 * 3
        manager = AddressManager(size)

        # Allocate all memory
        addr1, size1 = manager.allocate(4096)
        addr2, size2 = manager.allocate(4096)
        addr3, size3 = manager.allocate(4096)

        # Free middle block
        manager.free(addr2, size2)
        assert manager.get_free_size() == 4096
        assert manager.check_consistency()

        # Reallocate should succeed (freed block is available)
        addr_new, size_new = manager.allocate(4096)
        assert addr_new >= 0
        assert size_new == 4096
        assert manager.check_consistency()

    def test_coalescing_allows_larger_allocation(self):
        """Test that freeing adjacent blocks allows larger allocation."""
        size = 4096 * 4
        manager = AddressManager(size)

        # Allocate three blocks
        addr1, size1 = manager.allocate(4096)
        addr2, size2 = manager.allocate(4096)
        addr3, size3 = manager.allocate(4096)

        # Free blocks 2 and 3
        manager.free(addr3, size3)
        manager.free(addr2, size2)

        assert manager.get_free_size() == size - size1
        assert manager.check_consistency()

        # Should be able to allocate a block of size 2*4096
        # (only possible if blocks were coalesced)
        addr_large, size_large = manager.allocate(8192)
        assert addr_large >= 0
        assert size_large == 8192
        assert manager.check_consistency()

    def test_coalescing_predecessor(self):
        """Test that freeing blocks coalesces with predecessor."""
        size = 4096 * 4
        manager = AddressManager(size)

        # Allocate three blocks
        addr1, size1 = manager.allocate(4096)
        addr2, size2 = manager.allocate(4096)
        addr3, size3 = manager.allocate(4096)

        # Free blocks 1 and 2
        manager.free(addr1, size1)
        manager.free(addr2, size2)

        assert manager.get_free_size() == size - size3
        assert manager.check_consistency()

        # Should be able to allocate a block of size 2*4096
        addr_large, size_large = manager.allocate(8192)
        assert addr_large >= 0
        assert size_large == 8192
        assert manager.check_consistency()

    def test_coalescing_both_sides(self):
        """Test that freeing a block coalesces with both neighbors."""
        size = 4096 * 5
        manager = AddressManager(size)

        # Allocate four blocks
        addr1, size1 = manager.allocate(4096)
        addr2, size2 = manager.allocate(4096)
        addr3, size3 = manager.allocate(4096)
        addr4, size4 = manager.allocate(4096)

        # Free blocks 1, 3, then 2
        # Block 2 should coalesce with both block 1 and block 3
        manager.free(addr1, size1)
        manager.free(addr3, size3)
        manager.free(addr2, size2)

        assert manager.get_free_size() == size - size4
        assert manager.check_consistency()

        # Should be able to allocate a block of size 3*4096
        addr_large, size_large = manager.allocate(12288)
        assert addr_large >= 0
        assert size_large == 12288
        assert manager.check_consistency()

    def test_free_all(self):
        """Test freeing all allocated memory."""
        size = 4096 * 5
        manager = AddressManager(size)

        allocations: List[Tuple[int, int]] = []
        for _ in range(5):
            addr, alloc_size = manager.allocate(4096)
            allocations.append((addr, alloc_size))

        assert manager.get_free_size() == 0

        # Free in reverse order
        for addr, alloc_size in reversed(allocations):
            manager.free(addr, alloc_size)

        assert manager.get_free_size() == size
        assert manager.total_allocated_size == 0
        assert manager.check_consistency()

    def test_free_random_order(self):
        """Test freeing blocks in random order."""
        size = 4096 * 10
        manager = AddressManager(size)

        allocations: List[Tuple[int, int]] = []
        for _ in range(10):
            addr, alloc_size = manager.allocate(4096)
            allocations.append((addr, alloc_size))

        assert manager.get_free_size() == 0

        # Free in a shuffled order
        # Standard
        import random

        random.seed(42)
        shuffled = allocations.copy()
        random.shuffle(shuffled)

        for addr, alloc_size in shuffled:
            manager.free(addr, alloc_size)
            assert manager.check_consistency()

        assert manager.get_free_size() == size
        assert manager.total_allocated_size == 0
        assert manager.check_consistency()


class TestAddressManagerSbrk:
    """Test sbrk (expand) functionality."""

    def test_sbrk_basic(self):
        """Test basic sbrk expansion."""
        initial_size = 4096 * 2
        manager = AddressManager(initial_size)

        # Allocate all initial memory
        manager.allocate(4096)
        manager.allocate(4096)

        assert manager.get_free_size() == 0

        # Expand
        manager.sbrk(4096 * 2)

        # Should now be able to allocate more
        addr, alloc_size = manager.allocate(4096)
        assert addr >= 0
        assert alloc_size == 4096
        assert manager.check_consistency()

    def test_sbrk_increases_free_size(self):
        """Test that sbrk increases free size."""
        size = 4096 * 3
        manager = AddressManager(size)

        # Allocate two blocks
        manager.allocate(4096)
        manager.allocate(4096)

        free_before = manager.get_free_size()
        assert free_before == 4096

        # Expand
        manager.sbrk(4096)
        assert manager.get_free_size() == free_before + 4096
        assert manager.check_consistency()

    def test_sbrk_allows_larger_allocation(self):
        """Test that sbrk allows allocating larger blocks."""
        size = 4096 * 2
        manager = AddressManager(size)

        # Allocate one block
        manager.allocate(4096)

        # Try to allocate 2 pages - should fail
        with pytest.raises(RuntimeError):
            manager.allocate(8192)

        # Expand
        manager.sbrk(4096 * 2)

        # Now it should succeed
        addr, alloc_size = manager.allocate(8192)
        assert addr >= 0
        assert alloc_size == 8192
        assert manager.check_consistency()

    def test_sbrk_after_fragmentation(self):
        """Test sbrk after memory fragmentation."""
        size = 4096 * 4
        manager = AddressManager(size)

        # Allocate all
        addrs = []
        for _ in range(4):
            addr, alloc_size = manager.allocate(4096)
            addrs.append((addr, alloc_size))

        # Create fragmentation by freeing alternating blocks
        manager.free(addrs[1][0], addrs[1][1])
        manager.free(addrs[3][0], addrs[3][1])

        # Expand
        manager.sbrk(4096)
        assert manager.check_consistency()

        # Free size should include the expansion
        assert manager.get_free_size() == 4096 * 3  # 2 freed + 1 expanded


class TestAddressManagerConsistency:
    """Test consistency checking."""

    def test_consistency_empty(self):
        """Test consistency on empty manager."""
        manager = AddressManager(4096 * 10)
        assert manager.check_consistency()

    def test_consistency_full(self):
        """Test consistency on fully allocated manager."""
        manager = AddressManager(4096 * 3)

        manager.allocate(4096)
        manager.allocate(4096)
        manager.allocate(4096)

        assert manager.check_consistency()

    def test_consistency_after_operations(self):
        """Test consistency after various operations."""
        manager = AddressManager(4096 * 10)

        # Series of operations
        _ = manager.allocate(4096)
        a2 = manager.allocate(8192)
        _ = manager.allocate(4096)
        assert manager.check_consistency()

        manager.free(a2[0], a2[1])
        assert manager.check_consistency()

        manager.allocate(4096)
        assert manager.check_consistency()

        manager.sbrk(4096)
        assert manager.check_consistency()

    def test_free_size_plus_allocated_equals_total(self):
        """Test that free_size + allocated_size equals total size."""
        size = 4096 * 10
        manager = AddressManager(size)

        # After various operations, invariant should hold
        allocations = []
        for _ in range(5):
            addr, alloc_size = manager.allocate(4096)
            allocations.append((addr, alloc_size))
            assert manager.get_free_size() + manager.total_allocated_size == size

        for addr, alloc_size in allocations[:3]:
            manager.free(addr, alloc_size)
            assert manager.get_free_size() + manager.total_allocated_size == size

        # After sbrk, total size increases
        manager.sbrk(4096 * 5)
        new_total = size + 4096 * 5
        assert manager.get_free_size() + manager.total_allocated_size == new_total


class TestAddressManagerThreadSafety:
    """Test thread-safety of AddressManager."""

    def test_concurrent_allocations(self):
        """Test that concurrent allocations are thread-safe."""
        # Large enough for many concurrent allocations
        size = 4096 * 1000
        manager = AddressManager(size)

        num_threads = 50
        allocations_per_thread = 10
        results: List[List[Tuple[int, int]]] = [[] for _ in range(num_threads)]
        errors: List[Exception] = []

        def allocate_worker(thread_id: int):
            try:
                for _ in range(allocations_per_thread):
                    addr, alloc_size = manager.allocate(4096)
                    results[thread_id].append((addr, alloc_size))
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=allocate_worker, args=(i,))
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Check no errors occurred
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify all allocations succeeded
        all_allocations = [a for thread_results in results for a in thread_results]
        assert len(all_allocations) == num_threads * allocations_per_thread

        # Verify no overlapping addresses
        assert _no_overlap(all_allocations)

        # Verify consistency
        assert manager.check_consistency()
        assert (
            manager.total_allocated_size == num_threads * allocations_per_thread * 4096
        )

    def test_concurrent_frees(self):
        """Test that concurrent frees are thread-safe."""
        size = 4096 * 100
        manager = AddressManager(size)

        # Pre-allocate blocks
        allocations = []
        for _ in range(100):
            addr, alloc_size = manager.allocate(4096)
            allocations.append((addr, alloc_size))

        errors: List[Exception] = []

        def free_worker(allocs: List[Tuple[int, int]]):
            try:
                for addr, alloc_size in allocs:
                    manager.free(addr, alloc_size)
            except Exception as e:
                errors.append(e)

        # Split allocations among threads
        num_threads = 10
        chunk_size = len(allocations) // num_threads
        threads = []
        for i in range(num_threads):
            start = i * chunk_size
            end = start + chunk_size if i < num_threads - 1 else len(allocations)
            t = threading.Thread(target=free_worker, args=(allocations[start:end],))
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Check no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify all memory is freed
        assert manager.total_allocated_size == 0
        assert manager.get_free_size() == size
        assert manager.check_consistency()

    def test_concurrent_alloc_and_free(self):
        """Test concurrent allocations and frees."""
        size = 4096 * 500
        manager = AddressManager(size)

        num_iterations = 100
        errors: List[Exception] = []
        allocated_lock = threading.Lock()
        allocated: List[Tuple[int, int]] = []

        def alloc_worker():
            try:
                for _ in range(num_iterations):
                    try:
                        addr, alloc_size = manager.allocate(4096)
                        with allocated_lock:
                            allocated.append((addr, alloc_size))
                    except RuntimeError:
                        # Out of memory is expected sometimes
                        pass
            except Exception as e:
                errors.append(e)

        def free_worker():
            try:
                for _ in range(num_iterations):
                    to_free = None
                    with allocated_lock:
                        if allocated:
                            to_free = allocated.pop()
                    if to_free:
                        manager.free(to_free[0], to_free[1])
            except Exception as e:
                errors.append(e)

        # Run allocators and freers concurrently
        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=alloc_worker))
            threads.append(threading.Thread(target=free_worker))

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Check no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify consistency
        assert manager.check_consistency()

        # Free remaining allocations
        for addr, alloc_size in allocated:
            manager.free(addr, alloc_size)

        assert manager.total_allocated_size == 0
        assert manager.get_free_size() == size
        assert manager.check_consistency()

    def test_concurrent_with_thread_pool(self):
        """Test using ThreadPoolExecutor for concurrent operations."""
        size = 4096 * 200
        manager = AddressManager(size)

        def alloc_and_free():
            addr, alloc_size = manager.allocate(4096)
            # Small delay to increase chance of race conditions
            manager.free(addr, alloc_size)
            return True

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(alloc_and_free) for _ in range(100)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert all(results)
        assert manager.total_allocated_size == 0
        assert manager.get_free_size() == size
        assert manager.check_consistency()

    def test_stress_test_concurrent_operations(self):
        """Stress test with many concurrent operations."""
        size = 4096 * 2000
        manager = AddressManager(size)

        num_threads = 20
        operations_per_thread = 50
        errors: List[Exception] = []

        allocated_lock = threading.Lock()
        allocated: List[Tuple[int, int]] = []

        def worker(thread_id: int):
            try:
                for i in range(operations_per_thread):
                    if i % 2 == 0:
                        # Allocate
                        try:
                            addr, alloc_size = manager.allocate(4096)
                            with allocated_lock:
                                allocated.append((addr, alloc_size))
                        except RuntimeError:
                            pass
                    else:
                        # Free
                        to_free = None
                        with allocated_lock:
                            if allocated:
                                to_free = allocated.pop()
                        if to_free:
                            manager.free(to_free[0], to_free[1])
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(i,)) for i in range(num_threads)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert manager.check_consistency()

        # Clean up remaining allocations
        for addr, alloc_size in allocated:
            manager.free(addr, alloc_size)

        assert manager.total_allocated_size == 0
        assert manager.check_consistency()

    def test_concurrent_batched_allocations(self):
        """Test that concurrent batched_allocate calls are thread-safe."""
        size = 4096 * 2000
        manager = AddressManager(size)

        num_threads = 20
        batch_per_thread = 10
        results: List[List[Tuple[int, int]]] = [[] for _ in range(num_threads)]
        errors: List[Exception] = []

        def batch_alloc_worker(thread_id: int):
            try:
                batch = manager.batched_allocate(4096, batch_per_thread)
                results[thread_id] = batch
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=batch_alloc_worker, args=(i,))
            for i in range(num_threads)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"

        all_allocations = [a for thread_results in results for a in thread_results]
        assert len(all_allocations) == num_threads * batch_per_thread
        assert _no_overlap(all_allocations)
        assert manager.check_consistency()
        assert manager.total_allocated_size == num_threads * batch_per_thread * 4096

    def test_concurrent_batched_alloc_and_free(self):
        """Test concurrent batched_allocate and free operations."""
        size = 4096 * 1000
        manager = AddressManager(size)

        num_iterations = 20
        errors: List[Exception] = []
        allocated_lock = threading.Lock()
        allocated: List[Tuple[int, int]] = []

        def batch_alloc_worker():
            try:
                for _ in range(num_iterations):
                    try:
                        batch = manager.batched_allocate(4096, 5)
                        with allocated_lock:
                            allocated.extend(batch)
                    except RuntimeError:
                        pass
            except Exception as e:
                errors.append(e)

        def free_worker():
            try:
                for _ in range(num_iterations * 5):
                    to_free = None
                    with allocated_lock:
                        if allocated:
                            to_free = allocated.pop()
                    if to_free:
                        manager.free(to_free[0], to_free[1])
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=batch_alloc_worker))
            threads.append(threading.Thread(target=free_worker))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert manager.check_consistency()

        # Clean up remaining allocations
        for addr, alloc_size in allocated:
            manager.free(addr, alloc_size)

        assert manager.total_allocated_size == 0
        assert manager.check_consistency()


class TestAddressManagerEdgeCases:
    """Test edge cases."""

    def test_zero_size_manager(self):
        """Test manager with zero size."""
        manager = AddressManager(0)
        assert manager.get_free_size() == 0
        assert manager.total_allocated_size == 0
        assert manager.check_consistency()

        with pytest.raises(RuntimeError):
            manager.allocate(1)

    def test_alignment_edge_cases(self):
        """Test allocation with sizes at alignment boundaries."""
        align = 4096
        manager = AddressManager(align * 10, align_bytes=align)

        # Exactly one byte less than alignment
        addr1, size1 = manager.allocate(align - 1)
        assert size1 == align

        # Exactly one byte more than alignment
        addr2, size2 = manager.allocate(align + 1)
        assert size2 == align * 2

        # Verify no overlap
        assert _no_overlap([(addr1, size1), (addr2, size2)])
        assert manager.check_consistency()

    def test_many_small_allocations(self):
        """Test many small allocations."""
        size = 4096 * 100
        manager = AddressManager(size)

        allocations = []
        for _ in range(100):
            addr, alloc_size = manager.allocate(1)  # Will be aligned to 4096
            allocations.append((addr, alloc_size))

        assert manager.get_free_size() == 0
        assert _no_overlap(allocations)
        assert manager.check_consistency()

        # Free all
        for addr, alloc_size in allocations:
            manager.free(addr, alloc_size)

        assert manager.get_free_size() == size
        assert manager.check_consistency()

    def test_single_large_allocation(self):
        """Test allocating entire space at once."""
        size = 4096 * 10
        manager = AddressManager(size)

        addr, alloc_size = manager.allocate(size)
        assert addr >= 0
        assert alloc_size == size
        assert manager.get_free_size() == 0
        assert manager.check_consistency()

        manager.free(addr, alloc_size)
        assert manager.get_free_size() == size
        assert manager.check_consistency()

    def test_repeated_alloc_free_cycle(self):
        """Test repeated allocation and free cycles."""
        size = 4096 * 5
        manager = AddressManager(size)

        for _ in range(100):
            addr, alloc_size = manager.allocate(4096 * 3)
            assert manager.check_consistency()
            manager.free(addr, alloc_size)
            assert manager.check_consistency()

        assert manager.get_free_size() == size
        assert manager.total_allocated_size == 0

    def test_varied_allocation_sizes(self):
        """Test allocations of varied sizes."""
        size = 4096 * 100
        manager = AddressManager(size)

        allocations = []
        # Allocate blocks of different sizes
        for num_pages in [1, 2, 3, 5, 8, 13]:
            addr, alloc_size = manager.allocate(4096 * num_pages)
            allocations.append((addr, alloc_size))

        assert _no_overlap(allocations)
        assert manager.check_consistency()

        # Free all
        for addr, alloc_size in allocations:
            manager.free(addr, alloc_size)

        assert manager.get_free_size() == size
        assert manager.check_consistency()

    def test_batched_allocate_varied_batch_sizes(self):
        """Test batched_allocate with varied batch sizes."""
        size = 4096 * 100
        manager = AddressManager(size)

        all_allocs: List[Tuple[int, int]] = []

        for batch_size in [1, 2, 5, 10, 3]:
            results = manager.batched_allocate(4096, batch_size)
            assert len(results) == batch_size
            all_allocs.extend(results)

        assert _no_overlap(all_allocs)
        assert manager.total_allocated_size == 4096 * (1 + 2 + 5 + 10 + 3)
        assert manager.check_consistency()

        # Free all
        for addr, alloc_size in all_allocs:
            manager.free(addr, alloc_size)

        assert manager.get_free_size() == size
        assert manager.check_consistency()

    def test_batched_allocate_from_zero_size_manager(self):
        """Test batched_allocate on a zero-size manager."""
        manager = AddressManager(0)

        with pytest.raises(RuntimeError):
            manager.batched_allocate(1, 1)

        assert manager.check_consistency()

    def test_batched_allocate_repeated_cycles(self):
        """Test repeated batched_allocate and free cycles."""
        size = 4096 * 10
        manager = AddressManager(size)

        for _ in range(50):
            results = manager.batched_allocate(4096, 5)
            assert len(results) == 5
            assert manager.check_consistency()

            for addr, alloc_size in results:
                manager.free(addr, alloc_size)
            assert manager.check_consistency()

        assert manager.get_free_size() == size
        assert manager.total_allocated_size == 0


def _no_overlap(allocations: List[Tuple[int, int]]) -> bool:
    """
    Check that no allocations overlap.

    Args:
        allocations: List of (address, size) tuples

    Returns:
        True if no overlaps, False otherwise
    """
    # Sort by address
    sorted_allocs = sorted(allocations, key=lambda x: x[0])

    for i in range(len(sorted_allocs) - 1):
        addr1, size1 = sorted_allocs[i]
        addr2, size2 = sorted_allocs[i + 1]
        # Check that first block ends before second block starts
        if addr1 + size1 > addr2:
            return False

    return True
