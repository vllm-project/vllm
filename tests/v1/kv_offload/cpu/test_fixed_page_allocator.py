"""Focused unit tests for FixedPageAllocator."""

import pytest

from vllm.v1.kv_offload.cpu.fixed_page_allocator import FixedPageAllocator


def test_scattered_pages_allocate_without_contiguous_fit() -> None:
    allocator = FixedPageAllocator(8 * 256, 256)
    allocations = [allocator.allocate(256) for _ in range(8)]
    assert all(allocation is not None for allocation in allocations)
    for idx in (1, 3, 5, 7):
        allocator.free(allocations[idx])
    large = allocator.allocate(1024)
    assert large is not None
    assert large.page_ids == (1, 3, 5, 7)
    assert allocator.fragmentation == 0.0


def test_simulation_validates_candidate_ownership_and_is_atomic() -> None:
    allocator = FixedPageAllocator(1024, 256)
    resident = allocator.allocate(768)
    assert resident is not None
    assert not allocator.simulate_batch_allocation([512])
    assert allocator.simulate_batch_allocation([512], [resident])
    other = FixedPageAllocator(1024, 256).allocate(256)
    with pytest.raises(ValueError, match="different page allocator"):
        allocator.simulate_batch_allocation([256], [other])
    assert allocator.used_bytes == 768


def test_page_spans_coalesce_adjacent_pages_and_preserve_logical_tail() -> None:
    allocator = FixedPageAllocator(8 * 256, 256)
    pages = [allocator.allocate(256) for _ in range(8)]
    for idx in (1, 2, 5, 6):
        allocator.free(pages[idx])
    allocation = allocator.allocate(700)
    assert allocation is not None
    assert allocation.page_ids == (1, 2, 5)
    assert allocator.page_spans(allocation) == (
        (256, 512, 512),
        (1280, 188, 256),
    )


def test_reset_rejects_stale_allocation() -> None:
    allocator = FixedPageAllocator(1024, 256)
    allocation = allocator.allocate(256)
    allocator.reset()
    with pytest.raises(ValueError, match="unknown"):
        allocator.free(allocation)


def test_allocation_size_validation() -> None:
    allocator = FixedPageAllocator(1024, 256)
    with pytest.raises(ValueError, match="positive"):
        allocator.pages_required(0)
    with pytest.raises(ValueError, match="positive"):
        allocator.pages_required(-1)
    assert allocator.pages_required(1) == 1
    assert allocator.pages_required(256) == 1
    assert allocator.pages_required(257) == 2


def test_allocate_returns_none_when_full() -> None:
    allocator = FixedPageAllocator(512, 256)
    assert allocator.allocate(256) is not None
    assert allocator.allocate(256) is not None
    assert allocator.allocate(256) is None


def test_free_nonexistent_allocation_raises() -> None:
    allocator = FixedPageAllocator(1024, 256)
    other = FixedPageAllocator(1024, 256)
    other_alloc = other.allocate(256)
    with pytest.raises(ValueError, match="different page allocator"):
        allocator.free(other_alloc)

    # Double-free raises
    a = allocator.allocate(256)
    allocator.free(a)
    with pytest.raises(ValueError, match="unknown"):
        allocator.free(a)


def test_simulate_batch_allocation_edge_cases() -> None:
    allocator = FixedPageAllocator(1024, 256)
    # Non-positive sizes raise ValueError
    with pytest.raises(ValueError, match="allocation size must be positive"):
        allocator.simulate_batch_allocation([0])
    with pytest.raises(ValueError, match="allocation size must be positive"):
        allocator.simulate_batch_allocation([-1])
    with pytest.raises(ValueError, match="allocation size must be positive"):
        allocator.simulate_batch_allocation([256, 0, 128])
    # Empty sizes with no frees: always possible
    assert allocator.simulate_batch_allocation([])
    # Duplicate candidate in frees raises
    a = allocator.allocate(256)
    with pytest.raises(ValueError, match="duplicate"):
        allocator.simulate_batch_allocation([256], [a, a])


def test_page_spans_single_page() -> None:
    allocator = FixedPageAllocator(1024, 256)
    a = allocator.allocate(100)
    assert a is not None
    spans = allocator.page_spans(a)
    assert spans == ((0, 100, 256),)


def test_page_spans_wrong_allocator() -> None:
    """page_spans raises explicit error for foreign allocator."""
    allocator = FixedPageAllocator(1024, 256)
    other = FixedPageAllocator(1024, 256)
    other_alloc = other.allocate(256)
    with pytest.raises(ValueError, match="different page allocator"):
        allocator.page_spans(other_alloc)


def test_page_spans_scattered_non_consecutive() -> None:
    """Scattered non-adjacent page_ids produce separate uncoalesced spans."""
    allocator = FixedPageAllocator(8 * 256, 256)
    pages = [allocator.allocate(256) for _ in range(8)]
    for idx in (0, 2, 4, 6):
        allocator.free(pages[idx])
    allocation = allocator.allocate(700)
    assert allocation is not None
    # Non-consecutive page_ids (0, 2, 4)
    assert allocation.page_ids == (0, 2, 4)
    spans = allocator.page_spans(allocation)
    assert len(spans) == 3
    assert spans[0] == (0, 256, 256)
    assert spans[1] == (512, 256, 256)  # page 2 starts at byte 512
    assert spans[2] == (1024, 188, 256)  # page 4 starts at byte 1024


def test_properties_reflect_state() -> None:
    allocator = FixedPageAllocator(1024, 256)
    assert allocator.total_bytes == 1024
    assert allocator.page_size == 256
    assert allocator.free_bytes == 1024
    assert allocator.used_bytes == 0
    assert allocator.num_active_handles == 0
    assert allocator.largest_free_block == 1024

    a = allocator.allocate(256)
    assert a is not None
    assert allocator.free_bytes == 768
    assert allocator.used_bytes == 256
    assert allocator.num_active_handles == 1

    allocator.free(a)
    assert allocator.free_bytes == 1024
    assert allocator.used_bytes == 0
    assert allocator.num_active_handles == 0
