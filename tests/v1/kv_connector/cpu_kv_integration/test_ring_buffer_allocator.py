# SPDX-License-Identifier: Apache-2.0
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl_cpu_utils import (
    RingBufferAllocator)


def test_basic_allocation():
    """Test basic allocation and deallocation behavior."""
    # Create a buffer with 1024 bytes, aligned to 256 bytes
    allocator = RingBufferAllocator(size=1024, align_to=256)

    # Allocate 100 bytes - should be aligned to 256
    addr1, buffer1 = allocator.allocate(100)
    assert addr1 >= 0  # Valid address
    assert buffer1 is not None
    assert len(buffer1) == 100
    assert allocator.high_watermark == 256  # Aligned to 256
    assert allocator.low_watermark == 0

    # Allocate another 100 bytes
    addr2, buffer2 = allocator.allocate(100)
    assert addr2 >= 0  # Valid address
    assert buffer2 is not None
    assert len(buffer2) == 100
    assert allocator.high_watermark == 512  # Next aligned position

    # Verify buffers don't overlap
    assert buffer1.data_ptr() + len(buffer1) <= buffer2.data_ptr()


def test_alignment():
    """Test that allocations are properly aligned."""
    allocator = RingBufferAllocator(size=1024, align_to=256)

    # Allocate various sizes and verify alignment
    sizes = [10, 100, 200, 50]
    addresses = []
    buffers = []

    for size in sizes:
        addr, buf = allocator.allocate(size)
        assert addr >= 0  # Valid address
        assert buf is not None
        addresses.append(addr)
        buffers.append(buf)
        # High watermark should always be aligned to 256
        assert allocator.high_watermark % 256 == 0


def test_wraparound():
    """Test buffer wraparound behavior."""
    allocator = RingBufferAllocator(size=1024, align_to=256)

    # Fill most of the buffer
    addr1, buffer1 = allocator.allocate(300)  # Takes 512 bytes aligned
    addr2, buffer2 = allocator.allocate(300)  # Takes 512 bytes aligned
    assert addr1 >= 0 and addr2 >= 0  # Valid addresses
    assert buffer1 is not None and buffer2 is not None

    # This allocation should fail as we don't have enough contiguous space
    addr3, buffer3 = allocator.allocate(300)
    assert addr3 == -1  # Invalid address
    assert buffer3 is None

    # Free the first buffer
    allocator.free(addr1)  # Free first 512 bytes

    # Now we should be able to allocate again by wrapping around
    addr4, buffer4 = allocator.allocate(200)
    assert addr4 >= 0  # Valid address
    assert buffer4 is not None
    assert allocator.high_watermark >= allocator._size  # Wrapped around
    assert allocator.high_watermark % allocator._size < 512  # Using freed space


def test_fragmentation():
    """Test handling of fragmentation."""
    allocator = RingBufferAllocator(size=1024, align_to=256)

    # Allocate several buffers
    addr1, buffer1 = allocator.allocate(100)  # 256 bytes aligned
    addr2, buffer2 = allocator.allocate(100)  # 256 bytes aligned
    addr3, buffer3 = allocator.allocate(100)  # 256 bytes aligned
    assert all(addr >= 0 for addr in [addr1, addr2, addr3])  # Valid addresses
    assert all(buf is not None for buf in [buffer1, buffer2, buffer3])

    # Free buffer2, creating a gap
    allocator.free(addr2)  # Free middle buffer

    # Try to allocate a buffer larger than the gap
    addr4, buffer4 = allocator.allocate(300)
    assert addr4 == -1  # Invalid address
    assert buffer4 is None  # Should fail due to fragmentation

    # Allocate a buffer that fits in the gap
    # This should also fail as we don't track gaps in current implementation
    addr5, buffer5 = allocator.allocate(100)
    assert addr5 == -1  # Invalid address
    assert buffer5 is None  # Should fail due to fragmentation

    # Free buffer1
    allocator.free(addr1)  # Free first buffer

    # Now we should be able to allocate again
    addr6, buffer6 = allocator.allocate(100)
    assert addr6 >= 0  # Valid address
    assert buffer6 is not None
    assert allocator.high_watermark >= allocator._size  # Wrapped around
    assert allocator.high_watermark % allocator._size < 512  # Using freed space


def test_full_buffer():
    """Test behavior when buffer is completely full."""
    allocator = RingBufferAllocator(size=1024, align_to=256)

    # Fill the entire buffer
    addresses = []
    buffers = []
    while True:
        addr, buf = allocator.allocate(200)
        if addr == -1:  # Invalid address indicates allocation failure
            break
        addresses.append(addr)
        buffers.append(buf)

    # Verify we can't allocate more
    addr, buf = allocator.allocate(10)
    assert addr == -1
    assert buf is None

    # Free everything
    for addr in addresses:
        allocator.free(addr)

    # Should be able to allocate again
    addr, buffer = allocator.allocate(200)
    assert addr >= 0  # Valid address
    assert buffer is not None


def test_invalid_free():
    """Test that freeing invalid addresses raises an error."""
    allocator = RingBufferAllocator(size=1024, align_to=256)

    # Allocate a buffer
    addr, buffer = allocator.allocate(100)
    assert addr >= 0  # Valid address
    assert buffer is not None

    # Try to free an invalid address
    with pytest.raises(AssertionError):
        allocator.free(100)  # Invalid address
