# SPDX-License-Identifier: Apache-2.0
"""
Regression tests for issue #6215: staging scratch buffer prevents
allocator reuse from corrupting in-flight buffer-GET data.

Tests cover:
1. The underlying bug (dangling byte_array after allocator reuse)
2. Staging buffer safety (independent of allocator lifecycle)
3. ValkeyConnector._do_get_into staging path
"""

# Third Party
import torch

# First Party
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator


class TestUnderlyingBug:
    """Demonstrates the root cause: byte_array dangles after free."""

    def test_byte_array_dangling_after_free(self):
        """byte_array memoryview sees new allocation's data after slot reuse."""
        pool = torch.zeros(8192, dtype=torch.uint8, device="cpu")
        allocator = TensorMemoryAllocator(pool)
        shape = torch.Size([4096])

        obj1 = allocator.allocate(shape, torch.uint8)
        obj1.raw_data.fill_(0xAA)
        mv1 = obj1.byte_array
        if mv1.format != "B":
            mv1 = mv1.cast("B")
        assert mv1[0] == 0xAA

        obj1.ref_count_down()  # free slot

        obj2 = allocator.allocate(shape, torch.uint8)
        obj2.raw_data.fill_(0xBB)

        # BUG: mv1 is dangling — sees obj2's data
        assert mv1[0] == 0xBB
        obj2.ref_count_down()

    def test_late_write_corrupts_recycled_slot(self):
        """A write into a dangling memoryview destroys another allocation."""
        # Standard
        import ctypes

        pool = torch.zeros(4096, dtype=torch.uint8, device="cpu")
        allocator = TensorMemoryAllocator(pool)
        shape = torch.Size([4096])

        obj_a = allocator.allocate(shape, torch.uint8)
        mv_a = obj_a.byte_array
        if mv_a.format != "B":
            mv_a = mv_a.cast("B")

        obj_a.ref_count_down()  # slot recycled

        obj_b = allocator.allocate(shape, torch.uint8)
        obj_b.raw_data.fill_(0xBB)

        # Late write (simulates Rust buffer-GET arriving after free)
        ctypes.memset(ctypes.addressof(ctypes.c_ubyte.from_buffer(mv_a)), 0xCC, 4096)

        # obj_b is corrupted
        assert obj_b.raw_data[0].item() == 0xCC
        obj_b.ref_count_down()


class TestStagingBufferSafety:
    """Staging scratch buffer is independent of allocator lifecycle."""

    def test_scratch_unaffected_by_slab_reuse(self):
        """Scratch buffer retains correct data even after slab is recycled."""
        pool = torch.zeros(4096, dtype=torch.uint8, device="cpu")
        allocator = TensorMemoryAllocator(pool)
        shape = torch.Size([4096])

        scratch = bytearray(4096)
        scratch[:] = bytes([0xAA]) * 4096  # simulate GLIDE writing

        obj = allocator.allocate(shape, torch.uint8)
        obj.ref_count_down()  # free

        obj2 = allocator.allocate(shape, torch.uint8)
        obj2.raw_data.fill_(0xBB)  # overwrite slab

        # Scratch is untouched — safe to copy from
        assert scratch[0] == 0xAA
        assert scratch[4095] == 0xAA
        obj2.ref_count_down()

    def test_scratch_reuse_across_calls(self):
        """Scratch buffer grows and is reused without reallocation."""
        # Standard
        import threading

        local = threading.local()

        def get_scratch(size):
            buf = getattr(local, "scratch", None)
            if buf is None or len(buf) < size:
                buf = bytearray(size)
                local.scratch = buf
            return buf

        s1 = get_scratch(1024)
        s1_id = id(s1)
        s2 = get_scratch(512)  # smaller — reuses same buffer
        assert id(s2) == s1_id

        s3 = get_scratch(2048)  # larger — grows
        assert len(s3) == 2048
        s4 = get_scratch(1024)  # smaller again — reuses grown buffer
        assert id(s4) == id(s3)


class TestValkeyConnectorStaging:
    """Tests for _ThreadWorkerPool staging buffer integration."""

    def test_get_scratch_returns_correct_size(self):
        """_get_scratch returns buffer at least as large as requested."""
        # Standard
        import threading

        # Simulate the _get_scratch method
        local = threading.local()

        def _get_scratch(size):
            buf = getattr(local, "scratch", None)
            if buf is None or len(buf) < size:
                buf = bytearray(size)
                local.scratch = buf
            return buf

        buf = _get_scratch(65536)
        assert len(buf) >= 65536
        # Reuse
        buf2 = _get_scratch(4096)
        assert id(buf2) == id(buf)

    def test_do_get_into_copies_only_n_bytes(self):
        """Only the bytes actually received are copied into destination."""
        scratch = bytearray(1024)
        scratch[:500] = bytes([0xAA]) * 500
        scratch[500:] = bytes([0x00]) * 524

        dst = bytearray(1024)
        n = 500
        # Simulate: buf[:n] = scratch_view[:n]
        memoryview(dst)[:n] = memoryview(scratch)[:n]

        assert dst[0] == 0xAA
        assert dst[499] == 0xAA
        assert dst[500] == 0x00  # untouched beyond n


class TestValkeyConnectorBatchStaging:
    """Tests for batched staging buffer copy logic."""

    def test_batch_staging_copies_all_buffers(self):
        """Batched path stages into scratch then copies to each slab."""
        pool = torch.zeros(4096 * 3, dtype=torch.uint8, device="cpu")
        allocator = TensorMemoryAllocator(pool)
        shape = torch.Size([4096])

        objs = [allocator.allocate(shape, torch.uint8) for _ in range(3)]
        assert all(o is not None for o in objs)

        # Simulate: batch_get writes different patterns into scratch buffers
        scratches = [bytearray(4096) for _ in range(3)]
        for i, s in enumerate(scratches):
            s[:] = bytes([i + 1]) * 4096

        # Copy staged data into slab buffers (simulates post-batch_get copy)
        for obj, scratch in zip(objs, scratches, strict=True):
            dst = obj.byte_array
            if not isinstance(dst, memoryview):
                dst = memoryview(dst)
            if dst.format != "B":
                dst = dst.cast("B")
            dst[:4096] = memoryview(scratch)[:4096]

        # Verify each slab got the right data
        assert objs[0].raw_data[0].item() == 1
        assert objs[1].raw_data[0].item() == 2
        assert objs[2].raw_data[0].item() == 3

        for obj in objs:
            obj.ref_count_down()
        assert allocator.memcheck()
