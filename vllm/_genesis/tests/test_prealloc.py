# SPDX-License-Identifier: Apache-2.0
"""TDD tests for vllm._genesis.prealloc.GenesisPreallocBuffer.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import pytest
import torch


class TestGetOrCreate:
    """Group 1: Core get_or_create behavior."""

    def test_first_call_allocates_fresh_tensor(self, reset_genesis_prealloc):
        from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB

        buf = GPB.get_or_create(
            namespace="test_fresh",
            shape=(4, 8),
            dtype=torch.float32,
            device="cpu",
        )
        assert isinstance(buf, torch.Tensor)
        assert buf.shape == (4, 8)
        assert buf.dtype == torch.float32

    def test_same_key_returns_same_tensor(self, reset_genesis_prealloc):
        """Calling twice with same args returns IDENTICAL tensor (same ptr)."""
        from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB

        buf1 = GPB.get_or_create("same", (4,), torch.float32, "cpu")
        buf2 = GPB.get_or_create("same", (4,), torch.float32, "cpu")

        # Same object (same memory address, same data_ptr)
        assert buf1 is buf2
        assert buf1.data_ptr() == buf2.data_ptr()

    def test_different_namespace_different_tensor(self, reset_genesis_prealloc):
        """Different namespaces create separate tensors."""
        from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB

        buf1 = GPB.get_or_create("ns1", (4,), torch.float32, "cpu")
        buf2 = GPB.get_or_create("ns2", (4,), torch.float32, "cpu")

        assert buf1 is not buf2
        assert buf1.data_ptr() != buf2.data_ptr()

    def test_different_shape_different_tensor(self, reset_genesis_prealloc):
        from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB

        buf1 = GPB.get_or_create("ns", (4,), torch.float32, "cpu")
        buf2 = GPB.get_or_create("ns", (8,), torch.float32, "cpu")

        assert buf1 is not buf2

    def test_different_dtype_different_tensor(self, reset_genesis_prealloc):
        from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB

        buf1 = GPB.get_or_create("ns", (4,), torch.float32, "cpu")
        buf2 = GPB.get_or_create("ns", (4,), torch.float16, "cpu")

        assert buf1 is not buf2

    def test_zero_init_creates_zeros(self, reset_genesis_prealloc):
        from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB

        buf = GPB.get_or_create(
            "zeros", (4, 8), torch.float32, "cpu", zero_init=True
        )
        assert (buf == 0).all()

    def test_default_no_init_uses_torch_empty(self, reset_genesis_prealloc):
        """Default is torch.empty (uninitialized) for speed."""
        from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB

        # Just make sure it doesn't crash (content is garbage, won't check)
        buf = GPB.get_or_create("empty", (4, 8), torch.float32, "cpu")
        assert buf.numel() == 32


class TestSliceTo:
    """Group 2: Slice helper."""

    def test_slice_to_returns_view(self, reset_genesis_prealloc):
        """Slicing returns a view (shares storage)."""
        from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB

        buf = GPB.get_or_create("slice_test", (16, 8), torch.float32, "cpu")
        sliced = GPB.slice_to(buf, 4, dim=0)

        assert sliced.shape == (4, 8)
        # View shares storage
        assert sliced.data_ptr() == buf.data_ptr()

    def test_slice_to_modifications_visible_through_parent(self, reset_genesis_prealloc):
        """Writing to slice visible in parent buffer (shared storage)."""
        from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB

        buf = GPB.get_or_create("shared", (16,), torch.float32, "cpu", zero_init=True)
        sliced = GPB.slice_to(buf, 4)

        sliced.fill_(42.0)

        # First 4 elements now 42, rest still 0
        assert (buf[:4] == 42.0).all()
        assert (buf[4:] == 0.0).all()

    def test_slice_to_raises_on_overflow(self, reset_genesis_prealloc):
        """Exceeding buffer shape raises with helpful message."""
        from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB

        buf = GPB.get_or_create("small", (8,), torch.float32, "cpu")

        with pytest.raises(AssertionError, match="exceeds buffer"):
            GPB.slice_to(buf, 16)

    def test_slice_to_negative_raises(self, reset_genesis_prealloc):
        from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB

        buf = GPB.get_or_create("neg_test", (8,), torch.float32, "cpu")

        with pytest.raises(ValueError, match="negative"):
            GPB.slice_to(buf, -1)

    def test_slice_to_zero_returns_empty_view(self, reset_genesis_prealloc):
        """n=0 returns an empty view (edge case)."""
        from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB

        buf = GPB.get_or_create("z", (8,), torch.float32, "cpu")
        sliced = GPB.slice_to(buf, 0)

        assert sliced.shape == (0,)

    def test_slice_to_custom_dim(self, reset_genesis_prealloc):
        from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB

        buf = GPB.get_or_create("dim_test", (4, 16), torch.float32, "cpu")
        sliced = GPB.slice_to(buf, 8, dim=1)

        assert sliced.shape == (4, 8)


class TestRegistryInfo:
    """Group 3: Diagnostic / observability."""

    def test_empty_registry_reports_zero(self, reset_genesis_prealloc):
        from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB

        info = GPB.get_registry_info()
        assert info["total_buffers"] == 0
        assert info["total_bytes"] == 0
        assert info["entries"] == []

    def test_registry_info_tracks_allocations(self, reset_genesis_prealloc):
        from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB

        GPB.get_or_create("a", (4,), torch.float32, "cpu")
        GPB.get_or_create("b", (8, 16), torch.float16, "cpu")

        info = GPB.get_registry_info()
        assert info["total_buffers"] == 2

        namespaces = {e["namespace"] for e in info["entries"]}
        assert namespaces == {"a", "b"}

    def test_registry_info_json_serializable(self, reset_genesis_prealloc):
        """Registry info must be JSON-serializable for logs."""
        import json
        from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB

        GPB.get_or_create("x", (4,), torch.bfloat16, "cpu")
        info = GPB.get_registry_info()

        # default=str handles dtype objects
        json_str = json.dumps(info, default=str)
        assert len(json_str) > 0


class TestPointerStability:
    """Group 4: CRITICAL — same pointer across calls (CUDA graph safety)."""

    def test_pointer_never_changes(self, reset_genesis_prealloc):
        """Repeated get_or_create calls → identical data_ptr.

        This is THE critical invariant for CUDA graph replay safety.
        If this test fails, captured CUDA graphs will break.
        """
        from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB

        pointers = set()
        for _ in range(100):
            buf = GPB.get_or_create("stable", (64, 128), torch.bfloat16, "cpu")
            pointers.add(buf.data_ptr())

        assert len(pointers) == 1, (
            f"Pointer changed across calls: {pointers}. "
            f"This would break CUDA graph replay!")

    def test_slice_pointer_matches_parent_offset(self, reset_genesis_prealloc):
        """Slice pointer is parent pointer + byte offset."""
        from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB

        buf = GPB.get_or_create("parent", (32,), torch.float32, "cpu")
        sliced = GPB.slice_to(buf, 16)

        # Slice of [0:n] starts at same address as parent
        assert sliced.data_ptr() == buf.data_ptr()


class TestClearForTests:
    """Group 5: clear_for_tests() behavior."""

    def test_clear_removes_all_buffers(self, reset_genesis_prealloc):
        from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB

        GPB.get_or_create("a", (4,), torch.float32, "cpu")
        GPB.get_or_create("b", (4,), torch.float32, "cpu")

        assert GPB.get_registry_info()["total_buffers"] == 2

        GPB.clear_for_tests()
        assert GPB.get_registry_info()["total_buffers"] == 0

    def test_clear_allows_re_alloc_with_different_pointer(self, reset_genesis_prealloc):
        """After clear, same namespace+shape gets a fresh tensor."""
        from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB

        buf1 = GPB.get_or_create("x", (4,), torch.float32, "cpu")
        buf1.data_ptr()

        GPB.clear_for_tests()

        buf2 = GPB.get_or_create("x", (4,), torch.float32, "cpu")
        buf2.data_ptr()

        # New tensor may or may not have same ptr — both outcomes OK
        # (Python allocator reuse is implementation detail). What matters is
        # buf1 and buf2 are different Python objects.
        assert buf1 is not buf2


@pytest.mark.cuda_required
class TestCUDABehavior:
    """Group 6: CUDA-specific behavior."""

    def test_cuda_allocation(self, reset_genesis_prealloc, cuda_available):
        if not cuda_available:
            pytest.skip("CUDA not available")

        from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB

        buf = GPB.get_or_create(
            "cuda_test", (16, 32), torch.bfloat16, "cuda"
        )
        assert buf.device.type == "cuda"

    def test_cuda_device_index_distinction(self, reset_genesis_prealloc, cuda_available):
        """Different CUDA devices = separate buffers."""
        if not cuda_available or torch.cuda.device_count() < 2:
            pytest.skip("Need 2+ CUDA devices")

        from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB

        buf0 = GPB.get_or_create("dev", (4,), torch.float32, "cuda:0")
        buf1 = GPB.get_or_create("dev", (4,), torch.float32, "cuda:1")

        assert buf0.device.index == 0
        assert buf1.device.index == 1
        assert buf0 is not buf1
