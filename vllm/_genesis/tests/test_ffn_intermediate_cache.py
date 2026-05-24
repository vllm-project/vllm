# SPDX-License-Identifier: Apache-2.0
"""TDD tests for vllm._genesis.kernels.ffn_intermediate_cache.

PN12 migration target: pool transient SiluAndMul output buffers across layers
to close Cliff 1 (138 MiB OOM at long-ctx + tool-call on TQ3 path) which PN8
empirically does not address (different memory class — transient activation
peak vs persistent draft footprint).

Root cause: vllm/model_executor/layers/activation.py:146 SiluAndMul.forward_cuda
allocates [M, intermediate_size] BF16 transient PER LAYER × 64 layers (Lorbus
27B) = 4.7-18 GiB allocator churn per step.

Design invariants tested here:
  1. Single shared buffer per (intermediate_size, dtype, device).
  2. Returned tensor is a slice — caller writes in-place, no new allocation.
  3. Sequential layer execution (no concurrent overlap on same buffer).
  4. Idempotent on size growth (max-shape preallocate, slice on acquire).
  5. Returns IDENTICAL data_ptr across calls for same (key, M).
  6. Disabled when env GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL is unset.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import pytest
import torch


# ═══════════════════════════════════════════════════════════════════════════
#                       ENV GATE BEHAVIOR
# ═══════════════════════════════════════════════════════════════════════════

class TestEnvGate:
    """Group 1: should_apply() respects GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL."""

    def test_should_apply_returns_bool(self):
        from vllm._genesis.kernels.ffn_intermediate_cache import (
            FFNIntermediateCache,
        )
        assert isinstance(FFNIntermediateCache.should_apply(), bool)

    def test_should_apply_false_when_env_unset(self, monkeypatch):
        monkeypatch.delenv(
            "GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL", raising=False
        )
        from vllm._genesis.kernels.ffn_intermediate_cache import (
            FFNIntermediateCache,
        )
        assert FFNIntermediateCache.should_apply() is False

    def test_should_apply_false_when_env_zero(self, monkeypatch):
        monkeypatch.setenv("GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL", "0")
        from vllm._genesis.kernels.ffn_intermediate_cache import (
            FFNIntermediateCache,
        )
        assert FFNIntermediateCache.should_apply() is False

    def test_should_apply_true_when_env_one(self, monkeypatch):
        monkeypatch.setenv("GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL", "1")
        from vllm._genesis.kernels.ffn_intermediate_cache import (
            FFNIntermediateCache,
        )
        assert FFNIntermediateCache.should_apply() is True


# ═══════════════════════════════════════════════════════════════════════════
#                       BUFFER LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════

class TestBufferLifecycle:
    """Group 2: get_or_create returns shape-correct CPU tensors (graph-safe).

    These tests use CPU tensors — Cliff 1 is a memory-management issue,
    NOT a kernel correctness issue. CPU mock proves pointer-stability +
    slice semantics. GPU path is identical except for device argument.
    """

    def setup_method(self):
        # Reset registry between tests so namespaces don't bleed.
        from vllm._genesis.kernels.ffn_intermediate_cache import (
            FFNIntermediateCache,
        )
        FFNIntermediateCache._BUFFER_REGISTRY.clear()

    def test_acquire_silu_out_returns_correct_shape(self):
        from vllm._genesis.kernels.ffn_intermediate_cache import (
            FFNIntermediateCache,
        )
        out = FFNIntermediateCache.acquire_silu_out(
            num_tokens=128, intermediate_size=17408,
            dtype=torch.bfloat16, device=torch.device("cpu"),
        )
        assert out.shape == (128, 17408)
        assert out.dtype == torch.bfloat16
        assert out.device.type == "cpu"

    def test_acquire_returns_same_pointer_for_equal_shape(self):
        """Pointer-stable: same key + same M → same data_ptr (graph-safe)."""
        from vllm._genesis.kernels.ffn_intermediate_cache import (
            FFNIntermediateCache,
        )
        out1 = FFNIntermediateCache.acquire_silu_out(
            num_tokens=64, intermediate_size=8192,
            dtype=torch.bfloat16, device=torch.device("cpu"),
        )
        out2 = FFNIntermediateCache.acquire_silu_out(
            num_tokens=64, intermediate_size=8192,
            dtype=torch.bfloat16, device=torch.device("cpu"),
        )
        # Underlying storage MUST be the same object → cudagraph capture-safe
        assert out1.data_ptr() == out2.data_ptr()

    def test_acquire_smaller_M_returns_slice_of_same_buffer(self):
        """Subsequent smaller M → narrowed view of same backing buffer."""
        from vllm._genesis.kernels.ffn_intermediate_cache import (
            FFNIntermediateCache,
        )
        big = FFNIntermediateCache.acquire_silu_out(
            num_tokens=4096, intermediate_size=8192,
            dtype=torch.bfloat16, device=torch.device("cpu"),
        )
        small = FFNIntermediateCache.acquire_silu_out(
            num_tokens=128, intermediate_size=8192,
            dtype=torch.bfloat16, device=torch.device("cpu"),
        )
        assert small.shape == (128, 8192)
        # Both views point into the SAME underlying buffer
        assert small.data_ptr() == big.data_ptr()

    def test_acquire_larger_M_grows_buffer_once(self):
        """Subsequent larger M → buffer grows; pointer changes (one-time
        reallocation; subsequent same-large acquire stays stable)."""
        from vllm._genesis.kernels.ffn_intermediate_cache import (
            FFNIntermediateCache,
        )
        small = FFNIntermediateCache.acquire_silu_out(
            num_tokens=64, intermediate_size=8192,
            dtype=torch.bfloat16, device=torch.device("cpu"),
        )
        small_ptr = small.data_ptr()

        big = FFNIntermediateCache.acquire_silu_out(
            num_tokens=8192, intermediate_size=8192,
            dtype=torch.bfloat16, device=torch.device("cpu"),
        )
        # New ptr (we grew the buffer)
        assert big.data_ptr() != small_ptr

        big2 = FFNIntermediateCache.acquire_silu_out(
            num_tokens=8192, intermediate_size=8192,
            dtype=torch.bfloat16, device=torch.device("cpu"),
        )
        # Stable after growth
        assert big2.data_ptr() == big.data_ptr()

    def test_different_intermediate_size_uses_different_buffer(self):
        from vllm._genesis.kernels.ffn_intermediate_cache import (
            FFNIntermediateCache,
        )
        a = FFNIntermediateCache.acquire_silu_out(
            num_tokens=64, intermediate_size=8192,
            dtype=torch.bfloat16, device=torch.device("cpu"),
        )
        b = FFNIntermediateCache.acquire_silu_out(
            num_tokens=64, intermediate_size=17408,
            dtype=torch.bfloat16, device=torch.device("cpu"),
        )
        # Distinct keys → distinct backing buffers
        assert a.data_ptr() != b.data_ptr()
        assert a.shape == (64, 8192)
        assert b.shape == (64, 17408)

    def test_different_dtype_uses_different_buffer(self):
        from vllm._genesis.kernels.ffn_intermediate_cache import (
            FFNIntermediateCache,
        )
        a = FFNIntermediateCache.acquire_silu_out(
            num_tokens=64, intermediate_size=8192,
            dtype=torch.bfloat16, device=torch.device("cpu"),
        )
        b = FFNIntermediateCache.acquire_silu_out(
            num_tokens=64, intermediate_size=8192,
            dtype=torch.float16, device=torch.device("cpu"),
        )
        assert a.data_ptr() != b.data_ptr()
        assert a.dtype == torch.bfloat16
        assert b.dtype == torch.float16


# ═══════════════════════════════════════════════════════════════════════════
#                       CORRECTNESS (write semantics)
# ═══════════════════════════════════════════════════════════════════════════

class TestWriteSemantics:
    """Group 3: caller writes into the slice and reads back (numerical check)."""

    def setup_method(self):
        from vllm._genesis.kernels.ffn_intermediate_cache import (
            FFNIntermediateCache,
        )
        FFNIntermediateCache._BUFFER_REGISTRY.clear()

    def test_inplace_write_visible_in_returned_slice(self):
        from vllm._genesis.kernels.ffn_intermediate_cache import (
            FFNIntermediateCache,
        )
        out = FFNIntermediateCache.acquire_silu_out(
            num_tokens=4, intermediate_size=8,
            dtype=torch.bfloat16, device=torch.device("cpu"),
        )
        out.fill_(1.5)
        # Re-acquire same shape; should see the write
        out2 = FFNIntermediateCache.acquire_silu_out(
            num_tokens=4, intermediate_size=8,
            dtype=torch.bfloat16, device=torch.device("cpu"),
        )
        # Approximate equality — bfloat16 has limited precision
        assert torch.allclose(
            out2.float(),
            torch.full_like(out2.float(), 1.5),
            atol=1e-2,
        )

    def test_partial_write_does_not_corrupt_unused_region(self):
        """Smaller M acquire writes only the slice; bigger acquire later sees
        whatever was written previously in the slice + zeros (or undefined)
        in the unwritten region. Tests slice isolation, not zero-init."""
        from vllm._genesis.kernels.ffn_intermediate_cache import (
            FFNIntermediateCache,
        )
        big = FFNIntermediateCache.acquire_silu_out(
            num_tokens=8, intermediate_size=4,
            dtype=torch.bfloat16, device=torch.device("cpu"),
        )
        big.fill_(0.0)

        small = FFNIntermediateCache.acquire_silu_out(
            num_tokens=2, intermediate_size=4,
            dtype=torch.bfloat16, device=torch.device("cpu"),
        )
        small.fill_(7.0)

        # The big view sees: rows 0-1 → 7.0, rows 2-7 → 0.0 (untouched)
        big_again = FFNIntermediateCache.acquire_silu_out(
            num_tokens=8, intermediate_size=4,
            dtype=torch.bfloat16, device=torch.device("cpu"),
        )
        first_two_rows = big_again[:2].float()
        rest = big_again[2:].float()
        assert torch.allclose(first_two_rows,
                              torch.full((2, 4), 7.0), atol=1e-2)
        assert torch.allclose(rest, torch.zeros(6, 4), atol=1e-2)


# ═══════════════════════════════════════════════════════════════════════════
#                       MEMORY ACCOUNTING (the actual win)
# ═══════════════════════════════════════════════════════════════════════════

class TestMemoryAccounting:
    """Group 4: registry stays bounded — N calls do NOT create N buffers."""

    def setup_method(self):
        from vllm._genesis.kernels.ffn_intermediate_cache import (
            FFNIntermediateCache,
        )
        FFNIntermediateCache._BUFFER_REGISTRY.clear()

    def test_64_acquires_create_only_one_buffer(self):
        """Lorbus 27B has 64 FFN layers — each forward pass calls
        SiluAndMul once per layer. Old behavior: 64 fresh allocations.
        New behavior with PN12: ONE shared buffer."""
        from vllm._genesis.kernels.ffn_intermediate_cache import (
            FFNIntermediateCache,
        )
        outputs = []
        for _layer in range(64):
            out = FFNIntermediateCache.acquire_silu_out(
                num_tokens=4096, intermediate_size=17408,
                dtype=torch.bfloat16, device=torch.device("cpu"),
            )
            outputs.append(out)
        # All 64 must alias the same backing buffer
        ptrs = {o.data_ptr() for o in outputs}
        assert len(ptrs) == 1, (
            f"Expected exactly 1 unique data_ptr across 64 layers, got {len(ptrs)}"
        )

    def test_registry_size_one_per_unique_key(self):
        from vllm._genesis.kernels.ffn_intermediate_cache import (
            FFNIntermediateCache,
        )
        for _ in range(50):
            FFNIntermediateCache.acquire_silu_out(
                num_tokens=64, intermediate_size=8192,
                dtype=torch.bfloat16, device=torch.device("cpu"),
            )
        # Same key, 50 calls → registry should hold exactly 1 entry
        assert len(FFNIntermediateCache._BUFFER_REGISTRY) == 1

    def test_clear_registry_drops_all_entries(self):
        from vllm._genesis.kernels.ffn_intermediate_cache import (
            FFNIntermediateCache,
        )
        FFNIntermediateCache.acquire_silu_out(
            num_tokens=64, intermediate_size=8192,
            dtype=torch.bfloat16, device=torch.device("cpu"),
        )
        assert len(FFNIntermediateCache._BUFFER_REGISTRY) >= 1
        FFNIntermediateCache._BUFFER_REGISTRY.clear()
        assert len(FFNIntermediateCache._BUFFER_REGISTRY) == 0


# ═══════════════════════════════════════════════════════════════════════════
#                       BAD INPUT HANDLING
# ═══════════════════════════════════════════════════════════════════════════

class TestBadInput:
    """Group 5: invalid inputs raise informative errors (loud-fail principle)."""

    def setup_method(self):
        from vllm._genesis.kernels.ffn_intermediate_cache import (
            FFNIntermediateCache,
        )
        FFNIntermediateCache._BUFFER_REGISTRY.clear()

    def test_zero_num_tokens_raises(self):
        from vllm._genesis.kernels.ffn_intermediate_cache import (
            FFNIntermediateCache,
        )
        with pytest.raises((ValueError, RuntimeError, AssertionError)):
            FFNIntermediateCache.acquire_silu_out(
                num_tokens=0, intermediate_size=8192,
                dtype=torch.bfloat16, device=torch.device("cpu"),
            )

    def test_negative_intermediate_size_raises(self):
        from vllm._genesis.kernels.ffn_intermediate_cache import (
            FFNIntermediateCache,
        )
        with pytest.raises((ValueError, RuntimeError, AssertionError)):
            FFNIntermediateCache.acquire_silu_out(
                num_tokens=64, intermediate_size=-1,
                dtype=torch.bfloat16, device=torch.device("cpu"),
            )
