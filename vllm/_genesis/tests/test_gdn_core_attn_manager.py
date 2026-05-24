# SPDX-License-Identifier: Apache-2.0
"""TDD tests for `GdnCoreAttnManager` — Patch 28 core_attn_out prealloc.

Mirrors TurboQuantBufferManager test structure: platform guard, shape/dtype
contract, pointer stability, safety-net fallback when num_tokens exceeds
max, graceful skip on CPU.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import torch


class TestGdnCoreAttnManagerGuard:
    def test_should_apply_is_bool(self):
        from vllm._genesis.kernels.gdn_core_attn_manager import (
            GdnCoreAttnManager,
        )
        assert isinstance(GdnCoreAttnManager.should_apply(), bool)

    def test_get_or_create_returns_none_on_cpu(
        self, monkeypatch, reset_genesis_prealloc,
    ):
        from vllm._genesis.kernels import gdn_core_attn_manager as m
        # Clear any leftover cache from other tests and force platform=False
        m._SHOULD_APPLY_CACHED = False
        assert m.GdnCoreAttnManager.get_or_create(
            num_tokens_max=4096, num_v_heads=32, head_v_dim=128,
            device="cpu", dtype=torch.bfloat16,
        ) is None


class TestGdnCoreAttnManagerCompatible:
    """Platform guard forced True — behavioral contract."""

    def test_returns_right_shape_dtype(
        self, monkeypatch, reset_genesis_prealloc,
    ):
        from vllm._genesis.kernels import gdn_core_attn_manager as m
        # Post-redesign: should_apply reads module-level cache via
        # `should_apply()` function, which classmethod delegates to.
        m._SHOULD_APPLY_CACHED = True
        t = m.GdnCoreAttnManager.get_or_create(
            num_tokens_max=4096, num_v_heads=32, head_v_dim=128,
            device="cpu", dtype=torch.bfloat16,
        )
        assert t is not None
        assert t.shape == (4096, 32, 128)
        assert t.dtype == torch.bfloat16

    def test_pointer_stable(self, monkeypatch, reset_genesis_prealloc):
        from vllm._genesis.kernels import gdn_core_attn_manager as m
        # Post-redesign: should_apply reads module-level cache via
        # `should_apply()` function, which classmethod delegates to.
        m._SHOULD_APPLY_CACHED = True
        a = m.GdnCoreAttnManager.get_or_create(
            num_tokens_max=1024, num_v_heads=16, head_v_dim=64,
            device="cpu", dtype=torch.bfloat16,
        )
        b = m.GdnCoreAttnManager.get_or_create(
            num_tokens_max=1024, num_v_heads=16, head_v_dim=64,
            device="cpu", dtype=torch.bfloat16,
        )
        assert a is b, "must be pointer-stable for CUDA graph safety"

    def test_different_keys_different_buffers(
        self, monkeypatch, reset_genesis_prealloc,
    ):
        from vllm._genesis.kernels import gdn_core_attn_manager as m
        # Post-redesign: should_apply reads module-level cache via
        # `should_apply()` function, which classmethod delegates to.
        m._SHOULD_APPLY_CACHED = True
        a = m.GdnCoreAttnManager.get_or_create(
            num_tokens_max=1024, num_v_heads=16, head_v_dim=64,
            device="cpu", dtype=torch.bfloat16,
        )
        b = m.GdnCoreAttnManager.get_or_create(
            num_tokens_max=2048, num_v_heads=16, head_v_dim=64,
            device="cpu", dtype=torch.bfloat16,
        )
        assert a is not b


class TestAcquireSlice:
    """Forward-path entry: slice out of the shared pool + zero_()."""

    def test_returns_slice_of_correct_size(
        self, monkeypatch, reset_genesis_prealloc,
    ):
        from vllm._genesis.kernels import gdn_core_attn_manager as m
        # Post-redesign: should_apply reads module-level cache via
        # `should_apply()` function, which classmethod delegates to.
        m._SHOULD_APPLY_CACHED = True
        s = m.GdnCoreAttnManager.acquire_slice(
            num_tokens=512, num_v_heads=32, head_v_dim=128,
            device="cpu", dtype=torch.bfloat16,
            num_tokens_max=4096,
        )
        assert s.shape == (512, 32, 128)

    def test_slice_is_zeroed(self, monkeypatch, reset_genesis_prealloc):
        from vllm._genesis.kernels import gdn_core_attn_manager as m
        # Post-redesign: should_apply reads module-level cache via
        # `should_apply()` function, which classmethod delegates to.
        m._SHOULD_APPLY_CACHED = True
        s = m.GdnCoreAttnManager.acquire_slice(
            num_tokens=64, num_v_heads=8, head_v_dim=16,
            device="cpu", dtype=torch.bfloat16,
            num_tokens_max=512,
        )
        assert s.sum().item() == 0

    def test_slice_shares_storage_with_pool(
        self, monkeypatch, reset_genesis_prealloc,
    ):
        """Slice writes persist in the underlying pool; two acquire_slice
        calls for the same (shape, device, dtype) return slices of the
        SAME backing buffer (pointer-stable for CUDA graph safety)."""
        from vllm._genesis.kernels import gdn_core_attn_manager as m
        m.GdnCoreAttnManager.clear_for_tests()
        m._SHOULD_APPLY_CACHED = True
        # Use the _BufferRegistry directly to assert the same buf is
        # returned — this is the invariant callers rely on.
        device = torch.device("cpu")
        dtype = torch.bfloat16
        buf1 = m._BufferRegistry.get_or_create(
            max_num_tokens=256, num_v_heads=4, head_v_dim=8,
            device=device, dtype=dtype,
        )
        buf2 = m._BufferRegistry.get_or_create(
            max_num_tokens=256, num_v_heads=4, head_v_dim=8,
            device=device, dtype=dtype,
        )
        # Pointer-stable: same backing tensor.
        assert buf1 is buf2
        assert buf1.data_ptr() == buf2.data_ptr()

        # Slice via the acquire_slice API; also pointer-stable + zero_.
        s = m.GdnCoreAttnManager.acquire_slice(
            num_tokens=32, num_v_heads=4, head_v_dim=8,
            device="cpu", dtype=dtype,
            num_tokens_max=256,
        )
        s.fill_(1.0)
        s2 = m.GdnCoreAttnManager.acquire_slice(
            num_tokens=32, num_v_heads=4, head_v_dim=8,
            device="cpu", dtype=dtype,
            num_tokens_max=256,
        )
        # Both slices are views of buf1 → same data_ptr, and s2 was
        # zero_()'d, so s reads zeros too.
        assert s2.sum().item() == 0
        assert s.sum().item() == 0

    def test_overflow_falls_back_to_fresh_alloc(
        self, monkeypatch, reset_genesis_prealloc,
    ):
        """num_tokens > max → correctness fallback, no crash."""
        from vllm._genesis.kernels import gdn_core_attn_manager as m
        # Post-redesign: should_apply reads module-level cache via
        # `should_apply()` function, which classmethod delegates to.
        m._SHOULD_APPLY_CACHED = True
        s = m.GdnCoreAttnManager.acquire_slice(
            num_tokens=10000, num_v_heads=4, head_v_dim=8,
            device="cpu", dtype=torch.bfloat16,
            num_tokens_max=256,
        )
        assert s.shape == (10000, 4, 8)
        assert s.sum().item() == 0
        # Not pointer-stable (fresh alloc) — that's the trade-off.
        s2 = m.GdnCoreAttnManager.acquire_slice(
            num_tokens=10000, num_v_heads=4, head_v_dim=8,
            device="cpu", dtype=torch.bfloat16,
            num_tokens_max=256,
        )
        # Each overflow path gives a fresh tensor (not cached)
        assert s.data_ptr() != s2.data_ptr()

    def test_incompatible_platform_falls_back(
        self, monkeypatch, reset_genesis_prealloc,
    ):
        """On CPU-only (should_apply False), returns fresh zeros."""
        from vllm._genesis.kernels import gdn_core_attn_manager as m
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: False)
        s = m.GdnCoreAttnManager.acquire_slice(
            num_tokens=64, num_v_heads=4, head_v_dim=8,
            device="cpu", dtype=torch.bfloat16,
            num_tokens_max=256,
        )
        assert s.shape == (64, 4, 8)
        assert s.dtype == torch.bfloat16
        assert s.sum().item() == 0

    def test_env_override_max(
        self, monkeypatch, reset_genesis_prealloc,
    ):
        """GENESIS_GDN_MAX_BATCHED_TOKENS raises the budget."""
        from vllm._genesis.kernels import gdn_core_attn_manager as m
        m._reset_pin_for_tests()
        # Post-redesign: should_apply reads module-level cache via
        # `should_apply()` function, which classmethod delegates to.
        m._SHOULD_APPLY_CACHED = True
        monkeypatch.setenv("GENESIS_GDN_MAX_BATCHED_TOKENS", "8192")
        # Ask for acquire with no explicit max_hint
        s = m.GdnCoreAttnManager.acquire_slice(
            num_tokens=6000, num_v_heads=4, head_v_dim=8,
            device="cpu", dtype=torch.bfloat16,
        )
        assert s.shape == (6000, 4, 8)

    def test_env_invalid_falls_to_default(
        self, monkeypatch, reset_genesis_prealloc,
    ):
        from vllm._genesis.kernels import gdn_core_attn_manager as m
        # Post-redesign: should_apply reads module-level cache via
        # `should_apply()` function, which classmethod delegates to.
        m._SHOULD_APPLY_CACHED = True
        monkeypatch.setenv("GENESIS_GDN_MAX_BATCHED_TOKENS", "abc")
        # Default is 4096; 3000 fits
        s = m.GdnCoreAttnManager.acquire_slice(
            num_tokens=3000, num_v_heads=4, head_v_dim=8,
            device="cpu", dtype=torch.bfloat16,
        )
        assert s.shape == (3000, 4, 8)


class TestBudgetResolution:
    """Post-redesign: `_ENV_BUDGET` is resolved at module import time to be
    dynamo-safe (no os.environ inside traced regions). `resolve_max_batched_tokens`
    is a pure function consulted at `__init__` time.
    """

    def test_hint_wins_over_env(self, monkeypatch):
        from vllm._genesis.kernels import gdn_core_attn_manager as m
        # Simulate env was "8192" at import time
        m._ENV_BUDGET = 8192
        assert m.resolve_max_batched_tokens(hint=2048) == 2048

    def test_env_wins_over_default(self, monkeypatch):
        from vllm._genesis.kernels import gdn_core_attn_manager as m
        m._ENV_BUDGET = 8192
        assert m.resolve_max_batched_tokens() == 8192

    def test_default_when_no_env(self, monkeypatch):
        from vllm._genesis.kernels import gdn_core_attn_manager as m
        m._ENV_BUDGET = None
        assert m.resolve_max_batched_tokens() == 4096

    def test_env_read_once_at_module_import(self, monkeypatch):
        """Changing env at runtime does NOT affect cached `_ENV_BUDGET` —
        dynamo-safety requirement."""
        from vllm._genesis.kernels import gdn_core_attn_manager as m
        m._ENV_BUDGET = 4096
        monkeypatch.setenv("GENESIS_GDN_MAX_BATCHED_TOKENS", "99999")
        # resolve_max_batched_tokens does NOT re-read env — it consults
        # the cached _ENV_BUDGET module global.
        assert m.resolve_max_batched_tokens() == 4096


class TestRegistryInfo:
    def test_empty_registry(self, reset_genesis_prealloc):
        from vllm._genesis.kernels.gdn_core_attn_manager import (
            GdnCoreAttnManager,
        )
        GdnCoreAttnManager.clear_for_tests()
        info = GdnCoreAttnManager.get_registry_info()
        assert info["entries"] == []
        assert info["total_bytes"] == 0

    def test_registry_reflects_allocations(
        self, monkeypatch, reset_genesis_prealloc,
    ):
        from vllm._genesis.kernels import gdn_core_attn_manager as m
        m.GdnCoreAttnManager.clear_for_tests()
        m._SHOULD_APPLY_CACHED = True
        m.GdnCoreAttnManager.get_or_create(
            num_tokens_max=512, num_v_heads=4, head_v_dim=8,
            device="cpu", dtype=torch.bfloat16,
        )
        info = m.GdnCoreAttnManager.get_registry_info()
        assert len(info["entries"]) == 1
        assert info["entries"][0]["max_num_tokens"] == 512
        assert info["total_bytes"] > 0
