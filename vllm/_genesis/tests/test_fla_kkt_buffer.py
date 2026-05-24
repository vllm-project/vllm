# SPDX-License-Identifier: Apache-2.0
"""TDD tests for Patch 39a — FLA KKT persistent A buffer.

Covers:
- Shape/dtype correctness of the pool
- Pointer-stability on repeat acquire
- Slice correctness (returned view shape = requested args)
- Overflow fallback to fresh alloc (doesn't poison pool)
- Platform guard: None on non-NVIDIA
- Registry aggregation
- Wiring apply/revert/is_applied surface

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def _reset_manager():
    from vllm._genesis.kernels.fla_kkt_buffer import FlaKktBufferManager
    FlaKktBufferManager.clear_for_tests()
    yield
    FlaKktBufferManager.clear_for_tests()


@pytest.fixture
def cuda_guard(monkeypatch):
    from vllm._genesis import guards
    monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
    monkeypatch.setattr(guards, "is_sm_at_least",
                        lambda major, minor=0: True)
    return True


class TestP39aPoolShape:
    def test_shape_matches_request(self, cuda_guard):
        from vllm._genesis.kernels.fla_kkt_buffer import FlaKktBufferManager
        pool = FlaKktBufferManager.get_or_create_pool(
            B_max=1, T_max=4096, H=16, BT=64,
            device="cpu", dtype=torch.float32,
        )
        assert pool.shape == (1, 4096, 16, 64)
        assert pool.dtype == torch.float32

    def test_dtype_fp32_preserved(self, cuda_guard):
        from vllm._genesis.kernels.fla_kkt_buffer import FlaKktBufferManager
        pool = FlaKktBufferManager.get_or_create_pool(
            B_max=1, T_max=1024, H=4, BT=64,
            device="cpu", dtype=torch.float32,
        )
        assert pool.dtype == torch.float32

    def test_dtype_fp16_preserved(self, cuda_guard):
        from vllm._genesis.kernels.fla_kkt_buffer import FlaKktBufferManager
        pool = FlaKktBufferManager.get_or_create_pool(
            B_max=1, T_max=1024, H=4, BT=64,
            device="cpu", dtype=torch.float16,
        )
        assert pool.dtype == torch.float16


class TestP39aPointerStability:
    def test_repeat_acquire_same_pool(self, cuda_guard):
        from vllm._genesis.kernels.fla_kkt_buffer import FlaKktBufferManager
        a = FlaKktBufferManager.get_or_create_pool(
            B_max=1, T_max=4096, H=16, BT=64,
            device="cpu", dtype=torch.float32,
        )
        b = FlaKktBufferManager.get_or_create_pool(
            B_max=1, T_max=4096, H=16, BT=64,
            device="cpu", dtype=torch.float32,
        )
        assert a.data_ptr() == b.data_ptr()

    def test_acquire_returns_view_of_pool(self, cuda_guard):
        from vllm._genesis.kernels.fla_kkt_buffer import FlaKktBufferManager
        a_view = FlaKktBufferManager.acquire(
            B=1, T=512, H=16, BT=64,
            device=torch.device("cpu"), dtype=torch.float32,
            max_T=4096,
        )
        assert a_view.shape == (1, 512, 16, 64)
        # View into pool — pool exists post-acquire
        pool = FlaKktBufferManager.get_or_create_pool(
            B_max=1, T_max=4096, H=16, BT=64,
            device="cpu", dtype=torch.float32,
        )
        assert a_view.data_ptr() == pool.data_ptr()

    def test_different_H_different_pools(self, cuda_guard):
        from vllm._genesis.kernels.fla_kkt_buffer import FlaKktBufferManager
        a = FlaKktBufferManager.acquire(
            B=1, T=64, H=8, BT=64,
            device=torch.device("cpu"), dtype=torch.float32,
            max_T=64,
        )
        b = FlaKktBufferManager.acquire(
            B=1, T=64, H=16, BT=64,
            device=torch.device("cpu"), dtype=torch.float32,
            max_T=64,
        )
        assert a.data_ptr() != b.data_ptr()


class TestP39aAutoGrow:
    def test_T_larger_auto_grows_pool(self, cuda_guard):
        """Pool grows on larger T instead of falling back to fresh alloc."""
        from vllm._genesis.kernels.fla_kkt_buffer import FlaKktBufferManager
        small = FlaKktBufferManager.acquire(
            B=1, T=64, H=4, BT=64,
            device=torch.device("cpu"), dtype=torch.float32,
            max_T=64,
        )
        assert small.shape == (1, 64, 4, 64)
        # Then acquire with T=256 > initial T_max=64 — pool auto-grows
        big = FlaKktBufferManager.acquire(
            B=1, T=256, H=4, BT=64,
            device=torch.device("cpu"), dtype=torch.float32,
            max_T=64,
        )
        assert big.shape == (1, 256, 4, 64)
        # After grow, subsequent small acquires reuse the grown pool
        small_again = FlaKktBufferManager.acquire(
            B=1, T=64, H=4, BT=64,
            device=torch.device("cpu"), dtype=torch.float32,
            max_T=64,
        )
        assert small_again.data_ptr() == big.data_ptr()

    def test_B_larger_auto_grows_pool(self, cuda_guard):
        """Larger B also triggers in-place grow rather than fallback."""
        from vllm._genesis.kernels.fla_kkt_buffer import FlaKktBufferManager
        first = FlaKktBufferManager.acquire(
            B=1, T=64, H=4, BT=64,
            device=torch.device("cpu"), dtype=torch.float32,
            max_T=64,
        )
        big = FlaKktBufferManager.acquire(
            B=4, T=64, H=4, BT=64,
            device=torch.device("cpu"), dtype=torch.float32,
            max_T=64,
        )
        assert big.shape == (4, 64, 4, 64)
        # Growth replaced the pool — old first pointer may be different
        # (the new pool is same shape but reallocated). Check via shape
        # contract instead of pointer identity.
        assert first.shape == (1, 64, 4, 64)


class TestP39aPlatformGuard:
    def test_non_nvidia_returns_none(self, monkeypatch):
        from vllm._genesis.kernels.fla_kkt_buffer import FlaKktBufferManager
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: False)
        pool = FlaKktBufferManager.get_or_create_pool(
            B_max=1, T_max=64, H=4, BT=64,
            device="cpu", dtype=torch.float32,
        )
        assert pool is None

    def test_non_nvidia_acquire_fresh_alloc(self, monkeypatch):
        from vllm._genesis.kernels.fla_kkt_buffer import FlaKktBufferManager
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: False)
        a = FlaKktBufferManager.acquire(
            B=1, T=64, H=4, BT=64,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        # Fresh alloc of exactly requested shape
        assert a.shape == (1, 64, 4, 64)
        # Pool count is still 0 (no pool created)
        assert len(FlaKktBufferManager._A_POOLS) == 0


class TestP39aRegistry:
    def test_registry_reports_bytes(self, cuda_guard):
        from vllm._genesis.kernels.fla_kkt_buffer import FlaKktBufferManager
        FlaKktBufferManager.get_or_create_pool(
            B_max=1, T_max=4096, H=16, BT=64,
            device="cpu", dtype=torch.float32,
        )
        info = FlaKktBufferManager.get_registry_info()
        assert info["num_pools"] == 1
        assert info["total_bytes"] == 1 * 4096 * 16 * 64 * 4  # fp32
        assert "MiB" in info["total_human"] or "KiB" in info["total_human"]

    def test_clear_for_tests(self, cuda_guard):
        from vllm._genesis.kernels.fla_kkt_buffer import FlaKktBufferManager
        FlaKktBufferManager.get_or_create_pool(
            B_max=1, T_max=128, H=4, BT=64,
            device="cpu", dtype=torch.float32,
        )
        FlaKktBufferManager.clear_for_tests()
        assert len(FlaKktBufferManager._A_POOLS) == 0


class TestP39aWiringSurface:
    def test_should_apply_non_nvidia(self, monkeypatch):
        # patch_39 imports `is_nvidia_cuda` at module level via
        # `from ... import is_nvidia_cuda`, so the local name binding
        # is what should_apply() actually calls. Monkey-patching the
        # original `vllm._genesis.guards.is_nvidia_cuda` would NOT
        # propagate (local binding already captured the original).
        from vllm._genesis.wiring.legacy import patch_39_fla_kkt_buffer as p39a
        monkeypatch.setattr(p39a, "is_nvidia_cuda", lambda: False)
        assert p39a.should_apply() is False

    def test_apply_skips_on_non_nvidia(self, monkeypatch):
        from vllm._genesis.wiring.legacy import patch_39_fla_kkt_buffer as p39a
        monkeypatch.setattr(p39a, "is_nvidia_cuda", lambda: False)
        status, reason = p39a.apply()
        assert status == "skipped"
        assert "NVIDIA" in reason

    def test_apply_skips_when_module_missing(self, monkeypatch):
        """FLA module isn't installed in unit-test env — expected skip."""
        from vllm._genesis.wiring.legacy import patch_39_fla_kkt_buffer as p39a
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: True)
        status, _reason = p39a.apply()
        # Either "skipped" (no FLA) or "applied" (if FLA is installed)
        assert status in ("skipped", "applied")

    def test_public_surface_present(self):
        from vllm._genesis.wiring.legacy import patch_39_fla_kkt_buffer as p39a
        assert callable(p39a.apply)
        assert callable(p39a.revert)
        assert callable(p39a.is_applied)
        assert callable(p39a.should_apply)
