# SPDX-License-Identifier: Apache-2.0
"""TDD tests for P46 — GDN gating buffer pool (`fused_gdn_gating`
`g` / `beta_output` persistent buffers).

Covers:
- Pool-hit: same shape-key → same tensor (identity `is`)
- Pool-miss: new shape-key → new tensor; OLD tensor stays alive
  (important for CUDA-graph safety on workload transitions)
- Platform guard: non-NVIDIA falls back to fresh `torch.empty`
- `should_apply()` respects mocked guards
- Shape + dtype correctness for both g and beta
- Different dtypes yield different entries
- Registry info aggregates correctly
- `clear_for_tests` drops both pools
- Wiring surface: apply/is_applied/revert

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def _reset_manager():
    from vllm._genesis.kernels.gdn_gating_buffer import GdnGatingBufferManager
    GdnGatingBufferManager.clear_for_tests()
    yield
    GdnGatingBufferManager.clear_for_tests()


@pytest.fixture
def cuda_guard(monkeypatch):
    from vllm._genesis import guards
    monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
    monkeypatch.setattr(
        guards, "is_sm_at_least", lambda major, minor=0: True,
    )
    return True


class TestP46PoolHit:
    def test_same_key_returns_same_tensor(self, cuda_guard):
        from vllm._genesis.kernels.gdn_gating_buffer import (
            GdnGatingBufferManager,
        )
        t1 = GdnGatingBufferManager.acquire_g(
            batch=1, num_heads=16,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        t2 = GdnGatingBufferManager.acquire_g(
            batch=1, num_heads=16,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        assert t1 is t2, "pool-hit must return identical tensor"

    def test_beta_pool_hit(self, cuda_guard):
        from vllm._genesis.kernels.gdn_gating_buffer import (
            GdnGatingBufferManager,
        )
        t1 = GdnGatingBufferManager.acquire_beta(
            batch=1, num_heads=16,
            device=torch.device("cpu"), dtype=torch.float16,
        )
        t2 = GdnGatingBufferManager.acquire_beta(
            batch=1, num_heads=16,
            device=torch.device("cpu"), dtype=torch.float16,
        )
        assert t1 is t2


class TestP46PoolMiss:
    def test_new_batch_creates_new_tensor(self, cuda_guard):
        from vllm._genesis.kernels.gdn_gating_buffer import (
            GdnGatingBufferManager,
        )
        t1 = GdnGatingBufferManager.acquire_g(
            batch=1, num_heads=16,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        t2 = GdnGatingBufferManager.acquire_g(
            batch=4, num_heads=16,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        assert t1 is not t2
        assert t1.shape == (1, 1, 16)
        assert t2.shape == (1, 4, 16)

    def test_old_tensor_survives_after_new_key(self, cuda_guard):
        """CUDA-graph safety: allocating under new key must NOT
        invalidate previously-returned tensor."""
        from vllm._genesis.kernels.gdn_gating_buffer import (
            GdnGatingBufferManager,
        )
        t_old = GdnGatingBufferManager.acquire_g(
            batch=1, num_heads=16,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        old_ptr = t_old.data_ptr()
        # Allocate under DIFFERENT key
        GdnGatingBufferManager.acquire_g(
            batch=16, num_heads=16,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        # Original still live and has SAME pointer
        t_still = GdnGatingBufferManager.acquire_g(
            batch=1, num_heads=16,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        assert t_still.data_ptr() == old_ptr
        assert t_still is t_old

    def test_different_dtypes_different_pools(self, cuda_guard):
        from vllm._genesis.kernels.gdn_gating_buffer import (
            GdnGatingBufferManager,
        )
        t_f16 = GdnGatingBufferManager.acquire_beta(
            batch=1, num_heads=16,
            device=torch.device("cpu"), dtype=torch.float16,
        )
        t_bf16 = GdnGatingBufferManager.acquire_beta(
            batch=1, num_heads=16,
            device=torch.device("cpu"), dtype=torch.bfloat16,
        )
        assert t_f16 is not t_bf16
        assert t_f16.dtype == torch.float16
        assert t_bf16.dtype == torch.bfloat16


class TestP46PlatformGuard:
    def test_non_nvidia_returns_fresh(self, monkeypatch):
        from vllm._genesis.kernels.gdn_gating_buffer import (
            GdnGatingBufferManager,
        )
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: False)
        t = GdnGatingBufferManager.acquire_g(
            batch=1, num_heads=16,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        assert t.shape == (1, 1, 16)
        # No pool created — platform skip path returns fresh each time
        t2 = GdnGatingBufferManager.acquire_g(
            batch=1, num_heads=16,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        assert t is not t2

    def test_should_apply(self, monkeypatch):
        from vllm._genesis.kernels.gdn_gating_buffer import (
            GdnGatingBufferManager,
        )
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(
            guards, "is_sm_at_least", lambda major, minor=0: True,
        )
        assert GdnGatingBufferManager.should_apply() is True
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: False)
        assert GdnGatingBufferManager.should_apply() is False


class TestP46Registry:
    def test_registry_aggregates_both_pools(self, cuda_guard):
        from vllm._genesis.kernels.gdn_gating_buffer import (
            GdnGatingBufferManager,
        )
        GdnGatingBufferManager.acquire_g(
            batch=1, num_heads=16,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        GdnGatingBufferManager.acquire_beta(
            batch=1, num_heads=16,
            device=torch.device("cpu"), dtype=torch.float16,
        )
        info = GdnGatingBufferManager.get_registry_info()
        assert info["num_g_pools"] == 1
        assert info["num_beta_pools"] == 1
        # g=(1,1,16) fp32=4B: 64 bytes; beta=(1,1,16) fp16=2B: 32 bytes
        assert info["total_bytes"] == 96

    def test_clear_drops_both(self, cuda_guard):
        from vllm._genesis.kernels.gdn_gating_buffer import (
            GdnGatingBufferManager,
        )
        GdnGatingBufferManager.acquire_g(
            batch=1, num_heads=16,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        GdnGatingBufferManager.acquire_beta(
            batch=1, num_heads=16,
            device=torch.device("cpu"), dtype=torch.float16,
        )
        GdnGatingBufferManager.clear_for_tests()
        assert len(GdnGatingBufferManager._G_POOLS) == 0
        assert len(GdnGatingBufferManager._BETA_POOLS) == 0


class TestP46Wiring:
    def test_wiring_public_surface(self):
        from vllm._genesis.wiring.legacy import patch_46_gdn_gating_buffers as p46
        assert callable(p46.apply)
        assert callable(p46.is_applied)
        assert callable(p46.revert)
        assert callable(p46.should_apply)

    def test_apply_skips_on_non_nvidia(self, monkeypatch):
        from vllm._genesis.wiring.legacy import patch_46_gdn_gating_buffers as p46
        monkeypatch.setattr(p46, "is_nvidia_cuda", lambda: False)
        status, reason = p46.apply()
        assert status == "skipped"
        assert "non-NVIDIA" in reason or "CUDA" in reason

    def test_revert_always_false(self):
        from vllm._genesis.wiring.legacy import patch_46_gdn_gating_buffers as p46
        assert p46.revert() is False


class TestP44MixedAttnOut:
    """P44 — TQ mixed-batch attn_out pool (extension of P26 infrastructure)."""

    def test_acquire_returns_correct_shape(self, monkeypatch):
        from vllm._genesis.kernels.dequant_buffer import TurboQuantBufferManager
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(
            guards, "is_sm_at_least", lambda major, minor=0: True,
        )
        t = TurboQuantBufferManager.acquire_mixed_attn_out(
            num_tokens=128, num_q_heads=32, head_size=128,
            device=torch.device("cpu"), dtype=torch.float16,
            max_batched_tokens=4096,
        )
        assert t.shape == (128, 32, 128)
        assert t.dtype == torch.float16
        # Pool view is zeroed
        assert torch.all(t == 0.0)

    def test_acquire_overflow_falls_back_fresh(self, monkeypatch):
        from vllm._genesis.kernels.dequant_buffer import TurboQuantBufferManager
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(
            guards, "is_sm_at_least", lambda major, minor=0: True,
        )
        # Request MORE than max_batched_tokens → fallback to fresh alloc
        t = TurboQuantBufferManager.acquire_mixed_attn_out(
            num_tokens=10000, num_q_heads=32, head_size=128,
            device=torch.device("cpu"), dtype=torch.float16,
            max_batched_tokens=4096,
        )
        assert t.shape == (10000, 32, 128)

    def test_pool_reuses_buffer_under_budget(self, monkeypatch):
        from vllm._genesis.kernels.dequant_buffer import TurboQuantBufferManager
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(
            guards, "is_sm_at_least", lambda major, minor=0: True,
        )
        TurboQuantBufferManager.clear_for_tests()
        t1 = TurboQuantBufferManager.acquire_mixed_attn_out(
            num_tokens=100, num_q_heads=32, head_size=128,
            device=torch.device("cpu"), dtype=torch.float16,
            max_batched_tokens=4096,
        )
        t2 = TurboQuantBufferManager.acquire_mixed_attn_out(
            num_tokens=200, num_q_heads=32, head_size=128,
            device=torch.device("cpu"), dtype=torch.float16,
            max_batched_tokens=4096,
        )
        # Both views share same underlying pool → same data_ptr (storage)
        assert t1.storage().data_ptr() == t2.storage().data_ptr()

    def test_platform_skip_returns_fresh(self, monkeypatch):
        from vllm._genesis.kernels.dequant_buffer import TurboQuantBufferManager
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: False)
        t = TurboQuantBufferManager.acquire_mixed_attn_out(
            num_tokens=64, num_q_heads=8, head_size=64,
            device=torch.device("cpu"), dtype=torch.float16,
            max_batched_tokens=4096,
        )
        assert t.shape == (64, 8, 64)


class TestP44Wiring:
    def test_public_surface(self):
        from vllm._genesis.wiring.legacy import patch_44_tq_mixed_attn_out as p44
        assert callable(p44.apply)
        assert callable(p44.is_applied)
        assert callable(p44.revert)

    def test_apply_skips_on_non_nvidia(self, monkeypatch):
        from vllm._genesis.wiring.legacy import patch_44_tq_mixed_attn_out as p44
        monkeypatch.setattr(p44, "is_nvidia_cuda", lambda: False)
        status, _reason = p44.apply()
        assert status == "skipped"
