# SPDX-License-Identifier: Apache-2.0
"""TDD tests for vllm._genesis.kernels.tq_continuation_prefill (Patch 20).

Validates the FP16 rotation path, Pi_half caching, redundant-contiguous
removal, and platform guard.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def _clear_cache():
    """Each test starts with a clean Pi_half cache."""
    from vllm._genesis.kernels.tq_continuation_prefill import clear_pi_half_cache
    clear_pi_half_cache()
    yield
    clear_pi_half_cache()


class TestGetPiHalf:
    """Group 1: Pi_half caching behavior."""

    def test_fp32_gets_downcasted(self):
        from vllm._genesis.kernels.tq_continuation_prefill import get_pi_half

        Pi = torch.randn(8, 8, dtype=torch.float32)
        half = get_pi_half(Pi)
        assert half.dtype == torch.float16
        assert half.shape == Pi.shape

    def test_fp16_input_returned_unchanged(self):
        """No copy if already fp16 — data_ptr must match."""
        from vllm._genesis.kernels.tq_continuation_prefill import get_pi_half

        Pi = torch.randn(8, 8, dtype=torch.float16)
        half = get_pi_half(Pi)
        assert half.data_ptr() == Pi.data_ptr()

    def test_repeat_call_cached(self):
        """Same Pi → same underlying fp16 tensor (cached)."""
        from vllm._genesis.kernels.tq_continuation_prefill import get_pi_half

        Pi = torch.randn(16, 16, dtype=torch.float32)
        a = get_pi_half(Pi)
        b = get_pi_half(Pi)
        assert a.data_ptr() == b.data_ptr()

    def test_different_pi_different_cache(self):
        from vllm._genesis.kernels.tq_continuation_prefill import get_pi_half

        Pi1 = torch.randn(8, 8, dtype=torch.float32)
        Pi2 = torch.randn(8, 8, dtype=torch.float32)
        a = get_pi_half(Pi1)
        b = get_pi_half(Pi2)
        # Different source → different cached fp16 tensors
        assert a.data_ptr() != b.data_ptr()

    def test_bfloat16_also_downcasted_to_fp16(self):
        """bf16 input is not fp16 — must be cast."""
        from vllm._genesis.kernels.tq_continuation_prefill import get_pi_half

        Pi = torch.randn(8, 8, dtype=torch.bfloat16)
        half = get_pi_half(Pi)
        assert half.dtype == torch.float16


class TestContinuationPrefillFp16Rotate:
    """Group 2: FP16 rotation math correctness."""

    def test_shape_matches_upstream_expectation(self):
        from vllm._genesis.kernels.tq_continuation_prefill import (
            continuation_prefill_fp16_rotate,
        )

        Hk, D, cached_len = 2, 16, 128
        k_cached = torch.randn(1, Hk, 256, D, dtype=torch.float16)
        Pi = torch.randn(D, D, dtype=torch.float16)

        out = continuation_prefill_fp16_rotate(k_cached, Pi, Hk, D, cached_len)
        assert out.shape == (cached_len, Hk, D)
        assert out.dtype == torch.float16

    def test_dtype_never_fp32(self):
        """Critical: output must never become fp32 — that was the OOM cause."""
        from vllm._genesis.kernels.tq_continuation_prefill import (
            continuation_prefill_fp16_rotate,
        )

        Hk, D, cached_len = 2, 16, 64
        k_cached = torch.randn(1, Hk, 128, D, dtype=torch.float16)
        Pi_fp32 = torch.randn(D, D, dtype=torch.float32)

        out = continuation_prefill_fp16_rotate(k_cached, Pi_fp32, Hk, D, cached_len)
        # Even though Pi was fp32, rotation must produce fp16
        assert out.dtype == torch.float16

    def test_matches_equivalent_upstream_math(self):
        """Numerical equivalence to `k.float() @ Pi.float()` within fp16 precision."""
        from vllm._genesis.kernels.tq_continuation_prefill import (
            continuation_prefill_fp16_rotate,
        )

        Hk, D, cached_len = 2, 16, 32
        torch.manual_seed(0)
        k_cached = torch.randn(1, Hk, 64, D, dtype=torch.float16)
        Pi = torch.randn(D, D, dtype=torch.float16)

        # Reference: the fp16 path we WANT
        k_flat_ref = k_cached[0, :, :cached_len, :].reshape(-1, D)
        k_flat_ref = k_flat_ref @ Pi
        ref = k_flat_ref.reshape(Hk, cached_len, D).transpose(0, 1)

        out = continuation_prefill_fp16_rotate(k_cached, Pi, Hk, D, cached_len)
        assert torch.equal(out, ref)

    def test_handles_cached_len_equals_max(self):
        from vllm._genesis.kernels.tq_continuation_prefill import (
            continuation_prefill_fp16_rotate,
        )

        Hk, D, max_len = 2, 8, 64
        k_cached = torch.randn(1, Hk, max_len, D, dtype=torch.float16)
        Pi = torch.randn(D, D, dtype=torch.float16)

        out = continuation_prefill_fp16_rotate(k_cached, Pi, Hk, D, max_len)
        assert out.shape == (max_len, Hk, D)

    def test_handles_cached_len_1(self):
        from vllm._genesis.kernels.tq_continuation_prefill import (
            continuation_prefill_fp16_rotate,
        )

        Hk, D = 2, 8
        k_cached = torch.randn(1, Hk, 16, D, dtype=torch.float16)
        Pi = torch.randn(D, D, dtype=torch.float16)

        out = continuation_prefill_fp16_rotate(k_cached, Pi, Hk, D, 1)
        assert out.shape == (1, Hk, D)


class TestFp8BranchViews:
    """Group 3: FP8 branch view helpers (no rotation, no .contiguous())."""

    def test_k_view_fp8_is_non_contiguous(self):
        """Upstream's .contiguous() was redundant. Our view must be a view."""
        from vllm._genesis.kernels.tq_continuation_prefill import (
            continuation_prefill_k_view_fp8,
        )

        k_cached = torch.randn(1, 2, 32, 16, dtype=torch.float16)
        view = continuation_prefill_k_view_fp8(k_cached, cached_len=16)
        assert view.shape == (16, 2, 16)
        # transpose(0, 1) on non-leading-index-slice → non-contiguous
        assert not view.is_contiguous()

    def test_v_view_is_non_contiguous(self):
        from vllm._genesis.kernels.tq_continuation_prefill import (
            continuation_prefill_v_view,
        )

        v_cached = torch.randn(1, 2, 32, 16, dtype=torch.float16)
        view = continuation_prefill_v_view(v_cached, cached_len=20)
        assert view.shape == (20, 2, 16)
        assert not view.is_contiguous()

    def test_views_share_storage_with_source(self):
        """Views must NOT allocate new memory — that was the spike."""
        from vllm._genesis.kernels.tq_continuation_prefill import (
            continuation_prefill_k_view_fp8,
            continuation_prefill_v_view,
        )

        k_cached = torch.randn(1, 2, 32, 16, dtype=torch.float16)
        v_cached = torch.randn(1, 2, 32, 16, dtype=torch.float16)

        kv = continuation_prefill_k_view_fp8(k_cached, 16)
        vv = continuation_prefill_v_view(v_cached, 16)

        assert kv.data_ptr() >= k_cached.data_ptr()
        assert kv.data_ptr() < k_cached.data_ptr() + k_cached.element_size() * k_cached.numel()
        assert vv.data_ptr() >= v_cached.data_ptr()


class TestShouldApplyPlatformGuard:
    """Group 4: Platform guard — NVIDIA CUDA + SM 8.0+ only."""

    def test_false_on_non_nvidia(self, monkeypatch):
        from vllm._genesis.kernels import tq_continuation_prefill as t
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: False)
        assert t.should_apply() is False

    def test_false_on_pre_ampere(self, monkeypatch):
        from vllm._genesis.kernels import tq_continuation_prefill as t
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: False)
        assert t.should_apply() is False

    def test_true_on_ampere_plus(self, monkeypatch):
        from vllm._genesis.kernels import tq_continuation_prefill as t
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: True)
        assert t.should_apply() is True


class TestGetCacheInfo:
    """Group 5: Diagnostic helper."""

    def test_empty_cache_reports_zero(self):
        from vllm._genesis.kernels.tq_continuation_prefill import get_cache_info

        info = get_cache_info()
        assert info["num_entries"] == 0
        assert info["total_bytes"] == 0
        assert info["entries"] == []

    def test_populated_cache_reports_entries(self):
        from vllm._genesis.kernels.tq_continuation_prefill import (
            get_pi_half, get_cache_info,
        )

        Pi = torch.randn(16, 16, dtype=torch.float32)
        _ = get_pi_half(Pi)

        info = get_cache_info()
        assert info["num_entries"] == 1
        assert info["total_bytes"] > 0
        assert info["entries"][0]["shape"] == [16, 16]
        assert info["entries"][0]["dtype"] == "torch.float16"
