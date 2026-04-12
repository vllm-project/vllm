# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for dynamic Triton/CK dispatch multiplexing on ROCm.

Tests the multiplexing logic only — underlying GEMM correctness is
covered by test_block_fp8.py, test_rocm_skinny_gemms.py, etc.

Verifies that:
1. ``can_implement`` guards reject unsupported configurations (CPU-safe).
2. The fake/meta impl returns correct shapes for ``torch.compile`` tracing.
3. The real dispatch routes to Triton (small M) or CK (large M) on ROCm GPU.
4. The dispatch wrapper passes tensors through without corruption (bitwise).
5. The custom op registration is consistent (opcheck).

Run ``pytest tests/kernels/quantization/test_dynamic_aiter_triton_ck.py``.
"""

import importlib.util
from unittest.mock import MagicMock, patch

import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
    AiterFp8BlockScaledDynamicMMKernel,
    _dynamic_aiter_triton_ck_blockscale_gemm_fake,
    _dynamic_aiter_triton_ck_blockscale_gemm_impl,
)
from vllm.platforms import current_platform

aiter_available = importlib.util.find_spec("aiter") is not None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inputs(M: int, N: int, K: int, device: str = "cpu"):
    """Create synthetic tensors for testing dispatch logic."""
    A = torch.randn(M, K, device=device)
    B = torch.randn(N, K, device=device)
    As = torch.randn(M, K // 128, device=device)
    Bs = torch.randn(N // 128, K // 128, device=device)
    return A, B, As, Bs


# =========================================================================
# Section 1: CPU-safe tests (can_implement guards + fake impl)
# =========================================================================


class TestCanImplementGuards:
    """AiterFp8BlockScaledDynamicMMKernel.can_implement rejects bad configs.

    Pure logic tests — no GPU required.
    """

    pytestmark = pytest.mark.cpu_test

    @staticmethod
    def _make_config(n=7168, k=2048, group_shape=(1, 128)):
        config = MagicMock()
        config.weight_shape = (n, k)
        config.activation_quant_key.scale.group_shape = group_shape
        return config

    def test_rejects_non_tuned_shape(self):
        config = self._make_config(n=9999, k=9999)

        with patch("vllm.model_executor.kernels.linear.scaled_mm.aiter"
                   ".rocm_aiter_ops") as mock_aiter, \
             patch("vllm.model_executor.kernels.linear.scaled_mm.aiter"
                   ".current_platform") as mock_platform, \
             patch.object(
                 AiterFp8BlockScaledDynamicMMKernel.__mro__[1],
                 "can_implement", return_value=(True, None)):
            mock_aiter.is_triton_gemm_w8a8_tuned.return_value = False
            mock_platform.is_fp8_fnuz.return_value = False

            result, reason = (
                AiterFp8BlockScaledDynamicMMKernel.can_implement(config)
            )
        assert result is False
        assert "No tuned Triton config" in reason

    def test_rejects_fnuz_fp8(self):
        config = self._make_config()

        with patch("vllm.model_executor.kernels.linear.scaled_mm.aiter"
                   ".rocm_aiter_ops") as mock_aiter, \
             patch("vllm.model_executor.kernels.linear.scaled_mm.aiter"
                   ".current_platform") as mock_platform, \
             patch.object(
                 AiterFp8BlockScaledDynamicMMKernel.__mro__[1],
                 "can_implement", return_value=(True, None)):
            mock_aiter.is_triton_gemm_w8a8_tuned.return_value = True
            mock_platform.is_fp8_fnuz.return_value = True

            result, reason = (
                AiterFp8BlockScaledDynamicMMKernel.can_implement(config)
            )
        assert result is False
        assert "fnuz" in reason

    def test_rejects_wrong_group_shape(self):
        config = self._make_config(group_shape=(1, 64))

        with patch("vllm.model_executor.kernels.linear.scaled_mm.aiter"
                   ".rocm_aiter_ops") as mock_aiter, \
             patch("vllm.model_executor.kernels.linear.scaled_mm.aiter"
                   ".current_platform") as mock_platform, \
             patch.object(
                 AiterFp8BlockScaledDynamicMMKernel.__mro__[1],
                 "can_implement", return_value=(True, None)):
            mock_aiter.is_triton_gemm_w8a8_tuned.return_value = True
            mock_platform.is_fp8_fnuz.return_value = False

            result, reason = (
                AiterFp8BlockScaledDynamicMMKernel.can_implement(config)
            )
        assert result is False
        assert "group_shape" in reason

    def test_accepts_valid_config(self):
        config = self._make_config()

        with patch("vllm.model_executor.kernels.linear.scaled_mm.aiter"
                   ".rocm_aiter_ops") as mock_aiter, \
             patch("vllm.model_executor.kernels.linear.scaled_mm.aiter"
                   ".current_platform") as mock_platform, \
             patch.object(
                 AiterFp8BlockScaledDynamicMMKernel.__mro__[1],
                 "can_implement", return_value=(True, None)):
            mock_aiter.is_triton_gemm_w8a8_tuned.return_value = True
            mock_platform.is_fp8_fnuz.return_value = False

            result, reason = (
                AiterFp8BlockScaledDynamicMMKernel.can_implement(config)
            )
        assert result is True
        assert reason is None


class TestFakeImpl:
    """Fake/meta implementation produces correct output shape and dtype.

    CPU-safe — exercises the tracing stub only.
    """

    pytestmark = pytest.mark.cpu_test

    @pytest.mark.parametrize(
        "M,N,K",
        [(1, 7168, 2048), (32, 4608, 7168), (128, 2112, 7168), (512, 3072, 1536)],
    )
    def test_output_shape(self, M, N, K):
        A, B, As, Bs = _make_inputs(M, N, K)
        out = _dynamic_aiter_triton_ck_blockscale_gemm_fake(
            A, B, As, Bs, torch.bfloat16,
        )
        assert out.shape == (M, N)
        assert out.dtype == torch.bfloat16

    def test_output_dtype_float16(self):
        A, B, As, Bs = _make_inputs(8, 7168, 2048)
        out = _dynamic_aiter_triton_ck_blockscale_gemm_fake(
            A, B, As, Bs, torch.float16,
        )
        assert out.dtype == torch.float16


# =========================================================================
# Section 2: GPU tests (real dispatch on ROCm with aiter)
# =========================================================================

rocm_aiter_skip = pytest.mark.skipif(
    not (current_platform.is_rocm() and aiter_available),
    reason="Dynamic Triton/CK dispatch requires ROCm with aiter installed",
)


def _quantize_blockscale_fp8(
    x: torch.Tensor, group_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a float tensor to FP8 with per-group scales."""
    from vllm._aiter_ops import rocm_aiter_ops  # noqa: F811

    return rocm_aiter_ops.group_fp8_quant(x, group_size)


def _make_gpu_inputs(M: int, N: int, K: int):
    """Create FP8-quantized GEMM inputs on GPU for real kernel tests."""
    A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

    A_fp8, As = _quantize_blockscale_fp8(A_bf16)
    B_fp8, Bs = _quantize_blockscale_fp8(B_bf16)

    return A_fp8, B_fp8, As, Bs, A_bf16, B_bf16


@rocm_aiter_skip
class TestGPUDispatchRouting:
    """Verify the real dispatch routes to the correct backend on ROCm GPU."""

    N, K = 7168, 2048

    @pytest.mark.parametrize("M", [1, 4, 8, 16])
    @torch.inference_mode()
    def test_small_M_calls_triton(self, M):
        A_fp8, B_fp8, As, Bs, _, _ = _make_gpu_inputs(M, self.N, self.K)

        with patch.object(envs, "VLLM_ROCM_W8A8_TRITON_MAX_M", 16), \
             patch.object(envs, "VLLM_BATCH_INVARIANT", False):
            triton_called = False
            ck_called = False

            _orig_triton = torch.ops.vllm.rocm_aiter_triton_gemm_a8w8_blockscale
            _orig_ck = torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale

            def _spy_triton(*args, **kwargs):
                nonlocal triton_called
                triton_called = True
                return _orig_triton(*args, **kwargs)

            def _spy_ck(*args, **kwargs):
                nonlocal ck_called
                ck_called = True
                return _orig_ck(*args, **kwargs)

            with patch("torch.ops.vllm.rocm_aiter_triton_gemm_a8w8_blockscale",
                       side_effect=_spy_triton), \
                 patch("torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale",
                       side_effect=_spy_ck):
                out = _dynamic_aiter_triton_ck_blockscale_gemm_impl(
                    A_fp8, B_fp8, As, Bs, torch.bfloat16,
                )

            assert triton_called, "Expected Triton branch for small M"
            assert not ck_called, "CK branch should not fire for small M"
            assert out.shape == (M, self.N)
            assert out.dtype == torch.bfloat16

    @pytest.mark.parametrize("M", [17, 32, 64, 128])
    @torch.inference_mode()
    def test_large_M_calls_ck(self, M):
        A_fp8, B_fp8, As, Bs, _, _ = _make_gpu_inputs(M, self.N, self.K)

        with patch.object(envs, "VLLM_ROCM_W8A8_TRITON_MAX_M", 16), \
             patch.object(envs, "VLLM_BATCH_INVARIANT", False):
            triton_called = False
            ck_called = False

            _orig_triton = torch.ops.vllm.rocm_aiter_triton_gemm_a8w8_blockscale
            _orig_ck = torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale

            def _spy_triton(*args, **kwargs):
                nonlocal triton_called
                triton_called = True
                return _orig_triton(*args, **kwargs)

            def _spy_ck(*args, **kwargs):
                nonlocal ck_called
                ck_called = True
                return _orig_ck(*args, **kwargs)

            with patch("torch.ops.vllm.rocm_aiter_triton_gemm_a8w8_blockscale",
                       side_effect=_spy_triton), \
                 patch("torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale",
                       side_effect=_spy_ck):
                out = _dynamic_aiter_triton_ck_blockscale_gemm_impl(
                    A_fp8, B_fp8, As, Bs, torch.bfloat16,
                )

            assert ck_called, "Expected CK branch for large M"
            assert not triton_called, "Triton branch should not fire for large M"
            assert out.shape == (M, self.N)
            assert out.dtype == torch.bfloat16

    @pytest.mark.parametrize("M", [1, 16, 64])
    @torch.inference_mode()
    def test_batch_invariant_always_ck(self, M):
        A_fp8, B_fp8, As, Bs, _, _ = _make_gpu_inputs(M, self.N, self.K)

        with patch.object(envs, "VLLM_ROCM_W8A8_TRITON_MAX_M", 16), \
             patch.object(envs, "VLLM_BATCH_INVARIANT", True):
            triton_called = False
            ck_called = False

            _orig_triton = torch.ops.vllm.rocm_aiter_triton_gemm_a8w8_blockscale
            _orig_ck = torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale

            def _spy_triton(*args, **kwargs):
                nonlocal triton_called
                triton_called = True
                return _orig_triton(*args, **kwargs)

            def _spy_ck(*args, **kwargs):
                nonlocal ck_called
                ck_called = True
                return _orig_ck(*args, **kwargs)

            with patch("torch.ops.vllm.rocm_aiter_triton_gemm_a8w8_blockscale",
                       side_effect=_spy_triton), \
                 patch("torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale",
                       side_effect=_spy_ck):
                out = _dynamic_aiter_triton_ck_blockscale_gemm_impl(
                    A_fp8, B_fp8, As, Bs, torch.bfloat16,
                )

            assert ck_called, "BATCH_INVARIANT should force CK"
            assert not triton_called, "Triton should not fire with BATCH_INVARIANT"
            assert out.shape == (M, self.N)


@rocm_aiter_skip
class TestGPUDispatchPassthrough:
    """The dispatch wrapper doesn't corrupt output — bitwise match vs direct call.

    This is a multiplexer concern: verifies torch.cond routing passes
    tensors through without modification. Not testing GEMM correctness
    (that's covered by test_block_fp8.py and test_rocm_skinny_gemms.py).
    """

    @pytest.mark.parametrize(
        "M,N,K",
        [(8, 7168, 2048), (64, 7168, 2048)],
    )
    @torch.inference_mode()
    def test_dynamic_dispatch_matches_direct_call(self, M, N, K):
        A_fp8, B_fp8, As, Bs, _, _ = _make_gpu_inputs(M, N, K)
        threshold = 16

        with patch.object(envs, "VLLM_ROCM_W8A8_TRITON_MAX_M", threshold), \
             patch.object(envs, "VLLM_BATCH_INVARIANT", False):
            out_dispatch = _dynamic_aiter_triton_ck_blockscale_gemm_impl(
                A_fp8, B_fp8, As, Bs, torch.bfloat16,
            )

        if M <= threshold:
            out_direct = torch.ops.vllm.rocm_aiter_triton_gemm_a8w8_blockscale(
                A_fp8, B_fp8, As, Bs, torch.bfloat16,
            )
        else:
            out_direct = torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale(
                A_fp8, B_fp8, As, Bs, torch.bfloat16,
            )

        torch.testing.assert_close(out_dispatch, out_direct, atol=0, rtol=0)


@rocm_aiter_skip
class TestGPUOpcheck:
    """Validate that the custom op's fake impl is consistent with real impl.

    Uses ``torch.library.opcheck`` (same pattern as test_grouped_quant.py).
    """

    @torch.inference_mode()
    def test_opcheck_dynamic_dispatch(self):
        M, N, K = 32, 7168, 2048
        A_fp8, B_fp8, As, Bs, _, _ = _make_gpu_inputs(M, N, K)

        torch.library.opcheck(
            torch.ops.vllm.dynamic_aiter_triton_ck_blockscale_gemm,
            (A_fp8, B_fp8, As, Bs, torch.bfloat16),
            test_utils=("test_faketensor",),
        )
