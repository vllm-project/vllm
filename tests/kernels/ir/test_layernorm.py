# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for layernorm IR ops.

Each op has its own test class containing all related tests:
- Registration checks
- Symbolic tests (supports_args + SymInt compatibility)
- Numerical tests (impl correctness)
- Lowering tests (torch.compile integration)
"""

import pytest
import torch

# This registers op implementations
import vllm.kernels  # noqa: F401
from tests.ir.ir_test_utils import (
    COMMON_HIDDEN_SIZES,
    NUM_TOKENS,
    assert_close,
    assert_dispatch_matches_direct,
    assert_impl_numerical,
    assert_op_e2e_correctness,
    assert_supports_args_returns_bool,
    supported_providers,
)
from tests.kernels.allclose_default import get_default_rtol
from vllm import ir
from vllm.platforms import current_platform

# ============================================================
# RMSNorm
# ============================================================


class TestRmsNorm:
    """Tests for the rms_norm IR op."""

    @classmethod
    def setup_class(cls):
        torch.set_default_device(current_platform.device_type)

    # --- Registration ---

    @pytest.mark.skipif(
        not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
        reason="Currently only kernels on CUDA, ROCm and XPU",
    )
    def test_registration(self):
        expected = {
            "native": True,
            "vllm_c": current_platform.is_cuda_alike(),
            "aiter": current_platform.is_rocm(),
            "oink": current_platform.has_device_capability(100)
            and hasattr(torch.ops, "oink")
            and hasattr(torch.ops.oink, "rmsnorm"),
            "xpu_kernels": current_platform.is_xpu(),
        }

        actual = {
            provider: impl.supported for provider, impl in ir.ops.rms_norm.impls.items()
        }

        assert actual == expected

    # --- Symbolic ---

    @pytest.mark.parametrize(
        "provider", ["native"] + supported_providers(ir.ops.rms_norm)
    )
    def test_supports_args_returns_bool(self, provider: str):
        """Verify supports_args returns bool with unbacked SymInts."""
        assert_supports_args_returns_bool(
            ir.ops.rms_norm,
            provider,
            symbolic_params=["x"],
            num_tokens=8,
            hidden_size=64,
            dtype=torch.bfloat16,
            epsilon=1e-5,
        )

    # --- Numerical ---

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("n_tokens", NUM_TOKENS)
    @pytest.mark.parametrize("hidden_size", COMMON_HIDDEN_SIZES)
    @pytest.mark.parametrize("epsilon", [1e-6, 1e-5])
    @pytest.mark.skipif(
        not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
        reason="Currently only kernels on CUDA, ROCm and XPU",
    )
    def test_native_semantics(self, dtype, n_tokens, hidden_size, epsilon):
        rms_norm_native = ir.ops.rms_norm.impls["native"].impl_fn
        x, weight, epsilon = ir.ops.rms_norm.generate_inputs(
            num_tokens=4, hidden_size=8, dtype=dtype, epsilon=epsilon
        )
        out = rms_norm_native(x, weight, epsilon=epsilon)

        # Check shape, dtype, device
        assert out.shape == x.shape
        assert out.dtype == x.dtype
        assert out.device == x.device

        # Check the scaling property of rms norm
        out2 = rms_norm_native(x * 2.0, weight, epsilon=epsilon)
        torch.testing.assert_close(out2, out, rtol=get_default_rtol(out), atol=1e-3)

        # Mean square should be approximately 1 (ignoring epsilon and weight scaling)
        combined_norm = out.float() / weight.float()
        variance = combined_norm.pow(2).mean(dim=-1)
        # After RMS normalization, variance should be close to 1
        torch.testing.assert_close(
            variance, torch.ones_like(variance), rtol=1e-2, atol=1e-2
        )

        # Check behavior with and without weight
        weight1 = torch.ones_like(weight)
        out3 = rms_norm_native(x, weight1, epsilon=epsilon)
        out4 = rms_norm_native(x, None, epsilon=epsilon)
        assert_close(ir.ops.rms_norm, out3, out4)

    @pytest.mark.parametrize("provider", supported_providers(ir.ops.rms_norm))
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("n_tokens", NUM_TOKENS)
    @pytest.mark.parametrize("hidden_size", COMMON_HIDDEN_SIZES)
    @pytest.mark.skipif(
        not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
        reason="Currently only kernels on CUDA, ROCm and XPU",
    )
    def test_impl_numerical(self, provider, dtype, n_tokens, hidden_size):
        """Test impl produces same results as native."""
        args = ir.ops.rms_norm.generate_inputs(
            num_tokens=n_tokens, hidden_size=hidden_size, dtype=dtype, epsilon=1e-5
        )
        assert_impl_numerical(ir.ops.rms_norm, provider, args)

    @pytest.mark.parametrize("provider", supported_providers(ir.ops.rms_norm))
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("n_tokens", NUM_TOKENS)
    @pytest.mark.parametrize("hidden_size", COMMON_HIDDEN_SIZES)
    @pytest.mark.skipif(
        not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
        reason="Currently only kernels on CUDA, ROCm and XPU",
    )
    def test_dispatch_consistency(self, provider, dtype, n_tokens, hidden_size):
        """Test dispatch matches direct impl call."""
        args = ir.ops.rms_norm.generate_inputs(
            num_tokens=n_tokens, hidden_size=hidden_size, dtype=dtype, epsilon=1e-5
        )
        assert_dispatch_matches_direct(ir.ops.rms_norm, provider, args)

    @pytest.mark.parametrize("provider", supported_providers(ir.ops.rms_norm))
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("n_tokens", NUM_TOKENS)
    @pytest.mark.parametrize("hidden_size", COMMON_HIDDEN_SIZES)
    @pytest.mark.skipif(
        not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
        reason="Currently only kernels on CUDA, ROCm and XPU",
    )
    def test_variance_size_not_supported(self, provider, dtype, n_tokens, hidden_size):
        """Test that variance_size override is not supported."""
        args = ir.ops.rms_norm.generate_inputs(
            num_tokens=n_tokens, hidden_size=hidden_size, dtype=dtype, epsilon=1e-5
        )
        impl = ir.ops.rms_norm.impls[provider]
        x, weight, eps = args
        assert not impl.supports_args(x, weight, eps, 4)
        assert not impl.supports_args(x, weight, eps, variance_size=4)

    @pytest.mark.parametrize("provider", supported_providers(ir.ops.rms_norm))
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("n_tokens", NUM_TOKENS)
    @pytest.mark.parametrize("hidden_size", COMMON_HIDDEN_SIZES)
    @pytest.mark.skipif(
        not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
        reason="Currently only kernels on CUDA, ROCm and XPU",
    )
    def test_weight_none(self, provider, dtype, n_tokens, hidden_size):
        """Test weight=None equals weight=ones."""
        x, weight, eps = ir.ops.rms_norm.generate_inputs(
            num_tokens=n_tokens, hidden_size=hidden_size, dtype=dtype, epsilon=1e-5
        )
        impl = ir.ops.rms_norm.impls[provider]
        out_no_weight = impl.impl_fn(x, None, eps)
        out_unit_weight = impl.impl_fn(x, torch.ones_like(weight), eps)
        assert_close(ir.ops.rms_norm, out_no_weight, out_unit_weight)

    @pytest.mark.parametrize("provider", ["vllm_c", "aiter", "xpu_kernels", "native"])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.skipif(
        not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
        reason="Currently only kernels on CUDA, ROCm and XPU",
    )
    def test_torch_opcheck(self, dtype, provider):
        if not ir.ops.rms_norm.impls[provider].supported:
            pytest.skip(f"{provider} impl not supported on this platform")

        args = ir.ops.rms_norm.generate_inputs(
            num_tokens=32, hidden_size=2048, dtype=dtype, epsilon=1e-5
        )

        # When checking the torch op, we have to set priority and use dispatch
        with ir.ops.rms_norm.set_priority([provider, "native"]):
            torch.library.opcheck(torch.ops.vllm_ir.rms_norm, args)

    # --- Lowering ---

    @pytest.mark.parametrize(
        "provider", ["native"] + supported_providers(ir.ops.rms_norm)
    )
    def test_e2e_correctness(self, provider: str, default_vllm_config):
        """Verify lowering produces correct results."""
        args = ir.ops.rms_norm.generate_inputs(
            num_tokens=8, hidden_size=16, dtype=torch.bfloat16, epsilon=1e-5
        )
        assert_op_e2e_correctness(ir.ops.rms_norm, provider, args)

    # --- Platform-specific ---

    @pytest.mark.skipif(
        not current_platform.is_rocm(),
        reason="aiter is only supported on ROCm",
    )
    def test_aiter_rejects_unsupported_dtypes(self):
        impl = ir.ops.rms_norm.impls["aiter"]
        for dtype in [torch.float32, torch.float64]:
            args = ir.ops.rms_norm.generate_inputs(
                num_tokens=8, hidden_size=4096, dtype=dtype, epsilon=1e-5
            )
            assert not impl.supports_args(*args), f"aiter should reject dtype={dtype}"


# ============================================================
# FusedAddRmsNorm
# ============================================================


class TestFusedAddRmsNorm:
    """Tests for the fused_add_rms_norm IR op."""

    @classmethod
    def setup_class(cls):
        torch.set_default_device(current_platform.device_type)

    # --- Registration ---

    @pytest.mark.skipif(
        not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
        reason="Currently only kernels on CUDA, ROCm and XPU",
    )
    def test_registration(self):
        expected = {
            "native": True,
            "vllm_c": current_platform.is_cuda_alike(),
            "aiter": current_platform.is_rocm(),
            "oink": current_platform.has_device_capability(100)
            and hasattr(torch.ops, "oink")
            and hasattr(torch.ops.oink, "fused_add_rms_norm"),
            "xpu_kernels": current_platform.is_xpu(),
        }

        actual = {
            provider: impl.supported
            for provider, impl in ir.ops.fused_add_rms_norm.impls.items()
        }

        assert actual == expected

    # --- Symbolic ---

    @pytest.mark.parametrize(
        "provider", ["native"] + supported_providers(ir.ops.fused_add_rms_norm)
    )
    def test_supports_args_returns_bool(self, provider: str):
        """Verify supports_args returns bool with unbacked SymInts."""
        assert_supports_args_returns_bool(
            ir.ops.fused_add_rms_norm,
            provider,
            symbolic_params=["x", "x_residual"],
            num_tokens=8,
            hidden_size=64,
            dtype=torch.bfloat16,
            epsilon=1e-5,
        )

    # --- Numerical ---

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("n_tokens", NUM_TOKENS)
    @pytest.mark.parametrize("hidden_size", COMMON_HIDDEN_SIZES)
    @pytest.mark.parametrize("epsilon", [1e-6, 1e-5])
    @pytest.mark.skipif(
        not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
        reason="Currently only kernels on CUDA, ROCm and XPU",
    )
    def test_native_semantics(self, dtype, n_tokens, hidden_size, epsilon):
        rms_norm_native = ir.ops.rms_norm.impls["native"].impl_fn
        fused_add_rms_norm_native = ir.ops.fused_add_rms_norm.impls["native"].impl_fn
        x, x_residual, weight, eps = ir.ops.fused_add_rms_norm.generate_inputs(
            num_tokens=4, hidden_size=8, dtype=dtype, epsilon=epsilon
        )
        out, residual_out = fused_add_rms_norm_native(x, x_residual, weight, eps)

        # Check shape, dtype, device
        assert out.shape == x.shape
        assert out.dtype == x.dtype
        assert out.device == x.device
        assert residual_out.shape == x_residual.shape
        assert residual_out.dtype == x_residual.dtype
        assert residual_out.device == x_residual.device

        # Check that residual_out = x + x_residual
        expected_residual = (x.float() + x_residual.float()).to(dtype)
        torch.testing.assert_close(
            residual_out, expected_residual, rtol=1e-3, atol=1e-3
        )

        # Verify that the output is RMS normalized version of (x + x_residual)
        expected_out = rms_norm_native(expected_residual, weight, epsilon)
        assert_close(
            ir.ops.fused_add_rms_norm,
            (out, residual_out),
            (expected_out, expected_residual),
        )

        # Check the scaling property of rms norm
        out1, _ = fused_add_rms_norm_native(
            x, torch.zeros_like(x), weight, epsilon=epsilon
        )
        out2, _ = fused_add_rms_norm_native(
            x * 2.0, torch.zeros_like(x), weight, epsilon=epsilon
        )
        torch.testing.assert_close(out2, out1, rtol=get_default_rtol(out), atol=1e-3)

        # Check behavior with and without weight
        weight1 = torch.ones_like(weight)
        out3, _ = fused_add_rms_norm_native(x, x_residual, weight1, eps)
        out4, _ = fused_add_rms_norm_native(x, x_residual, None, eps)
        assert_close(ir.ops.fused_add_rms_norm, out3, out4)

    @pytest.mark.parametrize("provider", supported_providers(ir.ops.fused_add_rms_norm))
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("n_tokens", NUM_TOKENS)
    @pytest.mark.parametrize("hidden_size", COMMON_HIDDEN_SIZES)
    @pytest.mark.skipif(
        not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
        reason="Currently only kernels on CUDA, ROCm and XPU",
    )
    def test_impl_numerical(self, provider, dtype, n_tokens, hidden_size):
        """Test impl produces same results as native."""
        args = ir.ops.fused_add_rms_norm.generate_inputs(
            num_tokens=n_tokens, hidden_size=hidden_size, dtype=dtype, epsilon=1e-5
        )
        args = args + (None,)  # variance_size parameter
        assert_impl_numerical(ir.ops.fused_add_rms_norm, provider, args)

    @pytest.mark.parametrize("provider", supported_providers(ir.ops.fused_add_rms_norm))
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("n_tokens", NUM_TOKENS)
    @pytest.mark.parametrize("hidden_size", COMMON_HIDDEN_SIZES)
    @pytest.mark.skipif(
        not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
        reason="Currently only kernels on CUDA, ROCm and XPU",
    )
    def test_dispatch_consistency(self, provider, dtype, n_tokens, hidden_size):
        """Test dispatch matches direct impl call."""
        args = ir.ops.fused_add_rms_norm.generate_inputs(
            num_tokens=n_tokens, hidden_size=hidden_size, dtype=dtype, epsilon=1e-5
        )
        assert_dispatch_matches_direct(ir.ops.fused_add_rms_norm, provider, args)

    @pytest.mark.parametrize("provider", supported_providers(ir.ops.fused_add_rms_norm))
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("n_tokens", NUM_TOKENS)
    @pytest.mark.parametrize("hidden_size", COMMON_HIDDEN_SIZES)
    @pytest.mark.skipif(
        not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
        reason="Currently only kernels on CUDA, ROCm and XPU",
    )
    def test_variance_size_not_supported(self, provider, dtype, n_tokens, hidden_size):
        """Test that variance_size override is not supported."""
        x, x_residual, weight, eps = ir.ops.fused_add_rms_norm.generate_inputs(
            num_tokens=n_tokens, hidden_size=hidden_size, dtype=dtype, epsilon=1e-5
        )
        impl = ir.ops.fused_add_rms_norm.impls[provider]
        assert not impl.supports_args(x, x_residual, weight, eps, 4)
        assert not impl.supports_args(x, x_residual, weight, eps, variance_size=4)

    @pytest.mark.parametrize("provider", supported_providers(ir.ops.fused_add_rms_norm))
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("n_tokens", NUM_TOKENS)
    @pytest.mark.parametrize("hidden_size", COMMON_HIDDEN_SIZES)
    @pytest.mark.skipif(
        not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
        reason="Currently only kernels on CUDA, ROCm and XPU",
    )
    def test_weight_none(self, provider, dtype, n_tokens, hidden_size):
        """Test weight=None equals weight=ones."""
        x, x_residual, weight, eps = ir.ops.fused_add_rms_norm.generate_inputs(
            num_tokens=n_tokens, hidden_size=hidden_size, dtype=dtype, epsilon=1e-5
        )
        impl = ir.ops.fused_add_rms_norm.impls[provider]
        out_no_weight, residual_no_weight = impl.impl_fn(
            x.clone(), x_residual.clone(), None, eps
        )
        out_unit_weight, residual_unit_weight = impl.impl_fn(
            x.clone(), x_residual.clone(), torch.ones_like(weight), eps
        )
        assert_close(ir.ops.fused_add_rms_norm, out_no_weight, out_unit_weight)
        assert_close(
            ir.ops.fused_add_rms_norm, residual_no_weight, residual_unit_weight
        )

    @pytest.mark.parametrize("provider", ["vllm_c"])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.skipif(
        not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
        reason="Currently only kernels on CUDA, ROCm and XPU",
    )
    def test_inplace_semantics(self, dtype, provider):
        """Test that inplace implementations reuse inputs,
        for maybe_inplace overload but not for default overload."""
        impl = ir.ops.fused_add_rms_norm.impls[provider]
        if not impl.supported:
            pytest.skip(f"{provider} impl not supported on this platform")

        x, x_residual, weight, eps = ir.ops.fused_add_rms_norm.generate_inputs(
            num_tokens=32, hidden_size=2048, dtype=dtype, epsilon=1e-5
        )

        # Test default overload - should NOT modify inputs even with inplace impl
        x_default = x.clone()
        x_residual_default = x_residual.clone()
        x_default_ptr = x_default.data_ptr()
        x_residual_default_ptr = x_residual_default.data_ptr()

        with ir.ops.fused_add_rms_norm.set_priority([provider, "native"]):
            out_default, residual_default = ir.ops.fused_add_rms_norm(
                x_default, x_residual_default, weight, eps
            )

        # Default should NOT be inplace (even with inplace implementation)
        assert out_default.data_ptr() != x_default_ptr
        assert residual_default.data_ptr() != x_residual_default_ptr
        torch.testing.assert_close(x, x_default, rtol=0.0, atol=0.0)
        torch.testing.assert_close(x_residual, x_residual_default, rtol=0.0, atol=0.0)

        # Test maybe_inplace overload - should modify inputs with inplace impl
        x_inplace = x.clone()
        x_residual_inplace = x_residual.clone()
        x_inplace_ptr = x_inplace.data_ptr()
        x_residual_inplace_ptr = x_residual_inplace.data_ptr()

        with ir.ops.fused_add_rms_norm.set_priority([provider, "native"]):
            out_inplace, residual_inplace = ir.ops.fused_add_rms_norm.maybe_inplace(
                x_inplace, x_residual_inplace, weight, eps
            )

        # maybe_inplace should be inplace
        assert out_inplace.data_ptr() == x_inplace_ptr
        assert residual_inplace.data_ptr() == x_residual_inplace_ptr

        # Both should produce same results
        torch.testing.assert_close(out_default, out_inplace, atol=0.0, rtol=0.0)
        torch.testing.assert_close(
            residual_default, residual_inplace, atol=0.0, rtol=0.0
        )

    @pytest.mark.parametrize("provider", supported_providers(ir.ops.fused_add_rms_norm))
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.skipif(
        not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
        reason="Currently only kernels on CUDA, ROCm and XPU",
    )
    def test_torch_opcheck(self, dtype, provider):
        args = ir.ops.fused_add_rms_norm.generate_inputs(
            num_tokens=32, hidden_size=2048, dtype=dtype, epsilon=1e-5
        )
        args = args + (None,)  # Add variance_size parameter

        # When checking the torch op, we have to set priority and use dispatch
        with ir.ops.fused_add_rms_norm.set_priority([provider, "native"]):
            torch.library.opcheck(torch.ops.vllm_ir.fused_add_rms_norm.default, args)

            # Only test maybe_inplace with non-inplace implementations
            # Inplace implementations return aliases of inputs which is not allowed.
            # We break this invariant, but we also convert maybe_inplace to the default
            # overload during compilation, so maybe_inplace never reaches Inductor.
            if not ir.ops.fused_add_rms_norm.impls[provider].inplace:
                torch.library.opcheck(
                    torch.ops.vllm_ir.fused_add_rms_norm.maybe_inplace, args
                )

    # --- Lowering ---

    @pytest.mark.parametrize(
        "provider", ["native"] + supported_providers(ir.ops.fused_add_rms_norm)
    )
    def test_e2e_correctness(self, provider: str, default_vllm_config):
        """Verify lowering produces correct results."""
        args = ir.ops.fused_add_rms_norm.generate_inputs(
            num_tokens=8, hidden_size=16, dtype=torch.bfloat16, epsilon=1e-5
        )
        assert_op_e2e_correctness(ir.ops.fused_add_rms_norm, provider, args)
