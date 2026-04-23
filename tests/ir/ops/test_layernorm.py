# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for the RMSNorm IR op.

Three layers of testing:
1. TestRmsNormSymbolic — supports_args + SymInt compatibility (fast)
2. TestRmsNormNumerical — impl numerical correctness (parametrized)
3. TestRmsNormLowering — torch.compile integration (e2e)
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
from vllm import ir
from vllm.ir.ops.layernorm import rms_norm
from vllm.platforms import current_platform

rms_norm_native = ir.ops.rms_norm.impls["native"].impl_fn
_all_providers = supported_providers(rms_norm) + ["native"]


# ============================================================
# Platform/registration checks
# ============================================================


@pytest.mark.skipif(
    not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
    reason="Currently only kernels on CUDA, ROCm and XPU",
)
def test_rms_norm_registration():
    expected = {
        "native": True,
        "vllm_c": current_platform.is_cuda_alike(),
        "aiter": current_platform.is_rocm(),
        "oink": False,
        "xpu_kernels": current_platform.is_xpu(),
    }

    actual = {
        provider: impl.supported for provider, impl in ir.ops.rms_norm.impls.items()
    }

    assert actual == expected


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="aiter is only supported on ROCm",
)
def test_aiter_rejects_unsupported_dtypes():
    torch.set_default_device(current_platform.device_type)
    impl = ir.ops.rms_norm.impls["aiter"]
    for dtype in [torch.float32, torch.float64]:
        args = ir.ops.rms_norm.generate_inputs(
            num_tokens=8, hidden_size=4096, dtype=dtype, epsilon=1e-5
        )
        assert not impl.supports_args(*args), f"aiter should reject dtype={dtype}"


# ============================================================
# Symbolic tests — supports_args + SymInt compatibility
# ============================================================


class TestRmsNormSymbolic:
    @pytest.mark.parametrize("provider", _all_providers)
    def test_supports_args_returns_bool(self, provider: str):
        """Verify supports_args returns bool with unbacked SymInts."""
        assert_supports_args_returns_bool(
            rms_norm,
            provider,
            num_tokens=8,
            hidden_size=64,
            dtype=torch.bfloat16,
            epsilon=1e-5,
        )


# ============================================================
# Numerical tests — impl correctness
# ============================================================


@pytest.mark.skipif(
    not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
    reason="Currently only kernels on CUDA, ROCm and XPU",
)
class TestRmsNormNumerical:
    @classmethod
    def setup_class(cls):
        torch.set_default_device(current_platform.device_type)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("n_tokens", NUM_TOKENS)
    @pytest.mark.parametrize("hidden_size", COMMON_HIDDEN_SIZES)
    @pytest.mark.parametrize("epsilon", [1e-6, 1e-5])
    def test_native_semantics(self, dtype, n_tokens, hidden_size, epsilon):
        """Verify native rms_norm has correct shape/dtype/scaling."""
        x, weight, epsilon = ir.ops.rms_norm.generate_inputs(
            num_tokens=4, hidden_size=8, dtype=dtype, epsilon=epsilon
        )
        out = rms_norm_native(x, weight, epsilon=epsilon)

        assert out.shape == x.shape
        assert out.dtype == x.dtype
        assert out.device == x.device

        # Scaling property: rms_norm(2*x) ≈ rms_norm(x) (approximate due to epsilon)
        out2 = rms_norm_native(x * 2.0, weight, epsilon=epsilon)
        assert_close(rms_norm, out2, out)

        # weight=None == unit weight
        out3 = rms_norm_native(x, torch.ones_like(weight), epsilon=epsilon)
        out4 = rms_norm_native(x, None, epsilon=epsilon)
        assert_close(rms_norm, out3, out4)

    @pytest.mark.parametrize("provider", supported_providers(ir.ops.rms_norm))
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_impl_numerical(self, provider, dtype):
        """Verify impl matches native numerically."""
        args = ir.ops.rms_norm.generate_inputs(
            num_tokens=32, hidden_size=2048, dtype=dtype, epsilon=1e-5
        )
        assert_impl_numerical(rms_norm, provider, args)

    @pytest.mark.parametrize("provider", supported_providers(ir.ops.rms_norm))
    def test_dispatch_matches_direct(self, provider):
        """Verify priority-based dispatch works correctly."""
        args = ir.ops.rms_norm.generate_inputs(
            num_tokens=32, hidden_size=2048, dtype=torch.bfloat16, epsilon=1e-5
        )
        assert_dispatch_matches_direct(rms_norm, provider, args)

    @pytest.mark.parametrize("provider", supported_providers(ir.ops.rms_norm))
    def test_weight_none(self, provider):
        """Verify all impls handle weight=None correctly."""
        impl = ir.ops.rms_norm.impls[provider]
        args = ir.ops.rms_norm.generate_inputs(
            num_tokens=32, hidden_size=2048, dtype=torch.bfloat16, epsilon=1e-5
        )
        x, weight, eps = args

        out_no_weight = impl.impl_fn(x, None, eps)
        out_unit_weight = impl.impl_fn(x, torch.ones_like(weight), eps)
        assert_close(rms_norm, out_no_weight, out_unit_weight)

    @pytest.mark.parametrize("provider", ["vllm_c", "aiter", "xpu_kernels", "native"])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_torch_opcheck(self, provider, dtype):
        """Verify torch op schema is correct."""
        if not ir.ops.rms_norm.impls[provider].supported:
            pytest.skip(f"{provider} impl not supported on this platform")

        args = ir.ops.rms_norm.generate_inputs(
            num_tokens=32, hidden_size=2048, dtype=dtype, epsilon=1e-5
        )

        with ir.ops.rms_norm.set_priority([provider, "native"]):
            torch.library.opcheck(torch.ops.vllm_ir.rms_norm, args)


# ============================================================
# Lowering tests — torch.compile integration
# ============================================================


class TestRmsNormLowering:
    @pytest.mark.parametrize("provider", _all_providers)
    def test_e2e_correctness(self, provider: str, default_vllm_config):
        """Verify lowering produces correct results."""
        args = rms_norm.generate_inputs(
            num_tokens=8, hidden_size=16, dtype=torch.bfloat16, epsilon=1e-5
        )
        assert_op_e2e_correctness(rms_norm, provider, args)