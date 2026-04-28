# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

# This registers op implementations
import vllm.kernels  # noqa: F401
from tests.ir.ir_test_utils import (
    COMMON_HIDDEN_SIZES,
    NUM_TOKENS,
    assert_close,
    clone_args,
    supported_providers,
)
from tests.kernels.allclose_default import get_default_rtol
from vllm import ir
from vllm.platforms import current_platform

rms_norm_native = ir.ops.rms_norm.impls["native"].impl_fn


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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", COMMON_HIDDEN_SIZES)
@pytest.mark.parametrize("epsilon", [1e-6, 1e-5])
@pytest.mark.skipif(
    not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
    reason="Currently only kernels on CUDA, ROCm and XPU",
)
class TestRMSNorm:
    @classmethod
    def setup_class(cls, **kwargs):
        torch.set_default_device(current_platform.device_type)

    def test_native_semantics(self, dtype, n_tokens, hidden_size, epsilon):
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

        # Check behavior with and without weight
        weight1 = torch.ones_like(weight)
        out3 = rms_norm_native(x, weight1, epsilon=epsilon)
        out4 = rms_norm_native(x, None, epsilon=epsilon)
        torch.testing.assert_close(out3, out4)

    @pytest.mark.parametrize("provider", supported_providers(ir.ops.rms_norm))
    def test_impls(self, dtype, n_tokens, hidden_size, epsilon, provider):
        impl = ir.ops.rms_norm.impls[provider]
        x, weight, eps = ir.ops.rms_norm.generate_inputs(
            num_tokens=n_tokens, hidden_size=hidden_size, dtype=dtype, epsilon=epsilon
        )
        args = (x, weight, eps)

        if not impl.supports_args(*args):
            pytest.skip(f"{provider} does not support args")

        ref_output = rms_norm_native(*clone_args(args))
        output = impl.impl_fn(*clone_args(args))
        assert_close(ir.ops.rms_norm, output, ref_output)

        # check that dispatched call matches direct call
        with ir.ops.rms_norm.set_priority([provider, "native"]):
            out_dispatched = ir.ops.rms_norm(*args)
        out_direct = impl.impl_fn(*args)
        torch.testing.assert_close(out_dispatched, out_direct, rtol=0.0, atol=0.0)

        # none of these support variance_size override
        assert not impl.supports_args(x, weight, eps, 4)
        assert not impl.supports_args(x, weight, eps, variance_size=4)

        # test weight=None behavior
        out_no_weight = impl.impl_fn(x, None, eps)
        out_unit_weight = impl.impl_fn(x, torch.ones_like(weight), eps)
        assert_close(ir.ops.rms_norm, out_no_weight, out_unit_weight)

    @pytest.mark.parametrize("provider", ["vllm_c", "aiter", "xpu_kernels", "native"])
    def test_torch_opcheck(self, dtype, n_tokens, hidden_size, epsilon, provider):
        if not ir.ops.rms_norm.impls[provider].supported:
            pytest.skip(f"{provider} impl not supported on this platform")

        args = ir.ops.rms_norm.generate_inputs(
            num_tokens=n_tokens, hidden_size=hidden_size, dtype=dtype, epsilon=epsilon
        )

        # When checking the torch op, we have to set priority and use dispatch
        with ir.ops.rms_norm.set_priority([provider, "native"]):
            torch.library.opcheck(torch.ops.vllm_ir.rms_norm, args)


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
