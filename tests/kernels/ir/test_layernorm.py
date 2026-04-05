# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

# This registers op implementations
import vllm.kernels  # noqa: F401
from tests.kernels.allclose_default import get_default_rtol
from vllm import ir
from vllm.platforms import current_platform


def rms_norm_inputs(n_tokens: int, hidden_size: int, dtype: torch.dtype):
    x = torch.randn(n_tokens, hidden_size, dtype=dtype)
    weight = torch.rand(hidden_size, dtype=dtype)
    return x, weight


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
@pytest.mark.parametrize("n_tokens", [1, 8, 17])
@pytest.mark.parametrize("hidden_size", [16, 4096, 8192])
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
        x, weight = rms_norm_inputs(4, 8, dtype)
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

        # Native impl should support mixed dtypes and follow dtype promotion.
        mixed_weight_dtype = (
            torch.float32 if x.dtype != torch.float32 else torch.float16
        )
        mixed_weight = torch.rand(
            x.shape[-1], dtype=mixed_weight_dtype, device=x.device
        )
        out_mixed = rms_norm_native(x, mixed_weight, epsilon=epsilon)
        assert out_mixed.dtype == torch.promote_types(x.dtype, mixed_weight_dtype)

    @pytest.mark.parametrize("provider", ["vllm_c", "aiter", "xpu_kernels"])
    def test_impls(self, dtype, n_tokens, hidden_size, epsilon, provider):
        impl = ir.ops.rms_norm.impls[provider]
        if not impl.supported:
            pytest.skip(f"{provider} impl not supported on this platform")

        x, weight = rms_norm_inputs(n_tokens, hidden_size, dtype)
        args = (x, weight, epsilon, None)

        assert impl.supported

        if provider == "aiter" and dtype not in [torch.float16, torch.bfloat16]:
            assert not impl.supports_args(*args)
            return

        assert impl.supports_args(*args)

        out_impl = impl.impl_fn(*args)
        out_native = rms_norm_native(*args)

        torch.testing.assert_close(
            out_impl, out_native, rtol=get_default_rtol(out_impl), atol=1e-3
        )

        # check that dispatched call matches direct call
        with ir.ops.rms_norm.set_priority([provider, "native"]):
            out_impl2 = ir.ops.rms_norm(*args)

        # exact match
        torch.testing.assert_close(out_impl2, out_impl, rtol=0.0, atol=0.0)

        # none of these support variance_size override
        assert not impl.supports_args(x, weight, epsilon, 4)
        assert not impl.supports_args(x, weight, epsilon, variance_size=4)

        # test weight=None behavior
        out_impl_no_weight = impl.impl_fn(x, None, epsilon)
        out_impl_unit_weight = impl.impl_fn(x, torch.ones_like(weight), epsilon)
        torch.testing.assert_close(
            out_impl_no_weight,
            out_impl_unit_weight,
            rtol=get_default_rtol(out_impl_no_weight),
            atol=2e-4,
        )

    @pytest.mark.parametrize("provider", ["vllm_c", "aiter", "xpu_kernels"])
    def test_impls_reject_mixed_weight_dtype(
        self, dtype, n_tokens, hidden_size, epsilon, provider
    ):
        impl = ir.ops.rms_norm.impls[provider]
        if not impl.supported:
            pytest.skip(f"{provider} impl not supported on this platform")
        if provider == "aiter" and dtype not in [torch.float16, torch.bfloat16]:
            pytest.skip(f"{provider} only supports fp16/bf16 activations")

        x = torch.randn(
            n_tokens,
            hidden_size,
            dtype=dtype,
            device=current_platform.device_type,
        )
        mixed_weight_dtype = torch.float32 if dtype != torch.float32 else torch.float16
        weight = torch.rand(hidden_size, dtype=mixed_weight_dtype, device=x.device)
        args = (x, weight, epsilon, None)

        # Provider kernels currently require homogeneous x/weight dtype.
        assert not impl.supports_args(*args)

        out_native = rms_norm_native(*args)
        assert out_native.dtype == torch.promote_types(dtype, mixed_weight_dtype)

        # Dispatch should fall back to native for mixed dtype inputs.
        with ir.ops.rms_norm.set_priority([provider, "native"]):
            out_dispatch = ir.ops.rms_norm(*args)

        torch.testing.assert_close(
            out_dispatch, out_native, rtol=get_default_rtol(out_dispatch), atol=1e-3
        )

    @pytest.mark.parametrize("provider", ["vllm_c", "aiter", "xpu_kernels", "native"])
    def test_torch_opcheck(self, dtype, n_tokens, hidden_size, epsilon, provider):
        if not ir.ops.rms_norm.impls[provider].supported:
            pytest.skip(f"{provider} impl not supported on this platform")

        x, weight = rms_norm_inputs(n_tokens, hidden_size, dtype)
        args = (x, weight, epsilon, None)

        # When checking the torch op, we have to set priority and use dispatch
        with ir.ops.rms_norm.set_priority([provider, "native"]):
            torch.library.opcheck(torch.ops.vllm_ir.rms_norm, args)
