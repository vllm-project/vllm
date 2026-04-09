# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

import vllm.kernels  # noqa: F401
from tests.kernels.allclose_default import get_default_rtol
from vllm import ir
from vllm.platforms import current_platform


def silu_and_mul_inputs(n_tokens: int, d: int, dtype: torch.dtype):
    x = torch.randn(n_tokens, 2 * d, dtype=dtype)
    return (x,)


silu_and_mul_native = ir.ops.silu_and_mul.impls["native"].impl_fn


@pytest.mark.skipif(
    not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
    reason="Currently only kernels on CUDA, ROCm and XPU",
)
def test_silu_and_mul_registration():
    expected = {
        "native": True,
        "vllm_c": current_platform.is_cuda_alike(),
        "xpu_kernels": current_platform.is_xpu(),  # if registered
    }
    actual = {
        provider: impl.supported for provider, impl in ir.ops.silu_and_mul.impls.items()
    }
    assert actual == expected


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n_tokens", [1, 8, 17])
@pytest.mark.parametrize("d", [16, 4096, 8192])
@pytest.mark.skipif(
    not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
    reason="Currently only kernels on CUDA, ROCm and XPU",
)
class TestSiluAndMul:
    @classmethod
    def setup_class(cls):
        torch.set_default_device(current_platform.device_type)

    def test_native_semantics(self, dtype, n_tokens, d):
        (x,) = silu_and_mul_inputs(n_tokens, d, dtype)
        out = silu_and_mul_native(x)
        assert out.shape == x.shape[:-1] + (d,)
        assert out.dtype == x.dtype

    @pytest.mark.parametrize("provider", ["vllm_c", "xpu_kernels"])
    def test_impls(self, dtype, n_tokens, d, provider):
        impl = ir.ops.silu_and_mul.impls[provider]
        if not impl.supported:
            pytest.skip(f"{provider} not supported")
        (x,) = silu_and_mul_inputs(n_tokens, d, dtype)
        out_impl = impl.impl_fn(x)
        out_native = silu_and_mul_native(x)
        torch.testing.assert_close(
            out_impl, out_native, rtol=get_default_rtol(out_impl), atol=1e-3
        )
        with ir.ops.silu_and_mul.set_priority([provider, "native"]):
            out_dispatched = ir.ops.silu_and_mul(x)
        torch.testing.assert_close(out_dispatched, out_impl, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("provider", ["vllm_c", "xpu_kernels", "native"])
    def test_torch_opcheck(self, dtype, n_tokens, d, provider):
        if not ir.ops.silu_and_mul.impls[provider].supported:
            pytest.skip(f"{provider} not supported")
        (x,) = silu_and_mul_inputs(n_tokens, d, dtype)
        with ir.ops.silu_and_mul.set_priority([provider, "native"]):
            torch.library.opcheck(torch.ops.vllm_ir.silu_and_mul, (x,))
