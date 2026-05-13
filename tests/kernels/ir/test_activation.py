# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

import vllm.kernels  # noqa: F401
from tests.kernels.allclose_default import get_default_rtol
from vllm import ir
from vllm.platforms import current_platform


def mul_and_silu_inputs(n_tokens: int, d: int, dtype: torch.dtype):
    x = torch.randn(n_tokens, 2 * d, dtype=dtype)
    return (x,)


mul_and_silu_native = ir.ops.mul_and_silu.impls["native"].impl_fn


@pytest.mark.skipif(
    not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
    reason="Currently only kernels on CUDA, ROCm and XPU",
)
def test_mul_and_silu_registration():
    expected = {
        "native": True,
        "vllm_c": current_platform.is_cuda_alike(),
        "xpu_kernels": current_platform.is_xpu(),  # if registered
    }
    actual = {
        provider: impl.supported for provider, impl in ir.ops.mul_and_silu.impls.items()
    }
    assert actual == expected


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n_tokens", [1, 8, 17])
@pytest.mark.parametrize("d", [16, 4096, 8192])
@pytest.mark.skipif(
    not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
    reason="Currently only kernels on CUDA, ROCm and XPU",
)
class TestMulAndSilu:
    @classmethod
    def setup_class(cls):
        torch.set_default_device(current_platform.device_type)

    def test_native_semantics(self, dtype, n_tokens, d):
        (x,) = mul_and_silu_inputs(n_tokens, d, dtype)
        out = mul_and_silu_native(x)
        assert out.shape == x.shape[:-1] + (d,)
        assert out.dtype == x.dtype

    @pytest.mark.parametrize("provider", ["vllm_c", "xpu_kernels"])
    def test_impls(self, dtype, n_tokens, d, provider):
        impl = ir.ops.mul_and_silu.impls[provider]
        if not impl.supported:
            pytest.skip(f"{provider} not supported")
        (x,) = mul_and_silu_inputs(n_tokens, d, dtype)
        out_impl = impl.impl_fn(x)
        out_native = mul_and_silu_native(x)
        torch.testing.assert_close(
            out_impl, out_native, rtol=get_default_rtol(out_impl), atol=1e-3
        )
        with ir.ops.mul_and_silu.set_priority([provider, "native"]):
            out_dispatched = ir.ops.mul_and_silu(x)
        torch.testing.assert_close(out_dispatched, out_impl, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("compile", [False, True])
    def test_native_impl_compile(self, dtype, n_tokens, d, compile):
        impl = ir.ops.mul_and_silu.impls["native"]
        assert impl.supported, "native implementation must be supported!"
        (x,) = mul_and_silu_inputs(n_tokens, d, dtype)
        out_impl = impl.impl_fn(x)
        out_native = mul_and_silu_native(x)
        torch.testing.assert_close(
            out_impl, out_native, rtol=get_default_rtol(out_impl), atol=1e-3
        )
        with ir.ops.mul_and_silu.set_priority(["native"], compile=compile):
            out_dispatched = ir.ops.mul_and_silu(x)

            if compile:
                assert isinstance(
                    ir.ops.mul_and_silu.dispatch(x),
                    ir.op.IrOpImplCompiledWrapper,
                ), (
                    "When `set_priority` with compile=True, the implementation is "
                    "expected to be wrapped with compile."
                )
        torch.testing.assert_close(out_dispatched, out_impl, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("provider", ["vllm_c", "xpu_kernels", "native"])
    def test_torch_opcheck(self, dtype, n_tokens, d, provider):
        if not ir.ops.mul_and_silu.impls[provider].supported:
            pytest.skip(f"{provider} not supported")
        (x,) = mul_and_silu_inputs(n_tokens, d, dtype)
        with ir.ops.mul_and_silu.set_priority([provider, "native"]):
            torch.library.opcheck(torch.ops.vllm_ir.mul_and_silu, (x,))
