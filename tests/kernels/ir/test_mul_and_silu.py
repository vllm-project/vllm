# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.kernels  # noqa: F401
from tests.kernels.allclose_default import get_default_rtol
from vllm import ir
from vllm.platforms import current_platform

pytestmark = pytest.mark.skip_global_cleanup


def mul_and_silu_inputs(n_tokens: int, d: int, dtype: torch.dtype):
    x = torch.randn(n_tokens, 2 * d, dtype=dtype)
    return (x,)


mul_and_silu_native = ir.ops.mul_and_silu.impls["native"].impl_fn


def _priority(provider: str) -> list[str]:
    return [provider] if provider == "native" else [provider, "native"]


def test_mul_and_silu_registration():
    expected = {
        "native": True,
        "vllm_c": current_platform.is_cuda_alike(),
        "xpu_kernels": current_platform.is_xpu(),
    }
    actual = {
        provider: impl.supported for provider, impl in ir.ops.mul_and_silu.impls.items()
    }
    assert actual == expected


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n_tokens", [1, 8, 17])
@pytest.mark.parametrize("d", [16, 256, 2048])
class TestMulAndSilu:
    def test_native_semantics(self, dtype, n_tokens, d):
        (x,) = mul_and_silu_inputs(n_tokens, d, dtype)
        out_ir = ir.ops.mul_and_silu(x)
        out_native = mul_and_silu_native(x)
        torch.testing.assert_close(out_ir, out_native, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("provider", ["vllm_c", "xpu_kernels"])
    def test_impls(self, dtype, n_tokens, d, provider):
        impl = ir.ops.mul_and_silu.impls[provider]
        if not impl.supported:
            pytest.skip(f"{provider} not supported")

        torch.set_default_device(current_platform.device_type)
        (x,) = mul_and_silu_inputs(n_tokens, d, dtype)
        out_impl = impl.impl_fn(x)
        out_native = mul_and_silu_native(x)
        torch.testing.assert_close(
            out_impl, out_native, rtol=get_default_rtol(out_impl), atol=1e-3
        )

        with ir.ops.mul_and_silu.set_priority(_priority(provider)):
            out_dispatched = ir.ops.mul_and_silu(x)
        torch.testing.assert_close(out_dispatched, out_impl, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("provider", ["native", "vllm_c", "xpu_kernels"])
    def test_torch_opcheck(self, dtype, n_tokens, d, provider):
        if not ir.ops.mul_and_silu.impls[provider].supported:
            pytest.skip(f"{provider} not supported")

        (x,) = mul_and_silu_inputs(n_tokens, d, dtype)
        with ir.ops.mul_and_silu.set_priority(_priority(provider)):
            torch.library.opcheck(torch.ops.vllm_ir.mul_and_silu, (x,))
