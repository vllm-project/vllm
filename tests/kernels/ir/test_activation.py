# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

import vllm.kernels  # noqa: F401
from tests.kernels.allclose_default import get_default_rtol
from vllm import ir
from vllm.platforms import current_platform

SWIGLU_LIMITS = [3.0, 7.0, 15.0]


def silu_and_mul_with_clamp_inputs(
    n_tokens: int, d: int, dtype: torch.dtype, swiglu_limit: float
):
    # Scale the input so that clamping is actually exercised.
    x = torch.randn(n_tokens, 2 * d, dtype=dtype) * swiglu_limit * 2
    return (x, swiglu_limit)


silu_and_mul_with_clamp_native = ir.ops.silu_and_mul_with_clamp.impls["native"].impl_fn


@pytest.mark.skipif(
    not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
    reason="Currently only kernels on CUDA, ROCm and XPU",
)
def test_silu_and_mul_with_clamp_registration():
    expected = {
        "native": True,
        "vllm_c": current_platform.is_cuda_alike(),
        "xpu_kernels": current_platform.is_xpu(),  # if registered
    }
    actual = {
        provider: impl.supported
        for provider, impl in ir.ops.silu_and_mul_with_clamp.impls.items()
    }
    assert actual == expected


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n_tokens", [1, 8, 17])
@pytest.mark.parametrize("d", [16, 4096, 8192])
@pytest.mark.parametrize("swiglu_limit", SWIGLU_LIMITS)
@pytest.mark.skipif(
    not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
    reason="Currently only kernels on CUDA, ROCm and XPU",
)
class TestSiluAndMulWithClamp:
    @classmethod
    def setup_class(cls):
        torch.set_default_device(current_platform.device_type)

    def test_native_semantics(self, dtype, n_tokens, d, swiglu_limit):
        x, limit = silu_and_mul_with_clamp_inputs(n_tokens, d, dtype, swiglu_limit)
        out = silu_and_mul_with_clamp_native(x, limit)
        assert out.shape == x.shape[:-1] + (d,)
        assert out.dtype == x.dtype

        # Verify clamping is actually being applied: the clamped output must
        # differ from the unclamped SiluAndMul output when inputs are large.
        from vllm.model_executor.layers.activation import SiluAndMul

        unclamped = SiluAndMul.forward_native(x)
        assert not torch.equal(out.float(), unclamped.float()), (
            "Input was not large enough to exercise the clamp; increase scale"
        )

        # Gate clamping semantics: gate=large_val clamped to limit, up=1.0.
        x_gate = torch.tensor(
            [[limit * 20.0, 1.0]],
            dtype=torch.float32,
            device=current_platform.device_type,
        )
        out_gate = silu_and_mul_with_clamp_native(x_gate, limit)
        expected_gate = torch.nn.functional.silu(
            torch.tensor(limit, dtype=torch.float32)
        ).item()
        torch.testing.assert_close(
            out_gate,
            torch.tensor(
                [[expected_gate]],
                dtype=torch.float32,
                device=current_platform.device_type,
            ),
            atol=1e-3,
            rtol=1e-3,
        )

        # Up clamping semantics: up >> limit gets clamped to limit.
        x_up = torch.tensor(
            [[1.0, limit * 20.0]],
            dtype=torch.float32,
            device=current_platform.device_type,
        )
        out_up = silu_and_mul_with_clamp_native(x_up, limit)
        silu_1 = torch.nn.functional.silu(torch.tensor(1.0)).item()
        torch.testing.assert_close(
            out_up,
            torch.tensor(
                [[silu_1 * limit]],
                dtype=torch.float32,
                device=current_platform.device_type,
            ),
            atol=1e-3,
            rtol=1e-3,
        )

    @pytest.mark.parametrize("provider", ["vllm_c", "xpu_kernels"])
    def test_impls(self, dtype, n_tokens, d, swiglu_limit, provider):
        impl = ir.ops.silu_and_mul_with_clamp.impls[provider]
        if not impl.supported:
            pytest.skip(f"{provider} not supported")
        x, limit = silu_and_mul_with_clamp_inputs(n_tokens, d, dtype, swiglu_limit)
        out_impl = impl.impl_fn(x, limit)
        out_native = silu_and_mul_with_clamp_native(x, limit)
        torch.testing.assert_close(
            out_impl, out_native, rtol=get_default_rtol(out_impl), atol=1e-3
        )
        with ir.ops.silu_and_mul_with_clamp.set_priority([provider, "native"]):
            out_dispatched = ir.ops.silu_and_mul_with_clamp(x, limit)
        torch.testing.assert_close(out_dispatched, out_impl, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("compile", [False, True])
    def test_native_impl_compile(self, dtype, n_tokens, d, swiglu_limit, compile):
        impl = ir.ops.silu_and_mul_with_clamp.impls["native"]
        assert impl.supported, "native implementation must be supported!"
        x, limit = silu_and_mul_with_clamp_inputs(n_tokens, d, dtype, swiglu_limit)
        out_impl = impl.impl_fn(x, limit)
        out_native = silu_and_mul_with_clamp_native(x, limit)
        torch.testing.assert_close(
            out_impl, out_native, rtol=get_default_rtol(out_impl), atol=1e-3
        )
        with ir.ops.silu_and_mul_with_clamp.set_priority(["native"], compile=compile):
            out_dispatched = ir.ops.silu_and_mul_with_clamp(x, limit)

            if compile:
                assert isinstance(
                    ir.ops.silu_and_mul_with_clamp.dispatch(x, limit),
                    ir.op.IrOpImplCompiledWrapper,
                ), (
                    "When `set_priority` with compile=True, the implementation is "
                    "expected to be wrapped with compile."
                )
        torch.testing.assert_close(out_dispatched, out_impl, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("provider", ["vllm_c", "xpu_kernels", "native"])
    def test_torch_opcheck(self, dtype, n_tokens, d, swiglu_limit, provider):
        if not ir.ops.silu_and_mul_with_clamp.impls[provider].supported:
            pytest.skip(f"{provider} not supported")
        x, limit = silu_and_mul_with_clamp_inputs(n_tokens, d, dtype, swiglu_limit)
        with ir.ops.silu_and_mul_with_clamp.set_priority([provider, "native"]):
            torch.library.opcheck(torch.ops.vllm_ir.silu_and_mul_with_clamp, (x, limit))
