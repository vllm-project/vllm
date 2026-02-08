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
    not current_platform.is_cuda_alike(),
    reason="Currently only kernels on CUDA and ROCm",
)
def test_rms_norm_registration():
    expected = {"native": True, "vllm_c": True, "aiter": current_platform.is_rocm()}

    actual = {
        provider: impl.supported for provider, impl in ir.ops.rms_norm.impls.items()
    }

    assert actual == expected


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n_tokens", [1, 8, 17])
@pytest.mark.parametrize("hidden_size", [16, 4096, 8192])
@pytest.mark.parametrize("epsilon", [1e-6, 1e-5])
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Currently only kernels on CUDA and ROCm",
)
class TestRMSNorm:
    @classmethod
    def setup_class(cls, **kwargs):
        torch.set_default_device("cuda")

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

    @pytest.mark.parametrize("provider", ["vllm_c", "aiter"])
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
            out_impl, out_native, rtol=get_default_rtol(out_impl), atol=2e-4
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

    @pytest.mark.parametrize("provider", ["vllm_c", "aiter", "native"])
    def test_torch_opcheck(self, dtype, n_tokens, hidden_size, epsilon, provider):
        if not ir.ops.rms_norm.impls[provider].supported:
            pytest.skip(f"{provider} impl not supported on this platform")

        x, weight = rms_norm_inputs(n_tokens, hidden_size, dtype)
        args = (x, weight, epsilon, None)

        # When checking the torch op, we have to set priority and use dispatch
        with ir.ops.rms_norm.set_priority([provider, "native"]):
            torch.library.opcheck(torch.ops.vllm_ir.rms_norm, args)
