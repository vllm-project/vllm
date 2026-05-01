# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

# This registers op implementations
import vllm.kernels  # noqa: F401
from tests.ir.ir_test_utils import assert_close, clone_args, supported_providers
from vllm import ir
from vllm.platforms import current_platform

RELU2_HIDDEN_SIZES = [16, 128, 1024]
RELU2_NUM_TOKENS = [1, 17, 512]
VLLM_C_SUPPORTED = current_platform.is_cuda_alike() or current_platform.is_cpu()
relu2_native = ir.ops.relu2.impls["native"].impl_fn


def test_relu2_registration():
    expected = {
        "native": True,
        "vllm_c": VLLM_C_SUPPORTED,
        "xpu_kernels": False,
    }

    actual = {
        provider: impl.supported for provider, impl in ir.ops.relu2.impls.items()
    }

    assert actual == expected


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n_tokens", RELU2_NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", RELU2_HIDDEN_SIZES)
class TestRelu2:
    @classmethod
    def setup_class(cls, **kwargs):
        torch.set_default_device(current_platform.device_type)

    def test_native_semantics(self, dtype, n_tokens, hidden_size):
        (x,) = ir.ops.relu2.generate_inputs(
            num_tokens=n_tokens, hidden_size=hidden_size, dtype=dtype
        )
        out = relu2_native(x)

        assert out.shape == x.shape
        assert out.dtype == x.dtype
        assert out.device == x.device
        torch.testing.assert_close(
            out, torch.square(torch.relu(x)), rtol=0.0, atol=0.0
        )

    @pytest.mark.parametrize("provider", supported_providers(ir.ops.relu2))
    def test_impls(self, dtype, n_tokens, hidden_size, provider):
        impl = ir.ops.relu2.impls[provider]
        args = ir.ops.relu2.generate_inputs(
            num_tokens=n_tokens, hidden_size=hidden_size, dtype=dtype
        )

        if not impl.supports_args(*args):
            pytest.skip(f"{provider} does not support args")

        ref_output = relu2_native(*clone_args(args))
        output = impl.impl_fn(*clone_args(args))
        assert_close(ir.ops.relu2, output, ref_output)

        with ir.ops.relu2.set_priority([provider, "native"]):
            out_dispatched = ir.ops.relu2(*args)
        out_direct = impl.impl_fn(*args)
        torch.testing.assert_close(out_dispatched, out_direct, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("provider", ["vllm_c", "native"])
    def test_preserves_nan(self, dtype, n_tokens, hidden_size, provider):
        impl = ir.ops.relu2.impls[provider]
        if not impl.supported:
            pytest.skip(f"{provider} impl not supported on this platform")

        x = torch.tensor(
            [
                float("nan"),
                -2.0,
                -0.0,
                0.0,
                1.5,
                float("nan"),
                3.0,
                -4.0,
                2.5,
                -1.0,
            ],
            dtype=dtype,
            device=current_platform.device_type,
        ).reshape(2, 5)

        expected = relu2_native(x.clone())
        output = impl.impl_fn(x.clone())
        torch.testing.assert_close(
            output, expected, rtol=0.0, atol=0.0, equal_nan=True
        )

    @pytest.mark.parametrize("provider", ["vllm_c", "native"])
    def test_torch_opcheck(self, dtype, n_tokens, hidden_size, provider):
        if not ir.ops.relu2.impls[provider].supported:
            pytest.skip(f"{provider} impl not supported on this platform")

        args = ir.ops.relu2.generate_inputs(
            num_tokens=n_tokens, hidden_size=hidden_size, dtype=dtype
        )

        with ir.ops.relu2.set_priority([provider, "native"]):
            torch.library.opcheck(torch.ops.vllm_ir.relu2, args)
