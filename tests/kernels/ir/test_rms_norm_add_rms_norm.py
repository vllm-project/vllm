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
from tests.utils import set_random_seed
from vllm import ir
from vllm.platforms import current_platform

pytestmark = pytest.mark.skip_global_cleanup

OP = ir.ops.rms_norm_add_rms_norm
native = OP.impls["native"].impl_fn
rms_norm_native = ir.ops.rms_norm.impls["native"].impl_fn
fused_add_rms_norm_native = ir.ops.fused_add_rms_norm.impls["native"].impl_fn

IS_GPGPU_DEVICE = current_platform.is_cuda_alike() or current_platform.is_xpu()
DEVICE = current_platform.device_type


def test_registration():
    expected = {"native": True, "triton": current_platform.is_cuda()}
    actual = {provider: impl.supported for provider, impl in OP.impls.items()}
    assert actual == expected


def test_native_is_the_composition():
    """The op is defined as rms_norm followed by fused_add_rms_norm."""
    set_random_seed(0)
    x, x_residual, weight, weight_residual, eps = OP.generate_inputs(
        num_tokens=4, hidden_size=8, dtype=torch.float32, device="cpu"
    )
    out, residual_out = native(x, x_residual, weight, weight_residual, eps)
    expected = fused_add_rms_norm_native(
        rms_norm_native(x, weight, eps), x_residual, weight_residual, eps
    )
    torch.testing.assert_close(out, expected[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(residual_out, expected[1], rtol=0.0, atol=0.0)

    # weight=None behaves like a unit weight
    out_none, res_none = native(x, x_residual, None, None, eps)
    out_ones, res_ones = native(
        x, x_residual, torch.ones_like(weight), torch.ones_like(weight), eps
    )
    torch.testing.assert_close(out_none, out_ones)
    torch.testing.assert_close(res_none, res_ones)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", COMMON_HIDDEN_SIZES)
@pytest.mark.parametrize("epsilon", [1e-6, 1e-5])
@pytest.mark.skipif(not IS_GPGPU_DEVICE, reason="Kernels need a GPU")
class TestRMSNormAddRMSNorm:
    @pytest.mark.parametrize("provider", supported_providers(OP))
    def test_impls(self, dtype, n_tokens, hidden_size, epsilon, provider):
        set_random_seed(0)
        impl = OP.impls[provider]
        args = OP.generate_inputs(
            num_tokens=n_tokens,
            hidden_size=hidden_size,
            dtype=dtype,
            epsilon=epsilon,
            device=DEVICE,
        )
        if not impl.supports_args(*args):
            pytest.skip(f"{provider} does not support args")

        ref = native(*clone_args(args))
        out = impl.impl_fn(*clone_args(args))
        assert_close(OP, out, ref)

        # dispatched call matches the direct call
        with OP.set_priority([provider, "native"]):
            out_dispatched = OP(*args)
        out_direct = impl.impl_fn(*args)
        torch.testing.assert_close(out_dispatched, out_direct, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("provider", supported_providers(OP))
    def test_impls_fp32_weights(self, dtype, n_tokens, hidden_size, epsilon, provider):
        """GemmaRMSNorm passes fp32 (1 + w) weights with a low-precision x."""
        set_random_seed(0)
        impl = OP.impls[provider]
        x, x_residual, weight, weight_residual, eps = OP.generate_inputs(
            num_tokens=n_tokens,
            hidden_size=hidden_size,
            dtype=dtype,
            epsilon=epsilon,
            device=DEVICE,
        )
        args = (x, x_residual, weight.float() + 1.0, weight_residual.float() + 1.0, eps)
        if not impl.supports_args(*args):
            pytest.skip(f"{provider} does not support args")

        ref = native(*clone_args(args))
        out = impl.impl_fn(*clone_args(args))
        assert_close(OP, out, ref)

    @pytest.mark.parametrize("provider", supported_providers(OP))
    def test_impls_no_weights(self, dtype, n_tokens, hidden_size, epsilon, provider):
        set_random_seed(0)
        impl = OP.impls[provider]
        x, x_residual, _, _, eps = OP.generate_inputs(
            num_tokens=n_tokens,
            hidden_size=hidden_size,
            dtype=dtype,
            epsilon=epsilon,
            device=DEVICE,
        )
        args = (x, x_residual, None, None, eps)
        if not impl.supports_args(*args):
            pytest.skip(f"{provider} does not support args")

        ref = native(*clone_args(args))
        out = impl.impl_fn(*clone_args(args))
        assert_close(OP, out, ref)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Triton impl is CUDA-only")
def test_triton_supports_args():
    impl = OP.impls["triton"]
    x = torch.randn(4, 64, dtype=torch.bfloat16, device=DEVICE)
    r = torch.randn(4, 64, dtype=torch.bfloat16, device=DEVICE)
    w = torch.randn(64, dtype=torch.bfloat16, device=DEVICE)
    assert impl.supports_args(x, r, w, w, 1e-6)
    assert impl.supports_args(x, r, w.float(), None, 1e-6)
    assert impl.supports_args(x.view(2, 2, 64), r.view(2, 2, 64), w, w, 1e-6)
    # residual dtype must match
    assert not impl.supports_args(x, r.float(), w, w, 1e-6)
    # weights must be 1-D of hidden size, in x's dtype or fp32
    assert not impl.supports_args(x, r, w.half(), w, 1e-6)
    assert not impl.supports_args(x, r, w[:32], w, 1e-6)
    # non-contiguous inputs are left to the native implementation
    assert not impl.supports_args(x.t().contiguous().t(), r, w, w, 1e-6)
    # hidden size cap
    big = torch.randn(1, 32768, dtype=torch.bfloat16, device=DEVICE)
    assert not impl.supports_args(big, big, None, None, 1e-6)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Triton impl is CUDA-only")
@pytest.mark.parametrize("gemma", [False, True])
def test_layer_helper_matches_module_sequence(gemma):
    """rms_norm_add_rms_norm(post, pre, x, r) == pre(post(x), r) on the modules."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import (
        GemmaRMSNorm,
        RMSNorm,
        rms_norm_add_rms_norm,
    )

    set_random_seed(0)
    hidden = 2048
    with set_current_vllm_config(VllmConfig()):
        if gemma:
            post = GemmaRMSNorm(hidden, eps=1e-6).to(DEVICE)
            pre = GemmaRMSNorm(hidden, eps=1e-6).to(DEVICE)
            post.weight.data.normal_(std=0.1)
            pre.weight.data.normal_(std=0.1)
        else:
            post = RMSNorm(hidden, eps=1e-6, dtype=torch.bfloat16).to(DEVICE)
            pre = RMSNorm(hidden, eps=1e-6, dtype=torch.bfloat16).to(DEVICE)
            post.weight.data.normal_(mean=1.0, std=0.1)
            pre.weight.data.normal_(mean=1.0, std=0.1)
        x = torch.randn(16, hidden, dtype=torch.bfloat16, device=DEVICE)
        r = torch.randn(16, hidden, dtype=torch.bfloat16, device=DEVICE) * 4

        ref_out, ref_res = pre(post(x.clone()), r.clone())
        with OP.set_priority(["triton", "native"]):
            out, res = rms_norm_add_rms_norm(post, pre, x.clone(), r.clone())
        assert_close(OP, (out, res), (ref_out, ref_res))
