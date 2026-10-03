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


def test_native_rounds_residual_before_second_norm():
    x = torch.tensor([[1.0, 1.0]], dtype=torch.bfloat16)
    x_residual = torch.tensor([[-0.25, 0.01171875]], dtype=torch.bfloat16)
    weight = torch.ones(2, dtype=torch.bfloat16)
    epsilon = 1e-6

    out, residual_out = native(
        x,
        x_residual,
        weight,
        weight,
        epsilon,
        round_residual_before_norm=True,
    )
    post_norm = rms_norm_native(x, weight, epsilon)
    expected_residual = post_norm + x_residual
    expected_out = rms_norm_native(expected_residual, weight, epsilon)
    unrounded_out, _ = native(x, x_residual, weight, weight, epsilon)

    torch.testing.assert_close(residual_out, expected_residual, rtol=0.0, atol=0.0)
    torch.testing.assert_close(out, expected_out, rtol=0.0, atol=0.0)
    assert not torch.equal(out, unrounded_out)


def test_layer_helper_rounded_fallback_matches_module_sequence():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import (
        GemmaRMSNorm,
        rms_norm_add_rms_norm,
    )

    with set_current_vllm_config(VllmConfig()):
        post_norm = GemmaRMSNorm(2, eps=1e-6)
        pre_norm = GemmaRMSNorm(2, eps=1e-5)
        x = torch.tensor([[1.0, 1.0]], dtype=torch.bfloat16)
        residual = torch.tensor([[-0.25, 0.01171875]], dtype=torch.bfloat16)

        post_out = post_norm(x)
        expected_residual = post_out + residual
        expected_out = pre_norm(expected_residual)
        out, residual_out = rms_norm_add_rms_norm(
            post_norm,
            pre_norm,
            x,
            residual,
            round_residual_before_norm=True,
        )

    torch.testing.assert_close(out, expected_out, rtol=0.0, atol=0.0)
    torch.testing.assert_close(residual_out, expected_residual, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", COMMON_HIDDEN_SIZES)
@pytest.mark.parametrize("epsilon", [1e-6, 1e-5])
@pytest.mark.skipif(not IS_GPGPU_DEVICE, reason="Kernels need a GPU")
class TestRMSNormAddRMSNorm:
    @pytest.mark.parametrize("round_residual_before_norm", [False, True])
    @pytest.mark.parametrize("provider", supported_providers(OP))
    def test_impls(
        self,
        dtype,
        n_tokens,
        hidden_size,
        epsilon,
        provider,
        round_residual_before_norm,
    ):
        set_random_seed(0)
        impl = OP.impls[provider]
        args = (
            *OP.generate_inputs(
                num_tokens=n_tokens,
                hidden_size=hidden_size,
                dtype=dtype,
                epsilon=epsilon,
                device=DEVICE,
            ),
            round_residual_before_norm,
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
    # Inputs and weights on another device are left to the native provider.
    assert not impl.supports_args(x, r.cpu(), w, w, 1e-6)
    assert not impl.supports_args(x, r, w.cpu(), w, 1e-6)
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
@pytest.mark.parametrize("round_residual_before_norm", [False, True])
def test_layer_helper_matches_module_sequence(gemma, round_residual_before_norm):
    """The helper preserves each architecture's residual rounding boundary."""
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

        post_out = post(x.clone())
        if round_residual_before_norm:
            ref_res = post_out + r.clone()
            ref_out = pre(ref_res)
        else:
            ref_out, ref_res = pre(post_out, r.clone())
        with OP.set_priority(["triton", "native"]):
            out, res = rms_norm_add_rms_norm(
                post,
                pre,
                x.clone(),
                r.clone(),
                round_residual_before_norm=round_residual_before_norm,
            )
        assert_close(OP, (out, res), (ref_out, ref_res))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Triton impl is CUDA-only")
@pytest.mark.parametrize("round_residual_before_norm", [False, True])
@pytest.mark.parametrize("n_tokens", [1, 16])
def test_compiled_op_specializations(
    n_tokens, round_residual_before_norm, default_vllm_config
):
    """Compile each residual rounding mode at decode and short-prefill shapes."""
    from tests.compile.backend import TestBackend
    from vllm.compilation.passes.ir.lowering_pass import VllmIRLoweringPass
    from vllm.config import get_current_vllm_config

    set_random_seed(0)
    hidden_size = 5376
    x = torch.randn(n_tokens, hidden_size, dtype=torch.bfloat16, device=DEVICE)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=torch.bfloat16, device=DEVICE)
    weight_residual = torch.randn_like(weight)
    triton_impl = OP.impls["triton"]
    assert triton_impl.supported
    assert triton_impl.supports_args(
        x,
        residual,
        weight,
        weight_residual,
        1e-6,
        round_residual_before_norm,
    )

    def boundary(x_i, residual_i):
        return OP(
            x_i,
            residual_i,
            weight,
            weight_residual,
            1e-6,
            round_residual_before_norm=round_residual_before_norm,
        )

    lowering_pass = VllmIRLoweringPass(get_current_vllm_config())
    backend = TestBackend(lowering_pass)
    with (
        OP.set_priority(["triton"]),
        ir.enable_torch_wrap(True),
    ):
        compiled = torch.compile(boundary, backend=backend, fullgraph=True)
        actual = compiled(x, residual)
    backend.check_before_ops([OP.torch_op])
    assert lowering_pass.selected_impls[OP.name] == {OP.name: "triton"}
    expected = native(
        x,
        residual,
        weight,
        weight_residual,
        1e-6,
        round_residual_before_norm,
    )
    assert_close(OP, actual, expected)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Triton impl is CUDA-only")
def test_triton_rounding_mode_changes_low_precision_result():
    x = torch.ones((1, 2), dtype=torch.bfloat16, device=DEVICE)
    residual = torch.tensor([[-0.25, 0.01171875]], dtype=torch.bfloat16, device=DEVICE)
    weight = torch.ones(2, dtype=torch.bfloat16, device=DEVICE)
    args = (x, residual, weight, weight, 1e-6)
    triton_impl = OP.impls["triton"]
    assert triton_impl.supported
    assert triton_impl.supports_args(*args, False)
    assert triton_impl.supports_args(*args, True)

    with OP.set_priority(["triton"]):
        unrounded = OP(*args, False)
        rounded = OP(*args, True)
    expected_rounded = native(*args, True)

    assert not torch.equal(rounded[0], unrounded[0])
    torch.testing.assert_close(rounded[0], expected_rounded[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(rounded[1], expected_rounded[1], rtol=0.0, atol=0.0)
