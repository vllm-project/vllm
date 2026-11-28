# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util

import pytest
import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
    AiterPerTokenFp8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
)
from vllm.platforms import current_platform

aiter_available = importlib.util.find_spec("aiter") is not None

pytestmark = [
    pytest.mark.skipif(
        not (
            current_platform.is_rocm()
            and current_platform.supports_fp8()
            and not current_platform.is_fp8_fnuz()
            and aiter_available
        ),
        reason="Requires ROCm + OCP FP8 support + aiter",
    ),
    pytest.mark.usefixtures("default_vllm_config"),
]

# gfx950 (N, K) that AITER has tuned: CK `gemm_a8w8` CSV rows only, and Triton
# `gemm_a8w8_per_token_scale` configs only. Neither has preshuffled CSV rows,
# which an earlier kernel in the list would take.
CK_SHAPE = (3584, 8192)
TRITON_SHAPE = (32768, 8192)
UNTUNED_SHAPE = (1232, 4096)


def _skip_unless_gfx950() -> None:
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        pytest.skip("AITER's tuned configs for these shapes are for gfx950")


@pytest.fixture
def enable_kernel(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_LINEAR", "1")
    rocm_aiter_ops.refresh_env_variables()
    yield
    monkeypatch.undo()
    rocm_aiter_ops.refresh_env_variables()


def _make_layer(weight_shape: tuple[int, int]) -> torch.nn.Module:
    n, k = weight_shape
    layer = torch.nn.Module()
    # Online fp8_per_channel stores the weight as a [K, N] view of the [N, K]
    # tensor and one scale per output channel, [N, 1].
    w = (torch.randn(n, k, device="cuda") * 4).to(current_platform.fp8_dtype())
    layer.weight = torch.nn.Parameter(w.t(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(
        torch.rand(n, 1, dtype=torch.float32, device="cuda") * 0.01 + 0.005,
        requires_grad=False,
    )
    layer.input_scale = None
    layer.input_scale_ub = None
    return layer


# (N, K), output dtype, whether AITER's per-token kernel should take it. The
# untuned shape and fp16 output fall back to another kernel, with the same
# result.
_FORWARD_CASES = [
    pytest.param(CK_SHAPE, torch.bfloat16, True, id="ck-tuned"),
    pytest.param(TRITON_SHAPE, torch.bfloat16, True, id="triton-tuned"),
    pytest.param(UNTUNED_SHAPE, torch.bfloat16, False, id="untuned"),
    pytest.param(TRITON_SHAPE, torch.float16, False, id="fp16-out"),
]


@pytest.mark.parametrize("weight_shape, out_dtype, tuned", _FORWARD_CASES)
@pytest.mark.parametrize("num_tokens", [1, 8, 9, 64, 65, 256, 257, 1024, 1025])
@pytest.mark.parametrize("prequantized", [False, True])
def test_forward_matches_reference(
    enable_kernel,
    weight_shape: tuple[int, int],
    out_dtype: torch.dtype,
    tuned: bool,
    num_tokens: int,
    prequantized: bool,
):
    """The FP8 per-token-activation, per-channel-weight linear that vLLM selects
    matches the dequantized reference for tuned and untuned shapes."""
    _skip_unless_gfx950()
    if prequantized and not tuned:
        pytest.skip("pre-quantized activations are for the per-token kernel")
    kernel = init_fp8_linear_kernel(
        activation_quant_key=kFp8DynamicTokenSym,
        weight_quant_key=kFp8StaticChannelSym,
        input_dtype=torch.bfloat16,
        out_dtype=out_dtype,
        weight_shape=weight_shape,
    )
    if tuned:
        # The accuracy check cannot tell kernels apart; the tuned shapes are
        # meant for AITER's per-token GEMM.
        assert isinstance(kernel, AiterPerTokenFp8ScaledMMLinearKernel)
    layer = _make_layer(weight_shape)
    w_dq = layer.weight.t().float() * layer.weight_scale
    kernel.process_weights_after_loading(layer)

    x = torch.randn(num_tokens, weight_shape[1], dtype=torch.bfloat16, device="cuda")
    x_q, x_s = kernel.quant_fp8(x)
    if prequantized:
        x = QuantizedActivation(
            data=x_q,
            scale=x_s,
            orig_dtype=x.dtype,
            orig_shape=x.shape,
            quant_key=kFp8DynamicTokenSym,
        )
    out = kernel.apply_weights(layer, x)

    assert out.shape == (num_tokens, weight_shape[0])
    assert out.dtype == out_dtype
    ref = (x_q.float() * x_s) @ w_dq.t()
    assert (out.float() - ref).norm() / ref.norm() < 5e-3


@pytest.mark.parametrize("num_tokens", [8, 300])
def test_rocm_aiter_triton_per_token_fp8_gemm_fake_implementation(num_tokens: int):
    n, k = TRITON_SHAPE
    layer = _make_layer((n, k))
    x_q = torch.randn(num_tokens, k, device="cuda").to(current_platform.fp8_dtype())
    x_s = torch.rand(num_tokens, 1, dtype=torch.float32, device="cuda")

    torch.library.opcheck(
        torch.ops.vllm.rocm_aiter_triton_per_token_fp8_gemm,
        (
            x_q,
            layer.weight.t(),
            torch.bfloat16,
            x_s,
            layer.weight_scale,
            None,
            AiterPerTokenFp8ScaledMMLinearKernel.TRITON_MAX_M,
        ),
        test_utils=("test_faketensor",),
    )
