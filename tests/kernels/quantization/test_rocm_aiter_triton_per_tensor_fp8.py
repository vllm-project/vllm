# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util

import pytest
import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
    AiterTritonPerTensorFp8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.rocm import (
    ROCmFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
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

PARAM_NAMES = ("weight", "weight_scale", "input_scale", "input_scale_ub")
# Llama 3.1 8B decode GEMMs: qkv, o_proj, gate_up, down.
LLAMA_8B_SHAPES = [(6144, 4096), (4096, 4096), (28672, 4096), (4096, 14336)]


@pytest.fixture
def enable_kernel(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_LINEAR", "1")
    rocm_aiter_ops.refresh_env_variables()
    yield
    monkeypatch.undo()
    rocm_aiter_ops.refresh_env_variables()


def _make_config(
    *,
    activation_quant_key=kFp8StaticTensorSym,
    weight_quant_key=kFp8StaticTensorSym,
    out_dtype: torch.dtype = torch.bfloat16,
    weight_shape: tuple[int, int] = (4096, 4096),
) -> FP8ScaledMMLinearLayerConfig:
    return FP8ScaledMMLinearLayerConfig(
        weight_quant_key=weight_quant_key,
        activation_quant_key=activation_quant_key,
        weight_shape=weight_shape,
        input_dtype=torch.bfloat16,
        out_dtype=out_dtype,
    )


def _make_layer(weight_shape: tuple[int, int]) -> torch.nn.Module:
    n, k = weight_shape
    layer = torch.nn.Module()
    # Fp8LinearMethod stores the weight as a [K, N] view of the [N, K] tensor
    # and a single per-tensor weight and input scale.
    w = torch.randn(n, k, device="cuda").to(current_platform.fp8_dtype())
    layer.weight = torch.nn.Parameter(w.t(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(
        torch.tensor(0.01, dtype=torch.float32, device="cuda"), requires_grad=False
    )
    layer.input_scale = torch.nn.Parameter(
        torch.tensor(0.02, dtype=torch.float32, device="cuda"), requires_grad=False
    )
    layer.input_scale_ub = None
    return layer


@pytest.mark.parametrize(
    ("config", "expected_reason"),
    [
        (
            _make_config(activation_quant_key=kFp8DynamicTokenSym),
            "requires per tensor activation and weight scales.",
        ),
        (
            _make_config(weight_quant_key=kFp8StaticChannelSym),
            "requires per tensor activation and weight scales.",
        ),
        (
            _make_config(activation_quant_key=kFp8DynamicTensorSym),
            "requires static activation scales.",
        ),
        (
            _make_config(out_dtype=torch.float32),
            "requires bfloat16 output dtype.",
        ),
    ],
)
def test_can_implement_rejects_unsupported_configs(
    enable_kernel,
    config: FP8ScaledMMLinearLayerConfig,
    expected_reason: str,
):
    can_implement, reason = AiterTritonPerTensorFp8ScaledMMLinearKernel.can_implement(
        config
    )
    assert not can_implement
    assert reason == expected_reason


def test_can_implement_rejects_untuned_shape(enable_kernel):
    can_implement, reason = AiterTritonPerTensorFp8ScaledMMLinearKernel.can_implement(
        _make_config(weight_shape=(1232, 4096))
    )
    assert not can_implement
    assert reason is not None
    assert reason.startswith("requires a tuned configuration for N: 1232")


@pytest.mark.parametrize(
    ("weight_shape", "expected_kernel"),
    [
        ((4096, 4096), AiterTritonPerTensorFp8ScaledMMLinearKernel),
        ((1232, 4096), ROCmFP8ScaledMMLinearKernel),
    ],
)
def test_default_selection(
    enable_kernel,
    weight_shape: tuple[int, int],
    expected_kernel: type,
):
    kernel = init_fp8_linear_kernel(
        activation_quant_key=kFp8StaticTensorSym,
        weight_quant_key=kFp8StaticTensorSym,
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        weight_shape=weight_shape,
    )
    assert type(kernel) is expected_kernel


@pytest.mark.parametrize("weight_shape", LLAMA_8B_SHAPES)
@pytest.mark.parametrize("num_tokens", [1, 4, 5, 8, 16, 32, 64, 128, 256, 257])
def test_forward_matches_scaled_mm(
    enable_kernel,
    monkeypatch: pytest.MonkeyPatch,
    weight_shape: tuple[int, int],
    num_tokens: int,
):
    import aiter.ops.triton.gemm.basic.gemm_a8w8 as aiter_gemm_a8w8

    triton_calls = []
    original_gemm_a8w8 = aiter_gemm_a8w8.gemm_a8w8

    def counting_gemm_a8w8(*args, **kwargs):
        triton_calls.append(args[0].shape[0])
        return original_gemm_a8w8(*args, **kwargs)

    monkeypatch.setattr(aiter_gemm_a8w8, "gemm_a8w8", counting_gemm_a8w8)

    kernel = AiterTritonPerTensorFp8ScaledMMLinearKernel(
        _make_config(weight_shape=weight_shape), layer_param_names=PARAM_NAMES
    )
    layer = _make_layer(weight_shape)
    kernel.process_weights_after_loading(layer)

    x = torch.randn(num_tokens, weight_shape[1], dtype=torch.bfloat16, device="cuda")
    out = kernel.apply_weights(layer, x)

    x_q, x_s = kernel.quant_fp8(x, layer.input_scale, layer.input_scale_ub)
    expected = torch._scaled_mm(
        x_q,
        layer.weight,
        out_dtype=torch.bfloat16,
        scale_a=x_s,
        scale_b=layer.weight_scale,
    )

    assert out.shape == (num_tokens, weight_shape[0])
    torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)
    expect_triton = 5 <= num_tokens <= AiterTritonPerTensorFp8ScaledMMLinearKernel.MAX_M
    assert triton_calls == ([num_tokens] if expect_triton else [])


@pytest.mark.parametrize("num_tokens", [8, 300])
def test_rocm_aiter_triton_per_tensor_fp8_gemm_fake_implementation(num_tokens: int):
    n, k = 4096, 4096
    max_m = AiterTritonPerTensorFp8ScaledMMLinearKernel.MAX_M
    layer = _make_layer((n, k))
    x_q = torch.randn(num_tokens, k, device="cuda").to(current_platform.fp8_dtype())
    x_scale_vec = layer.input_scale.reshape(1).expand(max_m).contiguous()
    w_scale_vec = layer.weight_scale.reshape(1).expand(n).contiguous()

    torch.library.opcheck(
        torch.ops.vllm.rocm_aiter_triton_per_tensor_fp8_gemm,
        (
            x_q,
            layer.weight,
            torch.bfloat16,
            layer.input_scale,
            layer.weight_scale,
            None,
            x_scale_vec,
            w_scale_vec,
            max_m,
        ),
        test_utils=("test_faketensor",),
    )
