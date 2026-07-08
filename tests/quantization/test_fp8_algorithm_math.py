# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.platforms import current_platform
from vllm.model_executor.kernels.linear.scaled_mm import (
    FP8ScaledMMLinearLayerConfig,
    FlashInferFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticTensorSym,
)
from vllm.utils.flashinfer import flashinfer_scaled_fp8_mm, has_flashinfer


FP8_MAX = 448.0


def _quant_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (x / scale).to(torch.float8_e4m3fn).contiguous()


def _dequant_mm(
    a_fp8: torch.Tensor,
    b_fp8: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    return (a_fp8.float() * scale_a) @ (b_fp8.float() * scale_b)


def _relative_metrics(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    return float(diff.max()), float(diff.mean())


def _static_w8a8_reference(
    x_bf16: torch.Tensor,
    w_fp8: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    x_fp8 = _quant_fp8(x_bf16, input_scale)
    return _dequant_mm(x_fp8, w_fp8, input_scale, weight_scale)


def _bf16_weight_only_reference(
    x_bf16: torch.Tensor,
    w_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    return x_bf16.float() @ (w_fp8.float() * weight_scale)


def _make_flashinfer_kernel(n: int, k: int) -> FlashInferFP8ScaledMMLinearKernel:
    return FlashInferFP8ScaledMMLinearKernel(
        FP8ScaledMMLinearLayerConfig(
            weight_quant_key=kFp8StaticTensorSym,
            activation_quant_key=kFp8StaticTensorSym,
            weight_shape=(n, k),
            input_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
        ),
        layer_param_names=[
            "weight",
            "weight_scale",
            "input_scale",
            "input_scale_ub",
        ],
    )


def _make_layer(
    w_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor,
) -> torch.nn.Module:
    class Layer(torch.nn.Module):
        pass

    layer = Layer()
    layer.weight = torch.nn.Parameter(w_fp8, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
    layer.input_scale = torch.nn.Parameter(input_scale, requires_grad=False)
    return layer


@pytest.mark.skipif(
    not current_platform.is_cuda() or not has_flashinfer(),
    reason="FP8 FlashInfer math checks require CUDA and FlashInfer.",
)
@pytest.mark.parametrize(
    ("m", "n", "k"),
    [
        (1, 512, 3584),
        (8, 512, 3584),
        (1, 3584, 512),
        (8, 3584, 512),
        (1, 3584, 3584),
    ],
)
def test_flashinfer_fp8_static_w8a8_matches_dequant_reference(m, n, k):
    torch.manual_seed(0)
    a_bf16 = (torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.02)
    b_bf16 = (torch.randn(k, n, device="cuda", dtype=torch.bfloat16) * 0.02)

    scale_a = (a_bf16.abs().max().float().clamp_min(1e-8) / FP8_MAX).view(1)
    scale_b = (b_bf16.abs().max().float().clamp_min(1e-8) / FP8_MAX).view(1)
    a_fp8 = _quant_fp8(a_bf16, scale_a)
    b_fp8 = _quant_fp8(b_bf16, scale_b)

    expected = _dequant_mm(a_fp8, b_fp8, scale_a, scale_b)
    actual = flashinfer_scaled_fp8_mm(
        a_fp8,
        b_fp8,
        scale_a,
        scale_b,
        out_dtype=torch.bfloat16,
    )

    max_abs, mean_abs = _relative_metrics(actual, expected)
    assert max_abs <= 2e-3
    assert mean_abs <= 5e-5


@pytest.mark.skipif(
    not current_platform.is_cuda() or not has_flashinfer(),
    reason="FP8 FlashInfer math checks require CUDA and FlashInfer.",
)
@pytest.mark.parametrize(
    ("m", "n", "k"),
    [
        (1, 512, 3584),
        (8, 512, 3584),
        (1, 3584, 512),
        (8, 3584, 512),
    ],
)
def test_flashinfer_fp8_linear_kernel_apply_weights_matches_reference(m, n, k):
    torch.manual_seed(1)
    x_bf16 = (torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.02)
    w_bf16 = (torch.randn(k, n, device="cuda", dtype=torch.bfloat16) * 0.02)

    input_scale = (x_bf16.abs().max().float().clamp_min(1e-8) / FP8_MAX).view(1)
    weight_scale = (w_bf16.abs().max().float().clamp_min(1e-8) / FP8_MAX).view(1)
    w_fp8 = _quant_fp8(w_bf16, weight_scale)

    layer = _make_layer(w_fp8, weight_scale, input_scale)

    with set_current_vllm_config(VllmConfig()):
        kernel = _make_flashinfer_kernel(n, k)
        expected = _static_w8a8_reference(x_bf16, w_fp8, input_scale, weight_scale)
        actual = kernel.apply_weights(layer, x_bf16)

    max_abs, mean_abs = _relative_metrics(actual, expected)
    assert max_abs <= 2e-3
    assert mean_abs <= 5e-5


@pytest.mark.skipif(
    not current_platform.is_cuda() or not has_flashinfer(),
    reason="FP8 FlashInfer math checks require CUDA and FlashInfer.",
)
@pytest.mark.parametrize("input_scale_factor", [0.25, 0.5])
def test_flashinfer_fp8_static_scale_too_small_is_large_error(input_scale_factor):
    torch.manual_seed(2)
    m, n, k = 8, 1024, 3584
    x_bf16 = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.02
    w_bf16 = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) * 0.02

    calibrated_input_scale = (
        x_bf16.abs().max().float().clamp_min(1e-8) / FP8_MAX
    ).view(1)
    runtime_input_scale = calibrated_input_scale * input_scale_factor
    weight_scale = (w_bf16.abs().max().float().clamp_min(1e-8) / FP8_MAX).view(1)
    w_fp8 = _quant_fp8(w_bf16, weight_scale)

    layer = _make_layer(w_fp8, weight_scale, runtime_input_scale)
    with set_current_vllm_config(VllmConfig()):
        actual = _make_flashinfer_kernel(n, k).apply_weights(layer, x_bf16)

    weight_only_reference = _bf16_weight_only_reference(x_bf16, w_fp8, weight_scale)
    _, w8a8_vs_w8a16_mean = _relative_metrics(actual, weight_only_reference)
    assert w8a8_vs_w8a16_mean > 1e-3


@pytest.mark.skipif(
    not current_platform.is_cuda() or not has_flashinfer(),
    reason="FP8 FlashInfer math checks require CUDA and FlashInfer.",
)
@pytest.mark.parametrize("input_scale_factor", [2.0, 16.0])
def test_flashinfer_fp8_static_scale_too_large_matches_static_contract(
    input_scale_factor,
):
    torch.manual_seed(2)
    m, n, k = 8, 1024, 3584
    x_bf16 = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.02
    w_bf16 = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) * 0.02

    calibrated_input_scale = (
        x_bf16.abs().max().float().clamp_min(1e-8) / FP8_MAX
    ).view(1)
    runtime_input_scale = calibrated_input_scale * input_scale_factor
    weight_scale = (w_bf16.abs().max().float().clamp_min(1e-8) / FP8_MAX).view(1)
    w_fp8 = _quant_fp8(w_bf16, weight_scale)

    layer = _make_layer(w_fp8, weight_scale, runtime_input_scale)
    with set_current_vllm_config(VllmConfig()):
        actual = _make_flashinfer_kernel(n, k).apply_weights(layer, x_bf16)

    static_reference = _static_w8a8_reference(
        x_bf16, w_fp8, runtime_input_scale, weight_scale
    )
    weight_only_reference = _bf16_weight_only_reference(x_bf16, w_fp8, weight_scale)

    max_abs, mean_abs = _relative_metrics(actual, static_reference)
    assert max_abs <= 2e-3
    assert mean_abs <= 5e-5

    _, w8a8_vs_w8a16_mean = _relative_metrics(static_reference, weight_only_reference)
    assert w8a8_vs_w8a16_mean > 1e-9


@pytest.mark.skipif(
    not current_platform.is_cuda() or not has_flashinfer(),
    reason="FP8 FlashInfer math checks require CUDA and FlashInfer.",
)
def test_flashinfer_fp8_prequantized_activation_scale_is_trusted():
    torch.manual_seed(3)
    m, n, k = 8, 1024, 3584
    x_bf16 = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.02
    w_bf16 = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) * 0.02

    true_input_scale = (x_bf16.abs().max().float().clamp_min(1e-8) / FP8_MAX).view(1)
    wrong_input_scale = true_input_scale * 1024.0
    weight_scale = (w_bf16.abs().max().float().clamp_min(1e-8) / FP8_MAX).view(1)
    w_fp8 = _quant_fp8(w_bf16, weight_scale)
    x_fp8_wrong = _quant_fp8(x_bf16, wrong_input_scale)

    qa = QuantizedActivation(
        data=x_fp8_wrong,
        scale=wrong_input_scale,
        orig_dtype=x_bf16.dtype,
        orig_shape=x_bf16.shape,
        quant_key=kFp8StaticTensorSym,
    )
    layer = _make_layer(w_fp8, weight_scale, true_input_scale)

    with set_current_vllm_config(VllmConfig()):
        actual = _make_flashinfer_kernel(n, k).apply_weights(layer, qa)

    trusted_qa_reference = _dequant_mm(
        x_fp8_wrong, w_fp8, wrong_input_scale, weight_scale
    )
    layer_scale_reference = _static_w8a8_reference(
        x_bf16, w_fp8, true_input_scale, weight_scale
    )

    max_abs, mean_abs = _relative_metrics(actual, trusted_qa_reference)
    assert max_abs <= 2e-3
    assert mean_abs <= 5e-5

    _, trusted_vs_layer_mean = _relative_metrics(
        trusted_qa_reference, layer_scale_reference
    )
    assert trusted_vs_layer_mean > 1e-6


@pytest.mark.skipif(
    not current_platform.is_cuda() or not has_flashinfer(),
    reason="FP8 FlashInfer compile checks require CUDA and FlashInfer.",
)
@pytest.mark.parametrize(
    ("m", "n", "k"),
    [
        (8, 512, 3584),
        (8, 3584, 512),
        (8, 3584, 3584),
    ],
)
def test_flashinfer_fp8_linear_kernel_torch_compile_matches_eager(m, n, k):
    torch.manual_seed(4)
    x_bf16 = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.02
    w_bf16 = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) * 0.02

    input_scale = (x_bf16.abs().max().float().clamp_min(1e-8) / FP8_MAX).view(1)
    weight_scale = (w_bf16.abs().max().float().clamp_min(1e-8) / FP8_MAX).view(1)
    w_fp8 = _quant_fp8(w_bf16, weight_scale)
    layer = _make_layer(w_fp8, weight_scale, input_scale)

    with set_current_vllm_config(VllmConfig()):
        kernel = _make_flashinfer_kernel(n, k)

        def apply(x: torch.Tensor) -> torch.Tensor:
            return kernel.apply_weights(layer, x)

        eager = apply(x_bf16)
        compiled_apply = torch.compile(apply, fullgraph=False)
        # Run twice: the first call compiles, the second exercises the cached
        # graph that is closer to the no-eager server path.
        compiled_apply(x_bf16)
        compiled = compiled_apply(x_bf16)

    max_abs, mean_abs = _relative_metrics(compiled, eager)
    assert torch.isfinite(compiled.float()).all()
    assert max_abs <= 2e-3
    assert mean_abs <= 5e-5
