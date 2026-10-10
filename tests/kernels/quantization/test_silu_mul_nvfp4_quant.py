# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused gated activation + NVFP4 quantization kernels
(silu_and_mul_nvfp4_quant, gelu_tanh_and_mul_nvfp4_quant)."""

import pytest
import torch
import torch.nn.functional as F

from tests.kernels.quantization.nvfp4_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    dequantize_nvfp4_to_dtype,
)
from vllm._custom_ops import scaled_fp4_quant
from vllm.model_executor.layers.activation import GeluAndMul, SiluAndMul
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="Nvfp4 Requires compute capability of 10 or above.",
        allow_module_level=True,
    )

FP4_DTYPE = torch.uint8
FP8_DTYPE = current_platform.fp8_dtype()

DTYPES = [torch.float16, torch.bfloat16]
SHAPES = [
    (128, 256),
    (128, 128),
    (256, 256),
    (256, 128),
    (1, 256),
    (64, 256),
]
# Gemma-4 26B-A4B dense MLP (2 * 2112) and Gemma-4 31B down_proj input
# (2 * 21504, grid.y > 1 in the kernel). Only the single-rounding test runs
# them: test_act_mul_nvfp4_quant compares with a reference that rounds twice
# (activation to dtype, then the product), and at 1e5 elements a one-code
# FP4 flip at large magnitude exceeds its tolerance for silu and gelu alike.
LARGE_SHAPES = SHAPES + [(3, 4224), (5, 43008)]
BLOCK_SIZE = 16

# activation -> (layer factory, unfused CUDA op, fused CUDA op)
ACTIVATIONS = {
    "silu": (SiluAndMul, "silu_and_mul", "silu_and_mul_nvfp4_quant"),
    "gelu_tanh": (
        lambda: GeluAndMul(approximate="tanh"),
        "gelu_tanh_and_mul",
        "gelu_tanh_and_mul_nvfp4_quant",
    ),
}


def _op(name: str):
    if not hasattr(torch.ops._C, name):
        pytest.skip(f"{name} is not compiled in")
    return getattr(torch.ops._C, name)


def _global_scale(t: torch.Tensor) -> torch.Tensor:
    return (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.abs(t).max().to(torch.float32)


@pytest.mark.parametrize("activation", list(ACTIVATIONS))
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@torch.inference_mode()
def test_act_mul_nvfp4_quant(
    default_vllm_config,
    activation: str,
    dtype: torch.dtype,
    shape: tuple[int, int],
) -> None:
    """The fused kernel dequantizes to the native activation quantized apart."""
    make_layer, _, fused_name = ACTIVATIONS[activation]
    fused = _op(fused_name)
    set_random_seed(42)
    device = "cuda:0"
    torch.set_default_device(device)

    x = torch.randn(shape, dtype=dtype)

    # ref op
    ref_output = make_layer().forward_native(x)
    ref_global_scale = _global_scale(ref_output)
    ref_output_quant, ref_block_scale = scaled_fp4_quant(ref_output, ref_global_scale)

    # fused op
    fused_output_quant = torch.empty_like(ref_output_quant)
    fused_block_scale = torch.empty_like(ref_block_scale)
    fused(fused_output_quant, fused_block_scale, x, ref_global_scale)

    # check dtype
    assert ref_output_quant.dtype == FP4_DTYPE
    assert fused_output_quant.dtype == FP4_DTYPE
    assert ref_output_quant.shape == fused_output_quant.shape

    assert ref_block_scale.dtype == FP8_DTYPE
    assert fused_block_scale.dtype == FP8_DTYPE
    assert ref_block_scale.shape == fused_block_scale.shape

    # check dequantized output
    ref_output_dequant = dequantize_nvfp4_to_dtype(
        ref_output_quant, ref_block_scale, ref_global_scale, dtype, device
    )
    fused_output_dequant = dequantize_nvfp4_to_dtype(
        fused_output_quant, fused_block_scale, ref_global_scale, dtype, device
    )

    atol, rtol = 3e-1, 3e-1
    torch.testing.assert_close(
        ref_output_dequant, fused_output_dequant, atol=atol, rtol=rtol
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", LARGE_SHAPES)
@torch.inference_mode()
def test_gelu_tanh_mul_nvfp4_quant_single_rounding(
    default_vllm_config,
    dtype: torch.dtype,
    shape: tuple[int, int],
) -> None:
    """The fused kernel evaluates gelu(gate) * up in fp32 and rounds once to the
    input dtype, like compute_silu_mul and the Inductor-fused native path
    (gelu_tanh_and_mul rounds the activation before the multiply, so it is not
    the reference here). Against an fp32 single-rounding reference the FP4
    codes must agree on all but a vanishing fraction of elements; the only
    expected source of differences is fma contraction inside the kernel, and
    such a difference is a single FP4 code step, so no magnitude bound is
    asserted on top of the fraction."""
    fused = _op("gelu_tanh_and_mul_nvfp4_quant")
    set_random_seed(0)
    device = "cuda:0"
    torch.set_default_device(device)

    x = torch.randn(shape, dtype=dtype)
    d = shape[1] // 2
    gate, up = x[:, :d].float(), x[:, d:].float()
    ref_act = (F.gelu(gate, approximate="tanh") * up).to(dtype)
    global_scale = _global_scale(ref_act)
    ref_quant, ref_scale = scaled_fp4_quant(ref_act, global_scale)

    out_quant = torch.empty_like(ref_quant)
    out_scale = torch.empty_like(ref_scale)
    fused(out_quant, out_scale, x, global_scale)

    # Compare through dequantization: it reads only the valid region of the
    # swizzled scale tensor, whose padding is uninitialized in both arms.
    ref_dequant = dequantize_nvfp4_to_dtype(
        ref_quant, ref_scale, global_scale, dtype, device
    )
    out_dequant = dequantize_nvfp4_to_dtype(
        out_quant, out_scale, global_scale, dtype, device
    )
    mismatch = (out_dequant != ref_dequant).float().mean().item()
    assert mismatch <= 1e-3, f"{mismatch:.2e} of the elements differ"
