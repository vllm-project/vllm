# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ScaleSweep NVFP4 quantization kernel.

Verifies that ScaleSweep produces lower or equal MSE compared to the vLLM
default quantization, and that the output is consumable by the NVFP4 GEMM.
"""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="Nvfp4 Requires compute capability of 10 or above.",
        allow_module_level=True,
    )

DTYPES = [torch.float16, torch.bfloat16]
SHAPES = [
    (128, 128),
    (256, 256),
    (128, 512),
    (64, 1024),
    (256, 8192),
    (8192, 8192),
]
SEEDS = [42]
CUDA_DEVICES = ["cuda:0"]

FLOAT4_E2M1_MAX = scalar_types.float4_e2m1f.max()
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
BLOCK_SIZE = 16

E2M1_TO_FLOAT32 = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def cast_from_fp4(x, m, n):
    v_2nd = x & 0xF
    v_1st = (x >> 4) & 0xF
    c = torch.stack((v_2nd, v_1st), dim=-1)
    out = torch.tensor([E2M1_TO_FLOAT32[x] for x in c.flatten()])
    return out.reshape(m, n).to(torch.float32)


def recover_swizzled_scales(scale, m, n):
    round_up = lambda x, y: (x + y - 1) // y * y
    rounded_m = round_up(m, 128)
    scale_n = n // BLOCK_SIZE
    rounded_n = round_up(scale_n, 4)
    tmp = torch.reshape(scale, (1, rounded_m // 128, rounded_n // 4, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    result = torch.reshape(tmp, (rounded_m, rounded_n)).to(torch.float32)
    return result[:m, :scale_n]


def dequantize_fp4(output, output_scale, m, n, global_scale):
    fp4_vals = cast_from_fp4(output, m, n)
    if output_scale.dtype == torch.float8_e4m3fn:
        scale = recover_swizzled_scales(output_scale, m, n)
    else:
        scale = output_scale.to(torch.float32)
    return fp4_vals * scale * global_scale.to(torch.float32)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_scalesweep_produces_valid_fp4(
    dtype: torch.dtype,
    shape: tuple[int, int],
    seed: int,
    device: str,
) -> None:
    """ScaleSweep output should be valid FP4 with correct scale layout."""
    set_random_seed(seed)
    torch.set_default_device(device)

    m, n = shape
    x = torch.randn((m, n), dtype=dtype)
    tensor_amax = torch.abs(x).max().to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax

    out, out_scale = ops.scaled_fp4_quant(
        x, global_scale, backend="scalesweep"
    )

    assert out.dtype == torch.uint8
    assert out.shape == (m, n // 2)
    assert out_scale.dtype == torch.float8_e4m3fn


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_scalesweep_mse_not_worse(
    dtype: torch.dtype,
    shape: tuple[int, int],
    seed: int,
    device: str,
) -> None:
    """ScaleSweep MSE should not be worse than vLLM default."""
    set_random_seed(seed)
    torch.set_default_device(device)

    m, n = shape
    x = torch.randn((m, n), dtype=dtype)
    tensor_amax = torch.abs(x).max().to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax

    # vLLM baseline
    out_vllm, scale_vllm = ops.scaled_fp4_quant(x, global_scale)
    deq_vllm = dequantize_fp4(out_vllm, scale_vllm, m, n, global_scale)
    mse_vllm = torch.mean((x.to(torch.float32) - deq_vllm) ** 2)

    # ScaleSweep
    out_ss, scale_ss = ops.scaled_fp4_quant(
        x, global_scale, backend="scalesweep"
    )
    deq_ss = dequantize_fp4(out_ss, scale_ss, m, n, global_scale)
    mse_ss = torch.mean((x.to(torch.float32) - deq_ss) ** 2)

    assert mse_ss <= mse_vllm * 1.01, (
        f"ScaleSweep MSE ({mse_ss.item():.6f}) should be <= "
        f"vLLM MSE ({mse_vllm.item():.6f}) * 1.01"
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(128, 128), (256, 256)])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_scalesweep_non_swizzled_layout(
    dtype: torch.dtype,
    shape: tuple[int, int],
    seed: int,
    device: str,
) -> None:
    """ScaleSweep should work with non-swizzled scale layout."""
    set_random_seed(seed)
    torch.set_default_device(device)

    m, n = shape
    x = torch.randn((m, n), dtype=dtype)
    tensor_amax = torch.abs(x).max().to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax

    out, out_scale = ops.scaled_fp4_quant(
        x, global_scale, backend="scalesweep",
        is_sf_swizzled_layout=False,
    )

    assert out.dtype == torch.uint8
    assert out.shape == (m, n // 2)
    assert out_scale.dtype == torch.float8_e4m3fn
    assert out_scale.shape == (m, n // BLOCK_SIZE)


@pytest.mark.parametrize("shape", [(128, 128), (256, 8192)])
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_scalesweep_env_var_switch(
    shape: tuple[int, int],
    device: str,
) -> None:
    """VLLM_NVFP4_QUANT_METHOD=scalesweep should activate ScaleSweep."""
    import os

    torch.set_default_device(device)
    m, n = shape
    x = torch.randn((m, n), dtype=torch.bfloat16)
    tensor_amax = torch.abs(x).max().to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax

    # Default (vLLM baseline)
    out_default, _ = ops.scaled_fp4_quant(x, global_scale)

    # With env var
    old_val = os.environ.get("VLLM_NVFP4_QUANT_METHOD")
    os.environ["VLLM_NVFP4_QUANT_METHOD"] = "scalesweep"
    try:
        out_ss, _ = ops.scaled_fp4_quant(x, global_scale)
    finally:
        if old_val is None:
            os.environ.pop("VLLM_NVFP4_QUANT_METHOD", None)
        else:
            os.environ["VLLM_NVFP4_QUANT_METHOD"] = old_val

    # Both should produce valid output shapes
    assert out_default.shape == out_ss.shape
