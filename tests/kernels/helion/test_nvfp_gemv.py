# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import Tensor
from typing import Any
from vllm.utils.import_utils import has_helion

# Skip entire module if helion is not available
if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )

from vllm.kernels.helion.ops.nvfp4_gemv import (
    _dequant_e2m1,
    _fp4_storage,
    nvfp4_gemv_fp4in,
    swizzle_fp8_scales,
)


def reference_nvfp4_gemv_fp4in(
    weight_packed: Tensor,
    x_packed: Tensor,
    weight_scale: Tensor,
    x_scale: Tensor,
    alpha: float = 1.0,
) -> Tensor:
    """Pure PyTorch mathematical reference for NVFP4-in, NVFP4-weight GEMV."""
    w_storage = _fp4_storage(weight_packed).to(torch.int32)
    x_storage = _fp4_storage(x_packed).to(torch.int32)
    
    M, K_bytes = w_storage.shape
    K_groups = K_bytes

    weight_lo = _dequant_e2m1(w_storage & 0xF)
    weight_hi = _dequant_e2m1((w_storage >> 4) & 0xF)

    x_lo = _dequant_e2m1(x_storage & 0xF)
    x_hi = _dequant_e2m1((x_storage >> 4) & 0xF)

    scale_cols = K_bytes // 8
    w_scale_f32 = weight_scale.to(torch.float32)
    x_scale_f32 = x_scale.to(torch.float32)

    out = torch.zeros((M,), device=weight_packed.device, dtype=torch.float32)
    
    # Import swizzled offset utility from implementation module
    from vllm.kernels.helion.ops.nvfp4_gemv import swizzled_scale_offsets

    for m in range(M):
        acc = 0.0
        for k_g in range(K_groups):
            contrib = (weight_lo[m, k_g] * x_lo[k_g]) + (weight_hi[m, k_g] * x_hi[k_g])
            col_tile_idx = k_g // 8
            w_off = swizzled_scale_offsets(m, col_tile_idx, scale_cols)
            x_off = swizzled_scale_offsets(0, col_tile_idx, scale_cols)
            
            scale = w_scale_f32[w_off] * x_scale_f32[x_off]
            acc += contrib * scale
            
        out[m] = acc * alpha

    return out.to(torch.bfloat16)

#TODO: add more shapes.
def test_nvfp4_gemv_fp4in_correctness():
    """Validates the fp4in entrypoint directly against the torch reference."""
    M = 64  
    K = 2048
    K_bytes = K // 2  
    scale_cols = K_bytes // 8

    # Mock inputs on target device
    device = "cuda"
    weight_packed = torch.randint(0, 255, (M, K_bytes), device=device, dtype=torch.uint8)
    x_packed = torch.randint(0, 255, (K_bytes,), device=device, dtype=torch.uint8)

    # Generate logical scale coordinates 
    w_scale_logical = torch.rand((M, scale_cols), device=device, dtype=torch.float32) * 0.1
    x_scale_logical = torch.rand((1, scale_cols), device=device, dtype=torch.float32) * 0.1

    # Convert to swizzled representations
    weight_scale = swizzle_fp8_scales(w_scale_logical.to(torch.float8_e4m3fn))
    x_scale = swizzle_fp8_scales(x_scale_logical.to(torch.float8_e4m3fn))

    # Invoke the Helion kernel directly (passing output buffer if the kernel expects it)
    # Note: Ensure nvfp4_gemv_fp4in signature handles mutations or returns the tensor directly
    result = nvfp4_gemv_fp4in(weight_packed, x_packed, weight_scale, x_scale, 1.0)

    # Compute reference
    expected = reference_nvfp4_gemv_fp4in(weight_packed, x_packed, weight_scale, x_scale, 1.0)

    # Taken fron helion nvfp4_gemv
    torch.testing.assert_close(
        result.to(torch.float32),
        expected.to(torch.float32),
        rtol=2e-1,
        atol=4.0
    )

# TODO: do we need this test to compare it to cutlass- i do have it in quantization.. so maybe remove it.
import pytest
import torch
from tests.kernels.quantization.nvfp4_utils import FLOAT4_E2M1_MAX, FLOAT8_E4M3_MAX, dequantize_nvfp4_to_dtype

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="Nvfp4 Requires compute capability of 10 or above.",
        allow_module_level=True,
    )
import sys

DTYPES = [ torch.bfloat16]
# m, n, k
SHAPES = [(64, 2048), (256, 128), (128, 4096), (4096, 4096), (2048, 7168)]


SEEDS = [42]
CUDA_DEVICES = ["cuda:0"]


def get_ref_results(
    a_fp4,
    b_fp4,
    a_sf,
    b_sf,
    a_global_scale,
    b_global_scale,
    m,
    n,
    dtype,
    block_size,
    device,
):
    _, m_k = a_fp4.shape
    _, n_k = b_fp4.shape
    assert m_k == n_k
    a_in_dtype = dequantize_nvfp4_to_dtype(
        a_fp4, a_sf, a_global_scale, dtype=dtype, device=device, block_size=block_size
    )
    b_in_dtype = dequantize_nvfp4_to_dtype(
        b_fp4, b_sf, b_global_scale, dtype=dtype, device=device, block_size=block_size
    )
    return torch.matmul(a_in_dtype, b_in_dtype.t())


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_nvfp4_gemm(
    dtype: torch.dtype,
    shape: tuple[int, int, int],
    seed: int,
    device: str,
) -> None:
    set_random_seed(seed)
    m = 1 # batch_size=1
    n, packed_k = shape
    
    k = packed_k * 2
    block_size = 16
    a_dtype = torch.randn((m, k), dtype=dtype, device=device)
    b_dtype = torch.randn((n, k), dtype=dtype, device=device)
    a_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(a_dtype.flatten(), dim=-1)
    ).to(torch.float32)
    b_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(b_dtype.flatten(), dim=-1)
    ).to(torch.float32)
    alpha = 1.0 / (a_global_scale * b_global_scale)
    # ops.scaled_fp4_quant returns swizzled scales, while weights
    # from checkpoints are in linear scales.
    a_fp4, a_scale_interleaved = ops.scaled_fp4_quant(a_dtype, a_global_scale)
    b_fp4, b_scale_interleaved = ops.scaled_fp4_quant(b_dtype, b_global_scale)

    # get_ref_results unswizzles the scales internally.
    expected_out = get_ref_results(
        a_fp4,
        b_fp4,
        a_scale_interleaved,
        b_scale_interleaved,
        a_global_scale,
        b_global_scale,
        m,
        n,
        dtype,
        block_size,
        device,
    )
    out = ops.cutlass_scaled_fp4_mm(
        a_fp4, b_fp4, a_scale_interleaved, b_scale_interleaved, alpha, dtype
    )
    
    torch.testing.assert_close(out, expected_out.to(dtype=dtype), atol=1e-1, rtol=1e-1)


    # Helion FP4×FP4 GEMV
    k_bytes = b_fp4.shape[1]
    backend = "triton" #"cute" if k_bytes % 2048 == 0 else "triton"
    alpha_helion = float(1.0 / (a_global_scale * b_global_scale))

    helion_out = nvfp4_gemv_fp4in(
        b_fp4,
        a_fp4,
        b_scale_interleaved,
        a_scale_interleaved,
        alpha=alpha_helion,
    ).unsqueeze(0)

    # Compare to CUTLASS 
    torch.testing.assert_close(helion_out, out, atol=1e-1, rtol=1e-1)
    print(f"Helion FP4×FP4 GEMV M={m}, N={n}, K={k}, backend={backend}")