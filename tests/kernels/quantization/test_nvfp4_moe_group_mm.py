# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
import os
import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from .nvfp4_utils import dequantize_nvfp4_to_dtype

# Only run on Blackwell GeForce (SM_120) since this targets SM_120 kernels
if not current_platform.is_cuda():
    pytest.skip(reason="CUDA required", allow_module_level=True)

# Require CC >= 120 to exercise this kernel
if not current_platform.has_device_capability(120):
    pytest.skip(reason="NVFP4 MoE grouped GEMM requires SM_120", allow_module_level=True)

DTYPES = [torch.float16, torch.bfloat16]


def _round_up(x: int, y: int) -> int:
    return (x + y - 1) // y * y


@torch.inference_mode()
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("E", [2, 4])
@pytest.mark.parametrize("K", [128, 256])
@pytest.mark.parametrize("N", [128, 256])
@pytest.mark.parametrize("device", ["cuda:0"])
def test_nvfp4_moe_group_mm_sm120(dtype: torch.dtype, E: int, K: int, N: int, device: str) -> None:
    torch.manual_seed(0)
    # Per-expert token counts (varying M per expert)
    Ms = [i * 16 for i in range(1, E + 1)]  # multiples of 16
    M_total = sum(Ms)

    # Build expert_offsets and sf_offsets
    expert_offsets = torch.zeros(E + 1, dtype=torch.int32, device=device)
    for i in range(E):
        expert_offsets[i + 1] = expert_offsets[i] + Ms[i]

    # a scales use rounded_m (to 128) rows per expert in swizzled layout
    rounded_ms = [_round_up(m, 128) for m in Ms]
    sf_offsets = torch.zeros(E + 1, dtype=torch.int32, device=device)
    for i in range(E):
        sf_offsets[i + 1] = sf_offsets[i] + rounded_ms[i]

    # A: concatenate per-expert activations along M
    a_dtype = torch.randn((M_total, K), dtype=dtype, device=device)

    # Quantize A per expert to NVFP4 using experts quant (produces swizzled FP8 scales)
    # Build per-expert global scales (alpha_a) for quantization and later GEMM alpha
    a_global_scale = torch.empty((E,), dtype=torch.float32, device=device)
    start = 0
    for i, m in enumerate(Ms):
        a_slice = a_dtype[start:start + m]
        a_global_scale[i] = (torch.finfo(torch.float8_e4m3fn).max * torch.finfo(torch.float16).max) / torch.amax(a_slice.abs())
        start += m

    # Prepare tensors for experts quant
    blockscale_offsets = sf_offsets.clone()
    # Quantize activations across experts; wrapper allocates outputs
    a_fp4, a_scales = ops.scaled_fp4_experts_quant(
        input_tensor=a_dtype,
        input_global_scale=a_global_scale,
        expert_offsets=expert_offsets,
        blockscale_offsets=blockscale_offsets,
        topk=1,
    )

    # B: per-expert weights [E, N, K]
    b_dtype = torch.randn((E, N, K), dtype=dtype, device=device)

    # Quantize each expert's B to NVFP4 row-wise; produce packed fp4 and swizzled scales
    b_fp4 = torch.empty((E, N, K // 2), dtype=torch.uint8, device=device)
    b_scales_list = []
    b_global_scales = torch.empty((E,), dtype=torch.float32, device=device)
    for e in range(E):
        b_global_scales[e] = (torch.finfo(torch.float8_e4m3fn).max * torch.finfo(torch.float16).max) / torch.amax(b_dtype[e].abs())
        b_fp4[e], b_scale = ops.scaled_fp4_quant(b_dtype[e], b_global_scales[e])
        # b_scale is [round_up(N,128), round_up(K/16,4)/4] as float8 view; reshape to [N, K//16] for kernel
        # It is already in swizzled layout that the kernel understands via layout metadata
        # We keep the 2D tensor with leading dim N for each expert
        b_scales_list.append(b_scale)
    # Stack into [E, N, K//16(swizzled with padding aggregated into last dim/4 as int32 packed)]
    b_scales = torch.stack(b_scales_list, dim=0)

    # Output tensor and alpha per expert (combine global scales into alpha)
    out_dtype = dtype
    out = torch.empty((M_total, N), dtype=out_dtype, device=device)

    # problem sizes per expert (M, N, K)
    problem_sizes = torch.empty((E, 3), dtype=torch.int32, device=device)
    for e in range(E):
        problem_sizes[e, 0] = Ms[e]
        problem_sizes[e, 1] = N
        problem_sizes[e, 2] = K

    # GEMM alpha per expert equals 1.0 / (a_global_scale * b_global_scales[e])
    alphas = (1.0 / (a_global_scale * b_global_scales)).to(torch.float32)

    # Call the kernel
    ops.cutlass_fp4_moe_mm(
        out_tensors=out,
        a_tensors=a_fp4,
        b_tensors=b_fp4,
        a_scales=a_scales,
        b_scales=b_scales,
        alphas=alphas,
        problem_sizes=problem_sizes,
        expert_offsets=expert_offsets[:-1],
        sf_offsets=sf_offsets[:-1],
    )

    # Sanity check shape
    assert out.shape == (M_total, N)

    # Optional numerical check on a single expert slice
    # Compare expert 0 slice to dequantized reference within loose tolerance
    m0 = Ms[0]
    ref_a0 = dequantize_nvfp4_to_dtype(a_fp4[:m0], a_scales[:rounded_ms[0]], a_global_scale[0], dtype=dtype, device=device, block_size=16)
    # Dequantize all B0 rows
    ref_b0 = dequantize_nvfp4_to_dtype(b_fp4[0], b_scales[0], b_global_scales[0], dtype=dtype, device=device, block_size=16)
    ref = ref_a0 @ ref_b0.t()
    torch.testing.assert_close(out[:m0], ref.to(dtype), rtol=1e-1, atol=1e-1)
