#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    pad_nvfp4_weight_for_cutlass,
    swizzle_blockscale,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.triton_utils import triton

if not current_platform.has_device_capability(100):
    raise RuntimeError("NVFP4 requires compute capability 10.0+")

DTYPE = torch.bfloat16
DEVICE = "cuda"
K = 7152  # misaligned, triggers activation padding
N = 8192  # aligned, avoids mixing in output slicing
BATCH_SIZES = [1, 8, 32, 128, 512, 2048, 8192]

FLOAT4_E2M1_MAX = scalar_types.float4_e2m1f.max()
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def global_scale(x: torch.Tensor) -> torch.Tensor:
    return FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / torch.abs(x).max().to(torch.float32)


weight = torch.randn((N, K), device=DEVICE, dtype=DTYPE)
weight_gs = global_scale(weight)
weight_fp4, weight_scale = ops.scaled_fp4_quant(weight, weight_gs)
weight_fp4, weight_pad_bytes = pad_nvfp4_weight_for_cutlass(weight_fp4)
weight_scale = swizzle_blockscale(weight_scale)
padded_k = K + weight_pad_bytes * 2

print(f"N={N} K={K} padded_k={padded_k} pad_bytes={weight_pad_bytes}")
print("batch | old_us | new_us | speedup")
print("----- | ------ | ------ | -------")

for m in BATCH_SIZES:
    x = torch.randn((m, K), device=DEVICE, dtype=DTYPE)
    x_gs = global_scale(x)
    alpha = 1.0 / (x_gs * weight_gs)

    def old(x=x, x_gs=x_gs, alpha=alpha):
        x_fp4, x_scale = ops.scaled_fp4_quant(
            x, x_gs, is_sf_swizzled_layout=True, backend="cutlass"
        )
        x_fp4 = torch.nn.functional.pad(x_fp4, (0, weight_pad_bytes)).contiguous()
        return ops.cutlass_scaled_fp4_mm(
            x_fp4, weight_fp4, x_scale, weight_scale, alpha, DTYPE
        )

    def new(x=x, x_gs=x_gs, alpha=alpha):
        x_fp4, x_scale = ops.scaled_fp4_quant(
            x,
            x_gs,
            is_sf_swizzled_layout=True,
            backend="cutlass",
            padded_n=padded_k,
        )
        return ops.cutlass_scaled_fp4_mm(
            x_fp4, weight_fp4, x_scale, weight_scale, alpha, DTYPE
        )

    old_ms, _, _ = triton.testing.do_bench_cudagraph(old, quantiles=[0.5, 0.2, 0.8])
    new_ms, _, _ = triton.testing.do_bench_cudagraph(new, quantiles=[0.5, 0.2, 0.8])
    old_us = old_ms * 1000
    new_us = new_ms * 1000
    print(f"{m:5d} | {old_us:6.2f} | {new_us:6.2f} | {old_us / new_us:7.3f}x")
