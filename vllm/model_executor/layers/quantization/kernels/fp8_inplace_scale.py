# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl


@triton.jit
def triton_fp8_inplace_scale_kernel(x_ptr, scale, M, N, stride_xm, stride_xn,
                                    BLOCK_SIZE_M: tl.constexpr,
                                    BLOCK_SIZE_N: tl.constexpr,
                                    FP8_MIN: tl.constexpr,
                                    FP8_MAX: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs = stride_xm * offs_m[:, None] + stride_xn * offs_n[None, :]
    masks_m = offs_m < M
    masks_n = offs_n < N
    masks = masks_m[:, None] & masks_n[None, :]
    ptrs = x_ptr + offs
    values = tl.load(ptrs, mask=masks)
    values = scale * values.to(tl.float16)
    values = tl.clamp(values, FP8_MIN, FP8_MAX).to(tl.float8e4b8)
    tl.store(ptrs, values, mask=masks)


def triton_fp8_inplace_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    from vllm.platforms import current_platform
    M, N = x.shape
    dtype = current_platform.fp8_dtype()
    finfo = torch.finfo(dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]),
                         triton.cdiv(
                             N,
                             meta["BLOCK_SIZE_N"],
                         ))
    stride_xm = x.stride(0)
    stride_xn = x.stride(1)
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 256
    triton_fp8_inplace_scale_kernel[grid](x, 1.0 / scale, M, N, stride_xm,
                                          stride_xn, BLOCK_SIZE_M,
                                          BLOCK_SIZE_N, finfo.min, finfo.max)
    return x
