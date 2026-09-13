# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Triton implementation of NVFP4 GEMM (FP4 x FP4 -> BF16/FP16).

Uses tl.dot_scaled for block-scaled FP4 matrix multiplication on SM100+.
This serves as a portable reference/fallback when CUTLASS kernels are
not available (e.g., SM120 without CUTLASS support).

Data format (NVFP4):
  - FP4 values: E2M1 format, 2 packed per uint8 byte along K dimension
  - Block scales: float8_e4m3fn, 1 scale per 16 FP4 elements
  - Global scale: float32 scalar
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _triton_nvfp4_gemm_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Pointers to block scales
    a_scale_ptr,
    b_scale_ptr,
    # Global alpha = 1/(global_scale_a * global_scale_b)
    alpha,
    # Matrix dimensions
    M,
    N,
    K,  # logical K (unpacked)
    # Strides (in elements, accounting for packing)
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Scale strides
    stride_a_scale_m,
    stride_b_scale_n,
    # Tile sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # NVFP4 block scale group size (elements per scale)
    VEC_SIZE: tl.constexpr,
):
    """Triton kernel for NVFP4 GEMM: C = alpha * (A @ B^T).

    A is [M, K//2] uint8 (packed FP4, row-major)
    B is [N, K//2] uint8 (packed FP4, row-major, weight format)
    C is [M, N] output (bf16/fp16)
    a_scale is [M, K//VEC_SIZE] float8_e4m3fn (linear layout)
    b_scale is [N, K//VEC_SIZE] float8_e4m3fn (linear layout)

    Computes C = alpha * dot_scaled(A, B^T) where dot_scaled handles
    block-scale dequantization internally via tensor core instructions.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    # Offsets for the M and N tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # K is packed: 2 FP4 values per uint8 byte
    BLOCK_K_PACKED: tl.constexpr = BLOCK_K // 2
    offs_k = tl.arange(0, BLOCK_K_PACKED)

    # Scale offsets: one scale per VEC_SIZE elements along K
    SCALE_K: tl.constexpr = BLOCK_K // VEC_SIZE
    offs_scale_k = tl.arange(0, SCALE_K)

    # Initialize pointers for A [M, K//2] and B^T [K//2, N]
    # A is row-major: a[m, k] = a_ptr + m * stride_am + k * stride_ak
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # B is [N, K//2] row-major, but we want B^T [K//2, N]
    # b^T[k, n] = b[n, k] = b_ptr + n * stride_bn + k * stride_bk
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Scale pointers
    # a_scale [M, K//VEC_SIZE]: a_scale[m, g] = a_scale_ptr + m * stride + g
    a_scale_ptrs = (a_scale_ptr + offs_m[:, None] * stride_a_scale_m +
                    offs_scale_k[None, :])
    # b_scale [N, K//VEC_SIZE]: b_scale[n, g] = b_scale_ptr + n * stride + g
    b_scale_ptrs = (b_scale_ptr + offs_n[:, None] * stride_b_scale_n +
                    offs_scale_k[None, :])

    # Accumulator in float32
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main loop over K dimension
    for _ in tl.range(0, tl.cdiv(K, BLOCK_K)):
        # Load A tile [BLOCK_M, BLOCK_K_PACKED] and B^T tile [BLOCK_K_PACKED, BLOCK_N]
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

        # Load block scales
        scale_a = tl.load(a_scale_ptrs)
        scale_b = tl.load(b_scale_ptrs)

        # Block-scaled FP4 dot product
        # tl.dot_scaled handles: dequant(a, scale_a) @ dequant(b, scale_b)
        accumulator = tl.dot_scaled(
            a,
            scale_a,
            "e2m1",
            b,
            scale_b,
            "e2m1",
            accumulator,
            lhs_k_pack=True,
            rhs_k_pack=True,
        )

        # Advance pointers
        a_ptrs += BLOCK_K_PACKED * stride_ak
        b_ptrs += BLOCK_K_PACKED * stride_bk
        a_scale_ptrs += SCALE_K
        b_scale_ptrs += SCALE_K

    # Apply global scale: C = alpha * accumulator
    c = (accumulator * alpha).to(c_ptr.dtype.element_ty)

    # Store output with bounds checking
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def triton_scaled_fp4_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    alpha: float | torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Triton-based NVFP4 GEMM: C = alpha * dequant(A) @ dequant(B)^T.

    Args:
        a:       Packed FP4 activations [M, K//2] uint8
        b:       Packed FP4 weights    [N, K//2] uint8
        a_scale: Block scales for A    [M, K//16] float8_e4m3fn (linear layout)
        b_scale: Block scales for B    [N, K//16] float8_e4m3fn (linear layout)
        alpha:   Global scale = 1 / (global_scale_a * global_scale_b)
        out_dtype: Output dtype (torch.bfloat16 or torch.float16)

    Returns:
        C: [M, N] tensor in out_dtype
    """
    assert a.ndim == 2 and b.ndim == 2
    assert a.dtype == torch.uint8 and b.dtype == torch.uint8

    M, K_packed = a.shape
    N, K_packed_b = b.shape
    assert K_packed == K_packed_b, (
        f"K dimension mismatch: A has {K_packed}, B has {K_packed_b}"
    )
    K = K_packed * 2  # Logical K (unpacked)

    # Validate scale shapes
    VEC_SIZE = 16  # NVFP4 block scale group size
    assert a_scale.shape == (M, K // VEC_SIZE), (
        f"a_scale shape {a_scale.shape} != expected ({M}, {K // VEC_SIZE})"
    )
    assert b_scale.shape == (N, K // VEC_SIZE), (
        f"b_scale shape {b_scale.shape} != expected ({N}, {K // VEC_SIZE})"
    )

    # Convert alpha to float
    if isinstance(alpha, torch.Tensor):
        alpha_val = alpha.item()
    else:
        alpha_val = float(alpha)

    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)

    # Tile sizes — tuned for SM100+ tensor cores
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128  # Must be multiple of VEC_SIZE (16)

    # Adjust block sizes for small matrices
    if M < BLOCK_M:
        BLOCK_M = max(16, triton.next_power_of_2(M))
    if N < BLOCK_N:
        BLOCK_N = max(16, triton.next_power_of_2(N))

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    # Strides for A [M, K//2] — row-major
    stride_am = a.stride(0)
    stride_ak = a.stride(1)

    # Strides for B [N, K//2] — row-major
    # For B^T access: b^T[k, n] = b[n, k]
    stride_bn = b.stride(0)  # stride to move along N (rows of B)
    stride_bk = b.stride(1)  # stride to move along K (cols of B)

    # Scale strides
    stride_a_scale_m = a_scale.stride(0)
    stride_b_scale_n = b_scale.stride(0)

    _triton_nvfp4_gemm_kernel[grid](
        a,
        b,
        c,
        a_scale,
        b_scale,
        alpha_val,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        c.stride(0),
        c.stride(1),
        stride_a_scale_m,
        stride_b_scale_n,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        VEC_SIZE=VEC_SIZE,
    )

    return c
