# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton

GPTQ_TRITON_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]


@triton.jit
def gptq_gemm_kernel(
    a_ptr,  # input activations [M, K] (already reordered if g_idx present)
    b_qweight_ptr,  # quantized weights [K // 8, N], int32
    # (already shuffled if g_idx present)
    b_scales_ptr,  # scales [K // group_size, N], float16
    c_ptr,  # output [M, N]
    M,  # batch size
    N,  # output features
    K,  # input features
    group_size,  # quantization group size
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """
    GPTQ Triton GEMM kernel for 4-bit symmetric quantization.

    Supports:
    - 4-bit weights only
    - Symmetric quantization (no zero points)
    - Group sizes: -1 (channelwise), 32, 64, 128

    Note: Input activations and weights should be pre-processed (reordered/shuffled)
    if g_idx is present. This kernel assumes inputs are already in the correct order.
    """
    pid = tl.program_id(axis=0)
    pid_z = tl.program_id(1)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    accumulator_dtype = c_ptr.type.element_ty
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accumulator_dtype)

    # Offsets and masks for input A [M, K]
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    masks_am = offsets_am < M

    # Offsets for output N dimension
    offsets_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    masks_bn = offsets_bn < N

    # Offsets for scales [K // group_size, N]
    offsets_sn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    masks_sn = offsets_sn < N

    offsets_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offsets_a = K * offsets_am[:, None] + offsets_k[None, :]
    a_ptrs = a_ptr + offsets_a
    b_ptrs = b_qweight_ptr + offsets_k[:, None] // 8 * N + offsets_bn[None, :]

    # Main loop over K dimension
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        masks_k = offsets_k < K
        masks_a = masks_am[:, None] & masks_k[None, :]
        a = tl.load(a_ptrs, mask=masks_a, other=0.0)

        masks_b = masks_k[:, None] & masks_bn[None, :]
        b_qweight = tl.load(b_ptrs, mask=masks_b, other=0)

        # Unpack 4-bit weights from int32 packed along K dimension.
        # Each int32 corresponds to 8 K elements for a given N.
        k_shifts = (offsets_k % 8) * 4
        shifts = tl.broadcast_to(k_shifts[:, None], (BLOCK_SIZE_K, BLOCK_SIZE_N))
        b_qweight = (b_qweight >> shifts) & 0xF

        # Convert from uint4 [0, 15] to int4 [-8, 7] for GPTQ symmetric quantization
        # GPTQ uses bias of 8, so subtract 8 to get signed range
        b_qweight = b_qweight.to(tl.int8) - 8
        b_qweight = b_qweight.to(accumulator_dtype)

        # Load and apply scales
        # Scales are per group, compute which group each K element belongs to
        group_idx = offsets_k // group_size
        offsets_s = N * group_idx[:, None] + offsets_sn[None, :]
        masks_sk = group_idx < K // group_size
        masks_s = masks_sk[:, None] & masks_sn[None, :]
        scales_ptrs = b_scales_ptr + offsets_s
        scales = tl.load(scales_ptrs, mask=masks_s, other=1.0)
        scales = tl.broadcast_to(scales, (BLOCK_SIZE_K, BLOCK_SIZE_N))

        # Dequantize: weight = (quantized - 8) * scale
        # We already subtracted 8 above, so just multiply by scale
        b_qweight = b_qweight * scales

        # Accumulate results
        accumulator = tl.dot(a, b_qweight, accumulator, out_dtype=accumulator_dtype)

        offsets_k += BLOCK_SIZE_K * SPLIT_K
        a_ptrs += BLOCK_SIZE_K * SPLIT_K
        b_ptrs = b_qweight_ptr + offsets_k[:, None] // 8 * N + offsets_bn[None, :]

    # Store results
    c = accumulator.to(c_ptr.type.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + pid_z * N * M + N * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def gptq_gemm_triton(
    input: torch.Tensor,  # [M, K] (should be reordered if g_idx present)
    qweight: torch.Tensor,  # [K // 8, N], int32 (should be shuffled if g_idx present)
    scales: torch.Tensor,  # [K // group_size, N], float16
    group_size: int,
    split_k_iters: int = 1,
    block_size_m: int = 32,
    block_size_n: int = 32,
    block_size_k: int = 32,
) -> torch.Tensor:
    """
    GPTQ Triton GEMM for 4-bit symmetric quantization.

    Args:
        input: Input activations [M, K], float16. Should be reordered using g_idx
               if activation reordering is enabled.
        qweight: Quantized weights [K // 8, N], int32. Should be shuffled using
                 gptq_shuffle if activation reordering is enabled.
        scales: Scales [K // group_size, N], float16
        group_size: Quantization group size. Use -1 for channelwise.
        split_k_iters: Parallelism along K-dimension (power of 2, <= 32)
        block_size_m: Block size for M dimension
        block_size_n: Block size for N dimension
        block_size_k: Block size for K dimension

    Returns:
        Output tensor [M, N], float16
    """
    M, K = input.shape
    N = qweight.shape[1]

    # Validate inputs
    assert N > 0 and K > 0 and M > 0
    assert qweight.shape[0] == K // 8 and qweight.shape[1] == N
    assert scales.shape[1] == N
    assert split_k_iters & (split_k_iters - 1) == 0 and split_k_iters != 0
    assert split_k_iters <= 32

    # Handle group_size
    effective_group_size = group_size if group_size != -1 else K
    assert scales.shape[0] == K // effective_group_size
    assert effective_group_size <= K
    assert (
        effective_group_size in GPTQ_TRITON_SUPPORTED_GROUP_SIZES
        or effective_group_size == K
    )

    # Ensure inputs are contiguous
    input = input.contiguous()
    qweight = qweight.contiguous()
    scales = scales.contiguous()

    # Launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        split_k_iters,
    )

    result = torch.zeros((split_k_iters, M, N), dtype=scales.dtype, device=input.device)

    gptq_gemm_kernel[grid](
        input,
        qweight,
        scales,
        result,
        M,
        N,
        K,
        effective_group_size,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        SPLIT_K=split_k_iters,
    )

    # Sum across split_k dimension
    result = result.sum(0)

    return result
