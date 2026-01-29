# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext

import torch
import torch._dynamo

import vllm.envs as envs
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

AWQ_TRITON_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]


@triton.jit
def awq_dequantize_kernel(
    qweight_ptr,  # quantized matrix
    scales_ptr,  # scales, per group
    zeros_ptr,  # zeros, per group
    group_size,  # Should always be one of the supported group sizes
    result_ptr,  # Output matrix
    num_cols,  # input num cols in qweight
    num_rows,  # input num rows in qweight
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    # Set up the pids.
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)

    # Compute offsets and masks for qweight_ptr.
    offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offsets = num_cols * offsets_y[:, None] + offsets_x[None, :]

    masks_y = offsets_y < num_rows
    masks_x = offsets_x < num_cols

    masks = masks_y[:, None] & masks_x[None, :]

    # Compute offsets and masks for result output ptr.
    result_offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    result_offsets_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(0, BLOCK_SIZE_X * 8)
    result_offsets = (
        8 * num_cols * result_offsets_y[:, None] + result_offsets_x[None, :]
    )

    result_masks_y = result_offsets_y < num_rows
    result_masks_x = result_offsets_x < num_cols * 8
    result_masks = result_masks_y[:, None] & result_masks_x[None, :]

    # Load the weights.
    iweights = tl.load(qweight_ptr + offsets, masks, 0.0)
    iweights = tl.interleave(iweights, iweights)
    iweights = tl.interleave(iweights, iweights)
    iweights = tl.interleave(iweights, iweights)

    # Create reverse AWQ order as tensor: [0, 4, 1, 5, 2, 6, 3, 7]
    # that will map given indices to the correct order.
    reverse_awq_order_tensor = (
        (tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]
    ).reshape(8)

    # Use this to compute a set of shifts that can be used to unpack and
    # reorder the values in iweights and zeros.
    shifts = reverse_awq_order_tensor * 4
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_Y * BLOCK_SIZE_X, 8))
    shifts = tl.reshape(shifts, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    # Unpack and reorder: shift out the correct 4-bit value and mask.
    iweights = (iweights >> shifts) & 0xF

    # Compute zero offsets and masks.
    zero_offsets_y = pid_y * BLOCK_SIZE_Y // group_size + tl.arange(0, 1)
    zero_offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    zero_offsets = num_cols * zero_offsets_y[:, None] + zero_offsets_x[None, :]

    zero_masks_y = zero_offsets_y < num_rows // group_size
    zero_masks_x = zero_offsets_x < num_cols
    zero_masks = zero_masks_y[:, None] & zero_masks_x[None, :]

    # Load the zeros.
    zeros = tl.load(zeros_ptr + zero_offsets, zero_masks, 0.0)
    zeros = tl.interleave(zeros, zeros)
    zeros = tl.interleave(zeros, zeros)
    zeros = tl.interleave(zeros, zeros)
    zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    # Unpack and reorder: shift out the correct 4-bit value and mask.
    zeros = (zeros >> shifts) & 0xF

    # Compute scale offsets and masks.
    scale_offsets_y = pid_y * BLOCK_SIZE_Y // group_size + tl.arange(0, 1)
    scale_offsets_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(0, BLOCK_SIZE_X * 8)
    scale_offsets = num_cols * 8 * scale_offsets_y[:, None] + scale_offsets_x[None, :]
    scale_masks_y = scale_offsets_y < num_rows // group_size
    scale_masks_x = scale_offsets_x < num_cols * 8
    scale_masks = scale_masks_y[:, None] & scale_masks_x[None, :]

    # Load the scales.
    scales = tl.load(scales_ptr + scale_offsets, scale_masks, 0.0)
    scales = tl.broadcast_to(scales, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    # Dequantize.
    iweights = (iweights - zeros) * scales
    iweights = iweights.to(result_ptr.type.element_ty)

    # Finally, store.
    tl.store(result_ptr + result_offsets, iweights, result_masks)


@triton.jit
def awq_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    zeros_ptr,
    scales_ptr,
    M,
    N,
    K,
    group_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    NUM_K_TILES: tl.constexpr,
    K_GROUPS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pid_z = tl.program_id(1)

    # NOTE: This doesn't work in TRITON_INTERPRET=1 mode.  Use below instead.
    # num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    accumulator_dtype = c_ptr.type.element_ty

    # NOTE: This doesn't work in TRITON_INTERPRET=1 mode.  Use below instead.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accumulator_dtype)

    # Create reverse AWQ order as tensor: [0, 4, 1, 5, 2, 6, 3, 7]
    # that will map given indices to the correct order.
    reverse_awq_order_tensor = (
        (tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]
    ).reshape(8)

    # Create the necessary shifts to use to unpack.
    shifts = reverse_awq_order_tensor * 4
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_K * (BLOCK_SIZE_N // 8), 8))
    shifts = tl.reshape(shifts, (BLOCK_SIZE_K, BLOCK_SIZE_N))

    # Offsets and masks.
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    masks_am = offsets_am < M

    offsets_bn = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
    masks_bn = offsets_bn < N // 8

    offsets_zn = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
    masks_zn = offsets_zn < N // 8

    offsets_sn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    masks_sn = offsets_sn < N

    offsets_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offsets_a = K * offsets_am[:, None] + offsets_k[None, :]
    offsets_b = (N // 8) * offsets_k[:, None] + offsets_bn[None, :]

    a_ptrs = a_ptr + offsets_a
    b_ptrs = b_ptr + offsets_b

    for k in range(0, NUM_K_TILES):
        masks_k = offsets_k < K
        masks_a = masks_am[:, None] & masks_k[None, :]
        a = tl.load(a_ptrs, mask=masks_a, other=0.0)

        masks_b = masks_k[:, None] & masks_bn[None, :]
        b = tl.load(b_ptrs, mask=masks_b, other=0.0)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)

        # Dequantize b.
        offsets_szk = (
            BLOCK_SIZE_K * SPLIT_K * k + pid_z * BLOCK_SIZE_K
        ) // group_size + tl.arange(0, 1)
        offsets_z = (N // 8) * offsets_szk[:, None] + offsets_zn[None, :]
        masks_zk = offsets_szk < K_GROUPS
        masks_z = masks_zk[:, None] & masks_zn[None, :]
        zeros_ptrs = zeros_ptr + offsets_z
        zeros = tl.load(zeros_ptrs, mask=masks_z, other=0.0)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_K, BLOCK_SIZE_N))

        offsets_s = N * offsets_szk[:, None] + offsets_sn[None, :]
        masks_sk = offsets_szk < K_GROUPS
        masks_s = masks_sk[:, None] & masks_sn[None, :]
        scales_ptrs = scales_ptr + offsets_s
        scales = tl.load(scales_ptrs, mask=masks_s, other=0.0)
        scales = tl.broadcast_to(scales, (BLOCK_SIZE_K, BLOCK_SIZE_N))

        b = (b >> shifts) & 0xF
        zeros = (zeros >> shifts) & 0xF
        b = (b - zeros) * scales
        b = b.to(c_ptr.type.element_ty)

        # Accumulate results.
        accumulator = tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)

        offsets_k += BLOCK_SIZE_K * SPLIT_K
        a_ptrs += BLOCK_SIZE_K * SPLIT_K
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * (N // 8)

    c = accumulator.to(c_ptr.type.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + pid_z * N * M + N * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def awq_gemv_kernel(
    input_ptr,  # [K] fp16
    qweight_ptr,  # [K, N//8] int32 packed
    qzeros_ptr,  # [K//G, N//8] int32 packed
    scales_ptr,  # [K//G, N] fp16
    output_ptr,  # [N] fp16
    K: tl.constexpr,
    N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    use_fp32_accumulator: tl.constexpr,
):
    """
    Optimized N-split AWQ kernel WITHOUT masking.

    Assumes BLOCK_N divides N evenly to eliminate exec mask manipulation.
    This should generate much cleaner assembly.
    """
    pid = tl.program_id(0)
    n_start = pid * BLOCK_N

    N_packed = N // 8
    num_groups = K // GROUP_SIZE

    # Output column indices [BLOCK_N] - NO MASK needed
    n_offs = n_start + tl.arange(0, BLOCK_N)

    # Packed column indices
    n_packed_offs = n_offs // 8
    n_in_pack = n_offs % 8

    # AWQ shift amounts
    shifts = (n_in_pack // 2) * 4 + (n_in_pack % 2) * 16

    accumulator_dtype = tl.float32 if use_fp32_accumulator else tl.float16
    # Accumulator tensor [BLOCK_N]
    acc = tl.zeros([BLOCK_N], dtype=accumulator_dtype)

    for g in tl.range(num_groups, flatten=True):
        # Load scales [BLOCK_N] - NO MASK
        scale_ptrs = scales_ptr + g * N + n_offs
        scales = tl.load(scale_ptrs).to(tl.float16)

        # Load zeros [BLOCK_N//8] and unpack to [BLOCK_N] - NO MASK
        qz_ptrs = qzeros_ptr + g * N_packed + n_packed_offs
        qz = tl.load(qz_ptrs).to(tl.int32)
        zeros = (qz >> shifts) & 0xF

        # Precompute bias
        # bias = -zeros * scales
        # bias = -(zeros).to(tl.float16) * scales
        # bias = -zeros.to(tl.float32) * scales.to(tl.float32)

        # Inner loop over K in this group
        for k in tl.range(g * GROUP_SIZE, (g + 1) * GROUP_SIZE):
            # Load input scalar and broadcast
            x = tl.load(input_ptr + k).to(tl.float16)

            # Load weights - NO MASK
            qw_ptrs = qweight_ptr + k * N_packed + n_packed_offs
            qw = tl.load(qw_ptrs).to(tl.uint32)
            w = (qw >> shifts) & 0xF
            w2 = (w - zeros.to(tl.float16)) * scales

            # Accumulate
            acc += x * w2
            # acc += x * (w * scales + bias)

    # Store results - NO MASK
    tl.store(output_ptr + n_offs, acc.to(output_ptr.type.element_ty))


@triton.jit
def awq_gemv_kernel_split_k(
    input_ptr,  # [K] fp16
    qweight_ptr,  # [K, N//8] int32 packed
    qzeros_ptr,  # [K//G, N//8] int32 packed
    scales_ptr,  # [K//G, N] fp16
    output_ptr,  # [N] fp16 or [split_k, N] fp16 if split_k > 1
    K: tl.constexpr,
    N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr = 1,
    use_fp32_accumulator: tl.constexpr = True,
):
    """
    Optimized N-split AWQ kernel WITHOUT masking, with optional split-K.

    Assumes BLOCK_N divides N evenly to eliminate exec mask manipulation.

    Split-K parallelization:
    - SPLIT_K=1: Normal operation, each workgroup processes all K
    - SPLIT_K>1: Each workgroup processes K/SPLIT_K elements, writes partial results
    """
    pid_n = tl.program_id(0)  # N dimension
    pid_k = tl.program_id(1)  # K split dimension (0 if SPLIT_K=1)

    n_start = pid_n * BLOCK_N

    N_packed = N // 8
    num_groups = K // GROUP_SIZE

    # Output column indices [BLOCK_N] - NO MASK needed
    n_offs = n_start + tl.arange(0, BLOCK_N)

    # Packed column indices
    n_packed_offs = n_offs // 8
    n_in_pack = n_offs % 8

    # AWQ shift amounts
    shifts = (n_in_pack // 2) * 4 + (n_in_pack % 2) * 16

    # Accumulator tensor [BLOCK_N]
    accumulator_dtype = tl.float32 if use_fp32_accumulator else tl.float16
    acc = tl.zeros([BLOCK_N], dtype=accumulator_dtype)

    if SPLIT_K == 1:
        # Fast path: no split-K, use static loops
        for g in tl.range(num_groups, flatten=True):
            # Load scales [BLOCK_N] - NO MASK
            scale_ptrs = scales_ptr + g * N + n_offs
            scales = tl.load(scale_ptrs).to(tl.float16)

            # Load zeros [BLOCK_N//8] and unpack to [BLOCK_N] - NO MASK
            qz_ptrs = qzeros_ptr + g * N_packed + n_packed_offs
            qz = tl.load(qz_ptrs).to(tl.int32)
            zeros = (qz >> shifts) & 0xF

            # Inner loop over K in this group
            for k in tl.range(g * GROUP_SIZE, (g + 1) * GROUP_SIZE):
                # Load input scalar and broadcast
                x = tl.load(input_ptr + k).to(tl.float16)

                # Load weights - NO MASK
                qw_ptrs = qweight_ptr + k * N_packed + n_packed_offs
                qw = tl.load(qw_ptrs).to(tl.uint32)
                w = (qw >> shifts) & 0xF
                w2 = (w - zeros.to(tl.float16)) * scales

                # Accumulate
                acc += x * w2
    else:
        # Split-K path: each split processes K/SPLIT_K elements
        # k_per_split = K // SPLIT_K
        # k_start = pid_k * k_per_split

        groups_per_split = num_groups // SPLIT_K
        g_start = pid_k * groups_per_split
        g_end = g_start + groups_per_split

        for g in tl.range(g_start, g_end):
            # Load scales [BLOCK_N] - NO MASK
            scale_ptrs = scales_ptr + g * N + n_offs
            scales = tl.load(scale_ptrs).to(tl.float16)

            # Load zeros [BLOCK_N//8] and unpack to [BLOCK_N] - NO MASK
            qz_ptrs = qzeros_ptr + g * N_packed + n_packed_offs
            qz = tl.load(qz_ptrs).to(tl.int32)
            zeros = (qz >> shifts) & 0xF

            # Inner loop over K in this group - clean loop, no conditionals
            for k in tl.range(g * GROUP_SIZE, (g + 1) * GROUP_SIZE):
                # Load input scalar and broadcast
                x = tl.load(input_ptr + k).to(tl.float16)

                # Load weights - NO MASK
                qw_ptrs = qweight_ptr + k * N_packed + n_packed_offs
                qw = tl.load(qw_ptrs).to(tl.uint32)
                w = (qw >> shifts) & 0xF
                w2 = (w - zeros.to(tl.float16)) * scales

                # Accumulate
                acc += x * w2

    # Store results - NO MASK
    if SPLIT_K == 1:
        # Direct write to output
        tl.store(output_ptr + n_offs, acc.to(tl.float16))
    else:
        # Write partial results: output[pid_k, n_offs]
        out_offs = pid_k * N + n_offs
        tl.store(output_ptr + out_offs, acc.to(tl.float16))


@triton.jit
def awq_gemv_kernel_k_blocked(
    input_ptr,  # [K] fp16
    qweight_ptr,  # [K, N//8] int32 packed
    qzeros_ptr,  # [K//G, N//8] int32 packed
    scales_ptr,  # [K//G, N] fp16
    output_ptr,  # [split_k, N] fp16
    K: tl.constexpr,
    N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # Process BLOCK_K elements at once
    SPLIT_K: tl.constexpr,
    use_fp32_accumulator: tl.constexpr = True,
):
    """
    K-blocked AWQ kernel for better instruction-level parallelism.

    Processes BLOCK_K K-elements at a time to reduce loop overhead and
    enable better pipelining of loads.
    """
    pid_n = tl.program_id(0)  # N dimension
    pid_k = tl.program_id(1)  # K split dimension

    n_start = pid_n * BLOCK_N
    N_packed = N // 8

    # Output column indices [BLOCK_N] - NO MASK needed
    n_offs = n_start + tl.arange(0, BLOCK_N)

    # Packed column indices
    n_packed_offs = n_offs // 8
    n_in_pack = n_offs % 8

    # AWQ shift amounts
    shifts = (n_in_pack // 2) * 4 + (n_in_pack % 2) * 16

    # Accumulator tensor [BLOCK_N]
    accumulator_dtype = tl.float32 if use_fp32_accumulator else tl.float16
    acc = tl.zeros([BLOCK_N], dtype=accumulator_dtype)

    # Calculate K range for this split
    k_per_split = K // SPLIT_K
    k_start = pid_k * k_per_split
    # k_end = k_start + k_per_split

    # Iterate over K in blocks of BLOCK_K
    num_k_blocks = k_per_split // BLOCK_K

    for kb in tl.range(num_k_blocks):
        k_block_start = k_start + kb * BLOCK_K

        # Which group does this block belong to?
        g = k_block_start // GROUP_SIZE

        # Load scales [BLOCK_N] - NO MASK
        scale_ptrs = scales_ptr + g * N + n_offs
        scales = tl.load(scale_ptrs).to(tl.float16)

        # Load zeros and unpack to [BLOCK_N] - NO MASK
        qz_ptrs = qzeros_ptr + g * N_packed + n_packed_offs
        qz = tl.load(qz_ptrs).to(tl.int32)
        zeros = ((qz >> shifts) & 0xF).to(tl.float16)

        # Process BLOCK_K elements
        for ki in tl.range(BLOCK_K):
            k = k_block_start + ki

            # Load input scalar
            x = tl.load(input_ptr + k).to(tl.float16)

            # Load weights - NO MASK
            qw_ptrs = qweight_ptr + k * N_packed + n_packed_offs
            qw = tl.load(qw_ptrs).to(tl.uint32)
            w = (qw >> shifts) & 0xF
            w2 = (w - zeros) * scales

            # Accumulate
            acc += x * w2

    # Store partial results
    out_offs = pid_k * N + n_offs
    tl.store(output_ptr + out_offs, acc.to(tl.float16))


def awq_gemv_k_blocked(
    input,
    qweight,
    qzeros,
    scales,
    group_size,
    use_fp32_accumulator,
    num_warps=None,
    split_k=None,
    block_n=None,
    block_k=None,
):
    """Wrapper for K-blocked kernel with automatic reduction."""
    M, K = input.shape
    N = qweight.shape[1] * 8

    if block_n is None:
        block_n = 256
    if split_k is None:
        split_k = 8
    if num_warps is None:
        num_warps = 2
    if block_k is None:
        # BLOCK_K must divide GROUP_SIZE evenly
        block_k = min(32, group_size)

    # Validate constraints
    k_per_split = K // split_k
    assert k_per_split % block_k == 0, (
        f"k_per_split={k_per_split} must be divisible by block_k={block_k}"
    )
    assert block_k <= group_size, (
        f"block_k={block_k} must be <= group_size={group_size}"
    )
    assert group_size % block_k == 0, (
        f"group_size={group_size} must be divisible by block_k={block_k}"
    )

    partial_output = torch.zeros(split_k, N, dtype=torch.float16, device="cuda")

    grid = (triton.cdiv(N, block_n), split_k)
    awq_gemv_kernel_k_blocked[grid](
        input,
        qweight,
        qzeros,
        scales,
        partial_output,
        K=K,
        N=N,
        GROUP_SIZE=group_size,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        SPLIT_K=split_k,
        use_fp32_accumulator=use_fp32_accumulator,
        num_warps=num_warps,
    )

    # Reduce partial results
    result = torch.zeros((M, N), dtype=scales.dtype, device=input.device)
    reduce_split_k_kernel[(triton.cdiv(N, block_n),)](
        partial_output, result, N=N, SPLIT_K=split_k, BLOCK_N=block_n, num_warps=1
    )

    return result


# qweights - [K     , M // 8], int32
# scales   - [K // G, M     ], float16
# zeros    - [K // G, M // 8], int32
def awq_dequantize_triton(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    block_size_x: int = 128,
    block_size_y: int = 32,
) -> torch.Tensor:
    K = qweight.shape[0]
    M = scales.shape[1]
    group_size = qweight.shape[0] // scales.shape[0]

    assert K > 0 and M > 0
    assert scales.shape[0] == K // group_size and scales.shape[1] == M
    assert zeros.shape[0] == K // group_size and zeros.shape[1] == M // 8
    assert group_size <= K
    assert group_size in AWQ_TRITON_SUPPORTED_GROUP_SIZES or group_size == K

    # Result tensor:
    # number of rows = same as input tensor
    # number of cols = 8 x input tensor num cols
    result = torch.empty(
        qweight.shape[0],
        qweight.shape[1] * 8,
        device=qweight.device,
        dtype=scales.dtype,
    )

    Y = qweight.shape[0]  # num rows
    X = qweight.shape[1]  # num cols

    grid = lambda META: (
        triton.cdiv(X, META["BLOCK_SIZE_X"]),
        triton.cdiv(Y, META["BLOCK_SIZE_Y"]),
    )

    # print(f"qweight shape:",qweight.shape)
    # print(f"scale shape:",scales.shape)
    # print(f"zeros shape:",zeros.shape)
    # print(f"grid shape:{grid}")

    awq_dequantize_kernel[grid](
        qweight,
        scales,
        zeros,
        group_size,
        result,
        X,
        Y,
        BLOCK_SIZE_X=block_size_x,
        BLOCK_SIZE_Y=block_size_y,
    )

    return result


@triton.jit
def reduce_split_k_kernel(
    partial_ptr,  # [split_k, N] fp16
    output_ptr,  # [N] fp16
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Reduce partial results from split-K parallelization.

    Each workgroup reduces BLOCK_N columns across SPLIT_K partial results.
    """
    pid = tl.program_id(0)
    n_start = pid * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)

    # Accumulate across splits
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k_split in range(SPLIT_K):
        partial_offs = k_split * N + n_offs
        partial = tl.load(partial_ptr + partial_offs).to(tl.float32)
        acc += partial

    # Store final result
    tl.store(output_ptr + n_offs, acc.to(tl.float16))


def _get_valid_split_k_values(K: int, group_size: int) -> list:
    """
    Get all valid split_k values that:
    1. Divide num_groups evenly (critical for clean inner loops without conditionals)
    2. Ensure k_per_split is divisible by 32 for proper alignment
    """
    num_groups = K // group_size
    valid_split_k = []
    for sk in range(1, num_groups + 1):
        if num_groups % sk == 0:
            k_per_split = K // sk
            if k_per_split % 32 == 0:
                valid_split_k.append(sk)
    return valid_split_k if valid_split_k else [1]


def _choose_optimal_config(K: int, N: int, group_size: int) -> tuple:
    """
    Choose optimal (split_k, block_n, num_warps) based on shape.

    Empirically tuned configurations (ROCm/AMD GPU MI300):
    - K=512, N=4096:    block_n=64,  warps=2, split_k=2  -> 43.0 GB/s
    - K=512, N=11008:   block_n=256, warps=8, split_k=2  -> 84.1 GB/s
    - K=1536, N=4096:   block_n=256, warps=8, split_k=6  -> 85.0 GB/s
    - K=2560, N=6144:   block_n=64,  warps=2, split_k=4  -> 99.8 GB/s
    - K=2752, N=4096:   block_n=64,  warps=1, split_k=43 -> 91.5 GB/s (group_size=32)
    - K=4096, N=2560:   block_n=64,  warps=2, split_k=8  -> 90.5 GB/s
    - K=4096, N=4096:   block_n=64,  warps=1, split_k=16 -> 102.0 GB/s
    - K=4096, N=11008:  block_n=256, warps=2, split_k=43 -> 103.5 GB/s
    - K=4096, N=12288:  block_n=256, warps=8, split_k=2  -> 107.1 GB/s
    - K=4096, N=22016:  block_n=256, warps=2, split_k=4  -> 125.4 GB/s
    - K=9728, N=2560:   block_n=64,  warps=2, split_k=38 -> 100.2 GB/s
    - K=11008, N=4096:  block_n=256, warps=2, split_k=43 -> 103.5 GB/s
    - K=2560, N=19456:  block_n=256, warps=2, split_k=5  -> 107.3 GB/s
    - K=9728, N=19456:  block_n=128, warps=1, split_k=19 -> 164.9 GB/s
    """
    valid_split_k = _get_valid_split_k_values(K, group_size)
    num_groups = K // group_size

    # Heuristics based on exhaustive search results
    if num_groups <= 4:
        # Very small K (e.g., K=512): use split_k=2
        target_sk = 2
        if N >= 8000:
            block_n = 256
            num_warps = 8
        else:
            block_n = 64
            num_warps = 2
    elif num_groups <= 7:
        # Small K (e.g., K=896): maximize parallelism with split_k=num_groups
        # For K=896 (num_groups=7): split_k=7 gives best results
        target_sk = num_groups
        if N >= 8000:
            block_n = 256
            num_warps = 2
        elif N >= 1000:
            block_n = 64
            num_warps = 1
        else:
            block_n = 128
            num_warps = 2
    elif num_groups <= 15:
        # Small K (e.g., K=1536): use ~50% of num_groups
        target_sk = max(2, num_groups // 2)
        block_n = 256
        num_warps = 8
    elif num_groups <= 25:
        # Medium-small K (e.g., K=2560): prefer smaller split_k
        target_sk = max(4, num_groups // 4)
        if N >= 10000:
            block_n = 256
            num_warps = 2
        else:
            block_n = 64
            num_warps = 2
    elif num_groups <= 40:
        # Medium K (e.g., K=4096, num_groups=32)
        if N >= 20000:
            # Very large N (e.g., N=22016): split_k=4, warps=2
            target_sk = 4
            block_n = 256
            num_warps = 2
        elif N >= 12000:
            # Large N (e.g., N=12288): split_k=2, warps=8
            target_sk = 2
            block_n = 256
            num_warps = 8
        elif N >= 8000:
            # Medium-large N (e.g., N=11008): split_k=8, warps=2
            target_sk = 8
            block_n = 64
            num_warps = 2
        elif N >= 4000:
            # Medium N (e.g., N=4096): split_k=16, warps=1
            target_sk = 16
            block_n = 64
            num_warps = 1
        elif N >= 2000:
            # Small-medium N (e.g., N=2560): split_k=8, warps=2
            target_sk = 8
            block_n = 64
            num_warps = 2
        else:
            # Very small N (e.g., N=896): need high split_k for parallelism
            # For K=4864, N=896 (num_groups=38): split_k=19, warps=2 works best
            target_sk = num_groups // 2
            block_n = 64
            num_warps = 2
    else:
        # Large K (e.g., K=9728, K=11008)
        if N >= 10000:
            target_sk = 19
            block_n = 128
            num_warps = 1
        else:
            # Small N with large K
            target_sk = min(43, num_groups // 2)
            block_n = 256
            num_warps = 2

    # Special case: small group_size with many groups
    if group_size <= 32 and num_groups > 40:
        target_sk = num_groups // 2
        block_n = 64
        num_warps = 1

    # Find the valid split_k closest to target
    split_k = min(valid_split_k, key=lambda sk: abs(sk - target_sk))

    return split_k, block_n, num_warps


def awq_gemv_no_split_k(
    input,
    qweight,
    qzeros,
    scales,
    group_size,
    use_fp32_accumulator,
    block_n=128,
    num_warps=4,
):
    """Non-split-k GEMV kernel - faster for small K with large N."""
    M, K = input.shape
    N = qweight.shape[1] * 8
    result = torch.zeros((M, N), dtype=scales.dtype, device=input.device)

    grid = (triton.cdiv(N, block_n),)
    awq_gemv_kernel[grid](
        input,
        qweight,
        qzeros,
        scales,
        result,
        K=K,
        N=N,
        GROUP_SIZE=group_size,
        BLOCK_N=block_n,
        use_fp32_accumulator=use_fp32_accumulator,
        num_warps=num_warps,
    )

    return result


def awq_gemv_split_k(
    input,
    qweight,
    qzeros,
    scales,
    group_size,
    use_fp32_accumulator,
    num_warps=None,
    split_k=None,
    block_n=None,
):
    M, K = input.shape
    N = qweight.shape[1] * 8
    result = torch.zeros((M, N), dtype=scales.dtype, device=input.device)

    # Auto-select optimal configuration if not specified
    if split_k is None or block_n is None or num_warps is None:
        auto_split_k, auto_block_n, auto_num_warps = _choose_optimal_config(
            K, N, group_size
        )
        if split_k is None:
            split_k = auto_split_k
        if block_n is None:
            block_n = auto_block_n
        if num_warps is None:
            num_warps = auto_num_warps

    partial_output = torch.zeros(split_k, N, dtype=torch.float16, device="cuda")

    # Use the optimized split_k kernel with proper configuration
    # The key optimization is choosing split_k that divides num_groups evenly
    # to avoid conditionals in the inner loop
    grid = (triton.cdiv(N, block_n), split_k)
    awq_gemv_kernel_split_k[grid](
        input,
        qweight,
        qzeros,
        scales,
        partial_output,
        K=K,
        N=N,
        GROUP_SIZE=group_size,
        BLOCK_N=block_n,
        SPLIT_K=split_k,
        use_fp32_accumulator=use_fp32_accumulator,
        num_warps=num_warps,
    )

    reduce_split_k_kernel[(triton.cdiv(N, block_n),)](
        partial_output, result, N=N, SPLIT_K=split_k, BLOCK_N=block_n, num_warps=1
    )

    return result


# input   - [M, K]
# qweight - [K, N // 8]
# qzeros  - [K // G, N // 8]
# scales  - [K // G, N]
# split_k_iters - parallelism along K-dimension, int, power of 2.
def _awq_gemm_triton(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    split_k_iters: int,
    # NOTE: These defaults materially impact prefill performance.
    # The older ROCm build (f37e8938) used 32/32/32; keep as default to avoid
    # regressions on common AWQ prefill shapes (e.g., M=128, K=4096, N in
    # {4096, 12288, 22016} with chunked prefill).
    block_size_m: int = 32,
    block_size_n: int = 32,
    block_size_k: int = 32,
) -> torch.Tensor:
    M, K = input.shape
    N = qweight.shape[1] * 8
    weight_K = qweight.shape[0]  # May be > K if weights were padded
    group_size = weight_K // qzeros.shape[0]

    assert N > 0 and K > 0 and M > 0
    assert split_k_iters & (split_k_iters - 1) == 0 and split_k_iters != 0
    assert split_k_iters <= 32
    assert group_size in AWQ_TRITON_SUPPORTED_GROUP_SIZES or group_size == weight_K

    # if M == 1 and N % 512 == 0 and N == 19456:
    # assert isinstance(M, int)

    use_fp32_accumulator = False  # K > 2560

    # Use GEMV kernel for M=1 if N is divisible by any reasonable BLOCK_N
    # The kernel uses BLOCK_N of 64, 128, or 256, so N just needs to divide evenly
    # Exception: for very small shapes (K <= 1024 and N <= 2000), the generic gemm
    # kernel is faster due to lower overhead
    use_gemv = envs.VLLM_USE_TRITON_AWQ_GEMV and M == 1 and N % 64 == 0
    if use_gemv and K <= 1024 and N <= 2000:
        use_gemv = False  # Fall back to gemm for tiny shapes

    # Try HIP-optimized GEMV kernel on ROCm
    # Supports: M=1, group_size=128, any K divisible by group_size
    # Uses K-padding to enable higher split-k factors for better parallelism
    # Skip for small N (<1500) where Triton's parallelization is more efficient
    if (
        use_gemv
        and group_size == 128
        and N % 8 == 0
        and K % group_size == 0
        and N >= 1500
    ):
        from vllm.platforms import current_platform

        if current_platform.is_rocm():
            try:
                from vllm._custom_ops import awq_gemv_hip

                # Check if weights are padded (qweight.K > activation.K)
                padded_K = qweight.shape[0]
                act = input.squeeze(0)

                if padded_K > K:
                    # Weights were padded during preprocessing - pad activation
                    act_padded = torch.zeros(
                        padded_K, dtype=act.dtype, device=act.device
                    )
                    act_padded[:K] = act
                    act = act_padded

                ctx = (
                    nullcontext()
                    if torch.compiler.is_compiling()
                    else torch.profiler.record_function(
                        f"awq_gemv_hip {N}x{padded_K} gs={group_size}"
                    )
                )
                with ctx:
                    return awq_gemv_hip(act, qweight, scales, qzeros).unsqueeze(0)
            except (RuntimeError, AttributeError):
                # Fall through to Triton kernels if HIP kernel not available
                pass

    # Pad activation if weights were padded during preprocessing
    if weight_K > K:
        input_padded = torch.zeros(
            (M, weight_K), dtype=input.dtype, device=input.device
        )
        input_padded[:, :K] = input
        input = input_padded
        K = weight_K  # Update K to match padded weights

    # Validate tensor dimensions
    assert qweight.shape[1] == N // 8
    assert qzeros.shape[0] == K // group_size and qzeros.shape[1] == N // 8
    assert scales.shape[0] == K // group_size and scales.shape[1] == N
    assert group_size <= K

    if use_gemv:
        # For K in [768, 1024] with very large N (e.g., 896x9728), non-split-k gemv
        # is faster because split-k overhead exceeds its benefits with few groups,
        # but N provides enough parallelism. This is a narrow sweet spot.
        num_groups = K // group_size
        if 6 <= num_groups <= 8 and N > 8000:
            ctx = (
                nullcontext()
                if torch.compiler.is_compiling()
                else torch.profiler.record_function(
                    f"awq_gemv_no_split_k {N}x{K} gs={group_size}"
                )
            )
            with ctx:
                return awq_gemv_no_split_k(
                    input,
                    qweight,
                    qzeros,
                    scales,
                    group_size,
                    use_fp32_accumulator,
                    block_n=128,
                    num_warps=4,
                )
        # Use optimized split-k GEMV kernel for all other shapes
        ctx = (
            nullcontext()
            if torch.compiler.is_compiling()
            else torch.profiler.record_function(
                f"awq_gemv_split_k {N}x{K} gs={group_size}"
            )
        )
        with ctx:
            return awq_gemv_split_k(
                input, qweight, qzeros, scales, group_size, use_fp32_accumulator
            )

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        split_k_iters,
    )

    result = torch.zeros((split_k_iters, M, N), dtype=scales.dtype, device=input.device)

    # A = input, B = qweight, C = result
    # A = M x K, B = K x N, C = M x N
    ctx = (
        nullcontext()
        if torch.compiler.is_compiling()
        else torch.profiler.record_function(f"awq_gemm {M}x{N}x{K}")
    )
    num_k_tiles = (K + block_size_k * split_k_iters - 1) // (
        block_size_k * split_k_iters
    )
    k_group = K // group_size

    with ctx:
        if envs.VLLM_USE_TRITON_AWQ_GEMV and M == 1:
            block_size_m = 1

        awq_gemm_kernel[grid](
            input,
            qweight,
            result,
            qzeros,
            scales,
            M,
            N,
            K,
            group_size,
            BLOCK_SIZE_M=block_size_m,
            BLOCK_SIZE_N=block_size_n,
            BLOCK_SIZE_K=block_size_k,
            SPLIT_K=split_k_iters,
            NUM_K_TILES=num_k_tiles,
            K_GROUPS=k_group,
            num_stages=1,
            num_warps=4,
        )

        result = result.sum(0)

    return result


def _awq_gemm_triton_fake(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    split_k_iters: int,
) -> torch.Tensor:
    M, N = input.shape[0], qweight.shape[1] * 8
    return torch.empty((M, N), dtype=scales.dtype, device=input.device)


direct_register_custom_op(
    op_name="awq_gemm_triton",
    op_func=_awq_gemm_triton,
    fake_impl=_awq_gemm_triton_fake,
)
awq_gemm_triton = torch.ops.vllm.awq_gemm_triton
