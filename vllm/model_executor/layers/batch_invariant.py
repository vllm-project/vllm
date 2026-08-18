# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
import os
from collections.abc import Callable
from functools import lru_cache
from typing import Any

import torch

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.mem_utils import get_max_shared_memory_bytes
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import is_torch_equal_or_newer


def _matmul_launch_metadata(
    grid: Callable[..., Any], kernel: Any, args: dict[str, Any]
) -> dict[str, Any]:
    ret = {}
    m, n, k = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={m}, N={n}, K={k}]"

    bytes_per_elem = args["c_ptr"].element_size()
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * m * n * k
    ret["bytes"] = bytes_per_elem * (m * k + n * k + m * n)
    return ret


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


def _vector_stride(tensor: torch.Tensor) -> int:
    """Stride between entries of a scalar or single-axis tensor.

    Flattening instead would be wrong: ``reshape(-1)`` on an ``(M, 1)`` slice
    returns a strided view that a kernel would read as unit stride.
    """
    strides = [s for s, size in zip(tensor.stride(), tensor.shape) if size > 1]
    return strides[0] if strides else 1


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_persistent(
    a_ptr,
    b_ptr,
    c_ptr,  #
    bias_ptr,
    M,
    N,
    K,  #
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bias,
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        if A_LARGE:
            offs_am = offs_am.to(tl.int64)
        if B_LARGE:
            offs_bn = offs_bn.to(tl.int64)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            if A_LARGE or B_LARGE:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
            else:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (
                offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
            )
            b_ptrs = b_ptr + (
                offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
            )

            a = tl.load(
                a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0
            )
            b = tl.load(
                b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0
            )
            accumulator = tl.dot(a, b, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(
            tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M
        )
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        if C_LARGE:
            offs_cm = offs_cm.to(tl.int64)
            offs_cn = offs_cn.to(tl.int64)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if HAS_BIAS:
            bias_ptrs = bias_ptr + offs_cn * stride_bias
            bias = tl.load(bias_ptrs, mask=offs_cn < N, other=0.0).to(tl.float32)
            accumulator += bias
        c = accumulator.to(c_ptr.dtype.element_ty)
        tl.store(c_ptrs, c, mask=c_mask)


def matmul_persistent(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None
):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    assert bias is None or bias.dim() == 1, (
        "Currently assuming bias is 1D, let Horace know if you run into this"
    )
    NUM_SMS = num_compute_units(a.device.index)
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=dtype)

    # 1D launch kernel where each block gets its own program.
    def grid(META):
        return (
            min(
                NUM_SMS,
                triton.cdiv(M, META["BLOCK_SIZE_M"])
                * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            ),
        )

    configs = {
        torch.bfloat16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": _fp16_block_size_n(),
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float32: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
    }
    matmul_kernel_persistent[grid](
        a,
        b,
        c,  #
        bias,
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        _vector_stride(bias) if bias is not None else 1,
        NUM_SMS=NUM_SMS,  #
        A_LARGE=a.numel() > 2**31,
        B_LARGE=b.numel() > 2**31,
        C_LARGE=c.numel() > 2**31,
        HAS_BIAS=bias is not None,
        **configs[dtype],
    )
    return c


@triton.jit
def bmm_kernel(
    a_ptr,  # (*, ) pointer to A, (B, M, K)
    b_ptr,  # (*, ) pointer to B, (B, K, N)
    c_ptr,  # (*, ) pointer to C, (B, M, N)
    B,  # int, batch size
    M,  # int, output rows
    N,  # int, output cols
    K,  # int, reduction dim
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
):
    """Batched GEMM: (B, M, K) x (B, K, N) -> (B, M, N)

    Each program computes one (batch_idx, tile_m, tile_n) tile, accumulating
    along K in a fixed order to preserve batch invariance.
    """
    pid_b = tl.program_id(0)
    pid = tl.program_id(1)

    if pid_b >= B:
        return

    # number of tiles along M / N
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    if pid_m >= num_pid_m or pid_n >= num_pid_n:
        return

    # offs_m / offs_n: raw global row/col indices for this tile
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # masks for valid logical rows/cols within (M, N)
    mask_m = offs_m < M  # [BLOCK_SIZE_M]
    mask_n = offs_n < N  # [BLOCK_SIZE_N]

    if A_LARGE or B_LARGE or C_LARGE:
        offs_m = offs_m.to(tl.int64)
        offs_n = offs_n.to(tl.int64)

    offs_m = tl.where(mask_m, offs_m, 0)
    offs_n = tl.where(mask_n, offs_n, 0)

    # hint for triton contiguous memory
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N)

    # base pointers for current batch, shape-wise:
    #   a_batch_ptr points to A[pid_b, 0, 0]
    #   b_batch_ptr points to B[pid_b, 0, 0]
    #   c_batch_ptr points to C[pid_b, 0, 0]
    a_batch_ptr = a_ptr + pid_b * stride_ab
    b_batch_ptr = b_ptr + pid_b * stride_bb
    c_batch_ptr = c_ptr + pid_b * stride_cb

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # number of K-blocks this tile iterates over
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    offs_k_mask = tl.arange(0, BLOCK_SIZE_K)

    for ki in range(k_tiles):
        if A_LARGE or B_LARGE:
            # offs_k: [BLOCK_SIZE_K], global K indices
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
        else:
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        # a_ptrs: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        #   element (i, j) points to A[pid_b, offs_m[i], offs_k[j]]
        a_ptrs = a_batch_ptr + (
            offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        )
        # b_ptrs: [BLOCK_SIZE_K, BLOCK_SIZE_N]
        #   element (i, j) points to B[pid_b, offs_k[i], offs_n[j]]
        b_ptrs = b_batch_ptr + (
            offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        )

        # valid K lanes for this block
        k_valid = offs_k_mask < (K - ki * BLOCK_SIZE_K)
        # A mask within (M, K): [BLOCK_SIZE_M, BLOCK_SIZE_K]
        a_mask = mask_m[:, None] & k_valid[None, :]
        # B mask within (K, N): [BLOCK_SIZE_K, BLOCK_SIZE_N]
        b_mask = k_valid[:, None] & mask_n[None, :]

        # a: [BLOCK_SIZE_M, BLOCK_SIZE_K] from A[offs_m, offs_k]
        a = tl.load(
            a_ptrs,
            mask=a_mask,
            other=0.0,
        )
        # b: [BLOCK_SIZE_K, BLOCK_SIZE_N] from B[offs_k, offs_n]
        b = tl.load(
            b_ptrs,
            mask=b_mask,
            other=0.0,
        )
        accumulator = tl.dot(a, b, accumulator)

    # c_m / c_n: [BLOCK_SIZE_M] / [BLOCK_SIZE_N], row/col indices for C
    c_m = offs_m
    c_n = offs_n
    if C_LARGE:
        c_m = c_m.to(tl.int64)
        c_n = c_n.to(tl.int64)

    # c_ptrs: [BLOCK_SIZE_M, BLOCK_SIZE_N]
    #   element (i, j) points to C[pid_b, c_m[i], c_n[j]]
    c_ptrs = c_batch_ptr + stride_cm * c_m[:, None] + stride_cn * c_n[None, :]
    # mask out elements that fall outside logical (M, N) range
    c_mask = mask_m[:, None] & mask_n[None, :]
    # cast FP32 accumulator back to original dtype of C
    c = accumulator.to(c_ptr.dtype.element_ty)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def _softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    input_col_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    LOG: tl.constexpr = False,
):
    """
    Compute softmax, or log_softmax when ``LOG``, along the last dimension of a
    2D tensor. Each block handles one row of the input tensor, in a fixed block
    order, so the result never depends on the row count.
    """
    # Get the row index for this block
    row_idx = tl.program_id(0).to(tl.int64)

    # Compute base pointers for input and output rows
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    # Step 1: Find maximum value in the row for numerical stability
    max_val = -float("inf")
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(
            row_start_ptr + col_idx * input_col_stride, mask=mask, other=-float("inf")
        )

        # Update maximum
        max_val = tl.max(tl.maximum(vals, max_val))

    # Step 2: Compute sum of exp(x - max_val)
    sum_exp = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx * input_col_stride, mask=mask, other=0.0)

        # Compute exp(x - max_val) and accumulate
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(mask, exp_vals, 0.0))

    log_sum_exp = tl.log(sum_exp) if LOG else 0.0

    # Step 3: normalise
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx * input_col_stride, mask=mask)

        if LOG:
            output = vals - max_val - log_sum_exp
        else:
            output = tl.exp(vals - max_val) / sum_exp

        # Store results
        tl.store(output_row_start_ptr + col_idx, output, mask=mask)


def log_softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute log_softmax using Triton kernel.

    Args:
        input: Input tensor
        dim: Dimension along which to compute log_softmax
             (only -1 or last dim supported)

    Returns:
        Tensor with log_softmax applied along the specified dimension
    """
    if dim != -1 and dim != input.ndim - 1:
        raise ValueError(
            "This implementation only supports log_softmax along the last dimension"
        )

    # Flatten all dimensions except the last one
    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1])

    n_rows, n_cols = input_2d.shape

    # Allocate output tensor
    output = torch.empty((n_rows, n_cols), dtype=input_2d.dtype, device=input_2d.device)

    # Choose block size based on the number of columns
    BLOCK_SIZE = 1024

    # Launch kernel with one block per row
    grid = (n_rows,)
    _softmax_kernel[grid](
        input_2d,
        output,
        input_2d.stride(0),
        input_2d.stride(1),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        LOG=True,
    )
    # Reshape output back to original shape
    return output.reshape(original_shape)


def softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute softmax along the last dimension using a Triton kernel."""
    if dim != -1 and dim != input.ndim - 1:
        raise ValueError(
            "This implementation only supports softmax along the last dimension"
        )

    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1])
    n_rows, n_cols = input_2d.shape
    output = torch.empty((n_rows, n_cols), dtype=input_2d.dtype, device=input_2d.device)

    _softmax_kernel[(n_rows,)](
        input_2d,
        output,
        input_2d.stride(0),
        input_2d.stride(1),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=1024,
    )
    return output.reshape(original_shape)


@triton.jit
def mean_kernel(
    input_ptr,
    output_ptr,
    input_stride0,
    input_stride1,
    input_stride2,
    output_stride0,
    output_stride1,
    M,  # size before reduction dim
    N,  # size of reduction dim
    K,  # size after reduction dim
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for computing mean along a single dimension.
    Input is viewed as (M, N, K) where N is the dimension being reduced.
    """
    # Program ID gives us which output element we're computing
    pid = tl.program_id(0)

    # Compute output indices
    m_idx = pid // K
    k_idx = pid % K

    # Bounds check
    if m_idx >= M or k_idx >= K:
        return

    # Accumulate sum across reduction dimension
    acc = 0.0
    for n_start in range(0, N, BLOCK_SIZE):
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
        mask = n_offsets < N

        # Calculate input indices
        input_idx = (
            m_idx * input_stride0 + n_offsets * input_stride1 + k_idx * input_stride2
        )

        # Load and accumulate
        vals = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
        acc += tl.sum(vals)

    # Compute mean and store
    mean_val = acc / N
    output_idx = m_idx * output_stride0 + k_idx * output_stride1
    tl.store(output_ptr + output_idx, mean_val)


def mean_dim(
    input: torch.Tensor,
    dim: int,
    keepdim: bool = False,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Triton implementation of torch.mean with single dimension reduction.

    Args:
        input: Input tensor
        dim: Single dimension along which to compute mean
        keepdim: Whether to keep the reduced dimension
        dtype: Output dtype. If None, uses input dtype
               (or float32 for integer inputs)

    Returns:
        Tensor with mean values along specified dimension
    """
    # Validate inputs
    assert -input.ndim <= dim < input.ndim, (
        f"Invalid dimension {dim} for tensor with {input.ndim} dimensions"
    )

    # Handle negative dim
    if dim < 0:
        dim = dim + input.ndim

    # Handle dtype
    if dtype is None:
        if input.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            dtype = torch.float32
        else:
            dtype = input.dtype

    # Convert input to appropriate dtype if needed
    if input.dtype != dtype:
        input = input.to(dtype)

    # Get input shape and strides
    shape = list(input.shape)

    # Calculate dimensions for kernel
    M = 1
    for i in range(dim):
        M *= shape[i]

    N = shape[dim]

    K = 1
    for i in range(dim + 1, len(shape)):
        K *= shape[i]

    # Reshape input to 3D view (M, N, K)
    input_3d = input.reshape(M, N, K)

    # Create output shape
    if keepdim:
        output_shape = shape.copy()
        output_shape[dim] = 1
    else:
        output_shape = shape[:dim] + shape[dim + 1 :]

    # Create output tensor
    output = torch.empty(output_shape, dtype=dtype, device=input.device)

    # Reshape output for kernel
    output_2d = output.reshape(M, 1, K).squeeze(1) if keepdim else output.reshape(M, K)

    # Launch kernel
    grid = (M * K,)
    BLOCK_SIZE = 1024

    mean_kernel[grid](
        input_3d,
        output_2d,
        input_3d.stride(0),
        input_3d.stride(1),
        input_3d.stride(2),
        output_2d.stride(0),
        output_2d.stride(1) if output_2d.ndim > 1 else 0,
        M,
        N,
        K,
        BLOCK_SIZE,
    )

    return output


def mm_batch_invariant(a, b):
    return matmul_persistent(a, b)


def matmul_batch_invariant(a, b, *, out=None):
    # torch.matmul can handle various dimensions
    # For 2D x 2D, it's the same as mm
    if a.ndim == 2 and b.ndim == 2:
        result = matmul_persistent(a, b)
        if out is not None:
            out.copy_(result)
            return out
        return result
    elif b.ndim == 2:
        # Handle ND x 2D: Common for linear layers
        # (..., batch, seq, hidden) @ (hidden, out) -> (..., batch, seq, out)
        batch_dims = a.shape[:-1]
        hidden = a.shape[-1]
        out_dim = b.shape[-1]
        a_2d = a.reshape(-1, hidden)
        result_2d = matmul_persistent(a_2d, b)
        result = result_2d.reshape(batch_dims + (out_dim,))
        if out is not None:
            out.copy_(result)
            return out
        return result
    elif a.ndim >= 2 and b.ndim >= 3:
        # Generic handler for 2D x ND and ND x ND (except 1D)
        # Broadcast dims to ensure both matrices have the same shape
        # If 2D x ND, then unsqueeze to add a dim to a
        if a.ndim == 2:
            a = a.unsqueeze(0)
        broadcast_shape = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
        a = a.expand(broadcast_shape + a.shape[-2:])
        b = b.expand(broadcast_shape + b.shape[-2:])
        batch_dim = math.prod(broadcast_shape)
        # Reuse broadcast shape to get all dims except mm dims
        a_3d = a.reshape(batch_dim, a.shape[-2], a.shape[-1])
        b_3d = b.reshape(batch_dim, b.shape[-2], b.shape[-1])
        # Do batched matmul
        result_3d = bmm_batch_invariant(a_3d, b_3d)
        # Reshape back to [broadcast_shape, seq_a, seq_b]
        result = result_3d.reshape(broadcast_shape + (a.shape[-2], b.shape[-1]))
        if out is not None:
            out.copy_(result)
            return out
        return result
    else:
        raise ValueError(
            f"matmul_batch_invariant requires both inputs be at least 2D "
            f"got shapes {a.shape} and {b.shape}"
        )


def bmm_batch_invariant(a, b, *, out=None):
    # Batched matrix multiply: (B, M, K) x (B, K, N) -> (B, M, N)
    if not (a.ndim == 3 and b.ndim == 3):
        raise ValueError(
            f"bmm_batch_invariant expects 3D tensors, "
            f"got shapes {a.shape} and {b.shape}"
        )

    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"Batch dimensions of tensors must match, "
            f"but got {a.shape[0]} and {b.shape[0]}."
        )
    if a.shape[2] != b.shape[1]:
        raise ValueError(
            f"Incompatible inner dimensions for matmul: got {a.shape} and {b.shape}."
        )
    if a.dtype != b.dtype:
        raise ValueError(f"Incompatible dtypes: got {a.dtype} and {b.dtype}.")

    B, M, K = a.shape
    _, _, N = b.shape
    dtype = a.dtype

    if out is None:
        c = torch.empty((B, M, N), device=a.device, dtype=dtype)
    else:
        assert out.shape == (B, M, N), "out tensor has incorrect shape"
        assert out.dtype == dtype and out.device == a.device, "out tensor mismatch"
        c = out

    configs = {
        torch.bfloat16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": _fp16_block_size_n(),
            "BLOCK_SIZE_K": 64,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float32: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "num_stages": 3,
            "num_warps": 8,
        },
    }

    cfg = configs[dtype]
    # grid = (B, num_tiles_per_matrix)
    grid = (
        B,
        triton.cdiv(M, cfg["BLOCK_SIZE_M"]) * triton.cdiv(N, cfg["BLOCK_SIZE_N"]),
    )

    bmm_kernel[grid](
        a,
        b,
        c,
        B,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        A_LARGE=a.numel() > 2**31,
        B_LARGE=b.numel() > 2**31,
        C_LARGE=c.numel() > 2**31,
        **cfg,
    )

    return c


def _addmm_fused_bias(bias, beta, alpha, n):
    # The kernel adds bias into its fp32 accumulator indexed by output column,
    # so it can only absorb a row vector of width n, unscaled by alpha.
    if beta == 0 or alpha != 1:
        return None
    if bias.dim() == 2 and bias.shape[0] == 1:
        bias = bias.squeeze(0)
    if bias.dim() != 1 or bias.shape[0] != n:
        return None
    # fp32 keeps the scaling exact; the kernel casts the bias to fp32 anyway.
    return bias if beta == 1 else bias.float() * beta


def _addmm_impl(bias, a, b, beta, alpha, out):
    fused_bias = _addmm_fused_bias(bias, beta, alpha, b.shape[1])
    result = matmul_persistent(a, b, bias=fused_bias)
    if fused_bias is None:
        if alpha != 1:
            result = result * alpha
        if beta != 0:
            result = result + (bias if beta == 1 else beta * bias)
    if out is None:
        return result
    return out.copy_(result)


def addmm_batch_invariant(bias, a, b, *, beta=1, alpha=1):
    return _addmm_impl(bias, a, b, beta, alpha, None)


# Inductor lowers mm/addmm/bmm to ``extern_kernels.<op>(..., out=buf)``, which
# dispatches the ``.out`` overload rather than the default one. Registering only
# the default leaves the compiled path on the vendor GEMM, so a compiled
# ``torch.addmm`` is batch variant while the eager one is not.


def mm_out_batch_invariant(a, b, *, out):
    return matmul_batch_invariant(a, b, out=out)


def addmm_out_batch_invariant(bias, a, b, *, beta=1, alpha=1, out):
    return _addmm_impl(bias, a, b, beta, alpha, out)


def bmm_out_batch_invariant(a, b, *, out):
    return bmm_batch_invariant(a, b, out=out)


def _log_softmax_batch_invariant(input, dim, _half_to_float):
    if _half_to_float:
        return log_softmax(input.float(), dim=dim)
    return log_softmax(input, dim=dim)


def softmax_batch_invariant(input, dim, dtype=None):
    if dim == -1 or dim == input.ndim - 1:
        return softmax(input, dim=-1)

    # Reducing over an interior dimension: torch.sum picks its split count from
    # the tensor shape, so this path is only incidentally batch invariant.
    input_max = torch.amax(input, dim=dim, keepdim=True)
    # First subtract max for numerical stability (standard practice)
    input = input - input_max
    exp_x = torch.exp(input)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp_x


def mean_batch_invariant(input, dim, keepdim=False, dtype: torch.dtype | None = None):
    assert dtype is None or dtype == torch.float32, f"unsupported dtype: {dtype}"

    result = input.to(torch.float32)

    if len(dim) == 0:
        dim = [i for i in range(len(input.shape))]

    # Sort dimensions to reduce from largest to smallest to handle shifting dims
    # during iterative reduction.
    sorted_dims = sorted([d % input.ndim for d in dim], reverse=True)

    # Iteratively apply a deterministic mean.
    for d in sorted_dims:
        result = mean_dim(result, dim=d, keepdim=True)

    if not keepdim:
        # Squeeze the reduced dimensions.
        for d in sorted_dims:
            result = result.squeeze(d)

    return result


@triton.jit
def _rms_norm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
):
    """
    Compute RMS normalization along the last dimension of a 2D tensor.
    RMS Norm: y = x / sqrt(mean(x^2) + eps) * weight
    Each block handles one row of the input tensor.
    """
    row_idx = tl.program_id(0).to(tl.int64)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    # Step 1: Compute sum of squares in float32 to avoid overflow
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
        # Convert to float32 for accumulation to prevent overflow
        vals_f32 = vals.to(tl.float32)
        sq_vals = vals_f32 * vals_f32
        sum_sq += tl.sum(tl.where(mask, sq_vals, 0.0))

    # Step 2: Compute RMS (root mean square) in float32
    mean_sq = sum_sq / n_cols
    rms = tl.sqrt(mean_sq + eps)
    inv_rms = 1.0 / rms

    # Step 3: Normalize and apply weight
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
        # Compute in float32 then convert back to input dtype
        vals_f32 = vals.to(tl.float32)
        output_f32 = vals_f32 * inv_rms
        if HAS_WEIGHT:
            weight = tl.load(weight_ptr + col_idx, mask=mask, other=1.0)
            output_f32 = output_f32 * weight.to(tl.float32)
        output = output_f32.to(vals.dtype)
        tl.store(output_row_start_ptr + col_idx, output, mask=mask)


def rms_norm_batch_invariant(
    input: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float = 1e-6,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Compute RMS normalization using Triton kernel.


    Args:
        input: Input tensor of shape (..., hidden_size)
        weight: Weight tensor of shape (hidden_size,), or None to skip the
            per-channel multiply (``RMSNorm(has_weight=False)``)
        eps: Small constant for numerical stability
        residual: Optional residual tensor fused into the normalization path

    Returns:
        RMS normalized tensor, or ``(output, residual_out)`` when ``residual``
        is provided
    """
    if residual is not None:
        assert input.shape == residual.shape, (
            f"Input shape {input.shape} must match residual shape {residual.shape}"
        )
        import vllm._custom_ops as ops

        ops.fused_add_rms_norm(input, residual, weight, eps)
        return input, residual

    if weight is not None:
        assert weight.dim() == 1, "Weight must be 1-dimensional"
        assert input.shape[-1] == weight.shape[0], (
            f"Input last dimension ({input.shape[-1]}) must match "
            f"weight dimension ({weight.shape[0]})"
        )
        weight = weight.contiguous()

    # Flatten all dimensions except the last one
    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1])
    input_2d = input_2d.contiguous()

    n_rows, n_cols = input_2d.shape

    output = torch.empty_like(input_2d)
    BLOCK_SIZE = 1024
    grid = (n_rows,)
    _rms_norm_kernel[grid](
        input_2d,
        weight if weight is not None else input_2d,
        output,
        input_2d.stride(0),
        output.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_WEIGHT=weight is not None,
    )
    return output.reshape(original_shape)


def linear_batch_invariant(input, weight, bias=None):
    # Fold bias into the matmul's fp32 accumulator instead of adding it to the
    # rounded product. matmul_persistent is 2D-only, so flatten the batch dims.
    output = matmul_persistent(
        input.reshape(-1, input.shape[-1]), weight.t(), bias=bias
    )
    return output.view(*input.shape[:-1], output.shape[-1])


@triton.jit
def _fixed_order_sum_kernel(
    src_ptr,  # (WORLD_SIZE, numel) gathered contributions, rank-major
    out_ptr,  # (numel,)
    numel,
    stride_rank,
    WORLD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LARGE: tl.constexpr,
):
    """Sum ``WORLD_SIZE`` contributions in ascending rank order, fp32 accumulator.

    An output element sums exactly ``WORLD_SIZE`` values in a compile-time
    constant order and rounds once, so the result depends only on the rank
    ordering and never on how many elements are being reduced.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if LARGE:
        offs = offs.to(tl.int64)
    mask = offs < numel

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for r in tl.static_range(WORLD_SIZE):
        vals = tl.load(src_ptr + r * stride_rank + offs, mask=mask, other=0.0)
        acc += vals.to(tl.float32)

    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def all_reduce_batch_invariant(
    input_: torch.Tensor, group: "torch.distributed.ProcessGroup | None" = None
) -> torch.Tensor:
    """Sum all-reduce whose result does not depend on the message size.

    Library all-reduces (NCCL/RCCL) pick their algorithm, channel count and chunk
    boundaries from the message size, so the order in which a given element's
    contributions are summed changes with the number of tokens in the batch.

    Instead, all-gather the contributions -- pure data movement, so bitwise
    reproducible at any size -- and reduce them with ``_fixed_order_sum_kernel``.
    """
    import torch.distributed as dist

    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input_

    x = input_.contiguous()
    numel = x.numel()
    gathered = torch.empty((world_size, numel), dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(gathered, x.view(-1), group=group)

    out = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(numel, BLOCK_SIZE),)
    _fixed_order_sum_kernel[grid](
        gathered,
        out.view(-1),
        numel,
        numel,
        WORLD_SIZE=world_size,
        BLOCK_SIZE=BLOCK_SIZE,
        LARGE=gathered.numel() > 2**31,
    )
    return out


def reduce_scatter_batch_invariant(
    input_: torch.Tensor,
    group: "torch.distributed.ProcessGroup | None" = None,
    sizes: list[int] | None = None,
) -> torch.Tensor:
    """Sum reduce-scatter over dim 0 whose result does not depend on the size.

    ``ncclReduceScatter`` picks its chunking from the message size just as
    ``ncclAllReduce`` does. Route around it the same way
    ``all_reduce_batch_invariant`` does: an all-to-all sends rank ``d`` exactly
    the rows it owns from every rank -- pure data movement, so bitwise
    reproducible however the library chunks it -- and lands them rank-major,
    which is the layout ``_fixed_order_sum_kernel`` already consumes.

    ``sizes`` (the reduce-scatterv case) changes only *which* rows a rank
    receives, never how the received contributions are summed, so variable
    shard sizes are invariant for the same reason.

    Args:
        input_: Contiguous full-size input; dim 0 is the scattered axis.
        group: Process group, defaults to the world group.
        sizes: Per-rank row counts. Uniform split when ``None``.

    Returns:
        This rank's shard of the sum.
    """
    import torch.distributed as dist

    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input_

    x = input_.contiguous()
    if sizes is None:
        assert x.shape[0] % world_size == 0
        sizes = [x.shape[0] // world_size] * world_size
    else:
        assert len(sizes) == world_size
        assert x.shape[0] == sum(sizes)

    rows = sizes[dist.get_rank(group)]
    out = torch.empty((rows, *x.shape[1:]), dtype=x.dtype, device=x.device)
    recv = torch.empty(
        (world_size * rows, *x.shape[1:]), dtype=x.dtype, device=x.device
    )
    # Every rank takes part even when its own shard is empty: it still has rows
    # to send to the others.
    dist.all_to_all_single(
        recv,
        x,
        output_split_sizes=[rows] * world_size,
        input_split_sizes=sizes,
        group=group,
    )

    numel = out.numel()
    if numel == 0:
        return out
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(numel, BLOCK_SIZE),)
    _fixed_order_sum_kernel[grid](
        recv.view(-1),
        out.view(-1),
        numel,
        numel,
        WORLD_SIZE=world_size,
        BLOCK_SIZE=BLOCK_SIZE,
        LARGE=recv.numel() > 2**31,
    )
    return out


_batch_invariant_MODE = False
_batch_invariant_LIB = None


# Save the eager path from constantly calling get_max_shared_memory_bytes
# torch.compiler.assume_constant_result is necessary for Dynamo to not trace
@lru_cache(maxsize=1)
@torch.compiler.assume_constant_result
def _fp16_block_size_n() -> int:
    if current_platform.is_xpu() or get_max_shared_memory_bytes() <= 106496:
        return 128
    return 256


def enable_batch_invariant_mode():
    global _batch_invariant_MODE, _batch_invariant_LIB

    if _batch_invariant_MODE:
        return

    _batch_invariant_MODE = True
    _batch_invariant_LIB = torch.library.Library("aten", "IMPL")

    key = current_platform.dispatch_key

    if current_platform.is_cuda():
        if current_platform.is_device_capability_family(80):
            # SM80 (Ampere) cannot rely on cuBLASLt-only determinism; install the
            # triton persistent matmul overrides for mm/addmm/matmul/linear.
            _batch_invariant_LIB.impl("aten::mm", mm_batch_invariant, key)
            _batch_invariant_LIB.impl("aten::addmm", addmm_batch_invariant, key)
            _batch_invariant_LIB.impl("aten::mm.out", mm_out_batch_invariant, key)
            _batch_invariant_LIB.impl("aten::addmm.out", addmm_out_batch_invariant, key)
            _batch_invariant_LIB.impl("aten::matmul", matmul_batch_invariant, key)
            _batch_invariant_LIB.impl("aten::linear", linear_batch_invariant, key)
        else:
            # Hopper (SM90) and Blackwell (SM100): the only source of batch
            # variance is split-k, which we disable via the cuBLAS workspace
            # config.
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
            os.environ["CUBLASLT_WORKSPACE_SIZE"] = "1"

    elif current_platform.is_rocm():
        _batch_invariant_LIB.impl("aten::mm", mm_batch_invariant, key)
        _batch_invariant_LIB.impl("aten::addmm", addmm_batch_invariant, key)
        _batch_invariant_LIB.impl("aten::mm.out", mm_out_batch_invariant, key)
        _batch_invariant_LIB.impl("aten::addmm.out", addmm_out_batch_invariant, key)
        _batch_invariant_LIB.impl("aten::matmul", matmul_batch_invariant, key)
        _batch_invariant_LIB.impl("aten::linear", linear_batch_invariant, key)

    elif current_platform.is_xpu():
        _batch_invariant_LIB.impl("aten::mm", mm_batch_invariant, key)
        _batch_invariant_LIB.impl("aten::addmm", addmm_batch_invariant, key)
        _batch_invariant_LIB.impl("aten::mm.out", mm_out_batch_invariant, key)
        _batch_invariant_LIB.impl("aten::addmm.out", addmm_out_batch_invariant, key)
        # TODO: register matmul and linear for XPU
        # once suitable Triton kernels are implemented

    _batch_invariant_LIB.impl("aten::_log_softmax", _log_softmax_batch_invariant, key)
    _batch_invariant_LIB.impl("aten::softmax", softmax_batch_invariant, key)
    _batch_invariant_LIB.impl("aten::_softmax", softmax_batch_invariant, key)
    _batch_invariant_LIB.impl("aten::mean.dim", mean_batch_invariant, key)
    # torch 2.12+ registers a built-in Triton bmm kernel for CUDA
    # (torch._native.ops.bmm_outer_product), so we need allow_override
    # to replace it at the dispatcher level.
    _batch_invariant_LIB.impl(
        "aten::bmm", bmm_batch_invariant, key, allow_override=True
    )
    _batch_invariant_LIB.impl(
        "aten::bmm.out", bmm_out_batch_invariant, key, allow_override=True
    )
    torch.bmm = bmm_batch_invariant

    reduced_precision_val = (
        (False, False) if is_torch_equal_or_newer("2.10.0") else False
    )
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
        reduced_precision_val
    )
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
        reduced_precision_val
    )
    if current_platform.is_cuda():
        torch.backends.cuda.preferred_blas_library(backend="cublaslt")


def override_envs_for_invariance():
    if not current_platform.is_rocm():
        # Symmetric memory is only reachable behind an is_cuda() check and
        # hipBLASLt does not read CUBLAS_WORKSPACE_CONFIG. NVLS is NVIDIA-only,
        # CollNet is a multi-node network offload, and RCCL spells the P2P knob
        # RCCL_P2P_NET_DISABLE.
        os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["NCCL_LAUNCH_MODE"] = "GROUP"
        os.environ["NCCL_COLLNET_ENABLE"] = "0"
        os.environ["NCCL_NVLS_ENABLE"] = "0"
        os.environ["NCCL_P2P_NET_DISABLE"] = "1"

        # Pinning the algorithm, protocol and channel count serialises NCCL's
        # reduction so that it does not depend on the message size.
        os.environ["NCCL_MIN_NCHANNELS"] = "1"
        os.environ["NCCL_MAX_NCHANNELS"] = "1"
        os.environ["NCCL_PROTO"] = "Simple"
        os.environ["NCCL_ALGO"] = "allreduce:tree"
        os.environ["NCCL_NTHREADS"] = "1"
        os.environ["NCCL_SOCKET_NTHREADS"] = "1"
    else:
        os.environ["VLLM_ROCM_USE_SKINNY_GEMM"] = "0"

    # torch.compile settings
    os.environ["VLLM_USE_AOT_COMPILE"] = "0"


def init_batch_invariance():
    # this will hit all the csrc overrides as well
    if envs.VLLM_BATCH_INVARIANT:
        override_envs_for_invariance()
        enable_batch_invariant_mode()

        # Disable TF32 for batch invariance - it causes non-deterministic rounding
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        torch.backends.cudnn.rnn.fp32_precision = "ieee"
