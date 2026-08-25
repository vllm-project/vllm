# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import direct_register_custom_op, is_torch_equal_or_newer
from vllm.v1.attention.backends.registry import AttentionBackendEnum

logger = init_logger(__name__)

_ROUTER_GEMM_K = 2048
_ROUTER_GEMM_N = 128
_FULL_K_MAX_M = {
    torch.bfloat16: 2048,
    torch.float16: 2048,
    # The FP32 one-row kernel deliberately trades weight reuse for a stable,
    # non-TF32 per-row reduction. Past this decode-oriented range persistent
    # is faster, so auto must fall back before a prefill regression.
    torch.float32: 128,
}
_FP32_FULL_K_RTOL = 1e-5
_FP32_FULL_K_ATOL = 1e-6


def _matmul_launch_metadata(
    grid: Callable[..., Any], kernel: Any, args: dict[str, Any]
) -> dict[str, Any]:
    ret = {}
    m, n, k = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={m}, N={n}, K={k}]"
    if "tiles_per_update" in args:
        ret["name"] = (
            f"{kernel.name} [M={m}, N={n}, K={k}, "
            f"tiles_per_update={args['tiles_per_update']:02}]"
        )
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * m * n * k
    ret["bytes"] = bytes_per_elem * (m * k + n * k + m * n)
    return ret


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


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
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
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
            tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        if C_LARGE:
            offs_cm = offs_cm.to(tl.int64)
            offs_cn = offs_cn.to(tl.int64)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if HAS_BIAS:
            bias_ptrs = bias_ptr + offs_cn
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
            "BLOCK_SIZE_N": 256,
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
    # print(a.device, b.device, c.device)
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
        NUM_SMS=NUM_SMS,  #
        A_LARGE=a.numel() > 2**31,
        B_LARGE=b.numel() > 2**31,
        C_LARGE=c.numel() > 2**31,
        HAS_BIAS=bias is not None,
        **configs[dtype],
    )
    return c


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_full_k(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    FP32_INPUT: tl.constexpr,
):
    """One CTA per output tile, with a fixed, ascending full-K reduction."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    safe_m = tl.where(offs_m < M, offs_m, 0)
    safe_n = tl.where(offs_n < N, offs_n, 0)
    safe_m = tl.max_contiguous(tl.multiple_of(safe_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
    safe_n = tl.max_contiguous(tl.multiple_of(safe_n, BLOCK_SIZE_N), BLOCK_SIZE_N)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
        a = tl.load(
            a_ptr + safe_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=offs_k[None, :] < K,
            other=0.0,
        )
        b = tl.load(
            b_ptr + offs_k[:, None] * stride_bk + safe_n[None, :] * stride_bn,
            mask=offs_k[:, None] < K,
            other=0.0,
        )
        if FP32_INPUT:
            accumulator = tl.dot(a, b, accumulator, input_precision="ieee")
        else:
            accumulator = tl.dot(a, b, accumulator)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        accumulator.to(c_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


_FULL_K_CONFIGS = {
    torch.bfloat16: {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 16,
        "BLOCK_SIZE_K": 64,
        "num_stages": 4,
        "num_warps": 2,
    },
    torch.float16: {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 16,
        "BLOCK_SIZE_K": 64,
        "num_stages": 4,
        "num_warps": 2,
    },
}

_FP32_FULL_K_ONE_ROW_CONFIG = {
    # One CTA owns one logical row and a 32-column tile. Keeping this layout
    # independent of M makes a row's FP32 reduction bitwise stable as batches
    # grow or move the row to a different position.
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 64,
    "num_stages": 3,
    "num_warps": 4,
}


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_full_k_fp32_one_row(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """One row per CTA with a fixed IEEE FP32 reduction over full K."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    accumulator = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
        a = tl.load(
            a_ptr + pid_m * stride_am + offs_k * stride_ak,
            mask=offs_k < K,
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(a[:, None] * b, axis=0)

    tl.store(
        c_ptr + pid_m * stride_cm + offs_n * stride_cn,
        accumulator,
        mask=offs_n < N,
    )


def matmul_full_k(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Deterministic GEMM with no split-K, atomics, or cross-CTA reduction."""
    assert a.ndim == 2 and b.ndim == 2, "matmul_full_k expects 2D tensors"
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    assert bias is None, "matmul_full_k does not support bias"
    M, K = a.shape
    _, N = b.shape
    assert K == _ROUTER_GEMM_K and N == _ROUTER_GEMM_N, (
        "matmul_full_k is specialized for the M0 Router shape"
    )
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    if dtype == torch.float32:
        cfg = _FP32_FULL_K_ONE_ROW_CONFIG
        grid = (M, triton.cdiv(N, cfg["BLOCK_SIZE_N"]))
        matmul_kernel_full_k_fp32_one_row[grid](
            a,
            b,
            c,
            M,
            N=N,
            K=K,
            stride_am=a.stride(0),
            stride_ak=a.stride(1),
            stride_bk=b.stride(0),
            stride_bn=b.stride(1),
            stride_cm=c.stride(0),
            stride_cn=c.stride(1),
            **cfg,
        )
        return c

    cfg = _FULL_K_CONFIGS[dtype]
    grid = (
        triton.cdiv(M, cfg["BLOCK_SIZE_M"]),
        triton.cdiv(N, cfg["BLOCK_SIZE_N"]),
    )
    matmul_kernel_full_k[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        FP32_INPUT=False,
        **cfg,
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
def _log_softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute log_softmax along the last dimension of a 2D tensor.
    Each block handles one row of the input tensor.
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
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=-float("inf"))

        # Update maximum
        max_val = tl.max(tl.maximum(vals, max_val))

    # Step 2: Compute sum of exp(x - max_val)
    sum_exp = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)

        # Compute exp(x - max_val) and accumulate
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(mask, exp_vals, 0.0))

    # Compute log(sum_exp)
    log_sum_exp = tl.log(sum_exp)

    # Step 3: Compute final log_softmax values: x - max_val - log_sum_exp
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask)

        # Compute log_softmax
        output = vals - max_val - log_sum_exp

        # Store results
        tl.store(output_row_start_ptr + col_idx, output, mask=mask)


def log_softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute log_softmax using Triton kernel.

    Args:
        input: Input tensor
        dim: Dimension along which to compute log_softmax
             (only -1 or last dim supported)
    >> Stashed changes
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
    input_2d = input_2d.contiguous()

    n_rows, n_cols = input_2d.shape

    # Allocate output tensor
    output = torch.empty_like(input_2d)

    # Choose block size based on the number of columns
    BLOCK_SIZE = 1024

    # Launch kernel with one block per row
    grid = (n_rows,)
    _log_softmax_kernel[grid](
        input_2d,
        output,
        input_2d.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Reshape output back to original shape
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
    elif a.ndim == 3 and b.ndim == 3:
        # Handle batched case like bmm
        return bmm_batch_invariant(a, b, out=out)
    elif a.ndim == 3 and b.ndim == 2:
        # Handle 3D x 2D: common for linear layers
        # (batch, seq, hidden) @ (hidden, out) -> (batch, seq, out)
        # Reshape to 2D, do mm, reshape back
        batch, seq, hidden = a.shape
        a_2d = a.reshape(-1, hidden)
        result_2d = matmul_persistent(a_2d, b)
        result = result_2d.reshape(batch, seq, -1)
        if out is not None:
            out.copy_(result)
            return out
        return result
    elif a.ndim == 2 and b.ndim == 3:
        # Handle 2D x 3D: (M, K) @ (B, K, N) -> (B, M, N)
        # By broadcasting `a` to 3D, we can reuse the batched matrix
        # multiplication logic.
        a_expanded = a.unsqueeze(0).expand(b.shape[0], -1, -1)
        return bmm_batch_invariant(a_expanded, b, out=out)
    elif a.ndim == 4 and b.ndim == 4:
        # Handle 4D attention tensors: [batch, heads, seq, dim]
        # Reshape to 3D, process, reshape back
        batch, heads, seq_a, dim_a = a.shape
        _, _, dim_b, seq_b = b.shape

        # Reshape to [batch*heads, seq_a, dim_a]
        a_3d = a.reshape(batch * heads, seq_a, dim_a)
        b_3d = b.reshape(batch * heads, dim_b, seq_b)

        # Do batched matmul
        result_3d = bmm_batch_invariant(a_3d, b_3d)

        # Reshape back to [batch, heads, seq_a, seq_b]
        result = result_3d.reshape(batch, heads, seq_a, seq_b)

        if out is not None:
            out.copy_(result)
            return out
        return result
    else:
        raise ValueError(
            f"matmul_batch_invariant currently only supports 2D x 2D, 3D x 3D, "
            f"3D x 2D, 2D x 3D, and 4D x 4D, "
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
            "BLOCK_SIZE_N": 256,
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


def addmm_batch_invariant(bias, a, b):
    return matmul_persistent(a, b, bias=bias)


def _log_softmax_batch_invariant(input, dim, _half_to_float):
    assert not _half_to_float, "not implemented"
    return log_softmax(input, dim=dim)


def softmax_batch_invariant(input, dim, dtype=None):
    # Compute softmax in a deterministic way
    # First subtract max for numerical stability (standard practice)
    input_max = torch.amax(input, dim=dim, keepdim=True)
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
        weight = tl.load(weight_ptr + col_idx, mask=mask, other=1.0)
        # Compute in float32 then convert back to input dtype
        vals_f32 = vals.to(tl.float32)
        weight_f32 = weight.to(tl.float32)
        output_f32 = vals_f32 * inv_rms * weight_f32
        output = output_f32.to(vals.dtype)
        tl.store(output_row_start_ptr + col_idx, output, mask=mask)


def rms_norm(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute RMS normalization using Triton kernel.

    RMS Norm normalizes the input by the root mean square and scales by weight:
    output = input / sqrt(mean(input^2) + eps) * weight

    Args:
        input: Input tensor of shape (..., hidden_size)
        weight: Weight tensor of shape (hidden_size,)
        eps: Small constant for numerical stability

    Returns:
        Tensor with RMS normalization applied along the last dimension
    """
    assert weight.dim() == 1, "Weight must be 1-dimensional"
    assert input.shape[-1] == weight.shape[0], (
        f"Input last dimension ({input.shape[-1]}) must match "
        f"weight dimension ({weight.shape[0]})"
    )

    # Flatten all dimensions except the last one
    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1])
    input_2d = input_2d.contiguous()
    weight = weight.contiguous()

    n_rows, n_cols = input_2d.shape

    output = torch.empty_like(input_2d)
    BLOCK_SIZE = 1024
    grid = (n_rows,)
    _rms_norm_kernel[grid](
        input_2d,
        weight,
        output,
        input_2d.stride(0),
        output.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output.reshape(original_shape)


def rms_norm_batch_invariant(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Batch-invariant wrapper for RMS normalization.

    This function provides a deterministic, batch-invariant implementation
    of RMS normalization for use with the batch_invariant mode.

    Args:
        input: Input tensor of shape (..., hidden_size)
        weight: Weight tensor of shape (hidden_size,)
        eps: Small constant for numerical stability

    Returns:
        RMS normalized tensor
    """
    return rms_norm(input, weight, eps=eps)


def linear_batch_invariant(input, weight, bias=None):
    output = matmul_batch_invariant(input, weight.t())

    if bias is not None:
        output = output + bias
    return output


_batch_invariant_MODE = False
_batch_invariant_LIB = None
_original_torch_bmm = None
_original_fp16_reduction_precision = None
_original_bf16_reduction_precision = None
_original_cublas_workspace_cfg = None
_original_cublaslt_workspace_size = None


def enable_batch_invariant_mode():
    global _batch_invariant_MODE, _batch_invariant_LIB, _original_torch_bmm
    global _original_fp16_reduction_precision, _original_bf16_reduction_precision
    global _original_cublas_workspace_cfg, _original_cublaslt_workspace_size
    if _batch_invariant_MODE:
        return

    _batch_invariant_MODE = True
    _batch_invariant_LIB = torch.library.Library("aten", "IMPL")

    if (
        current_platform.is_device_capability_family(100)
        or current_platform.is_device_capability(80)
        or current_platform.is_device_capability(89)
    ):
        # For PyTorch 2.9, B200 uses GEMV for bs=1
        # Requires https://github.com/pytorch/pytorch/pull/166735
        _batch_invariant_LIB.impl("aten::mm", mm_batch_invariant, "CUDA")
        _batch_invariant_LIB.impl("aten::addmm", addmm_batch_invariant, "CUDA")
        _batch_invariant_LIB.impl("aten::matmul", matmul_batch_invariant, "CUDA")
        _batch_invariant_LIB.impl("aten::linear", linear_batch_invariant, "CUDA")
    else:
        # Only source of batch invariance for Hopper is split-k, can disable through
        # cuBLAS workspace config
        _original_cublas_workspace_cfg = os.environ.get("CUBLAS_WORKSPACE_CONFIG", None)
        _original_cublaslt_workspace_size = os.environ.get(
            "CUBLASLT_WORKSPACE_SIZE", None
        )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        os.environ["CUBLASLT_WORKSPACE_SIZE"] = "1"

    _batch_invariant_LIB.impl(
        "aten::_log_softmax", _log_softmax_batch_invariant, "CUDA"
    )
    _batch_invariant_LIB.impl("aten::softmax", softmax_batch_invariant, "CUDA")
    _batch_invariant_LIB.impl("aten::_softmax", softmax_batch_invariant, "CUDA")
    _batch_invariant_LIB.impl("aten::mean.dim", mean_batch_invariant, "CUDA")

    # Also monkeypatch torch.bmm directly as a fallback
    _batch_invariant_LIB.impl("aten::bmm", bmm_batch_invariant, "CUDA")
    _original_torch_bmm = torch.bmm
    torch.bmm = bmm_batch_invariant

    _original_bf16_reduction_precision = (
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    )
    _original_fp16_reduction_precision = (
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    )

    reduced_precision_val = (
        (False, False) if is_torch_equal_or_newer("2.10.0") else False
    )
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
        reduced_precision_val
    )
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
        reduced_precision_val
    )
    torch.backends.cuda.preferred_blas_library(backend="cublaslt")


def _read_vllm_batch_invariant() -> bool:
    val = os.getenv("VLLM_BATCH_INVARIANT", "0")
    try:
        return int(val) != 0
    except ValueError:
        return False


VLLM_BATCH_INVARIANT: bool = _read_vllm_batch_invariant()

_ROUTER_GEMM_CHOICES = ("auto", "persistent", "full_k", "deepgemm")


def _read_router_gemm_choice() -> str:
    # Auto remains opt-in until every end-to-end promotion gate passes.
    value = os.getenv("VLLM_BATCH_INVARIANT_ROUTER_GEMM", "persistent").strip()
    if value not in _ROUTER_GEMM_CHOICES:
        raise ValueError(
            "VLLM_BATCH_INVARIANT_ROUTER_GEMM must be one of "
            f"{_ROUTER_GEMM_CHOICES}, got {value!r}"
        )
    return value


VLLM_BATCH_INVARIANT_ROUTER_GEMM = _read_router_gemm_choice()


@dataclass(frozen=True)
class RouterGemmSignature:
    device_type: str
    device_index: int | None
    weight_device_type: str
    weight_device_index: int | None
    input_dtype: torch.dtype
    weight_dtype: torch.dtype
    input_shape: tuple[int, ...]
    weight_shape: tuple[int, ...]
    input_stride: tuple[int, ...]
    weight_stride: tuple[int, ...]
    bias_shape: tuple[int, ...] | None
    bias_device_type: str | None
    bias_device_index: int | None
    bias_dtype: torch.dtype | None
    bias_stride: tuple[int, ...] | None


@dataclass(frozen=True)
class RouterGemmBackendDecision:
    requested: str
    selected: str | None
    reason: str
    signature: RouterGemmSignature
    preflighted: bool


_router_gemm_backend_cache: dict[
    tuple[str, RouterGemmSignature], RouterGemmBackendDecision
] = {}
_router_gemm_backend_lock = threading.Lock()
_router_deepgemm_load_lock = threading.Lock()
_router_deepgemm_impl: Callable[..., Any] | None = None
_router_deepgemm_load_error: str | None = None


def _router_gemm_signature(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> RouterGemmSignature:
    return RouterGemmSignature(
        device_type=input.device.type,
        device_index=input.device.index,
        weight_device_type=weight.device.type,
        weight_device_index=weight.device.index,
        input_dtype=input.dtype,
        weight_dtype=weight.dtype,
        input_shape=tuple(input.shape),
        weight_shape=tuple(weight.shape),
        input_stride=tuple(input.stride()),
        weight_stride=tuple(weight.stride()),
        bias_shape=None if bias is None else tuple(bias.shape),
        bias_device_type=None if bias is None else bias.device.type,
        bias_device_index=None if bias is None else bias.device.index,
        bias_dtype=None if bias is None else bias.dtype,
        bias_stride=None if bias is None else tuple(bias.stride()),
    )


def _reset_router_gemm_backend_cache(*, reset_deepgemm: bool = False) -> None:
    """Clear preflight decisions. Intended for tests and controlled benchmarks."""
    with _router_gemm_backend_lock:
        _router_gemm_backend_cache.clear()
    if reset_deepgemm:
        global _router_deepgemm_impl, _router_deepgemm_load_error
        with _router_deepgemm_load_lock:
            _router_deepgemm_impl = None
            _router_deepgemm_load_error = None


def _load_router_deepgemm_impl() -> Callable[..., Any]:
    global _router_deepgemm_impl, _router_deepgemm_load_error
    if _router_deepgemm_impl is not None:
        return _router_deepgemm_impl
    if _router_deepgemm_load_error is not None:
        raise RuntimeError(_router_deepgemm_load_error)

    with _router_deepgemm_load_lock:
        if _router_deepgemm_impl is not None:
            return _router_deepgemm_impl
        if _router_deepgemm_load_error is not None:
            raise RuntimeError(_router_deepgemm_load_error)
        try:
            import importlib

            deep_gemm = importlib.import_module("deep_gemm")
            impl = getattr(deep_gemm, "bf16_gemm_nt", None)
            if not callable(impl):
                raise AttributeError("deep_gemm.bf16_gemm_nt is not available")
        except Exception as exc:
            _router_deepgemm_load_error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(_router_deepgemm_load_error) from exc
        _router_deepgemm_impl = impl
        return impl


def _router_deepgemm_available() -> tuple[bool, str]:
    try:
        _load_router_deepgemm_impl()
    except Exception as exc:
        return False, str(exc)
    return True, "deep_gemm.bf16_gemm_nt is available"


def _router_deepgemm_op_impl(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    impl = _load_router_deepgemm_impl()
    output = torch.empty(
        (input.shape[0], weight.shape[0]),
        device=input.device,
        dtype=input.dtype,
    )
    impl(input, weight, output)
    return output


def _router_deepgemm_op_fake(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return input.new_empty((input.shape[0], weight.shape[0]))


direct_register_custom_op(
    op_name="router_deepgemm",
    op_func=_router_deepgemm_op_impl,
    fake_impl=_router_deepgemm_op_fake,
)


def _router_common_fast_path_error(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> str | None:
    if input.ndim != 2 or weight.ndim != 2:
        return f"expected 2D input/weight, got {input.ndim}D/{weight.ndim}D"
    if input.shape[0] == 0:
        return "M must be positive"
    if tuple(input.shape[1:]) != (_ROUTER_GEMM_K,):
        return f"expected input K={_ROUTER_GEMM_K}, got shape {tuple(input.shape)}"
    if tuple(weight.shape) != (_ROUTER_GEMM_N, _ROUTER_GEMM_K):
        return (
            f"expected weight shape ({_ROUTER_GEMM_N}, {_ROUTER_GEMM_K}), "
            f"got {tuple(weight.shape)}"
        )
    if bias is not None:
        return "bias is not supported by Router GEMM fast paths"
    if input.device.type != "cuda" or weight.device.type != "cuda":
        return f"expected CUDA tensors, got {input.device}/{weight.device}"
    if input.device != weight.device:
        return f"input/weight devices differ: {input.device}/{weight.device}"
    if input.dtype != weight.dtype:
        return f"input/weight dtypes differ: {input.dtype}/{weight.dtype}"
    if input.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return f"unsupported dtype {input.dtype}"
    if not input.is_contiguous() or not weight.is_contiguous():
        return "input and weight must use contiguous row-major layout"
    try:
        capability = torch.cuda.get_device_capability(input.device)
    except Exception as exc:
        return f"could not query CUDA capability: {type(exc).__name__}: {exc}"
    if capability != (9, 0):
        return f"expected SM90, got sm{capability[0]}{capability[1]}"
    return None


def _router_deepgemm_guard_error(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> str | None:
    common_error = _router_common_fast_path_error(input, weight, bias)
    if common_error is not None:
        return common_error
    if input.dtype != torch.bfloat16:
        return f"DeepGEMM requires torch.bfloat16, got {input.dtype}"
    available, reason = _router_deepgemm_available()
    if not available:
        return reason
    return None


def _router_full_k_guard_error(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> str | None:
    common_error = _router_common_fast_path_error(input, weight, bias)
    if common_error is not None:
        return common_error
    max_m = _FULL_K_MAX_M[input.dtype]
    if input.shape[0] > max_m:
        return f"M={input.shape[0]} exceeds full-K limit {max_m}"
    return None


def _auto_router_gemm_candidates(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple[list[str], list[str]]:
    common_error = _router_common_fast_path_error(input, weight, bias)
    if common_error is not None:
        return ["persistent"], [f"fast paths ineligible: {common_error}"]

    candidates: list[str] = []
    skipped: list[str] = []
    if input.dtype == torch.bfloat16:
        deepgemm_error = _router_deepgemm_guard_error(input, weight, bias)
        if deepgemm_error is None:
            candidates.append("deepgemm")
        else:
            skipped.append(f"deepgemm skipped: {deepgemm_error}")

    full_k_error = _router_full_k_guard_error(input, weight, bias)
    if full_k_error is None:
        candidates.append("full_k")
    else:
        skipped.append(f"full_k skipped: {full_k_error}")
    candidates.append("persistent")
    return candidates, skipped


def _select_router_gemm_backend(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    requested_mode: str | None = None,
) -> RouterGemmBackendDecision:
    """Return the first statically eligible backend without running a kernel."""
    requested = (
        VLLM_BATCH_INVARIANT_ROUTER_GEMM if requested_mode is None else requested_mode
    )
    if requested not in _ROUTER_GEMM_CHOICES:
        raise ValueError(
            f"requested_mode must be one of {_ROUTER_GEMM_CHOICES}, got {requested!r}"
        )
    signature = _router_gemm_signature(input, weight, bias)

    if requested == "persistent":
        selected = "persistent"
        reason = "persistent explicitly requested"
    elif requested == "deepgemm":
        error = _router_deepgemm_guard_error(input, weight, bias)
        if error is not None:
            raise ValueError(f"forced deepgemm Router GEMM is ineligible: {error}")
        selected = "deepgemm"
        reason = "forced DeepGEMM passed static guards"
    elif requested == "full_k":
        error = _router_full_k_guard_error(input, weight, bias)
        if error is not None:
            raise ValueError(f"forced full_k Router GEMM is ineligible: {error}")
        selected = "full_k"
        reason = "forced full-K passed static guards"
    else:
        candidates, skipped = _auto_router_gemm_candidates(input, weight, bias)
        selected = candidates[0]
        reason_parts = skipped + [f"auto first eligible backend: {selected}"]
        reason = "; ".join(reason_parts)

    return RouterGemmBackendDecision(
        requested=requested,
        selected=selected,
        reason=reason,
        signature=signature,
        preflighted=False,
    )


def _run_router_deepgemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    assert bias is None
    return torch.ops.vllm.router_deepgemm(input, weight)


def _run_router_full_k(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    assert bias is None
    return matmul_full_k(input, weight.t())


def _run_router_persistent(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    return linear_batch_invariant(input, weight, bias)


def _validate_router_gemm_output(
    output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
) -> None:
    expected_shape = (*input.shape[:-1], weight.shape[0])
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            "Router GEMM returned shape "
            f"{tuple(output.shape)}, expected {expected_shape}"
        )
    if output.dtype != input.dtype:
        raise RuntimeError(
            f"Router GEMM returned dtype {output.dtype}, expected {input.dtype}"
        )
    if output.device != input.device:
        raise RuntimeError(
            f"Router GEMM returned device {output.device}, expected {input.device}"
        )


def _validate_router_gemm_bitwise(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    comparison: str,
) -> None:
    if not torch.equal(actual, expected):
        raise RuntimeError(f"Router GEMM failed bitwise {comparison} validation")


def _validate_router_gemm_fp32_accuracy(
    output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
) -> None:
    """Reject an FP32 full-K result that is outside the FP64 error envelope.

    The persistent Triton kernel intentionally remains an independent fallback
    and uses its own arithmetic implementation, so it cannot be an exact FP32
    oracle.  This preflight runs before Graph capture and compares the full
    signature against a FP64 reference instead.
    """
    assert input.dtype == torch.float32
    reference = input.to(torch.float64) @ weight.to(torch.float64).t()
    actual = output.to(torch.float64)
    if torch.allclose(
        actual,
        reference,
        rtol=_FP32_FULL_K_RTOL,
        atol=_FP32_FULL_K_ATOL,
    ):
        return

    absolute_error = (actual - reference).abs()
    relative_error = absolute_error / reference.abs().clamp_min(
        torch.finfo(reference.dtype).eps
    )
    raise RuntimeError(
        "Router GEMM failed full-K/FP64 accuracy validation: "
        f"rtol={_FP32_FULL_K_RTOL:g}, atol={_FP32_FULL_K_ATOL:g}, "
        f"max_abs={absolute_error.max().item():.6g}, "
        f"max_rel={relative_error.max().item():.6g}"
    )


def _synchronize_router_gemm(input: torch.Tensor) -> None:
    if input.device.type == "cuda":
        torch.cuda.synchronize(input.device)


def _cuda_graph_smoke_router_backend(
    runner: Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor],
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> None:
    """Warm on a side stream, then capture and replay one isolated CUDA Graph."""
    if input.device.type != "cuda":
        return

    current_stream = torch.cuda.current_stream(input.device)
    warmup_stream = torch.cuda.Stream(device=input.device)
    warmup_stream.wait_stream(current_stream)
    with torch.cuda.stream(warmup_stream):
        warmup_output = runner(input, weight, bias)
    warmup_stream.synchronize()
    current_stream.wait_stream(warmup_stream)
    _validate_router_gemm_output(warmup_output, input, weight)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = runner(input, weight, bias)
    graph.replay()
    _synchronize_router_gemm(input)
    _validate_router_gemm_output(graph_output, input, weight)
    _validate_router_gemm_bitwise(
        graph_output,
        warmup_output,
        comparison="CUDA Graph/eager",
    )


def _preflight_deepgemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> None:
    output = _run_router_deepgemm(input, weight, bias)
    _synchronize_router_gemm(input)
    _validate_router_gemm_output(output, input, weight)
    reference = _run_router_persistent(input, weight, bias)
    _synchronize_router_gemm(input)
    _validate_router_gemm_bitwise(
        output,
        reference,
        comparison="DeepGEMM/persistent",
    )
    _cuda_graph_smoke_router_backend(_run_router_deepgemm, input, weight, bias)


def _preflight_full_k(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> None:
    output = _run_router_full_k(input, weight, bias)
    _synchronize_router_gemm(input)
    _validate_router_gemm_output(output, input, weight)
    if input.dtype in (torch.bfloat16, torch.float16):
        reference = _run_router_persistent(input, weight, bias)
        _synchronize_router_gemm(input)
        _validate_router_gemm_bitwise(
            output,
            reference,
            comparison="full-K/persistent",
        )
    elif input.dtype == torch.float32:
        _validate_router_gemm_fp32_accuracy(output, input, weight)
    _cuda_graph_smoke_router_backend(_run_router_full_k, input, weight, bias)


def _preflight_persistent(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> None:
    output = _run_router_persistent(input, weight, bias)
    _synchronize_router_gemm(input)
    _validate_router_gemm_output(output, input, weight)


def _preflight_router_backend(
    backend: str,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> None:
    if backend == "deepgemm":
        _preflight_deepgemm(input, weight, bias)
    elif backend == "full_k":
        _preflight_full_k(input, weight, bias)
    else:
        _preflight_persistent(input, weight, bias)


def _router_gemm_is_capturing_or_compiling(input: torch.Tensor) -> bool:
    if torch.compiler.is_compiling():
        return True
    if input.device.type != "cuda":
        return False
    try:
        return torch.cuda.is_current_stream_capturing()
    except RuntimeError:
        # Metadata-only policy tests and early process initialization may not
        # have a CUDA context yet. The actual preflight will still fail safely
        # before a decision is cached if CUDA is unavailable.
        return False


def prewarm_router_gemm_backend(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    requested_mode: str | None = None,
) -> RouterGemmBackendDecision:
    """Preflight and cache a Router GEMM decision before CUDA Graph capture."""
    requested = (
        VLLM_BATCH_INVARIANT_ROUTER_GEMM if requested_mode is None else requested_mode
    )
    signature = _router_gemm_signature(input, weight, bias)
    key = (requested, signature)
    cached = _router_gemm_backend_cache.get(key)
    if cached is not None:
        return cached
    if _router_gemm_is_capturing_or_compiling(input):
        raise RuntimeError(
            "Router GEMM signature was not preflighted before CUDA Graph capture "
            "or torch.compile; call prewarm_router_gemm_backend first"
        )

    with _router_gemm_backend_lock:
        cached = _router_gemm_backend_cache.get(key)
        if cached is not None:
            return cached

        failures: list[str] = []
        if requested == "auto":
            candidates, skipped = _auto_router_gemm_candidates(input, weight, bias)
            failures.extend(skipped)
        else:
            static_decision = _select_router_gemm_backend(
                input, weight, bias, requested_mode=requested
            )
            assert static_decision.selected is not None
            candidates = [static_decision.selected]

        selected: str | None = None
        for candidate in candidates:
            try:
                _preflight_router_backend(candidate, input, weight, bias)
            except Exception as exc:
                failure = f"{candidate} preflight failed: {type(exc).__name__}: {exc}"
                if requested != "auto" or candidate == "persistent":
                    raise RuntimeError(failure) from exc
                failures.append(failure)
                continue
            selected = candidate
            break

        assert selected is not None
        failures.append(f"selected {selected} after synchronized preflight")
        decision = RouterGemmBackendDecision(
            requested=requested,
            selected=selected,
            reason="; ".join(failures),
            signature=signature,
            preflighted=True,
        )
        _router_gemm_backend_cache[key] = decision
        logger.info(
            "BI Router GEMM requested=%s selected=%s reason=%s signature=%s",
            requested,
            selected,
            decision.reason,
            signature,
        )
        return decision


def get_router_gemm_backend_decision(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    requested_mode: str | None = None,
) -> RouterGemmBackendDecision:
    """Read a cached decision without importing, compiling, or running a backend."""
    requested = (
        VLLM_BATCH_INVARIANT_ROUTER_GEMM if requested_mode is None else requested_mode
    )
    if requested not in _ROUTER_GEMM_CHOICES:
        raise ValueError(
            f"requested_mode must be one of {_ROUTER_GEMM_CHOICES}, got {requested!r}"
        )
    signature = _router_gemm_signature(input, weight, bias)
    cached = _router_gemm_backend_cache.get((requested, signature))
    if cached is not None:
        return cached
    return RouterGemmBackendDecision(
        requested=requested,
        selected=None,
        reason="not_preflighted",
        signature=signature,
        preflighted=False,
    )


def _router_gemm_batch_invariant_op_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    decision = get_router_gemm_backend_decision(input, weight, bias)
    if not decision.preflighted:
        decision = prewarm_router_gemm_backend(input, weight, bias)

    if decision.selected == "deepgemm":
        return _run_router_deepgemm(input, weight, bias)
    if decision.selected == "full_k":
        return _run_router_full_k(input, weight, bias)
    return _run_router_persistent(input, weight, bias)


def _router_gemm_batch_invariant_op_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return input.new_empty((*input.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name="router_gemm_batch_invariant",
    op_func=_router_gemm_batch_invariant_op_impl,
    fake_impl=_router_gemm_batch_invariant_op_fake,
)


def router_gemm_batch_invariant(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Opaque compile boundary around the preflighted Router dispatch."""
    return torch.ops.vllm.router_gemm_batch_invariant(input, weight, bias)


def vllm_is_batch_invariant() -> bool:
    return VLLM_BATCH_INVARIANT


def override_envs_for_invariance(
    attention_backend: AttentionBackendEnum | None,
):
    decode_invariant_backends = [
        AttentionBackendEnum.FLASH_ATTN,  # best supported backend
        AttentionBackendEnum.TRITON_ATTN,
    ]
    supported_backends = decode_invariant_backends + [
        # FlashInfer temporarily disabled due to invariant CTA sizes.
        # See FlashInfer issue #2424
        # AttentionBackendEnum.FLASHINFER,
        AttentionBackendEnum.FLASH_ATTN_MLA,
        AttentionBackendEnum.TRITON_MLA,
        # Not yet supported MLA backends
        # AttentionBackendEnum.FLASHMLA,
        # AttentionBackendEnum.FLEX_ATTENTION,  # IMA issue
        # AttentionBackendEnum.FLASHINFER_MLA,  # PR #28967
    ]
    if attention_backend not in supported_backends:
        supported_names = [b.name for b in supported_backends]
        backend_name = attention_backend.name if attention_backend else None
        error = (
            "VLLM batch_invariant mode requires an attention backend in "
            f"{supported_names}, but got '{backend_name}'. "
            "Please use --attention-backend or attention_config to set "
            "one of the supported backends before enabling batch_invariant."
        )
        raise RuntimeError(error)
    if attention_backend not in decode_invariant_backends:
        warning = (
            "You are using a non-decode-invariant form of batch invariance. "
            "This will not be invariant between prefill and decode."
        )
        logger.warning_once(warning, scope="local")
    os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # NCCL determinism settings
    os.environ["NCCL_LAUNCH_MODE"] = "GROUP"
    os.environ["NCCL_COLLNET_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    os.environ["NCCL_P2P_NET_DISABLE"] = "1"
    os.environ["NCCL_MIN_NCHANNELS"] = "1"
    os.environ["NCCL_MAX_NCHANNELS"] = "1"
    os.environ["NCCL_PROTO"] = "Simple"
    os.environ["NCCL_ALGO"] = "allreduce:tree"
    os.environ["NCCL_NTHREADS"] = "1"
    os.environ["NCCL_SOCKET_NTHREADS"] = "1"

    # torch.compile settings
    os.environ["VLLM_USE_AOT_COMPILE"] = "0"


def init_batch_invariance(
    attention_backend: AttentionBackendEnum | None,
):
    # this will hit all the csrc overrides as well
    if vllm_is_batch_invariant():
        override_envs_for_invariance(attention_backend)
        enable_batch_invariant_mode()

        # Disable TF32 for batch invariance - it causes non-deterministic rounding
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        torch.backends.cudnn.rnn.fp32_precision = "ieee"
