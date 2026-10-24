# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton GEMM for activations with few rows (``y = x @ weight.T + bias``).

Used on RDNA4 (gfx1200, gfx1201) between the row counts wvSplitK covers and
the point where hipBLASLt catches up. Up to 128 rows, hipBLASLt runs these
BF16/FP16 GEMMs with tiles that leave much of the memory bandwidth unused,
although they only need to stream the weight once. This kernel sizes the M tile
to the row count and keeps N tiles narrow, so that enough workgroups stream the
weight.
"""

from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton

# wvSplitK covers 1 to 5 rows.
MIN_ROWS = 6
# Above 128 rows hipBLASLt picks better kernels and wins on some shapes.
MAX_ROWS = 128
# Narrower weights launch too few workgroups to beat hipBLASLt.
MIN_OUT_FEATURES = 512


@dataclass(frozen=True)
class SkinnyGemmConfig:
    block_m: int
    block_n: int
    block_k: int
    num_warps: int


# (max rows, min out features, (BLOCK_M, BLOCK_N, max BLOCK_K, num_warps)),
# first match wins. Tuned on an AI PRO R9700 (gfx1201, 64 CUs) for N from 512
# to 262144.
_TILES = (
    (16, 0, (16, 64, 256, 4)),
    (32, 0, (32, 32, 256, 4)),
    (64, 0, (64, 32, 256, 4)),
    # 128x64 tiles beat 64x32 ones only when N gives every CU a workgroup.
    (128, 4096, (128, 64, 128, 8)),
    (128, 0, (64, 32, 256, 4)),
)


@triton.jit(do_not_specialize=["M"])
def _skinny_gemm_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_wn,
    stride_ym,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Consecutive programs share an N tile, so the M tiles of one weight tile
    # run back to back and the weight is read from DRAM once.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    n_start = pid_n * BLOCK_N
    # The weight can exceed 2**31 elements (vocab x hidden).
    w_ptr += n_start.to(tl.int64) * stride_wn

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :]
    mask_m = offs_m[:, None] < M
    mask_n = offs_n[:, None] < N - n_start
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # K is a multiple of BLOCK_K.
    for _ in range(0, K // BLOCK_K):
        x = tl.load(x_ptrs, mask=mask_m, other=0.0)
        w = tl.load(w_ptrs, mask=mask_n, other=0.0)
        acc += tl.dot(x, tl.trans(w))
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    offs_yn = n_start + offs_n
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_yn, mask=offs_yn < N, other=0.0)
        acc += bias[None, :].to(tl.float32)
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_yn[None, :]
    tl.store(
        y_ptrs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_m & (offs_yn[None, :] < N),
    )


def skinny_gemm_config(m: int, n: int, k: int) -> SkinnyGemmConfig | None:
    """Tile config for an FP16/BF16 ``[m, k] x [n, k]^T`` GEMM, or None when
    hipBLASLt should run it."""
    if not MIN_ROWS <= m <= MAX_ROWS or n < MIN_OUT_FEATURES:
        return None
    block_m, block_n, max_block_k, num_warps = next(
        tile for rows, min_n, tile in _TILES if m <= rows and n >= min_n
    )
    block_k = next(
        (bk for bk in (256, 128, 64) if bk <= max_block_k and k % bk == 0), None
    )
    if block_k is None:
        return None
    return SkinnyGemmConfig(block_m, block_n, block_k, num_warps)


def skinny_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    config: SkinnyGemmConfig,
) -> torch.Tensor:
    """``torch.nn.functional.linear(x, weight, bias)`` for a ``weight`` whose
    rows are contiguous (row stride may be padded) and a contiguous ``bias``.
    """
    x_view = x.reshape(-1, x.size(-1))
    if x_view.stride(-1) != 1:
        x_view = x_view.contiguous()
    m, k = x_view.shape
    n = weight.shape[0]
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(m, config.block_m), triton.cdiv(n, config.block_n))
    _skinny_gemm_kernel[grid](
        x_view,
        weight,
        bias if bias is not None else out,
        out,
        m,
        n,
        k,
        x_view.stride(0),
        weight.stride(0),
        out.stride(0),
        HAS_BIAS=bias is not None,
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        num_warps=config.num_warps,
        num_stages=1,
    )
    return out.view(*x.shape[:-1], n)
