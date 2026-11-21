# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton GEMM for decode-sized BF16 linear layers: `x @ weight.T`.

At decode batch sizes (tens of tokens) these GEMMs are bound by reading the
weight, and the cuBLAS heuristic sometimes picks a kernel that reaches only
60-75% of memory bandwidth. This is a plain tiled kernel that reads the weight
in its stored `[N, K]` layout; per-shape tile configs come from measurement
(see skinny_gemm_plans.py).
"""

from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton


@dataclass(frozen=True, slots=True)
class TritonSkinnyGemmConfig:
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int


@triton.jit
def _skinny_gemm_kernel(
    a_ptr,
    w_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_wn,
    stride_cm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    # M tiles of the same N column are adjacent in launch order, so a weight
    # tile read by one of them is still in L2 for the others.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            w = tl.load(w_ptrs, mask=mask_n[None, :], other=0.0)
        else:
            mask_k = offs_k < K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            w = tl.load(w_ptrs, mask=mask_n[None, :] & mask_k[:, None], other=0.0)
        acc = tl.dot(a, w, acc)
        a_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
    tl.store(
        c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :]
    )


def triton_skinny_gemm(
    x: torch.Tensor, weight: torch.Tensor, config: TritonSkinnyGemmConfig
) -> torch.Tensor:
    """`x [M, K] @ weight [N, K].T`, both with unit stride along K."""
    M, K = x.shape
    N = weight.shape[0]
    assert x.stride(1) == 1 and weight.stride(1) == 1
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(M, config.block_m), triton.cdiv(N, config.block_n))
    _skinny_gemm_kernel[grid](
        x,
        weight,
        out,
        M,
        N,
        K,
        x.stride(0),
        weight.stride(0),
        out.stride(0),
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        EVEN_K=K % config.block_k == 0,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    return out
