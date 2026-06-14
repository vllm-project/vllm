# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Custom Triton scatter-reduce kernel for MoE output reduction.

Specialized for decode (M <= 64): performs unweighted summation of
expert outputs back to token positions. The gate weights (gammas) are
already applied inside matmul_ogs when scatter_indx is provided, so
this kernel only needs to sum the per-pair outputs:

    output[m, :] = sum_{j=0..top_k-1}(expert_output[m*top_k+j, :])

This replaces the proprietary _reduce scatter-accumulate path that
runs inside matmul_ogs when n_expts_act > 1, achieving ~2.3x speedup
at decode batch sizes (BS <= 64).
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def scatter_reduce_kernel(
    expert_out_ptr,
    output_ptr,
    stride_ek: tl.constexpr,
    stride_ok: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TOP_K: tl.constexpr,
):
    """
    Triton kernel that reduces expert pair outputs into token outputs.

    Grid: (M, ceil(K / BLOCK_K))

    Each program instance handles one token (axis 0) and one chunk of
    the hidden dimension (axis 1). It accumulates TOP_K expert outputs
    in fp32 and stores the result in bf16.

    Args:
        expert_out_ptr: Pointer to expert output tensor of shape
            (M * TOP_K, K), laid out so that rows
            [m*TOP_K .. m*TOP_K+TOP_K-1] are the TOP_K expert
            outputs for token m.
        output_ptr: Pointer to output tensor of shape (M, K).
        stride_ek: Stride along the row dimension of expert_out
            (number of elements between consecutive rows).
        stride_ok: Stride along the row dimension of output
            (number of elements between consecutive rows).
        K: Hidden dimension size.
        BLOCK_K: Tile size along K dimension (constexpr).
        TOP_K: Number of experts per token (constexpr).
    """
    token_id = tl.program_id(0)
    k_block = tl.program_id(1)

    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K

    # Accumulate in fp32 for numerical stability
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)

    base = token_id * TOP_K
    for j in range(TOP_K):
        row = base + j
        vals = tl.load(
            expert_out_ptr + row * stride_ek + k_offsets,
            mask=k_mask,
            other=0.0,
        )
        acc += vals.to(tl.float32)

    # Store result in bf16
    tl.store(
        output_ptr + token_id * stride_ok + k_offsets,
        acc.to(tl.bfloat16),
        mask=k_mask,
    )


def scatter_reduce(
    expert_output: torch.Tensor,
    output: torch.Tensor,
    M: int,
    K: int,
    top_k: int,
    BLOCK_K: int = 256,
) -> None:
    """
    Launch the scatter-reduce Triton kernel.

    Reduces expert_output of shape (M * top_k, K) into output of
    shape (M, K) by summing the top_k rows per token.

    Args:
        expert_output: Tensor of shape (M * top_k, K) containing
            per-expert-pair outputs. May be a reshaped view with
            a leading batch dimension of 1.
        output: Tensor of shape (M, K) or (1, M, K) to write the
            reduced result into.
        M: Number of tokens.
        K: Hidden dimension.
        top_k: Number of active experts per token.
        BLOCK_K: Tile size for the K dimension (default 256).
    """
    # Flatten to 2D if needed (matmul_ogs may produce (1, M*top_k, K))
    if expert_output.ndim == 3:
        expert_output = expert_output.view(-1, K)
    if output.ndim == 3:
        output = output.view(-1, K)

    assert expert_output.shape == (M * top_k, K), (
        f"expert_output shape {expert_output.shape} != "
        f"expected ({M * top_k}, {K})"
    )
    assert output.shape[0] >= M and output.shape[1] == K, (
        f"output shape {output.shape} incompatible with M={M}, K={K}"
    )

    assert expert_output.dtype == torch.bfloat16, (
        f"scatter_reduce requires bf16, got {expert_output.dtype}"
    )

    assert expert_output.stride(-1) == 1, "expert_output must be contiguous in last dim"
    assert output.stride(-1) == 1, "output must be contiguous in last dim"

    # Strides along the row dimension (elements between consecutive rows)
    stride_ek = expert_output.stride(0)
    stride_ok = output.stride(0)

    grid = (M, triton.cdiv(K, BLOCK_K))

    scatter_reduce_kernel[grid](
        expert_output,
        output,
        stride_ek=stride_ek,
        stride_ok=stride_ok,
        K=K,
        BLOCK_K=BLOCK_K,
        TOP_K=top_k,
    )
