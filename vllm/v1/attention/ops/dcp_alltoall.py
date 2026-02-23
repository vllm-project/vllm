# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DCP All-to-All communication backend for attention.

Provides All-to-All (A2A) communication as an alternative to
AllGather + ReduceScatter (AG+RS) for Decode Context Parallel (DCP).
Instead of gathering the full Q tensor and scattering partial outputs,
A2A exchanges partial attention outputs and their LSE values across
ranks, then combines them with exact LSE-weighted reduction.

This reduces the number of NCCL calls per attention layer from 3
(AG for Q, AG for K metadata, RS for output) to 2 (A2A for output,
A2A for LSE), lowering per-step communication overhead for long-context
decode where NCCL latency is a significant fraction of step time.

Usage:
    vllm serve model --tp 16 --dcp 16 --dcp-comm-backend a2a

Reference: https://arxiv.org/abs/2507.07120
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.v1.attention.ops.common import CPTritonContext


def _lse_weighted_combine(
    outputs: torch.Tensor,
    lses: torch.Tensor,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    CPU reference implementation for LSE-weighted combination.

    This is a pure PyTorch implementation used for testing and validation.
    For GPU execution, use dcp_lse_combine_triton instead.

    Args:
        outputs: Partial attention outputs [N, B, H, D]
                 N = number of KV shards (ranks)
                 B = batch size (num_tokens)
                 H = number of heads per rank
                 D = head dimension
        lses: Log-sum-exp values [N, B, H]
        return_lse: If True, also return the global LSE
        is_lse_base_on_e: If True, LSE is base e; if False, base 2

    Returns:
        Combined output [B, H, D], and optionally global LSE [B, H]
    """
    N, B, H, D = outputs.shape

    # Handle NaN and inf in LSEs
    lses = torch.where(
        torch.isnan(lses) | torch.isinf(lses),
        torch.tensor(float("-inf"), device=lses.device, dtype=lses.dtype),
        lses,
    )

    # Compute max LSE for numerical stability
    lse_max, _ = lses.max(dim=0)  # [B, H]
    lse_max = torch.where(
        lse_max == float("-inf"),
        torch.zeros_like(lse_max),
        lse_max,
    )

    # Compute weights: softmax over the N dimension
    if is_lse_base_on_e:
        weights = torch.exp(lses - lse_max.unsqueeze(0))  # [N, B, H]
    else:
        weights = torch.pow(2.0, lses - lse_max.unsqueeze(0))  # [N, B, H]

    # Handle NaN weights
    weights = torch.where(torch.isnan(weights), torch.zeros_like(weights), weights)

    # Normalize weights
    weight_sum = weights.sum(dim=0, keepdim=True)  # [1, B, H]
    weights = weights / weight_sum.clamp(min=1e-10)  # [N, B, H]

    # Weighted combination: sum over N dimension
    result = (outputs * weights.unsqueeze(-1)).sum(dim=0)  # [B, H, D]

    if return_lse:
        if is_lse_base_on_e:
            global_lse = torch.log(weight_sum.squeeze(0)) + lse_max  # [B, H]
        else:
            global_lse = torch.log2(weight_sum.squeeze(0)) + lse_max  # [B, H]
        return result, global_lse

    return result


@triton.jit
def _dcp_lse_combine_kernel(
    # Input pointers
    recv_output_ptr,
    recv_lse_ptr,
    # Output pointers
    out_ptr,
    out_lse_ptr,
    # Strides for recv_output [N, B, H_local, D]
    ro_stride_N,
    ro_stride_B,
    ro_stride_H,
    ro_stride_D,
    # Strides for recv_lse [N, B, H_local]
    rl_stride_N,
    rl_stride_B,
    rl_stride_H,
    # Strides for output [B, H_local, D]
    o_stride_B,
    o_stride_H,
    o_stride_D,
    # Constants
    N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_BASE_E: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    """
    Triton kernel for LSE-weighted combination of partial attention outputs.

    After All-to-All, each rank has:
    - recv_output [N, B, H_local, D]: partial outputs from all KV shards
    - recv_lse [N, B, H_local]: partial LSEs from all KV shards

    This kernel computes the weighted combination locally (no communication).

    Grid: (B, H_local)
    Each program handles one (batch, head) and processes all D elements.
    """
    batch_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)

    # Base offset for this (batch, head)
    base_lse_offset = batch_idx * rl_stride_B + head_idx * rl_stride_H
    base_out_offset = batch_idx * ro_stride_B + head_idx * ro_stride_H

    # First pass: find max LSE for numerical stability
    lse_max = -float("inf")
    for n in tl.static_range(N):
        lse_offset = n * rl_stride_N + base_lse_offset
        lse_val = tl.load(recv_lse_ptr + lse_offset)
        lse_val = tl.where(
            (lse_val != lse_val) | (lse_val == float("inf")),
            -float("inf"),
            lse_val,
        )
        lse_max = tl.maximum(lse_max, lse_val)

    lse_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)

    # Second pass: compute sum of exp(lse - max)
    lse_sum = 0.0
    for n in tl.static_range(N):
        lse_offset = n * rl_stride_N + base_lse_offset
        lse_val = tl.load(recv_lse_ptr + lse_offset)
        lse_val = tl.where(
            (lse_val != lse_val) | (lse_val == float("inf")),
            -float("inf"),
            lse_val,
        )
        if IS_BASE_E:
            lse_sum += tl.exp(lse_val - lse_max)
        else:
            lse_sum += tl.exp2(lse_val - lse_max)

    # Compute global LSE
    if IS_BASE_E:  # noqa: SIM108
        global_lse = tl.log(lse_sum) + lse_max
    else:
        global_lse = tl.log2(lse_sum) + lse_max

    # Third pass: weighted combination across D dimension
    d_offsets = tl.arange(0, HEAD_DIM)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for n in tl.static_range(N):
        lse_offset = n * rl_stride_N + base_lse_offset
        lse_val = tl.load(recv_lse_ptr + lse_offset)
        lse_val = tl.where(
            (lse_val != lse_val) | (lse_val == float("inf")),
            -float("inf"),
            lse_val,
        )
        if IS_BASE_E:
            weight = tl.exp(lse_val - global_lse)
        else:
            weight = tl.exp2(lse_val - global_lse)
        weight = tl.where(weight != weight, 0.0, weight)

        out_offsets = n * ro_stride_N + base_out_offset + d_offsets * ro_stride_D
        out_vals = tl.load(recv_output_ptr + out_offsets)
        acc += out_vals.to(tl.float32) * weight

    # Store result
    final_offsets = (
        batch_idx * o_stride_B + head_idx * o_stride_H + d_offsets * o_stride_D
    )
    tl.store(out_ptr + final_offsets, acc)

    if RETURN_LSE:
        tl.store(out_lse_ptr + base_lse_offset, global_lse)


def dcp_lse_combine_triton(
    recv_output: torch.Tensor,
    recv_lse: torch.Tensor,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Triton-accelerated LSE-weighted combination for DCP A2A.

    Args:
        recv_output: [N, B, H_local, D] - partial outputs from all KV shards
        recv_lse: [N, B, H_local] - partial LSEs from all KV shards
        return_lse: If True, also return the global LSE
        is_lse_base_on_e: If True, LSE is base e; if False, base 2

    Returns:
        Combined output [B, H_local, D]
        If return_lse=True, also returns global_lse [B, H_local]
    """
    N, B, H_local, D = recv_output.shape

    out = torch.empty(
        (B, H_local, D), device=recv_output.device, dtype=recv_output.dtype
    )

    if return_lse:
        out_lse = torch.empty(
            (B, H_local), device=recv_lse.device, dtype=recv_lse.dtype
        )
    else:
        out_lse = torch.empty(1, device=recv_lse.device, dtype=recv_lse.dtype)

    ro_stride_N, ro_stride_B, ro_stride_H, ro_stride_D = recv_output.stride()
    rl_stride_N, rl_stride_B, rl_stride_H = recv_lse.stride()
    o_stride_B, o_stride_H, o_stride_D = out.stride()

    grid = (B, H_local, 1)

    _dcp_lse_combine_kernel[grid](
        recv_output,
        recv_lse,
        out,
        out_lse,
        ro_stride_N,
        ro_stride_B,
        ro_stride_H,
        ro_stride_D,
        rl_stride_N,
        rl_stride_B,
        rl_stride_H,
        o_stride_B,
        o_stride_H,
        o_stride_D,
        N=N,
        HEAD_DIM=D,
        IS_BASE_E=is_lse_base_on_e,
        RETURN_LSE=return_lse,
    )

    if return_lse:
        return out, out_lse
    return out


def dcp_a2a_lse_reduce(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    ctx: CPTritonContext | None = None,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Combine partial attention outputs across DCP ranks using All-to-All.

    Each rank holds attention output for all heads but only a local shard
    of the KV cache. This function:
    1. Exchanges partial outputs across ranks via All-to-All
    2. Exchanges LSE values via All-to-All
    3. Combines them with exact LSE-weighted reduction (Triton kernel)

    Tensor flow:
        Input:  cp_attn_out [B, H, D] - all heads, local KV shard
        Reshape: [N, B, H/N, D] - split heads across ranks
        A2A:    Two all_to_all_single calls (output and LSE)
        Combine: recv [N, B, H/N, D] + lse [N, B, H/N] -> [B, H/N, D]

    Args:
        cp_attn_out: [B, H, D] where B=num_tokens, H=total_heads, D=head_dim
        cp_attn_lse: [B, H] log-sum-exp values (fp32)
        cp_group: GroupCoordinator for DCP communication
        ctx: CPTritonContext (unused, for signature compatibility)
        return_lse: If True, also return the combined global LSE
        is_lse_base_on_e: If True, LSE is base e; if False, base 2

    Returns:
        Combined output [B, H/N, D] (head-scattered)
        If return_lse=True, also returns global_lse [B, H/N]
    """
    world_size = cp_group.world_size

    if world_size == 1:
        if return_lse:
            return cp_attn_out, cp_attn_lse
        return cp_attn_out

    local_output = cp_attn_out.contiguous()
    local_lse = cp_attn_lse.contiguous()

    B, H, D = local_output.shape
    H_per_rank = H // world_size

    # Reshape for All-to-All: [B, H, D] -> [N, B, H/N, D]
    # Split heads into N chunks, each destined for a different rank
    send_output = (
        local_output.view(B, world_size, H_per_rank, D).permute(1, 0, 2, 3).contiguous()
    )
    recv_output = torch.empty_like(send_output)

    # Same for LSE: [B, H] -> [N, B, H/N]
    send_lse = local_lse.view(B, world_size, H_per_rank).permute(1, 0, 2).contiguous()
    recv_lse = torch.empty_like(send_lse)

    # All-to-All for partial attention outputs and LSE values (async overlap)
    work_output = dist.all_to_all_single(
        recv_output.view(-1),
        send_output.view(-1),
        group=cp_group.device_group,
        async_op=True,
    )
    work_lse = dist.all_to_all_single(
        recv_lse.view(-1),
        send_lse.view(-1),
        group=cp_group.device_group,
        async_op=True,
    )
    work_output.wait()
    work_lse.wait()

    # LSE-weighted combination via Triton kernel (local, no communication)
    return dcp_lse_combine_triton(
        recv_output,
        recv_lse,
        return_lse=return_lse,
        is_lse_base_on_e=is_lse_base_on_e,
    )
