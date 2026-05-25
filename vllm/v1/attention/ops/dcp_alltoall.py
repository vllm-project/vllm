# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DCP All-to-All communication backend for attention.

Provides All-to-All (A2A) communication as an alternative to
AllGather + ReduceScatter (AG+RS) for Decode Context Parallel (DCP).
Instead of gathering the full Q tensor and scattering partial outputs,
A2A exchanges partial attention outputs and their LSE values across
ranks, then combines them with exact LSE-weighted reduction.

This reduces the number of NCCL calls per attention layer by exchanging
the partial output and LSE in a single packed All-to-All payload.

Usage:
    vllm serve model --tp 16 --dcp 16 --dcp-comm-backend a2a

Reference: https://arxiv.org/abs/2507.07120
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm.triton_utils import tl, triton
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

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


def _dcp_a2a_lse_pack_dim(output_dtype: torch.dtype) -> int:
    bits = torch.finfo(output_dtype).bits
    if bits == 16:
        return 2
    if bits == 32:
        return 1
    raise ValueError(f"Cannot pack fp32 LSE into output dtype {output_dtype}.")


def _dcp_a2a_send_recv_buffers(
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if is_workspace_manager_initialized():
        send_buffer, recv_buffer = current_workspace_manager().get_simultaneous(
            (shape, dtype),
            (shape, dtype),
        )
        return send_buffer, recv_buffer

    return (
        torch.empty(shape, device=device, dtype=dtype),
        torch.empty(shape, device=device, dtype=dtype),
    )


@triton.jit
def _dcp_a2a_pack_send_kernel(
    out_ptr,
    lse_ptr,
    send_ptr,
    out_stride_B,
    out_stride_H,
    out_stride_D,
    lse_stride_B,
    lse_stride_H,
    send_stride_N,
    send_stride_B,
    send_stride_H,
    send_stride_D,
    N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    H_PER_RANK: tl.constexpr,
    LSE_PACK_DIM: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    local_head_idx = tl.program_id(1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)

    for rank_idx in tl.static_range(N):
        src_head_idx = rank_idx * H_PER_RANK + local_head_idx
        send_base = (
            rank_idx * send_stride_N
            + batch_idx * send_stride_B
            + local_head_idx * send_stride_H
        )

        out_offsets = (
            batch_idx * out_stride_B
            + src_head_idx * out_stride_H
            + d_offsets * out_stride_D
        )
        tl.store(
            send_ptr + send_base + d_offsets * send_stride_D,
            tl.load(out_ptr + out_offsets),
        )

        lse_val = tl.load(
            lse_ptr + batch_idx * lse_stride_B + src_head_idx * lse_stride_H
        )
        if LSE_PACK_DIM == 1:
            tl.store(
                send_ptr + send_base + HEAD_DIM * send_stride_D,
                lse_val.to(send_ptr.dtype.element_ty),
            )
        else:
            lse_bits = lse_val.to(tl.uint32, bitcast=True)
            lo = (lse_bits & 0xFFFF).to(tl.uint16)
            hi = ((lse_bits >> 16) & 0xFFFF).to(tl.uint16)
            tl.store(
                send_ptr + send_base + HEAD_DIM * send_stride_D,
                lo.to(send_ptr.dtype.element_ty, bitcast=True),
            )
            tl.store(
                send_ptr + send_base + (HEAD_DIM + 1) * send_stride_D,
                hi.to(send_ptr.dtype.element_ty, bitcast=True),
            )


@triton.jit
def _dcp_a2a_unpack_combine_kernel(
    recv_ptr,
    out_ptr,
    out_lse_ptr,
    recv_stride_N,
    recv_stride_B,
    recv_stride_H,
    recv_stride_D,
    out_stride_B,
    out_stride_H,
    out_stride_D,
    out_lse_stride_B,
    out_lse_stride_H,
    N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_BASE_E: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    LSE_PACK_DIM: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)

    lse_max = -float("inf")
    for rank_idx in tl.static_range(N):
        recv_base = (
            rank_idx * recv_stride_N
            + batch_idx * recv_stride_B
            + head_idx * recv_stride_H
        )
        if LSE_PACK_DIM == 1:
            lse_val = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D).to(
                tl.float32
            )
        else:
            lo_raw = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D)
            hi_raw = tl.load(recv_ptr + recv_base + (HEAD_DIM + 1) * recv_stride_D)
            lo = lo_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            hi = hi_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            lse_val = (lo | (hi << 16)).to(tl.float32, bitcast=True)
        lse_val = tl.where(
            (lse_val != lse_val) | (lse_val == float("inf")),
            -float("inf"),
            lse_val,
        )
        lse_max = tl.maximum(lse_max, lse_val)

    lse_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)

    lse_sum = 0.0
    for rank_idx in tl.static_range(N):
        recv_base = (
            rank_idx * recv_stride_N
            + batch_idx * recv_stride_B
            + head_idx * recv_stride_H
        )
        if LSE_PACK_DIM == 1:
            lse_val = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D).to(
                tl.float32
            )
        else:
            lo_raw = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D)
            hi_raw = tl.load(recv_ptr + recv_base + (HEAD_DIM + 1) * recv_stride_D)
            lo = lo_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            hi = hi_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            lse_val = (lo | (hi << 16)).to(tl.float32, bitcast=True)
        lse_val = tl.where(
            (lse_val != lse_val) | (lse_val == float("inf")),
            -float("inf"),
            lse_val,
        )
        if IS_BASE_E:
            lse_sum += tl.exp(lse_val - lse_max)
        else:
            lse_sum += tl.exp2(lse_val - lse_max)

    if IS_BASE_E:  # noqa: SIM108
        global_lse = tl.log(lse_sum) + lse_max
    else:
        global_lse = tl.log2(lse_sum) + lse_max

    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for rank_idx in tl.static_range(N):
        recv_base = (
            rank_idx * recv_stride_N
            + batch_idx * recv_stride_B
            + head_idx * recv_stride_H
        )
        if LSE_PACK_DIM == 1:
            lse_val = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D).to(
                tl.float32
            )
        else:
            lo_raw = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D)
            hi_raw = tl.load(recv_ptr + recv_base + (HEAD_DIM + 1) * recv_stride_D)
            lo = lo_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            hi = hi_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            lse_val = (lo | (hi << 16)).to(tl.float32, bitcast=True)
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
        acc += (
            tl.load(recv_ptr + recv_base + d_offsets * recv_stride_D).to(tl.float32)
            * weight
        )

    final_offsets = (
        batch_idx * out_stride_B + head_idx * out_stride_H + d_offsets * out_stride_D
    )
    tl.store(out_ptr + final_offsets, acc)

    if RETURN_LSE:
        out_lse_offset = batch_idx * out_lse_stride_B + head_idx * out_lse_stride_H
        tl.store(out_lse_ptr + out_lse_offset, global_lse)


def _dcp_a2a_pack_send(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    send_buffer: torch.Tensor,
    world_size: int,
    h_per_rank: int,
    head_dim: int,
    lse_pack_dim: int,
) -> None:
    grid = (cp_attn_out.shape[0], h_per_rank, 1)
    _dcp_a2a_pack_send_kernel[grid](
        cp_attn_out,
        cp_attn_lse,
        send_buffer,
        cp_attn_out.stride(0),
        cp_attn_out.stride(1),
        cp_attn_out.stride(2),
        cp_attn_lse.stride(0),
        cp_attn_lse.stride(1),
        send_buffer.stride(0),
        send_buffer.stride(1),
        send_buffer.stride(2),
        send_buffer.stride(3),
        N=world_size,
        HEAD_DIM=head_dim,
        H_PER_RANK=h_per_rank,
        LSE_PACK_DIM=lse_pack_dim,
    )


def _dcp_a2a_unpack_combine(
    recv_buffer: torch.Tensor,
    head_dim: int,
    lse_pack_dim: int,
    return_lse: bool,
    is_lse_base_on_e: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    world_size, num_tokens, h_per_rank, _ = recv_buffer.shape
    out = torch.empty(
        (num_tokens, h_per_rank, head_dim),
        device=recv_buffer.device,
        dtype=recv_buffer.dtype,
    )
    out_lse = torch.empty(
        (num_tokens, h_per_rank) if return_lse else (1, 1),
        device=recv_buffer.device,
        dtype=torch.float32 if return_lse else recv_buffer.dtype,
    )
    grid = (num_tokens, h_per_rank, 1)
    _dcp_a2a_unpack_combine_kernel[grid](
        recv_buffer,
        out,
        out_lse,
        recv_buffer.stride(0),
        recv_buffer.stride(1),
        recv_buffer.stride(2),
        recv_buffer.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out_lse.stride(0),
        out_lse.stride(1),
        N=world_size,
        HEAD_DIM=head_dim,
        IS_BASE_E=is_lse_base_on_e,
        RETURN_LSE=return_lse,
        LSE_PACK_DIM=lse_pack_dim,
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

    The output and fp32 LSE are packed into a single output-dtype buffer, sent
    with one All-to-All, then unpacked and combined with exact LSE weighting.

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

    B, H, D = cp_attn_out.shape
    if H % world_size != 0:
        raise ValueError(f"H={H} must be divisible by DCP world size {world_size}.")
    H_per_rank = H // world_size
    lse_pack_dim = _dcp_a2a_lse_pack_dim(cp_attn_out.dtype)

    send_buffer, recv_buffer = _dcp_a2a_send_recv_buffers(
        (world_size, B, H_per_rank, D + lse_pack_dim),
        device=cp_attn_out.device,
        dtype=cp_attn_out.dtype,
    )

    _dcp_a2a_pack_send(
        cp_attn_out,
        cp_attn_lse,
        send_buffer,
        world_size,
        H_per_rank,
        D,
        lse_pack_dim,
    )

    work = dist.all_to_all_single(
        recv_buffer.view(-1),
        send_buffer.view(-1),
        group=cp_group.device_group,
        async_op=True,
    )
    work.wait()

    return _dcp_a2a_unpack_combine(
        recv_buffer, D, lse_pack_dim, return_lse, is_lse_base_on_e
    )
