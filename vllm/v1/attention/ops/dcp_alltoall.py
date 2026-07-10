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

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.v1.attention.ops.common import CPTritonContext


# Module-level cache of the DCP A2A backend choice. Set by
# ``gpu_worker.py`` at workspace pre-init (where vllm_config is reliably
# available). The dispatcher reads from cache during forward, avoiding
# a dependency on ``get_current_vllm_config()`` that raises in V1's
# async scheduling path.
_DCP_A2A_BACKEND: str | None = None


def set_dcp_a2a_backend(backend: str) -> None:
    """Cache the DCP A2A backend on this worker process."""
    global _DCP_A2A_BACKEND
    _DCP_A2A_BACKEND = backend


def _get_dcp_a2a_backend() -> str:
    """Return the resolved DCP A2A kernel (``"nccl"`` or ``"flashinfer"``).

    There is no user-facing flag: the choice is made automatically at worker
    init (``gpu_worker._init_dcp_a2a_flashinfer_workspace``) — FlashInfer on
    Blackwell (sm_100+, where its fused LL128 kernel is CUDA-graph captured and
    wins), NCCL packed everywhere else — and cached here.
    """
    if _DCP_A2A_BACKEND is not None:
        return _DCP_A2A_BACKEND
    return "nccl"


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
    # Don't use the shared WorkspaceManager here. A FULL cudagraph bakes in the
    # buffer address at capture, but the workspace is growable and sized only to
    # the largest *captured* batch (the cudagraph capture cap). Any eager a2a
    # with a bigger batch regrows it, freeing that address and poisoning every
    # captured graph -> illegal memory access on replay. This bites the very
    # first request: the post-capture warmup runs an eager decode at
    # max_num_seqs (> the cap), so the graphs are already dangling before the
    # server is ready. torch.empty buffers instead live in the graph's private
    # pool and stay valid for its lifetime (as _dcp_a2a_unpack_combine and the
    # AG+RS combine path already rely on).
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
    # The pack kernel bit-casts the LSE as fp32; some MLA backends return it in
    # the activation dtype (bf16/fp16), so enforce the documented fp32 contract.
    if cp_attn_lse.dtype != torch.float32:
        cp_attn_lse = cp_attn_lse.to(torch.float32)
    lse_pack_dim = _dcp_a2a_lse_pack_dim(cp_attn_out.dtype)

    use_fi = _get_dcp_a2a_backend() == "flashinfer"
    if use_fi:
        # FlashInfer's LL128 kernel requires partial_o last-dim × elem_size
        # to be 16-byte aligned. Pad ``D + lse_pack_dim`` up to the next
        # 16-byte multiple. Pack writes only the first ``D + lse_pack_dim``
        # slots; the padding is unread by unpack_combine (which loads via
        # explicit offsets HEAD_DIM and HEAD_DIM+1).
        elem_size = cp_attn_out.element_size()
        elems_per_16B = 16 // elem_size
        D_prime_raw = D + lse_pack_dim
        D_prime = ((D_prime_raw + elems_per_16B - 1) // elems_per_16B) * elems_per_16B
    else:
        D_prime = D + lse_pack_dim

    send_buffer, recv_buffer = _dcp_a2a_send_recv_buffers(
        (world_size, B, H_per_rank, D_prime),
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

    if use_fi:
        from vllm.distributed.dcp_alltoall_flashinfer import (
            DCPAllToAllFlashInfer,
        )

        # send_buffer is [N, B, H_per_rank, D_prime]; permute to
        # FlashInfer's convention [B*H_per_rank, N, D_prime] (cp_size
        # is the second-to-last dim).
        send_for_fi = (
            send_buffer.permute(1, 2, 0, 3)
            .reshape(B * H_per_rank, world_size, D_prime)
            .contiguous()
        )
        # FlashInfer takes a softmax_stats arg as well. LSE is already
        # bit-packed inside send_for_fi; pass a small placeholder.
        placeholder = torch.zeros(
            B * H_per_rank,
            world_size,
            2,
            dtype=torch.float32,
            device=send_buffer.device,
        )
        mgr = DCPAllToAllFlashInfer.get(
            cp_rank=cp_group.rank_in_group,
            cp_size=world_size,
            cp_cpu_group=cp_group.cpu_group,
        )
        recv_fi, _ = mgr.run(send_for_fi, placeholder)
        recv_buffer.copy_(
            recv_fi.reshape(B, H_per_rank, world_size, D_prime)
            .permute(2, 0, 1, 3)
            .contiguous()
        )
    else:
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
