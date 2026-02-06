"""
Helix parallelism operations for attention.

Helix uses All-to-All communication instead of AllGather+ReduceScatter
for context parallel attention, which can reduce communication overhead
for long-context scenarios.

Reference: https://arxiv.org/abs/2507.07120
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

# Use packed single-A2A optimization (reduces NCCL call overhead)
# Set VLLM_HELIX_PACKED_A2A=1 to enable
_USE_PACKED_A2A = os.environ.get("VLLM_HELIX_PACKED_A2A", "0") == "1"


class HelixContext:
    """
    Persistent context for Helix All-to-All operations.
    
    Pre-allocates and reuses buffers to avoid allocation overhead
    at high concurrency.
    """
    
    def __init__(self):
        # Buffer cache keyed by (B, H, D, world_size, dtype, device)
        self._output_buffers: dict = {}
        self._lse_buffers: dict = {}
        # Packed buffer cache for single-A2A optimization
        self._packed_buffers: dict = {}
        # Compiled Triton kernel cache
        self._compiled_kernel = None
        self._kernel_key = None
    
    def get_buffers(
        self,
        B: int,
        H_per_rank: int,
        D: int,
        world_size: int,
        dtype: torch.dtype,
        lse_dtype: torch.dtype,
        device: torch.device,
    ):
        """Get or allocate send/recv buffers."""
        key = (B, H_per_rank, D, world_size, dtype, device)
        lse_key = (B, H_per_rank, world_size, lse_dtype, device)
        
        # Check if we can reuse existing buffers
        if key in self._output_buffers:
            send_output, recv_output = self._output_buffers[key]
        else:
            # Allocate new buffers
            shape = (world_size, B, H_per_rank, D)
            send_output = torch.empty(shape, dtype=dtype, device=device)
            recv_output = torch.empty(shape, dtype=dtype, device=device)
            self._output_buffers[key] = (send_output, recv_output)
        
        if lse_key in self._lse_buffers:
            send_lse, recv_lse = self._lse_buffers[lse_key]
        else:
            lse_shape = (world_size, B, H_per_rank)
            send_lse = torch.empty(lse_shape, dtype=lse_dtype, device=device)
            recv_lse = torch.empty(lse_shape, dtype=lse_dtype, device=device)
            self._lse_buffers[lse_key] = (send_lse, recv_lse)
        
        return send_output, recv_output, send_lse, recv_lse
    
    def get_packed_buffers(
        self,
        B: int,
        H_per_rank: int,
        D: int,
        world_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        """Get or allocate packed send/recv buffers for single-A2A optimization.
        
        Packs output [N, B, H/N, D] and LSE [N, B, H/N] into [N, B, H/N, D+1].
        """
        key = (B, H_per_rank, D, world_size, dtype, device)
        
        if key in self._packed_buffers:
            return self._packed_buffers[key]
        
        # Packed shape: [N, B, H/N, D+1] where last element is LSE
        packed_shape = (world_size, B, H_per_rank, D + 1)
        send_packed = torch.empty(packed_shape, dtype=dtype, device=device)
        recv_packed = torch.empty(packed_shape, dtype=dtype, device=device)
        self._packed_buffers[key] = (send_packed, recv_packed)
        
        return send_packed, recv_packed


# Global context for stateless function calls
_global_helix_ctx: HelixContext | None = None


def get_helix_context() -> HelixContext:
    """Get or create the global Helix context."""
    global _global_helix_ctx
    if _global_helix_ctx is None:
        _global_helix_ctx = HelixContext()
    return _global_helix_ctx


@triton.jit
def _helix_lse_combine_kernel(
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
    Triton kernel for Helix LSE-weighted combination.

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

    # Step 1: Load all LSEs and compute weights
    # We need to load LSEs one by one and compute global LSE
    # First pass: find max LSE
    lse_max = -float("inf")
    for n in tl.static_range(N):
        lse_offset = n * rl_stride_N + base_lse_offset
        lse_val = tl.load(recv_lse_ptr + lse_offset)
        # Handle NaN and inf
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
    if IS_BASE_E:
        global_lse = tl.log(lse_sum) + lse_max
    else:
        global_lse = tl.log2(lse_sum) + lse_max

    # Step 2: Weighted combination across D dimension
    d_offsets = tl.arange(0, HEAD_DIM)

    # Initialize accumulator
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    # Third pass: weighted sum
    for n in tl.static_range(N):
        # Compute weight for this shard
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

        # Load output for this shard and accumulate
        out_offsets = n * ro_stride_N + base_out_offset + d_offsets * ro_stride_D
        out_vals = tl.load(recv_output_ptr + out_offsets)
        acc += out_vals.to(tl.float32) * weight

    # Store result
    final_offsets = batch_idx * o_stride_B + head_idx * o_stride_H + d_offsets * o_stride_D
    tl.store(out_ptr + final_offsets, acc)

    # Optional: store global LSE
    if RETURN_LSE:
        tl.store(out_lse_ptr + base_lse_offset, global_lse)


def helix_lse_combine_triton(
    recv_output: torch.Tensor,
    recv_lse: torch.Tensor,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Triton-accelerated LSE-weighted combination for Helix.

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

    # Allocate output tensors
    out = torch.empty((B, H_local, D), device=recv_output.device, dtype=recv_output.dtype)

    if return_lse:
        out_lse = torch.empty(
            (B, H_local), device=recv_lse.device, dtype=recv_lse.dtype
        )
    else:
        # Dummy tensor (not used, but kernel expects it)
        out_lse = torch.empty(1, device=recv_lse.device, dtype=recv_lse.dtype)

    # Get strides
    ro_stride_N, ro_stride_B, ro_stride_H, ro_stride_D = recv_output.stride()
    rl_stride_N, rl_stride_B, rl_stride_H = recv_lse.stride()
    o_stride_B, o_stride_H, o_stride_D = out.stride()

    # Launch kernel (grid must be 3-tuple)
    grid = (B, H_local, 1)

    _helix_lse_combine_kernel[grid](
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


def helix_alltoall_lse_reduce(
    local_output: torch.Tensor,
    local_lse: torch.Tensor,
    kvp_group: GroupCoordinator,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
    ctx: HelixContext | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Perform Helix-style attention output combination using All-to-All.

    In DCP (Decode Context Parallel), each rank computes partial attention
    with its local KV cache shard. This function uses All-to-All communication
    to exchange partial outputs and combines them using LSE-weighted averaging.

    Communication pattern:
        1. Split outputs/LSEs by head groups
        2. All-to-All exchange (each rank sends chunks to all others)
        3. Local LSE-weighted combination via Triton kernel (no more communication)

    Tensor flow:
        Input:  local_output [B, H, D] - all heads, local KV shard
        Split:  [B, N, H/N, D] - split heads into N chunks
        A2A:    Each rank sends chunk[i] to rank i, receives from all ranks
        After:  recv_output [N, B, H/N, D] - all KV shards, local heads
        Combine: output [B, H/N, D] - LSE-weighted sum across KV shards (Triton)

    Args:
        local_output: Local attention output [B, H, D] where:
                      B = num_tokens, H = gathered_heads, D = kv_lora_rank
                      Each rank has output for the SAME tokens but computed
                      with DIFFERENT KV cache shards.
        local_lse: Local log-sum-exp values [B, H]
        kvp_group: GroupCoordinator for KV parallel communication
        return_lse: If True, also return the local portion of global LSE
        is_lse_base_on_e: If True, LSE is base e; if False, base 2
        ctx: Optional HelixContext for buffer reuse. If None, uses global context.

    Returns:
        Combined attention output [B, H/N, D] (scattered along head dimension)
        If return_lse=True, also returns local_lse [B, H/N]
    """
    world_size = kvp_group.world_size

    if world_size == 1:
        if return_lse:
            return local_output, local_lse
        return local_output

    # Ensure inputs are contiguous for view() to work correctly
    local_output = local_output.contiguous()
    local_lse = local_lse.contiguous()

    B, H, D = local_output.shape
    H_per_rank = H // world_size

    # Get or create context for buffer reuse
    if ctx is None:
        ctx = get_helix_context()

    # Get pre-allocated buffers (avoids allocation overhead at high concurrency)
    send_output, recv_output, send_lse, recv_lse = ctx.get_buffers(
        B, H_per_rank, D, world_size,
        local_output.dtype, local_lse.dtype, local_output.device
    )

    # Step 1: Reshape and transpose into send buffers
    # [B, H, D] -> [B, N, H/N, D] -> [N, B, H/N, D]
    # Use copy_ into pre-allocated buffer instead of creating new tensor
    local_output_view = local_output.view(B, world_size, H_per_rank, D)
    send_output.copy_(local_output_view.permute(1, 0, 2, 3))

    # [B, H] -> [B, N, H/N] -> [N, B, H/N]
    local_lse_view = local_lse.view(B, world_size, H_per_rank)
    send_lse.copy_(local_lse_view.permute(1, 0, 2))

    # Step 2: All-to-All exchange
    # After A2A: recv[i] contains rank i's partial output for MY local heads
    # Use async_op=True to overlap the two all-to-all operations
    work_output = dist.all_to_all_single(
        recv_output.view(-1),
        send_output.view(-1),
        group=kvp_group.device_group,
        async_op=True,
    )
    work_lse = dist.all_to_all_single(
        recv_lse.view(-1),
        send_lse.view(-1),
        group=kvp_group.device_group,
        async_op=True,
    )

    # Wait for both all-to-all operations to complete
    work_output.wait()
    work_lse.wait()

    # recv_output shape: [N, B, H/N, D]
    # recv_output[i] = rank i's partial output for my local heads

    # Step 3: LSE-weighted combination via Triton kernel (LOCAL, no communication)
    return helix_lse_combine_triton(
        recv_output,
        recv_lse,
        return_lse=return_lse,
        is_lse_base_on_e=is_lse_base_on_e,
    )


def helix_alltoall_lse_reduce_packed(
    local_output: torch.Tensor,
    local_lse: torch.Tensor,
    kvp_group: GroupCoordinator,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
    ctx: HelixContext | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized Helix All-to-All with packed output+LSE (single A2A call).
    
    Instead of two separate A2A calls for output and LSE, this version
    packs them into a single tensor [N, B, H/N, D+1] where the last
    element contains the LSE. This reduces NCCL call overhead.
    
    Args:
        local_output: Local attention output [B, H, D]
        local_lse: Local log-sum-exp values [B, H]
        kvp_group: GroupCoordinator for KV parallel communication
        return_lse: If True, also return the local portion of global LSE
        is_lse_base_on_e: If True, LSE is base e; if False, base 2
        ctx: Optional HelixContext for buffer reuse.
    
    Returns:
        Combined attention output [B, H/N, D]
        If return_lse=True, also returns local_lse [B, H/N]
    """
    world_size = kvp_group.world_size

    if world_size == 1:
        if return_lse:
            return local_output, local_lse
        return local_output

    # Ensure inputs are contiguous
    local_output = local_output.contiguous()
    local_lse = local_lse.contiguous()

    B, H, D = local_output.shape
    H_per_rank = H // world_size

    # Get or create context for buffer reuse
    if ctx is None:
        ctx = get_helix_context()

    # Get packed buffers [N, B, H/N, D+1]
    send_packed, recv_packed = ctx.get_packed_buffers(
        B, H_per_rank, D, world_size,
        local_output.dtype, local_output.device
    )

    # Step 1: Pack output and LSE into single tensor
    # [B, H, D] -> [B, N, H/N, D] -> [N, B, H/N, D]
    local_output_view = local_output.view(B, world_size, H_per_rank, D)
    send_packed[:, :, :, :D].copy_(local_output_view.permute(1, 0, 2, 3))
    
    # [B, H] -> [B, N, H/N] -> [N, B, H/N] -> [N, B, H/N, 1]
    local_lse_view = local_lse.view(B, world_size, H_per_rank)
    # Cast LSE to output dtype and pack into last dimension
    send_packed[:, :, :, D].copy_(
        local_lse_view.permute(1, 0, 2).to(local_output.dtype)
    )

    # Step 2: Single All-to-All exchange (instead of two)
    dist.all_to_all_single(
        recv_packed.view(-1),
        send_packed.view(-1),
        group=kvp_group.device_group,
    )

    # Step 3: Unpack output and LSE
    recv_output = recv_packed[:, :, :, :D].contiguous()
    recv_lse = recv_packed[:, :, :, D].to(local_lse.dtype).contiguous()

    # Step 4: LSE-weighted combination via Triton kernel
    return helix_lse_combine_triton(
        recv_output,
        recv_lse,
        return_lse=return_lse,
        is_lse_base_on_e=is_lse_base_on_e,
    )


def helix_alltoall_lse_reduce_auto(
    local_output: torch.Tensor,
    local_lse: torch.Tensor,
    kvp_group: GroupCoordinator,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
    ctx: HelixContext | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Auto-select Helix implementation based on environment variable.
    
    Set VLLM_HELIX_PACKED_A2A=1 to use single-A2A packed optimization.
    Default uses standard two-A2A implementation.
    """
    if _USE_PACKED_A2A:
        return helix_alltoall_lse_reduce_packed(
            local_output, local_lse, kvp_group,
            return_lse=return_lse,
            is_lse_base_on_e=is_lse_base_on_e,
            ctx=ctx,
        )
    else:
        return helix_alltoall_lse_reduce(
            local_output, local_lse, kvp_group,
            return_lse=return_lse,
            is_lse_base_on_e=is_lse_base_on_e,
            ctx=ctx,
        )
