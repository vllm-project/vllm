"""Deterministic Triton kernels for route permutation gradients.

These kernels only replace launch-heavy tensor indexing with equivalent
pointwise copies or one-program-per-token ordered reductions.  They do not use
atomics, and every visible output element has exactly one writer.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_routes_forward_kernel(
    hidden_states,
    output_index,
    output,
    num_slots,
    hidden_size: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    hidden_block = tl.program_id(0)
    first_slot = tl.program_id(1)
    num_slot_programs = tl.num_programs(1)
    hidden_offsets = hidden_block * BLOCK_D + tl.arange(0, BLOCK_D)
    hidden_mask = hidden_offsets < hidden_size

    for slot_i32 in range(first_slot, num_slots, num_slot_programs):
        slot = slot_i32.to(tl.int64)
        destination_i32 = tl.load(output_index + slot)
        destination = destination_i32.to(tl.int64)
        token = slot // topk
        valid = destination_i32 >= 0
        values = tl.load(
            hidden_states + token * hidden_size + hidden_offsets,
            mask=hidden_mask & valid,
            other=0.0,
        )
        tl.store(
            output + destination * hidden_size + hidden_offsets,
            values,
            mask=hidden_mask & valid,
        )


@triton.jit
def _scatter_routes_backward_kernel(
    grad_output,
    output_index,
    grad_input,
    num_tokens,
    grad_output_rows,
    hidden_size: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    hidden_block = tl.program_id(0)
    first_token = tl.program_id(1)
    num_token_programs = tl.num_programs(1)
    hidden_offsets = hidden_block * BLOCK_D + tl.arange(0, BLOCK_D)
    hidden_mask = hidden_offsets < hidden_size

    for token_i32 in range(first_token, num_tokens, num_token_programs):
        token = token_i32.to(tl.int64)
        accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
        for route_slot in range(topk):
            route_i32 = tl.load(output_index + token * topk + route_slot)
            valid = (route_i32 >= 0) & (route_i32 < grad_output_rows)
            safe_route = tl.maximum(route_i32, 0).to(tl.int64)
            route_grad = tl.load(
                grad_output + safe_route * hidden_size + hidden_offsets,
                mask=hidden_mask & valid,
                other=0.0,
            ).to(tl.float32)
            # The reference performs one BF16 in-place add per top-k slot.
            # Keep that observable rounding boundary instead of accumulating
            # all slots in FP32 and casting only once.
            accumulator = (accumulator + route_grad).to(tl.bfloat16).to(tl.float32)
        tl.store(
            grad_input + token * hidden_size + hidden_offsets,
            accumulator,
            mask=hidden_mask,
        )


@triton.jit
def _ordered_route_grad_kernel(
    grad_output,
    topk_weights,
    output_index,
    grad_routes,
    num_slots,
    hidden_size: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    hidden_block = tl.program_id(0)
    first_slot = tl.program_id(1)
    num_slot_programs = tl.num_programs(1)
    hidden_offsets = hidden_block * BLOCK_D + tl.arange(0, BLOCK_D)
    hidden_mask = hidden_offsets < hidden_size

    for slot_i32 in range(first_slot, num_slots, num_slot_programs):
        slot = slot_i32.to(tl.int64)
        route_i32 = tl.load(output_index + slot)
        route = route_i32.to(tl.int64)
        token = slot // topk
        valid = route_i32 >= 0
        weight = tl.load(topk_weights + slot).to(tl.float32)
        token_grad = tl.load(
            grad_output + token * hidden_size + hidden_offsets,
            mask=hidden_mask & valid,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            grad_routes + route * hidden_size + hidden_offsets,
            token_grad * weight,
            mask=hidden_mask & valid,
        )


@triton.jit
def _compact_route_positions_kernel(
    valid,
    prefix,
    positions,
    num_slots,
    topk: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    flat_slot = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = flat_slot < num_slots
    is_valid = tl.load(valid + flat_slot, mask=in_bounds, other=0).to(tl.int1)
    compact_row = tl.load(prefix + flat_slot, mask=in_bounds, other=0).to(tl.int64) - 1
    write_mask = in_bounds & is_valid
    tl.store(
        positions + compact_row * 2,
        flat_slot // topk,
        mask=write_mask,
    )
    tl.store(
        positions + compact_row * 2 + 1,
        flat_slot % topk,
        mask=write_mask,
    )


def _validate_common(
    value: torch.Tensor,
    output_index: torch.Tensor,
) -> None:
    if not value.is_cuda or not output_index.is_cuda:
        raise RuntimeError("deterministic route kernels require CUDA tensors")
    if value.ndim != 2 or value.dtype != torch.bfloat16:
        raise TypeError(
            "deterministic route kernels require BF16 [rows, hidden], got " f"{value.dtype} {tuple(value.shape)}"
        )
    if output_index.ndim != 2 or output_index.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            "deterministic route kernels require an integer [tokens, topk] mapping, got "
            f"{output_index.dtype} {tuple(output_index.shape)}"
        )
    if not value.is_contiguous() or not output_index.is_contiguous():
        raise RuntimeError("deterministic route kernel inputs must be contiguous")


def scatter_routes_forward(
    hidden_states: torch.Tensor,
    output_index: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Copy token rows into their unique expert-major route rows."""
    _validate_common(hidden_states, output_index)
    if output.ndim != 2 or output.dtype != hidden_states.dtype or not output.is_contiguous():
        raise TypeError("deterministic route-scatter output must be contiguous BF16 [rows, hidden]")
    hidden_size = hidden_states.shape[1]
    block_d = 1024 if hidden_size >= 1024 else triton.next_power_of_2(hidden_size)
    num_slot_programs = min(output_index.numel(), 8192)
    grid = (triton.cdiv(hidden_size, block_d), num_slot_programs)
    _scatter_routes_forward_kernel[grid](
        hidden_states,
        output_index,
        output,
        output_index.numel(),
        hidden_size=hidden_size,
        topk=output_index.shape[1],
        BLOCK_D=block_d,
        num_warps=4,
    )


def scatter_routes_backward(
    grad_output: torch.Tensor,
    output_index: torch.Tensor,
    grad_input: torch.Tensor,
) -> None:
    """Sum route gradients in fixed top-k order into token rows."""
    _validate_common(grad_output, output_index)
    if (
        grad_input.shape != (output_index.shape[0], grad_output.shape[1])
        or grad_input.dtype != grad_output.dtype
        or not grad_input.is_contiguous()
    ):
        raise TypeError("deterministic route-scatter input gradient has an invalid layout")
    hidden_size = grad_output.shape[1]
    block_d = 1024 if hidden_size >= 1024 else triton.next_power_of_2(hidden_size)
    num_token_programs = min(output_index.shape[0], 2048)
    grid = (triton.cdiv(hidden_size, block_d), num_token_programs)
    _scatter_routes_backward_kernel[grid](
        grad_output,
        output_index,
        grad_input,
        output_index.shape[0],
        grad_output.shape[0],
        hidden_size=hidden_size,
        topk=output_index.shape[1],
        BLOCK_D=block_d,
        num_warps=4,
    )


def ordered_route_grad(
    grad_output: torch.Tensor,
    topk_weights: torch.Tensor,
    output_index: torch.Tensor,
    grad_routes: torch.Tensor,
) -> None:
    """Write each token/slot gradient to its unique route row."""
    _validate_common(grad_output, output_index)
    if topk_weights.shape != output_index.shape or topk_weights.dtype != torch.float32:
        raise TypeError("ordered route gradients require FP32 [tokens, topk] weights")
    if not topk_weights.is_cuda or not topk_weights.is_contiguous():
        raise RuntimeError("ordered route-gradient weights must be contiguous CUDA tensors")
    if (
        grad_routes.ndim != 2
        or grad_routes.shape[1] != grad_output.shape[1]
        or grad_routes.dtype != grad_output.dtype
        or not grad_routes.is_cuda
        or not grad_routes.is_contiguous()
    ):
        raise TypeError("ordered route-gradient output has an invalid layout")
    hidden_size = grad_output.shape[1]
    block_d = 1024 if hidden_size >= 1024 else triton.next_power_of_2(hidden_size)
    num_slot_programs = min(output_index.numel(), 8192)
    grid = (triton.cdiv(hidden_size, block_d), num_slot_programs)
    _ordered_route_grad_kernel[grid](
        grad_output,
        topk_weights,
        output_index,
        grad_routes,
        output_index.numel(),
        hidden_size=hidden_size,
        topk=output_index.shape[1],
        BLOCK_D=block_d,
        num_warps=4,
    )


def compact_route_positions(
    valid: torch.Tensor,
    num_routes: int,
) -> torch.Tensor:
    """Compact a row-major fixed-top-k validity mask without a host sync."""
    if valid.ndim != 2 or valid.dtype != torch.bool or not valid.is_cuda:
        raise ValueError("route-position compaction requires a CUDA bool [tokens, topk] mask")
    if num_routes < 0 or num_routes > valid.numel():
        raise ValueError(f"invalid compact route count {num_routes} for {valid.numel()} slots")

    flat_valid = valid.reshape(-1).contiguous()
    prefix = torch.cumsum(flat_valid, dim=0, dtype=torch.int32)
    positions = torch.empty(
        (num_routes, 2),
        device=valid.device,
        dtype=torch.long,
    )
    if flat_valid.numel():
        torch._assert_async(
            prefix[-1] == num_routes,
            "DeepEP compact route count differs from its metadata handle",
        )
    if num_routes:
        block_size = 256
        grid = (triton.cdiv(flat_valid.numel(), block_size),)
        _compact_route_positions_kernel[grid](
            flat_valid,
            prefix,
            positions,
            flat_valid.numel(),
            topk=valid.shape[1],
            BLOCK_SIZE=block_size,
        )
    return positions
