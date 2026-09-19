# Copyright 2026, The FlagOS Contributors.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-safe Triton kernels for Qwen4 gated HyperConnection.

This module contains only the three self-developed Qwen4 HC device kernels.
The model wiring and vLLM custom-op registration stay in the model/plugin
repository.  The wrappers below are standalone FlagGems-vllm entry points and
fail closed when their accelerator/layout contract is not met.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

HAS_TRITON = True


def can_use_hc_triton(*tensors: torch.Tensor) -> bool:
    """Return whether the specialized contiguous inference path is usable."""

    return bool(
        HAS_TRITON
        and tensors
        and len({tensor.device for tensor in tensors}) == 1
        and all(
            tensor.device.type not in ("cpu", "meta")
            and tensor.dtype in (torch.bfloat16, torch.float16)
            and tensor.is_contiguous()
            for tensor in tensors
        )
    )


def _has_flat_row_layout(tensor: torch.Tensor) -> bool:
    """Return whether leading dims can be flattened using ``stride(-2)``.

    Packed HC projection output is split into ``[..., lowrank]`` and
    ``[..., hc_count]`` views.  The latter is contiguous within each row but
    retains the packed projection's wider row stride.  The injection kernel
    accepts that layout explicitly, while still rejecting transposes and
    layouts whose leading dimensions cannot be flattened into rows.
    """

    if tensor.ndim < 2 or tensor.stride(-1) != 1:
        return False
    if tensor.stride(-2) < tensor.shape[-1]:
        return False
    return all(
        tensor.stride(dim) == tensor.shape[dim + 1] * tensor.stride(dim + 1)
        for dim in range(tensor.ndim - 2)
    )


def can_use_hc_inject_triton(
    injection_logits: torch.Tensor,
    block_output: torch.Tensor,
    residual: torch.Tensor,
) -> bool:
    """Allow packed-projection logits with a wider row stride.

    ``_hc_inject_combine_kernel`` consumes ``stride_logits_row`` and only
    requires the branch dimension to be contiguous.  The other two inputs
    retain the stricter contiguous inference contract.
    """

    tensors = (injection_logits, block_output, residual)
    return bool(
        HAS_TRITON
        and all(
            tensor.device.type not in ("cpu", "meta")
            and tensor.dtype in (torch.bfloat16, torch.float16)
            for tensor in tensors
        )
        and len({tensor.device for tensor in tensors}) == 1
        and _has_flat_row_layout(injection_logits)
        and block_output.is_contiguous()
        and residual.is_contiguous()
    )


@triton.jit
def _grouped_gemma_rmsnorm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    hidden_size: tl.constexpr,
    hc_count: tl.constexpr,
    eps: tl.constexpr,
    block_h: tl.constexpr,
) -> None:
    group_row = tl.program_id(0)
    offsets = tl.arange(0, block_h)
    mask = offsets < hidden_size
    input_base = group_row * hidden_size
    weight_base = (group_row % hc_count) * hidden_size
    values = tl.load(input_ptr + input_base + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    weight = tl.load(weight_ptr + weight_base + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    inv_rms = tl.rsqrt(tl.sum(values * values, axis=0) / hidden_size + eps)
    tl.store(
        output_ptr + input_base + offsets,
        values * inv_rms * (1.0 + weight),
        mask=mask,
    )


@triton.jit
def _hc_gate_reduce_kernel(
    logits_ptr,
    normed_ptr,
    output_ptr,
    stride_logits_row,
    stride_normed_row,
    stride_output_row,
    hidden_size: tl.constexpr,
    hc_count: tl.constexpr,
    block_h: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.program_id(1) * block_h + tl.arange(0, block_h)
    mask = offsets < hidden_size
    accumulator = tl.zeros((block_h,), dtype=tl.float32)
    for branch in tl.static_range(0, hc_count):
        branch_offsets = branch * hidden_size + offsets
        logits = tl.load(
            logits_ptr + row * stride_logits_row + branch_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        normed = tl.load(
            normed_ptr + row * stride_normed_row + branch_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sigmoid(logits) * normed
    tl.store(
        output_ptr + row * stride_output_row + offsets,
        accumulator / hc_count,
        mask=mask,
    )


@triton.jit
def _hc_inject_combine_kernel(
    injection_logits_ptr,
    block_output_ptr,
    residual_ptr,
    output_ptr,
    stride_logits_row,
    stride_block_row,
    stride_residual_row,
    stride_output_row,
    hidden_size: tl.constexpr,
    hc_count: tl.constexpr,
    block_h: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    branch = tl.program_id(1)
    offsets = tl.program_id(2) * block_h + tl.arange(0, block_h)
    mask = offsets < hidden_size
    branch_offsets = branch * hidden_size + offsets
    logits = tl.load(injection_logits_ptr + row * stride_logits_row + branch).to(
        tl.float32
    )
    injection_weight = 2.0 * tl.sigmoid(logits / hc_count)
    block_output = tl.load(
        block_output_ptr + row * stride_block_row + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    residual = tl.load(
        residual_ptr + row * stride_residual_row + branch_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output_ptr + row * stride_output_row + branch_offsets,
        residual + block_output * injection_weight,
        mask=mask,
    )


def qwen4_grouped_gemma_rmsnorm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    hc_count: int,
    eps: float,
) -> torch.Tensor:
    if hidden_states.ndim < 2 or hidden_states.shape[-1] != weight.numel():
        raise ValueError("Qwen4 HC RMSNorm received incompatible input and weight")
    if hc_count <= 0 or weight.numel() % hc_count or eps <= 0:
        raise ValueError("Qwen4 HC RMSNorm requires a valid HC count")
    if not can_use_hc_triton(hidden_states, weight):
        raise RuntimeError("Qwen4 HC RMSNorm requires contiguous accelerator tensors")
    output = torch.empty_like(hidden_states)
    if not hidden_states.numel():
        return output
    hidden_size = weight.numel() // hc_count
    block_h = triton.next_power_of_2(hidden_size)
    rows = hidden_states.numel() // weight.numel()
    _grouped_gemma_rmsnorm_kernel[(rows * hc_count,)](
        hidden_states,
        weight,
        output,
        hidden_size=hidden_size,
        hc_count=hc_count,
        eps=eps,
        block_h=block_h,
        num_warps=8 if block_h > 2048 else 4,
    )
    return output


def qwen4_grouped_gemma_rmsnorm_fake(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    hc_count: int,
    eps: float,
) -> torch.Tensor:
    del weight, hc_count, eps
    return torch.empty_like(hidden_states)


def qwen4_hc_gate_reduce(
    logits: torch.Tensor,
    normed: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    if logits.shape != normed.shape or logits.ndim < 2:
        raise ValueError("Qwen4 HC gate logits and normalized input must match")
    if hc_count <= 0 or logits.shape[-1] % hc_count:
        raise ValueError("Qwen4 HC gate reduction requires a valid HC count")
    if not can_use_hc_triton(logits, normed):
        raise RuntimeError("Qwen4 HC gate reduction requires contiguous tensors")
    hidden_size = logits.shape[-1] // hc_count
    output = torch.empty(
        (*logits.shape[:-1], hidden_size), dtype=normed.dtype, device=normed.device
    )
    if not logits.numel():
        return output
    rows = logits.numel() // logits.shape[-1]
    block_h = 256
    _hc_gate_reduce_kernel[(rows, triton.cdiv(hidden_size, block_h))](
        logits,
        normed,
        output,
        logits.stride(-2),
        normed.stride(-2),
        output.stride(-2),
        hidden_size=hidden_size,
        hc_count=hc_count,
        block_h=block_h,
        num_warps=4,
    )
    return output


def qwen4_hc_gate_reduce_fake(
    logits: torch.Tensor,
    normed: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    del normed
    return torch.empty(
        (*logits.shape[:-1], logits.shape[-1] // hc_count),
        dtype=logits.dtype,
        device=logits.device,
    )


def qwen4_hc_inject_combine(
    injection_logits: torch.Tensor,
    block_output: torch.Tensor,
    residual: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    if injection_logits.shape != (*block_output.shape[:-1], hc_count):
        raise ValueError("Qwen4 HC injection logits have an invalid shape")
    if hc_count <= 0:
        raise ValueError("Qwen4 HC injection requires a positive HC count")
    if residual.shape != (
        *block_output.shape[:-1],
        hc_count * block_output.shape[-1],
    ):
        raise ValueError("Qwen4 HC residual and block output shapes are incompatible")
    if not can_use_hc_inject_triton(injection_logits, block_output, residual):
        raise RuntimeError(
            "Qwen4 HC injection received an unsupported accelerator layout"
        )
    output = torch.empty_like(residual)
    if not residual.numel():
        return output
    hidden_size = block_output.shape[-1]
    rows = block_output.numel() // hidden_size
    block_h = 256
    _hc_inject_combine_kernel[(rows, hc_count, triton.cdiv(hidden_size, block_h))](
        injection_logits,
        block_output,
        residual,
        output,
        injection_logits.stride(-2),
        block_output.stride(-2),
        residual.stride(-2),
        output.stride(-2),
        hidden_size=hidden_size,
        hc_count=hc_count,
        block_h=block_h,
        num_warps=4,
    )
    return output


def qwen4_hc_inject_combine_fake(
    injection_logits: torch.Tensor,
    block_output: torch.Tensor,
    residual: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    del injection_logits, block_output, hc_count
    return torch.empty_like(residual)


__all__ = [
    "can_use_hc_inject_triton",
    "can_use_hc_triton",
    "qwen4_grouped_gemma_rmsnorm",
    "qwen4_hc_gate_reduce",
    "qwen4_hc_inject_combine",
]
