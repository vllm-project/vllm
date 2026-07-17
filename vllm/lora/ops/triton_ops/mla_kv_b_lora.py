# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LoRA corrections for absorbed MLA ``kv_b_proj`` weights.

Absorbed MLA replaces ``kv_b_proj.forward()`` with BMMs using its base K/V
weights. These kernels add the routed low-rank delta without materializing it.
"""

from contextlib import suppress

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_BLOCK_M = 16
_STEP_A_BLOCK_K = 64
_STEP_A_BLOCK_N = 16
_STEP_B_BLOCK_K = 16
_STEP_B_BLOCK_N = 64


@triton.jit
def _mla_lora_step_a_kernel(
    x,
    weight,
    output,
    num_rows,
    contraction_size,
    output_size,
    x_stride_m,
    x_stride_h,
    x_stride_k,
    weight_stride_l,
    weight_stride_row,
    weight_stride_col,
    output_stride_m,
    output_stride_h,
    output_stride_n,
    token_indices_sorted_by_lora_ids,
    num_tokens_per_lora,
    lora_token_start_loc,
    lora_ids,
    full_head_dim: tl.constexpr,
    weight_is_k_slice: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)

    lora_id = tl.load(lora_ids + lora_idx)
    if lora_id == -1:
        return

    num_pid_n = tl.cdiv(output_size, block_n)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    lora_num_tokens = tl.load(num_tokens_per_lora + lora_idx)
    group_offset = pid_m * block_m
    if group_offset >= lora_num_tokens:
        return

    offsets_m = group_offset + tl.arange(0, block_m)
    offsets_n = pid_n * block_n + tl.arange(0, block_n)
    group_start = tl.load(lora_token_start_loc + lora_idx)
    token_rows = tl.load(
        token_indices_sorted_by_lora_ids + group_start + offsets_m,
        mask=offsets_m < lora_num_tokens,
        other=0,
    )
    row_mask = (offsets_m < lora_num_tokens) & (token_rows < num_rows)
    safe_rows = tl.minimum(token_rows, num_rows - 1)
    safe_n = tl.minimum(offsets_n, output_size - 1)

    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    offsets_k = tl.arange(0, block_k)
    for k_block in range(0, tl.cdiv(contraction_size, block_k)):
        current_k = k_block * block_k + offsets_k
        k_mask = current_k < contraction_size
        safe_k = tl.minimum(current_k, contraction_size - 1)
        x_tile = tl.load(
            x
            + safe_rows[:, None] * x_stride_m
            + head_id * x_stride_h
            + safe_k[None, :] * x_stride_k,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        if weight_is_k_slice:
            weight_rows = head_id * full_head_dim + safe_k[:, None]
            weight_cols = safe_n[None, :]
        else:
            weight_rows = safe_n[None, :]
            weight_cols = safe_k[:, None]
        weight_tile = tl.load(
            weight
            + lora_id * weight_stride_l
            + weight_rows * weight_stride_row
            + weight_cols * weight_stride_col,
            mask=k_mask[:, None] & (offsets_n[None, :] < output_size),
            other=0.0,
        )
        accumulator += tl.dot(x_tile, weight_tile)

    output_offsets = (
        safe_rows[:, None] * output_stride_m
        + head_id * output_stride_h
        + safe_n[None, :] * output_stride_n
    )
    output_mask = row_mask[:, None] & (offsets_n[None, :] < output_size)
    tl.store(output + output_offsets, accumulator, mask=output_mask)


@triton.jit
def _mla_lora_step_b_kernel(
    x,
    weight,
    output,
    num_rows,
    contraction_size,
    output_size,
    x_stride_m,
    x_stride_h,
    x_stride_k,
    weight_stride_l,
    weight_stride_row,
    weight_stride_col,
    output_stride_m,
    output_stride_h,
    output_stride_n,
    token_indices_sorted_by_lora_ids,
    num_tokens_per_lora,
    lora_token_start_loc,
    lora_ids,
    full_head_dim: tl.constexpr,
    value_offset: tl.constexpr,
    weight_is_v_slice: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)

    lora_id = tl.load(lora_ids + lora_idx)
    if lora_id == -1:
        return

    num_pid_n = tl.cdiv(output_size, block_n)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    lora_num_tokens = tl.load(num_tokens_per_lora + lora_idx)
    group_offset = pid_m * block_m
    if group_offset >= lora_num_tokens:
        return

    offsets_m = group_offset + tl.arange(0, block_m)
    offsets_n = pid_n * block_n + tl.arange(0, block_n)
    group_start = tl.load(lora_token_start_loc + lora_idx)
    token_rows = tl.load(
        token_indices_sorted_by_lora_ids + group_start + offsets_m,
        mask=offsets_m < lora_num_tokens,
        other=0,
    )
    row_mask = (offsets_m < lora_num_tokens) & (token_rows < num_rows)
    safe_rows = tl.minimum(token_rows, num_rows - 1)
    safe_n = tl.minimum(offsets_n, output_size - 1)

    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    offsets_k = tl.arange(0, block_k)
    for k_block in range(0, tl.cdiv(contraction_size, block_k)):
        current_k = k_block * block_k + offsets_k
        k_mask = current_k < contraction_size
        safe_k = tl.minimum(current_k, contraction_size - 1)
        x_tile = tl.load(
            x
            + safe_rows[:, None] * x_stride_m
            + head_id * x_stride_h
            + safe_k[None, :] * x_stride_k,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        if weight_is_v_slice:
            weight_rows = head_id * full_head_dim + value_offset + safe_n[None, :]
            weight_cols = safe_k[:, None]
        else:
            weight_rows = safe_k[:, None]
            weight_cols = safe_n[None, :]
        weight_tile = tl.load(
            weight
            + lora_id * weight_stride_l
            + weight_rows * weight_stride_row
            + weight_cols * weight_stride_col,
            mask=k_mask[:, None] & (offsets_n[None, :] < output_size),
            other=0.0,
        )
        accumulator += tl.dot(x_tile, weight_tile)

    output_offsets = (
        safe_rows[:, None] * output_stride_m
        + head_id * output_stride_h
        + safe_n[None, :] * output_stride_n
    )
    output_mask = row_mask[:, None] & (offsets_n[None, :] < output_size)
    accumulator += tl.load(output + output_offsets, mask=output_mask, other=0.0)
    tl.store(output + output_offsets, accumulator, mask=output_mask)


def _launch_step_a(
    x: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    num_active_loras: torch.Tensor,
    *,
    full_head_dim: int,
    weight_is_k_slice: bool,
) -> None:
    num_rows, num_heads, contraction_size = x.shape
    output_size = output.shape[-1]
    total_tokens = token_indices_sorted_by_lora_ids.shape[0]
    grid = (
        triton.cdiv(total_tokens, _BLOCK_M) * triton.cdiv(output_size, _STEP_A_BLOCK_N),
        num_heads,
        num_active_loras.item(),
    )
    _mla_lora_step_a_kernel[grid](
        x,
        weight,
        output,
        num_rows,
        contraction_size,
        output_size,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        full_head_dim=full_head_dim,
        weight_is_k_slice=weight_is_k_slice,
        block_m=_BLOCK_M,
        block_n=_STEP_A_BLOCK_N,
        block_k=_STEP_A_BLOCK_K,
    )


def _launch_step_b(
    x: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    num_active_loras: torch.Tensor,
    *,
    full_head_dim: int,
    value_offset: int,
    weight_is_v_slice: bool,
) -> None:
    num_rows, num_heads, contraction_size = x.shape
    output_size = output.shape[-1]
    total_tokens = token_indices_sorted_by_lora_ids.shape[0]
    grid = (
        triton.cdiv(total_tokens, _BLOCK_M) * triton.cdiv(output_size, _STEP_B_BLOCK_N),
        num_heads,
        num_active_loras.item(),
    )
    _mla_lora_step_b_kernel[grid](
        x,
        weight,
        output,
        num_rows,
        contraction_size,
        output_size,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        full_head_dim=full_head_dim,
        value_offset=value_offset,
        weight_is_v_slice=weight_is_v_slice,
        block_m=_BLOCK_M,
        block_n=_STEP_B_BLOCK_N,
        block_k=_STEP_B_BLOCK_K,
    )


def apply_mla_kv_b_lora_native(
    x: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    output: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    qk_nope_head_dim: int,
    *,
    query_side: bool,
) -> None:
    num_heads = x.shape[1]
    full_head_dim = lora_b.shape[2] // num_heads
    b_by_head = lora_b[:, 0].view(
        lora_b.shape[0], num_heads, full_head_dim, lora_b.shape[-1]
    )
    for lora_id in range(lora_a.shape[0]):
        token_mask = token_lora_mapping[: x.shape[0]] == lora_id
        if not torch.any(token_mask):
            continue
        a = lora_a[lora_id, 0]
        if query_side:
            b_k = b_by_head[lora_id, :, :qk_nope_head_dim]
            correction = torch.einsum("mhp,hpr,rl->mhl", x[token_mask], b_k, a)
        else:
            b_v = b_by_head[lora_id, :, qk_nope_head_dim:]
            correction = torch.einsum("mhl,rl,hvr->mhv", x[token_mask], a, b_v)
        output[token_mask] += correction


@torch.inference_mode()
def _mla_kv_b_lora_q(
    q_nope: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    output: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    num_active_loras: torch.Tensor,
    qk_nope_head_dim: int,
    v_head_dim: int,
) -> None:
    if no_lora_flag_cpu.item():
        return
    if not current_platform.is_cuda_alike():
        apply_mla_kv_b_lora_native(
            q_nope,
            lora_a,
            lora_b,
            output,
            token_lora_mapping,
            qk_nope_head_dim,
            query_side=True,
        )
        return

    a = lora_a.squeeze(1)
    b = lora_b.squeeze(1)
    intermediate = torch.empty(
        (*q_nope.shape[:2], b.shape[-1]),
        dtype=q_nope.dtype,
        device=q_nope.device,
    )
    full_head_dim = qk_nope_head_dim + v_head_dim
    _launch_step_a(
        q_nope,
        b,
        intermediate,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        num_active_loras,
        full_head_dim=full_head_dim,
        weight_is_k_slice=True,
    )
    _launch_step_b(
        intermediate,
        a,
        output,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        num_active_loras,
        full_head_dim=0,
        value_offset=0,
        weight_is_v_slice=False,
    )


@torch.inference_mode()
def _mla_kv_b_lora_v(
    latent_output: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    output: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    num_active_loras: torch.Tensor,
    qk_nope_head_dim: int,
    v_head_dim: int,
) -> None:
    if no_lora_flag_cpu.item():
        return
    if not current_platform.is_cuda_alike():
        apply_mla_kv_b_lora_native(
            latent_output,
            lora_a,
            lora_b,
            output,
            token_lora_mapping,
            qk_nope_head_dim,
            query_side=False,
        )
        return

    a = lora_a.squeeze(1)
    b = lora_b.squeeze(1)
    intermediate = torch.empty(
        (*latent_output.shape[:2], a.shape[1]),
        dtype=latent_output.dtype,
        device=latent_output.device,
    )
    full_head_dim = qk_nope_head_dim + v_head_dim
    _launch_step_a(
        latent_output,
        a,
        intermediate,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        num_active_loras,
        full_head_dim=0,
        weight_is_k_slice=False,
    )
    _launch_step_b(
        intermediate,
        b,
        output,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        num_active_loras,
        full_head_dim=full_head_dim,
        value_offset=qk_nope_head_dim,
        weight_is_v_slice=True,
    )


def _mla_kv_b_lora_fake(*args, **kwargs) -> None:
    return


for op_name, op_func in (
    ("mla_kv_b_lora_q", _mla_kv_b_lora_q),
    ("mla_kv_b_lora_v", _mla_kv_b_lora_v),
):
    with suppress(AttributeError):
        direct_register_custom_op(
            op_name=op_name,
            op_func=op_func,
            mutates_args=["output"],
            fake_impl=_mla_kv_b_lora_fake,
            tags=(torch.Tag.flexible_layout,),
        )

mla_kv_b_lora_q = getattr(torch.ops.vllm, "mla_kv_b_lora_q", _mla_kv_b_lora_q)
mla_kv_b_lora_v = getattr(torch.ops.vllm, "mla_kv_b_lora_v", _mla_kv_b_lora_v)
