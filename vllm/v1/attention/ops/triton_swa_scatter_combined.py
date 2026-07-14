# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _swa_scatter_combined_kernel(
    workspace_ptr,
    kv_c_normed_ptr,
    k_pe_ptr,
    ctx_dst_ptr,
    new_dst_ptr,
    combined_kv_c_ptr,
    combined_k_pe_ptr,
    total_windowed_tokens,
    workspace_row_stride: tl.int64,
    kv_c_normed_row_stride: tl.int64,
    k_pe_row_stride: tl.int64,
    combined_kv_c_row_stride: tl.int64,
    combined_k_pe_row_stride: tl.int64,
    L: tl.constexpr,
    R: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    is_ctx = pid < total_windowed_tokens
    offs_l = tl.arange(0, BLOCK_L)
    mask_l = offs_l < L
    offs_r = tl.arange(0, BLOCK_R)
    mask_r = offs_r < R

    if is_ctx:
        src_row = pid
        dst_row = tl.load(ctx_dst_ptr + pid)
        kv_c_src_base = workspace_ptr + src_row.to(tl.int64) * workspace_row_stride
        kv_c_vals = tl.load(kv_c_src_base + offs_l, mask=mask_l, other=0.0)
        k_pe_vals = tl.load(kv_c_src_base + L + offs_r, mask=mask_r, other=0.0)
    else:
        src_row = pid - total_windowed_tokens
        dst_row = tl.load(new_dst_ptr + src_row)
        kv_c_vals = tl.load(
            kv_c_normed_ptr + src_row.to(tl.int64) * kv_c_normed_row_stride + offs_l,
            mask=mask_l,
            other=0.0,
        )
        k_pe_vals = tl.load(
            k_pe_ptr + src_row.to(tl.int64) * k_pe_row_stride + offs_r,
            mask=mask_r,
            other=0.0,
        )

    tl.store(
        combined_kv_c_ptr + dst_row * combined_kv_c_row_stride + offs_l,
        kv_c_vals,
        mask=mask_l,
    )
    tl.store(
        combined_k_pe_ptr + dst_row * combined_k_pe_row_stride + offs_r,
        k_pe_vals,
        mask=mask_r,
    )


def swa_scatter_combined(
    workspace: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    ctx_dst: torch.Tensor,
    new_dst: torch.Tensor,
    combined_kv_c: torch.Tensor,
    combined_k_pe: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> None:
    """Fused scatter: replaces 4 separate index_put launches with one
    Triton kernel. Reads context from `workspace` (packed
    `[kv_lora_rank | qk_rope_head_dim]` per row) and new tokens from
    `kv_c_normed` / `k_pe`, writing both into the combined output
    buffers.
    """
    total_windowed_tokens = ctx_dst.shape[0]
    num_new_tokens = new_dst.shape[0]
    grid = (total_windowed_tokens + num_new_tokens,)
    block_l = triton.next_power_of_2(kv_lora_rank)
    block_r = triton.next_power_of_2(qk_rope_head_dim)
    _swa_scatter_combined_kernel[grid](
        workspace,
        kv_c_normed,
        k_pe,
        ctx_dst,
        new_dst,
        combined_kv_c,
        combined_k_pe,
        total_windowed_tokens,
        workspace.stride(0),
        kv_c_normed.stride(0),
        k_pe.stride(0),
        combined_kv_c.stride(0),
        combined_k_pe.stride(0),
        L=kv_lora_rank,
        R=qk_rope_head_dim,
        BLOCK_L=block_l,
        BLOCK_R=block_r,
    )
