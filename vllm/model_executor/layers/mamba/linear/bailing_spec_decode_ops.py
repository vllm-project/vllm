# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _bailing_linear_attn_decode_spec_step_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    kv_cache_ptr,
    slope_rate_ptr,
    state_indices_ptr,
    query_start_loc_ptr,
    num_accepted_tokens_ptr,
    output_ptr,
    q_start: tl.constexpr,
    D: tl.constexpr,
    q_b_stride,
    q_h_stride,
    q_d_stride,
    k_b_stride,
    k_h_stride,
    k_d_stride,
    v_b_stride,
    v_h_stride,
    v_d_stride,
    cache_b_stride,
    cache_h_stride,
    cache_d0_stride,
    cache_d1_stride,
    state_indices_b_stride,
    state_indices_t_stride,
    output_b_stride,
    output_d_stride,
    DRAFT_IDX: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_id = tl.program_id(0)
    head_id = tl.program_id(1)
    block_id = tl.program_id(2)

    req_start = tl.load(query_start_loc_ptr + req_id).to(tl.int64)
    req_end = tl.load(query_start_loc_ptr + req_id + 1).to(tl.int64)
    query_len = req_end - req_start
    if query_len <= DRAFT_IDX:
        return

    dst_slot = tl.load(
        state_indices_ptr
        + req_id * state_indices_b_stride
        + DRAFT_IDX * state_indices_t_stride
    ).to(tl.int64)
    if dst_slot == -1:
        return

    if DRAFT_IDX == 0:
        accepted_offset = tl.load(num_accepted_tokens_ptr + req_id).to(tl.int64) - 1
        accepted_offset = tl.maximum(accepted_offset, 0)
        accepted_offset = tl.minimum(accepted_offset, STATE_WIDTH - 1)
        src_slot = tl.load(
            state_indices_ptr
            + req_id * state_indices_b_stride
            + accepted_offset * state_indices_t_stride
        ).to(tl.int64)
    else:
        src_slot = tl.load(
            state_indices_ptr
            + req_id * state_indices_b_stride
            + (DRAFT_IDX - 1) * state_indices_t_stride
        ).to(tl.int64)
    if src_slot == -1:
        return

    token_idx = req_start - q_start + DRAFT_IDX
    qk_offsets = tl.arange(0, D)
    v_offsets = tl.arange(0, BLOCK_SIZE) + block_id * BLOCK_SIZE
    qk_mask = qk_offsets < D
    v_mask = v_offsets < D
    kv_mask = qk_mask[:, None] & v_mask[None, :]

    q = tl.load(
        q_ptr + token_idx * q_b_stride + head_id * q_h_stride + qk_offsets * q_d_stride,
        mask=qk_mask,
        other=0.0,
    )
    k = tl.load(
        k_ptr + token_idx * k_b_stride + head_id * k_h_stride + qk_offsets * k_d_stride,
        mask=qk_mask,
        other=0.0,
    )
    v = tl.load(
        v_ptr + token_idx * v_b_stride + head_id * v_h_stride + v_offsets * v_d_stride,
        mask=v_mask,
        other=0.0,
    )

    cache_offsets = (
        qk_offsets[:, None] * cache_d0_stride + v_offsets[None, :] * cache_d1_stride
    )
    src_cache_ptr = (
        kv_cache_ptr
        + src_slot * cache_b_stride
        + head_id * cache_h_stride
        + cache_offsets
    )
    dst_cache_ptr = (
        kv_cache_ptr
        + dst_slot * cache_b_stride
        + head_id * cache_h_stride
        + cache_offsets
    )

    slope = tl.load(slope_rate_ptr + head_id)
    decay = tl.exp(-slope)
    kv_old = tl.load(src_cache_ptr, mask=kv_mask, other=0.0)
    kv_new = k[:, None] * v[None, :] + decay * kv_old

    output = tl.sum(q[:, None].to(tl.float32) * kv_new, axis=0)
    tl.store(dst_cache_ptr, kv_new, mask=kv_mask)
    tl.store(
        output_ptr
        + token_idx * output_b_stride
        + (head_id * D + v_offsets) * output_d_stride,
        output,
        mask=v_mask,
    )


def bailing_linear_attention_decode_spec(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    slope_rate: torch.Tensor,
    state_indices_tensor: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    q_start: int,
    q_end: int | None,
    slot_start: int,
    slot_end: int | None,
    block_size: int,
) -> torch.Tensor:
    q_decode = q[q_start:q_end]
    k_decode = k[q_start:q_end]
    v_decode = v[q_start:q_end]
    hidden = torch.empty(
        (q_decode.shape[0], q.shape[1] * q.shape[2]),
        device=q.device,
        dtype=q.dtype,
    )
    hidden.zero_()

    state_indices_tensor = state_indices_tensor[slot_start:slot_end]
    query_start_loc = query_start_loc.to(device=q.device)

    batch_size = state_indices_tensor.shape[0]
    num_heads = q_decode.shape[1]
    head_dim = q_decode.shape[2]
    assert k_decode.shape == (q_decode.shape[0], num_heads, head_dim)
    assert v_decode.shape == (q_decode.shape[0], num_heads, head_dim)
    state_width = state_indices_tensor.shape[1]

    grid = (batch_size, num_heads, triton.cdiv(head_dim, block_size))
    for draft_idx in range(state_width):
        _bailing_linear_attn_decode_spec_step_kernel[grid](
            q_decode,
            k_decode,
            v_decode,
            kv_cache,
            slope_rate,
            state_indices_tensor,
            query_start_loc,
            num_accepted_tokens[:batch_size],
            hidden,
            q_start,
            head_dim,
            q_decode.stride(0),
            q_decode.stride(1),
            q_decode.stride(2),
            k_decode.stride(0),
            k_decode.stride(1),
            k_decode.stride(2),
            v_decode.stride(0),
            v_decode.stride(1),
            v_decode.stride(2),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            kv_cache.stride(3),
            state_indices_tensor.stride(0),
            state_indices_tensor.stride(1),
            hidden.stride(0),
            hidden.stride(1),
            DRAFT_IDX=draft_idx,
            STATE_WIDTH=state_width,
            BLOCK_SIZE=block_size,
        )

    return hidden
