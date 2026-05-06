# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable sparse MLA Triton kernels."""

import torch

from vllm.triton_utils import tl, triton


def sparse_mla_decode_head_block_size(num_decode_tokens: int) -> int:
    """Choose the SM12x sparse MLA head grouping for decode kernels.

    Single-token decode is latency sensitive and does best with one head per
    program. Once there are enough query tokens, grouping heads lets the kernel
    reuse each dequantized KV row across multiple heads.
    """

    if num_decode_tokens <= 4:
        return 1
    if num_decode_tokens < 16:
        return 2
    return 4


@triton.jit
def _accumulate_indexed_attention_chunk_kernel(
    q_ptr,
    kv_flat_ptr,
    indices_ptr,
    lens_ptr,
    max_score_ptr,
    denom_ptr,
    acc_ptr,
    stride_q_t: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_kv_t,
    stride_kv_d: tl.constexpr,
    stride_indices_t: tl.constexpr,
    stride_indices_c: tl.constexpr,
    stride_state_t: tl.constexpr,
    stride_state_h: tl.constexpr,
    stride_acc_t: tl.constexpr,
    stride_acc_h: tl.constexpr,
    stride_acc_d: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_candidates,
    candidate_offset,
    scale: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_D)
    dim_mask = offsets < head_dim

    q = tl.load(
        q_ptr + token_idx * stride_q_t + head_idx * stride_q_h + offsets * stride_q_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    state_offset = token_idx * stride_state_t + head_idx * stride_state_h
    acc_offset = (
        token_idx * stride_acc_t + head_idx * stride_acc_h + offsets * stride_acc_d
    )
    running_max = tl.load(max_score_ptr + state_offset)
    running_denom = tl.load(denom_ptr + state_offset)
    running_acc = tl.load(acc_ptr + acc_offset, mask=dim_mask, other=0.0).to(tl.float32)
    valid_len = tl.load(lens_ptr + token_idx)

    for candidate_idx in range(0, num_candidates):
        kv_index = tl.load(
            indices_ptr
            + token_idx * stride_indices_t
            + candidate_idx * stride_indices_c
        )
        is_valid = ((candidate_offset + candidate_idx) < valid_len) & (kv_index >= 0)

        if is_valid:
            kv = tl.load(
                kv_flat_ptr
                + kv_index.to(tl.int64) * stride_kv_t
                + offsets * stride_kv_d,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            score = tl.sum(q * kv, axis=0) * scale
            next_max = tl.maximum(running_max, score)
            previous_weight = tl.exp(running_max - next_max)
            candidate_weight = tl.exp(score - next_max)
            running_acc = running_acc * previous_weight + kv * candidate_weight
            running_denom = running_denom * previous_weight + candidate_weight
            running_max = next_max

    tl.store(max_score_ptr + state_offset, running_max)
    tl.store(denom_ptr + state_offset, running_denom)
    tl.store(acc_ptr + acc_offset, running_acc, mask=dim_mask)


def accumulate_indexed_sparse_mla_attention_chunk(
    q: torch.Tensor,
    kv_flat: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    scale: float,
    max_score: torch.Tensor,
    denom: torch.Tensor,
    acc: torch.Tensor,
    candidate_offset: int = 0,
) -> None:
    if q.dim() == 4:
        assert q.shape[1] == 1
        q = q[:, 0]

    assert q.dim() == 3, f"Expected q shape [T, H, D], got {q.shape}"
    assert kv_flat.dim() == 2
    assert indices.dim() == 2
    assert indices.shape[0] == q.shape[0]
    assert kv_flat.shape[-1] == q.shape[-1]
    assert lens.shape[0] == q.shape[0]
    assert max_score.shape[0] == q.shape[0]
    assert max_score.shape[1] <= q.shape[1]
    assert denom.shape == max_score.shape
    assert acc.shape == (*max_score.shape, q.shape[-1])
    assert max_score.dtype == torch.float32
    assert denom.dtype == torch.float32
    assert acc.dtype == torch.float32
    assert q.is_cuda and kv_flat.is_cuda and indices.is_cuda and lens.is_cuda
    assert max_score.is_cuda and denom.is_cuda and acc.is_cuda

    num_tokens, _, head_dim = q.shape
    num_heads = max_score.shape[1]
    num_candidates = indices.shape[1]
    block_d = min(1024, triton.next_power_of_2(head_dim))
    grid = (num_tokens, num_heads)
    _accumulate_indexed_attention_chunk_kernel[grid](
        q,
        kv_flat,
        indices,
        lens,
        max_score,
        denom,
        acc,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv_flat.stride(0),
        kv_flat.stride(1),
        indices.stride(0),
        indices.stride(1),
        max_score.stride(0),
        max_score.stride(1),
        acc.stride(0),
        acc.stride(1),
        acc.stride(2),
        num_heads,
        head_dim,
        num_candidates,
        candidate_offset,
        scale,
        BLOCK_D=block_d,
        num_warps=8,
    )


@triton.jit
def _accumulate_fp8ds_global_slots_attention_chunk_multihead_kernel(
    q_ptr,
    k_cache_ptr,
    slot_ids_ptr,
    lens_ptr,
    max_score_ptr,
    denom_ptr,
    acc_ptr,
    stride_q_t: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_slot_t: tl.constexpr,
    stride_slot_c: tl.constexpr,
    stride_state_t: tl.constexpr,
    stride_state_h: tl.constexpr,
    stride_acc_t: tl.constexpr,
    stride_acc_h: tl.constexpr,
    stride_acc_d: tl.constexpr,
    cache_block_size: tl.constexpr,
    token_data_size: tl.constexpr,
    block_stride: tl.constexpr,
    fp8_dim: tl.constexpr,
    scale_dim: tl.constexpr,
    quant_block: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_candidates,
    candidate_offset,
    scale: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)
    head_offsets = head_block_idx * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    dim_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < head_dim
    matrix_mask = head_mask[:, None] & dim_mask[None, :]

    q = tl.load(
        q_ptr
        + token_idx * stride_q_t
        + head_offsets[:, None] * stride_q_h
        + dim_offsets[None, :] * stride_q_d,
        mask=matrix_mask,
        other=0.0,
    ).to(tl.float32)

    state_offsets = token_idx * stride_state_t + head_offsets * stride_state_h
    acc_offsets = (
        token_idx * stride_acc_t
        + head_offsets[:, None] * stride_acc_h
        + dim_offsets[None, :] * stride_acc_d
    )
    running_max = tl.load(
        max_score_ptr + state_offsets,
        mask=head_mask,
        other=-float("inf"),
    )
    running_denom = tl.load(denom_ptr + state_offsets, mask=head_mask, other=0.0)
    running_acc = tl.load(acc_ptr + acc_offsets, mask=matrix_mask, other=0.0).to(
        tl.float32
    )
    valid_len = tl.load(lens_ptr + token_idx)

    fp8_mask = dim_offsets < fp8_dim
    rope_mask = (dim_offsets >= fp8_dim) & dim_mask
    rope_offsets = tl.maximum(dim_offsets - fp8_dim, 0)

    for candidate_idx in range(0, num_candidates):
        slot_id = tl.load(
            slot_ids_ptr + token_idx * stride_slot_t + candidate_idx * stride_slot_c
        )
        is_valid = ((candidate_offset + candidate_idx) < valid_len) & (slot_id >= 0)

        if is_valid:
            block_idx = slot_id // cache_block_size
            pos_in_block = slot_id % cache_block_size
            cache_block_ptr = k_cache_ptr + block_idx.to(tl.int64) * block_stride
            token_data_ptr = cache_block_ptr + pos_in_block * token_data_size
            token_scale_ptr = (
                cache_block_ptr
                + cache_block_size * token_data_size
                + pos_in_block * scale_dim
            )

            x_uint8 = tl.load(token_data_ptr + dim_offsets, mask=fp8_mask, other=0)
            x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
            x_float = x_fp8.to(tl.float32)
            scale_offsets = dim_offsets // quant_block
            encoded_scale = tl.load(
                token_scale_ptr + scale_offsets,
                mask=fp8_mask,
                other=127,
            )
            dequant_scale = tl.exp2(encoded_scale.to(tl.float32) - 127.0)
            x_dequant = x_float * dequant_scale

            rope_ptr = (token_data_ptr + fp8_dim).to(tl.pointer_type(tl.bfloat16))
            rope = tl.load(rope_ptr + rope_offsets, mask=rope_mask, other=0.0).to(
                tl.float32
            )
            kv = tl.where(fp8_mask, x_dequant, rope)
            kv = tl.where(dim_mask, kv, 0.0)

            score = tl.sum(q * kv[None, :], axis=1) * scale
            next_max = tl.maximum(running_max, score)
            previous_weight = tl.exp(running_max - next_max)
            candidate_weight = tl.exp(score - next_max)
            running_acc = (
                running_acc * previous_weight[:, None]
                + kv[None, :] * candidate_weight[:, None]
            )
            running_denom = running_denom * previous_weight + candidate_weight
            running_max = next_max

    tl.store(max_score_ptr + state_offsets, running_max, mask=head_mask)
    tl.store(denom_ptr + state_offsets, running_denom, mask=head_mask)
    tl.store(acc_ptr + acc_offsets, running_acc, mask=matrix_mask)


def accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    slot_ids: torch.Tensor,
    lens: torch.Tensor,
    block_size: int,
    scale: float,
    max_score: torch.Tensor,
    denom: torch.Tensor,
    acc: torch.Tensor,
    candidate_offset: int = 0,
    head_block_size: int = 2,
) -> None:
    if q.dim() == 4:
        assert q.shape[1] == 1
        q = q[:, 0]
    if slot_ids.dim() == 3:
        assert slot_ids.shape[1] == 1
        slot_ids = slot_ids[:, 0]

    assert q.dim() == 3, f"Expected q shape [T, H, D], got {q.shape}"
    assert q.shape[-1] == 512
    assert slot_ids.dim() == 2
    assert slot_ids.shape[0] == q.shape[0]
    assert lens.shape[0] == q.shape[0]
    assert max_score.shape[0] == q.shape[0]
    assert max_score.shape[1] <= q.shape[1]
    assert denom.shape == max_score.shape
    assert acc.shape == (*max_score.shape, q.shape[-1])
    assert head_block_size in (1, 2, 4)
    assert max_score.dtype == torch.float32
    assert denom.dtype == torch.float32
    assert acc.dtype == torch.float32
    assert k_cache.dtype == torch.uint8
    assert q.is_cuda and k_cache.is_cuda and slot_ids.is_cuda and lens.is_cuda
    assert max_score.is_cuda and denom.is_cuda and acc.is_cuda

    token_fp8_dim = 448
    token_bf16_dim = 64
    token_scale_dim = 8
    quant_block_size = 64
    token_data_size = token_fp8_dim + token_bf16_dim * 2

    num_tokens, _, head_dim = q.shape
    num_heads = max_score.shape[1]
    num_candidates = slot_ids.shape[1]
    block_d = min(1024, triton.next_power_of_2(head_dim))
    grid = (num_tokens, triton.cdiv(num_heads, head_block_size))
    _accumulate_fp8ds_global_slots_attention_chunk_multihead_kernel[grid](
        q,
        k_cache,
        slot_ids,
        lens,
        max_score,
        denom,
        acc,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        slot_ids.stride(0),
        slot_ids.stride(1),
        max_score.stride(0),
        max_score.stride(1),
        acc.stride(0),
        acc.stride(1),
        acc.stride(2),
        block_size,
        token_data_size,
        k_cache.stride(0),
        token_fp8_dim,
        token_scale_dim,
        quant_block_size,
        num_heads,
        head_dim,
        num_candidates,
        candidate_offset,
        scale,
        HEAD_BLOCK=head_block_size,
        BLOCK_D=block_d,
        num_warps=8,
    )


@triton.jit
def _accumulate_fp8ds_paged_attention_chunk_multihead_kernel(
    q_ptr,
    k_cache_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    block_table_ptr,
    max_score_ptr,
    denom_ptr,
    acc_ptr,
    stride_q_t: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_block_table_t,
    stride_state_t: tl.constexpr,
    stride_state_h: tl.constexpr,
    stride_acc_t: tl.constexpr,
    stride_acc_h: tl.constexpr,
    stride_acc_d: tl.constexpr,
    cache_block_size: tl.constexpr,
    token_data_size: tl.constexpr,
    block_stride: tl.constexpr,
    fp8_dim: tl.constexpr,
    scale_dim: tl.constexpr,
    quant_block: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_candidates,
    candidate_offset,
    scale: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)
    head_offsets = head_block_idx * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    dim_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < head_dim
    matrix_mask = head_mask[:, None] & dim_mask[None, :]

    q = tl.load(
        q_ptr
        + token_idx * stride_q_t
        + head_offsets[:, None] * stride_q_h
        + dim_offsets[None, :] * stride_q_d,
        mask=matrix_mask,
        other=0.0,
    ).to(tl.float32)

    state_offsets = token_idx * stride_state_t + head_offsets * stride_state_h
    acc_offsets = (
        token_idx * stride_acc_t
        + head_offsets[:, None] * stride_acc_h
        + dim_offsets[None, :] * stride_acc_d
    )
    running_max = tl.load(
        max_score_ptr + state_offsets,
        mask=head_mask,
        other=-float("inf"),
    )
    running_denom = tl.load(denom_ptr + state_offsets, mask=head_mask, other=0.0)
    running_acc = tl.load(acc_ptr + acc_offsets, mask=matrix_mask, other=0.0).to(
        tl.float32
    )

    seq_len = tl.load(seq_lens_ptr + token_idx)
    gather_len = tl.load(gather_lens_ptr + token_idx)
    start_pos = seq_len - gather_len
    fp8_mask = dim_offsets < fp8_dim
    rope_mask = (dim_offsets >= fp8_dim) & dim_mask
    rope_offsets = tl.maximum(dim_offsets - fp8_dim, 0)

    for candidate_idx in range(0, num_candidates):
        gather_idx = candidate_offset + candidate_idx
        is_valid = gather_idx < gather_len

        if is_valid:
            pos = start_pos + gather_idx
            block_in_seq = pos // cache_block_size
            pos_in_block = pos % cache_block_size
            physical_block = tl.load(
                block_table_ptr + token_idx * stride_block_table_t + block_in_seq
            )
            cache_block_ptr = k_cache_ptr + physical_block.to(tl.int64) * block_stride
            token_data_ptr = cache_block_ptr + pos_in_block * token_data_size
            token_scale_ptr = (
                cache_block_ptr
                + cache_block_size * token_data_size
                + pos_in_block * scale_dim
            )

            x_uint8 = tl.load(token_data_ptr + dim_offsets, mask=fp8_mask, other=0)
            x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
            x_float = x_fp8.to(tl.float32)
            scale_offsets = dim_offsets // quant_block
            encoded_scale = tl.load(
                token_scale_ptr + scale_offsets,
                mask=fp8_mask,
                other=127,
            )
            dequant_scale = tl.exp2(encoded_scale.to(tl.float32) - 127.0)
            x_dequant = x_float * dequant_scale

            rope_ptr = (token_data_ptr + fp8_dim).to(tl.pointer_type(tl.bfloat16))
            rope = tl.load(rope_ptr + rope_offsets, mask=rope_mask, other=0.0).to(
                tl.float32
            )
            kv = tl.where(fp8_mask, x_dequant, rope)
            kv = tl.where(dim_mask, kv, 0.0)

            score = tl.sum(q * kv[None, :], axis=1) * scale
            next_max = tl.maximum(running_max, score)
            previous_weight = tl.exp(running_max - next_max)
            candidate_weight = tl.exp(score - next_max)
            running_acc = (
                running_acc * previous_weight[:, None]
                + kv[None, :] * candidate_weight[:, None]
            )
            running_denom = running_denom * previous_weight + candidate_weight
            running_max = next_max

    tl.store(max_score_ptr + state_offsets, running_max, mask=head_mask)
    tl.store(denom_ptr + state_offsets, running_denom, mask=head_mask)
    tl.store(acc_ptr + acc_offsets, running_acc, mask=matrix_mask)


def accumulate_fp8ds_paged_sparse_mla_attention_chunk_multihead(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    scale: float,
    max_score: torch.Tensor,
    denom: torch.Tensor,
    acc: torch.Tensor,
    candidate_offset: int,
    num_candidates: int,
    head_block_size: int = 2,
) -> None:
    if q.dim() == 4:
        assert q.shape[1] == 1
        q = q[:, 0]

    assert q.dim() == 3, f"Expected q shape [T, H, D], got {q.shape}"
    assert q.shape[-1] == 512
    assert seq_lens.shape[0] == q.shape[0]
    assert gather_lens.shape[0] == q.shape[0]
    assert block_table.shape[0] == q.shape[0]
    assert max_score.shape[0] == q.shape[0]
    assert max_score.shape[1] <= q.shape[1]
    assert denom.shape == max_score.shape
    assert acc.shape == (*max_score.shape, q.shape[-1])
    assert head_block_size in (1, 2, 4)
    assert max_score.dtype == torch.float32
    assert denom.dtype == torch.float32
    assert acc.dtype == torch.float32
    assert k_cache.dtype == torch.uint8
    assert q.is_cuda and k_cache.is_cuda
    assert seq_lens.is_cuda and gather_lens.is_cuda and block_table.is_cuda
    assert max_score.is_cuda and denom.is_cuda and acc.is_cuda

    token_fp8_dim = 448
    token_bf16_dim = 64
    token_scale_dim = 8
    quant_block_size = 64
    token_data_size = token_fp8_dim + token_bf16_dim * 2

    num_tokens, _, head_dim = q.shape
    num_heads = max_score.shape[1]
    block_d = min(1024, triton.next_power_of_2(head_dim))
    grid = (num_tokens, triton.cdiv(num_heads, head_block_size))
    _accumulate_fp8ds_paged_attention_chunk_multihead_kernel[grid](
        q,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        max_score,
        denom,
        acc,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        block_table.stride(0),
        max_score.stride(0),
        max_score.stride(1),
        acc.stride(0),
        acc.stride(1),
        acc.stride(2),
        block_size,
        token_data_size,
        k_cache.stride(0),
        token_fp8_dim,
        token_scale_dim,
        quant_block_size,
        num_heads,
        head_dim,
        num_candidates,
        candidate_offset,
        scale,
        HEAD_BLOCK=head_block_size,
        BLOCK_D=block_d,
        num_warps=8,
    )


@triton.jit
def _finish_attention_state_with_sink_kernel(
    max_score_ptr,
    denom_ptr,
    acc_ptr,
    sink_ptr,
    output_ptr,
    stride_state_t: tl.constexpr,
    stride_state_h: tl.constexpr,
    stride_acc_t: tl.constexpr,
    stride_acc_h: tl.constexpr,
    stride_acc_d: tl.constexpr,
    stride_output_t: tl.constexpr,
    stride_output_h: tl.constexpr,
    stride_output_d: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_head = tl.program_id(0)
    block_d = tl.program_id(1)
    token_idx = token_head // num_heads
    head_idx = token_head - token_idx * num_heads
    offsets = block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    dim_mask = offsets < head_dim

    state_offset = token_idx * stride_state_t + head_idx * stride_state_h
    running_max = tl.load(max_score_ptr + state_offset)
    running_denom = tl.load(denom_ptr + state_offset)
    sink = tl.load(sink_ptr + head_idx)
    has_tokens = running_denom > 0.0
    has_sink = sink > -float("inf")
    valid_max = tl.where(has_tokens, running_max, -float("inf"))
    valid_sink = tl.where(has_sink, sink, -float("inf"))
    merge_max = tl.maximum(valid_max, valid_sink)
    has_any = has_tokens | has_sink
    safe_merge_max = tl.where(has_any, merge_max, 0.0)
    safe_running_max = tl.where(has_tokens, running_max, safe_merge_max)
    safe_sink = tl.where(has_sink, sink, safe_merge_max)
    subset_scale = tl.where(has_tokens, tl.exp(safe_running_max - safe_merge_max), 0.0)
    subset_weight = running_denom * subset_scale
    sink_weight = tl.where(has_sink, tl.exp(safe_sink - safe_merge_max), 0.0)
    total_weight = subset_weight + sink_weight
    inv_total = tl.where(total_weight > 0.0, 1.0 / total_weight, 0.0)

    acc_values = tl.load(
        acc_ptr
        + token_idx * stride_acc_t
        + head_idx * stride_acc_h
        + offsets * stride_acc_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    acc_values = tl.where(has_tokens, acc_values, 0.0)
    output = acc_values * subset_scale * inv_total
    tl.store(
        output_ptr
        + token_idx * stride_output_t
        + head_idx * stride_output_h
        + offsets * stride_output_d,
        output,
        mask=dim_mask,
    )


@triton.jit
def _finish_two_attention_states_with_sink_kernel(
    max0_ptr,
    denom0_ptr,
    acc0_ptr,
    max1_ptr,
    denom1_ptr,
    acc1_ptr,
    sink_ptr,
    output_ptr,
    stride_state0_t: tl.constexpr,
    stride_state0_h: tl.constexpr,
    stride_acc0_t: tl.constexpr,
    stride_acc0_h: tl.constexpr,
    stride_acc0_d: tl.constexpr,
    stride_state1_t: tl.constexpr,
    stride_state1_h: tl.constexpr,
    stride_acc1_t: tl.constexpr,
    stride_acc1_h: tl.constexpr,
    stride_acc1_d: tl.constexpr,
    stride_output_t: tl.constexpr,
    stride_output_h: tl.constexpr,
    stride_output_d: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_head = tl.program_id(0)
    block_d = tl.program_id(1)
    token_idx = token_head // num_heads
    head_idx = token_head - token_idx * num_heads
    offsets = block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    dim_mask = offsets < head_dim

    state0_offset = token_idx * stride_state0_t + head_idx * stride_state0_h
    state1_offset = token_idx * stride_state1_t + head_idx * stride_state1_h
    max0 = tl.load(max0_ptr + state0_offset)
    denom0 = tl.load(denom0_ptr + state0_offset)
    max1 = tl.load(max1_ptr + state1_offset)
    denom1 = tl.load(denom1_ptr + state1_offset)
    sink = tl.load(sink_ptr + head_idx)

    has0 = denom0 > 0.0
    has1 = denom1 > 0.0
    has_sink = sink > -float("inf")
    valid_max0 = tl.where(has0, max0, -float("inf"))
    valid_max1 = tl.where(has1, max1, -float("inf"))
    valid_sink = tl.where(has_sink, sink, -float("inf"))
    merge_max = tl.maximum(tl.maximum(valid_max0, valid_max1), valid_sink)
    has_any = has0 | has1 | has_sink
    safe_merge_max = tl.where(has_any, merge_max, 0.0)
    safe_max0 = tl.where(has0, max0, safe_merge_max)
    safe_max1 = tl.where(has1, max1, safe_merge_max)
    safe_sink = tl.where(has_sink, sink, safe_merge_max)
    scale0 = tl.where(has0, tl.exp(safe_max0 - safe_merge_max), 0.0)
    scale1 = tl.where(has1, tl.exp(safe_max1 - safe_merge_max), 0.0)
    sink_weight = tl.where(has_sink, tl.exp(safe_sink - safe_merge_max), 0.0)
    total_weight = denom0 * scale0 + denom1 * scale1 + sink_weight
    inv_total = tl.where(total_weight > 0.0, 1.0 / total_weight, 0.0)

    acc0 = tl.load(
        acc0_ptr
        + token_idx * stride_acc0_t
        + head_idx * stride_acc0_h
        + offsets * stride_acc0_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    acc1 = tl.load(
        acc1_ptr
        + token_idx * stride_acc1_t
        + head_idx * stride_acc1_h
        + offsets * stride_acc1_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    acc0 = tl.where(has0, acc0, 0.0)
    acc1 = tl.where(has1, acc1, 0.0)
    output = (acc0 * scale0 + acc1 * scale1) * inv_total
    tl.store(
        output_ptr
        + token_idx * stride_output_t
        + head_idx * stride_output_h
        + offsets * stride_output_d,
        output,
        mask=dim_mask,
    )


def finish_two_sparse_mla_attention_states_with_sink(
    max_score0: torch.Tensor,
    denom0: torch.Tensor,
    acc0: torch.Tensor,
    max_score1: torch.Tensor,
    denom1: torch.Tensor,
    acc1: torch.Tensor,
    attn_sink: torch.Tensor,
    output: torch.Tensor,
) -> None:
    assert max_score0.shape == denom0.shape
    assert max_score1.shape == denom1.shape
    assert max_score0.shape == max_score1.shape
    assert acc0.shape == acc1.shape
    assert acc0.shape[:2] == max_score0.shape
    assert output.shape[0] == acc0.shape[0]
    assert output.shape[1] >= acc0.shape[1]
    assert output.shape[2] == acc0.shape[2]
    assert attn_sink.shape[0] >= acc0.shape[1]
    assert max_score0.dtype == torch.float32
    assert denom0.dtype == torch.float32
    assert acc0.dtype == torch.float32
    assert max_score1.dtype == torch.float32
    assert denom1.dtype == torch.float32
    assert acc1.dtype == torch.float32
    assert max_score0.is_cuda and denom0.is_cuda and acc0.is_cuda
    assert max_score1.is_cuda and denom1.is_cuda and acc1.is_cuda
    assert attn_sink.is_cuda and output.is_cuda

    num_tokens, num_heads, head_dim = acc0.shape
    block_d = min(128, triton.next_power_of_2(head_dim))
    grid = (num_tokens * num_heads, triton.cdiv(head_dim, block_d))
    _finish_two_attention_states_with_sink_kernel[grid](
        max_score0,
        denom0,
        acc0,
        max_score1,
        denom1,
        acc1,
        attn_sink,
        output,
        max_score0.stride(0),
        max_score0.stride(1),
        acc0.stride(0),
        acc0.stride(1),
        acc0.stride(2),
        max_score1.stride(0),
        max_score1.stride(1),
        acc1.stride(0),
        acc1.stride(1),
        acc1.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        num_heads,
        head_dim,
        BLOCK_D=block_d,
        num_warps=4,
    )


def finish_sparse_mla_attention_with_sink(
    max_score: torch.Tensor,
    denom: torch.Tensor,
    acc: torch.Tensor,
    attn_sink: torch.Tensor,
    output: torch.Tensor,
) -> None:
    assert max_score.shape == denom.shape
    assert acc.shape[:2] == max_score.shape
    assert output.shape[0] == acc.shape[0]
    assert output.shape[1] >= acc.shape[1]
    assert output.shape[2] == acc.shape[2]
    assert attn_sink.shape[0] >= acc.shape[1]
    assert max_score.dtype == torch.float32
    assert denom.dtype == torch.float32
    assert acc.dtype == torch.float32
    assert max_score.is_cuda and denom.is_cuda and acc.is_cuda
    assert attn_sink.is_cuda and output.is_cuda

    num_tokens, num_heads, head_dim = acc.shape
    block_d = min(128, triton.next_power_of_2(head_dim))
    grid = (num_tokens * num_heads, triton.cdiv(head_dim, block_d))
    _finish_attention_state_with_sink_kernel[grid](
        max_score,
        denom,
        acc,
        attn_sink,
        output,
        max_score.stride(0),
        max_score.stride(1),
        acc.stride(0),
        acc.stride(1),
        acc.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        num_heads,
        head_dim,
        BLOCK_D=block_d,
        num_warps=4,
    )
