# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


@triton.jit
def _kda_gate(
    raw_g,
    dt_bias,
    A,
    lower_bound,
    USE_LOWER_BOUND: tl.constexpr,
):
    gate_input = raw_g + dt_bias
    if USE_LOWER_BOUND:
        return lower_bound * tl.sigmoid(A * gate_input)
    softplus_gate = tl.where(
        gate_input > 20.0,
        gate_input,
        tl.log(1.0 + tl.exp(gate_input)),
    )
    return -A * softplus_gate


@triton.jit
def _kda_replay_step(
    state,
    k,
    v,
    raw_g,
    raw_beta,
    dt_bias,
    A,
    lower_bound,
    USE_LOWER_BOUND: tl.constexpr,
):
    normalized_k = k * tl.rsqrt(tl.sum(k * k) + 1e-6)
    gate = _kda_gate(
        raw_g,
        dt_bias,
        A,
        lower_bound,
        USE_LOWER_BOUND,
    )

    state *= tl.exp(gate)[None, :]
    correction = v - tl.sum(state * normalized_k[None, :], axis=1)
    correction *= tl.sigmoid(raw_beta)
    return state + correction[:, None] * normalized_k[None, :], correction


@triton.jit
def _kda_replayssm_verify_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    raw_g_ptr,
    raw_beta_ptr,
    A_log_ptr,
    dt_bias_ptr,
    state_ptr,
    correction_cache_ptr,
    kg_cache_ptr,
    out_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    lower_bound,
    null_block_id,
    stride_q_token,
    stride_k_token,
    stride_v_token,
    stride_g_token,
    stride_beta_token,
    stride_state_block,
    stride_state_head,
    stride_state_v,
    stride_state_k,
    stride_correction_block,
    stride_correction_head,
    stride_correction_pos,
    stride_correction_dim,
    stride_kg_block,
    stride_kg_head,
    stride_kg_pos,
    stride_kg_dim,
    stride_out_token,
    stride_query_start_loc,
    stride_state_indices,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    bos = tl.load(query_start_loc_ptr + pid_b * stride_query_start_loc).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + (pid_b + 1) * stride_query_start_loc).to(
        tl.int64
    )
    query_len = eos - bos
    state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(tl.int64)

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]

    if state_idx <= null_block_id:
        for token_offset in tl.static_range(SPEC_QUERY_LEN):
            token_valid = token_offset < query_len
            tl.store(
                out_ptr + (bos + token_offset) * stride_out_token + pid_h * V + offs_v,
                tl.zeros([BV], dtype=tl.float32),
                mask=token_valid & mask_v,
            )
        return

    state_ptrs = (
        state_ptr
        + state_idx * stride_state_block
        + pid_h * stride_state_head
        + offs_v[:, None] * stride_state_v
        + offs_k[None, :] * stride_state_k
    )
    state = tl.load(state_ptrs, mask=mask_state, other=0.0).to(tl.float32)
    A = tl.exp(tl.load(A_log_ptr + pid_h).to(tl.float32))

    for token_offset in tl.static_range(SPEC_QUERY_LEN):
        token_valid = token_offset < query_len
        token = bos + token_offset
        q = tl.load(
            q_ptr + token * stride_q_token + pid_h * K + offs_k,
            mask=token_valid & mask_k,
            other=0.0,
        ).to(tl.float32)
        k = tl.load(
            k_ptr + token * stride_k_token + pid_h * K + offs_k,
            mask=token_valid & mask_k,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_ptr + token * stride_v_token + pid_h * V + offs_v,
            mask=token_valid & mask_v,
            other=0.0,
        ).to(tl.float32)
        raw_g = tl.load(
            raw_g_ptr + token * stride_g_token + pid_h * K + offs_k,
            mask=token_valid & mask_k,
            other=0.0,
        ).to(tl.float32)
        raw_beta = tl.load(
            raw_beta_ptr + token * stride_beta_token + pid_h,
            mask=token_valid,
            other=0.0,
        ).to(tl.float32)

        q *= tl.rsqrt(tl.sum(q * q) + 1e-6) * (K**-0.5)
        dt_bias = tl.load(dt_bias_ptr + pid_h * K + offs_k, mask=mask_k, other=0.0).to(
            tl.float32
        )
        updated_state, correction = _kda_replay_step(
            state,
            k,
            v,
            raw_g,
            raw_beta,
            dt_bias,
            A,
            lower_bound,
            USE_LOWER_BOUND,
        )
        state = tl.where(token_valid, updated_state, state)

        out = tl.sum(state * q[None, :], axis=1)
        tl.store(
            out_ptr + token * stride_out_token + pid_h * V + offs_v,
            out,
            mask=token_valid & mask_v,
        )

        correction_ptr = (
            correction_cache_ptr
            + state_idx * stride_correction_block
            + pid_h * stride_correction_head
            + token_offset * stride_correction_pos
        )
        tl.store(
            correction_ptr + offs_v * stride_correction_dim,
            correction,
            mask=token_valid & mask_v,
        )
        if pid_v == 0:
            kg_ptr = (
                kg_cache_ptr
                + state_idx * stride_kg_block
                + pid_h * stride_kg_head
                + token_offset * stride_kg_pos
            )
            tl.store(
                kg_ptr + offs_k * stride_kg_dim,
                k,
                mask=token_valid & mask_k,
            )
            tl.store(
                kg_ptr + (K + offs_k) * stride_kg_dim,
                raw_g,
                mask=token_valid & mask_k,
            )


@triton.heuristics(
    {
        "HAS_REQUEST_INDICES": lambda args: args["request_indices_ptr"] is not None,
        "ALIGN_MODE": lambda args: args["block_table_ptr"] is not None,
    }
)
@triton.jit
def _prepare_commit_plan_kernel(
    num_accepted_ptr,
    request_indices_ptr,
    state_indices_ptr,
    query_start_loc_ptr,
    block_table_ptr,
    num_computed_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_replay_counts_ptr,
    null_block_id,
    mamba_block_size,
    block_table_width,
    stride_num_accepted,
    stride_request_indices,
    stride_state_indices,
    stride_query_start_loc,
    stride_block_table_row,
    stride_block_table_col,
    stride_num_computed,
    SPEC_QUERY_LEN: tl.constexpr,
    HAS_REQUEST_INDICES: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
):
    spec_idx = tl.program_id(0)
    source_state_idx = tl.load(state_indices_ptr + spec_idx * stride_state_indices).to(
        tl.int64
    )
    request_idx = spec_idx
    if HAS_REQUEST_INDICES:
        request_idx = tl.load(
            request_indices_ptr + spec_idx * stride_request_indices
        ).to(tl.int64)
    num_accepted = tl.load(num_accepted_ptr + request_idx * stride_num_accepted).to(
        tl.int32
    )
    bos = tl.load(query_start_loc_ptr + spec_idx * stride_query_start_loc).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + (spec_idx + 1) * stride_query_start_loc).to(
        tl.int64
    )
    query_len = (eos - bos).to(tl.int32)
    commit_len = tl.minimum(tl.maximum(num_accepted, 0), query_len)
    commit_len = tl.minimum(commit_len, SPEC_QUERY_LEN)

    final_state_idx = source_state_idx
    boundary_state_idx = null_block_id
    boundary_replay_count = 0
    if ALIGN_MODE:
        num_computed = tl.load(num_computed_ptr + request_idx * stride_num_computed).to(
            tl.int32
        )
        final_num_computed = num_computed + commit_len
        final_state_col = tl.minimum(
            final_num_computed // mamba_block_size, block_table_width - 1
        )
        final_state_idx = tl.load(
            block_table_ptr
            + request_idx * stride_block_table_row
            + final_state_col * stride_block_table_col
        ).to(tl.int64)
        next_boundary = (num_computed // mamba_block_size + 1) * mamba_block_size
        crosses_boundary = final_num_computed >= next_boundary
        boundary_replay_count = next_boundary - num_computed
        boundary_state_idx = tl.load(
            block_table_ptr
            + request_idx * stride_block_table_row
            + (next_boundary // mamba_block_size - 1) * stride_block_table_col,
            mask=crosses_boundary,
            other=null_block_id,
        ).to(tl.int64)
    valid = (source_state_idx > null_block_id) & (commit_len > 0)
    tl.store(commit_lens_ptr + spec_idx, tl.where(valid, commit_len, 0))
    tl.store(
        final_state_indices_ptr + spec_idx,
        tl.where(valid, final_state_idx, null_block_id),
    )
    tl.store(
        boundary_state_indices_ptr + spec_idx,
        tl.where(valid, boundary_state_idx, null_block_id),
    )
    tl.store(
        boundary_replay_counts_ptr + spec_idx,
        tl.where(valid, boundary_replay_count, 0),
    )


@triton.jit
def _compact_conv_state_kernel(
    conv_state_ref_ptr,
    conv_state_base_addrs_ptr,
    conv_state_block_strides_ptr,
    conv_state_dim_strides_ptr,
    conv_state_token_strides_ptr,
    state_indices_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_replay_counts_ptr,
    null_block_id,
    conv_dim,
    conv_history_len,
    stride_state_indices,
    BLOCK_D: tl.constexpr,
    BLOCK_HISTORY: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_l = tl.program_id(2)
    source_state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(
        tl.int64
    )
    if source_state_idx <= null_block_id:
        return

    commit_len = tl.load(commit_lens_ptr + pid_b)
    if commit_len == 0:
        return
    final_state_idx = tl.load(final_state_indices_ptr + pid_b).to(tl.int64)
    boundary_state_idx = tl.load(boundary_state_indices_ptr + pid_b).to(tl.int64)
    boundary_replay_count = tl.load(boundary_replay_counts_ptr + pid_b)

    if final_state_idx <= null_block_id:
        return

    base_addr = tl.load(conv_state_base_addrs_ptr + pid_l)
    block_stride = tl.load(conv_state_block_strides_ptr + pid_l)
    dim_stride = tl.load(conv_state_dim_strides_ptr + pid_l)
    token_stride = tl.load(conv_state_token_strides_ptr + pid_l)
    conv_state_ptr = base_addr.to(tl.pointer_type(conv_state_ref_ptr.dtype.element_ty))
    source_ptr = conv_state_ptr + source_state_idx * block_stride
    final_ptr = conv_state_ptr + final_state_idx * block_stride

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_h = tl.arange(0, BLOCK_HISTORY)
    mask = (offs_d[:, None] < conv_dim) & (offs_h[None, :] < conv_history_len)
    final_values = tl.load(
        source_ptr
        + offs_d[:, None] * dim_stride
        + (commit_len - 1 + offs_h[None, :]) * token_stride,
        mask=mask,
    )
    if ALIGN_MODE:
        boundary_values = tl.load(
            source_ptr
            + offs_d[:, None] * dim_stride
            + (boundary_replay_count - 1 + offs_h[None, :]) * token_stride,
            mask=mask & (boundary_state_idx > null_block_id),
        )
        boundary_ptr = conv_state_ptr + boundary_state_idx * block_stride
        tl.store(
            boundary_ptr
            + offs_d[:, None] * dim_stride
            + offs_h[None, :] * token_stride,
            boundary_values,
            mask=mask & (boundary_state_idx > null_block_id),
        )
    tl.store(
        final_ptr + offs_d[:, None] * dim_stride + offs_h[None, :] * token_stride,
        final_values,
        mask=mask,
    )


@triton.jit
def _commit_kda_state_kernel(
    state_ref_ptr,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    correction_cache_ref_ptr,
    correction_cache_base_addrs_ptr,
    correction_cache_block_strides_ptr,
    kg_cache_ref_ptr,
    kg_cache_base_addrs_ptr,
    kg_cache_block_strides_ptr,
    A_log_ptr,
    dt_bias_ptr,
    state_indices_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_replay_counts_ptr,
    lower_bound,
    null_block_id,
    stride_state_head,
    stride_state_v,
    stride_state_k,
    stride_correction_cache_head,
    stride_correction_cache_pos,
    stride_correction_cache_dim,
    stride_kg_cache_head,
    stride_kg_cache_pos,
    stride_kg_cache_dim,
    stride_A_layer,
    stride_A_head,
    stride_dt_bias_layer,
    stride_dt_bias_head,
    stride_dt_bias_dim,
    stride_state_indices,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_lh = tl.program_id(2)
    pid_l = pid_lh // NUM_HEADS
    pid_h = pid_lh % NUM_HEADS

    source_state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(
        tl.int64
    )
    if source_state_idx <= null_block_id:
        return
    commit_len = tl.load(commit_lens_ptr + pid_b)
    if commit_len == 0:
        return
    final_state_idx = tl.load(final_state_indices_ptr + pid_b).to(tl.int64)
    boundary_state_idx = tl.load(boundary_state_indices_ptr + pid_b).to(tl.int64)
    boundary_replay_count = tl.load(boundary_replay_counts_ptr + pid_b)

    if final_state_idx <= null_block_id:
        return

    state_base_addr = tl.load(state_base_addrs_ptr + pid_l)
    state_block_stride = tl.load(state_block_strides_ptr + pid_l)
    state_ptr = state_base_addr.to(tl.pointer_type(state_ref_ptr.dtype.element_ty))
    source_state_ptr = (
        state_ptr + source_state_idx * state_block_stride + pid_h * stride_state_head
    )

    correction_cache_base_addr = tl.load(correction_cache_base_addrs_ptr + pid_l)
    correction_cache_block_stride = tl.load(correction_cache_block_strides_ptr + pid_l)
    correction_cache_ptr = correction_cache_base_addr.to(
        tl.pointer_type(correction_cache_ref_ptr.dtype.element_ty)
    )
    correction_cache_ptr += (
        source_state_idx * correction_cache_block_stride
        + pid_h * stride_correction_cache_head
    )
    kg_cache_base_addr = tl.load(kg_cache_base_addrs_ptr + pid_l)
    kg_cache_block_stride = tl.load(kg_cache_block_strides_ptr + pid_l)
    kg_cache_ptr = kg_cache_base_addr.to(
        tl.pointer_type(kg_cache_ref_ptr.dtype.element_ty)
    )
    kg_cache_ptr += (
        source_state_idx * kg_cache_block_stride + pid_h * stride_kg_cache_head
    )

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]
    state_ptrs = (
        source_state_ptr
        + offs_v[:, None] * stride_state_v
        + offs_k[None, :] * stride_state_k
    )
    initial_state = tl.load(state_ptrs, mask=mask_state, other=0.0).to(tl.float32)
    A = tl.exp(
        tl.load(A_log_ptr + pid_l * stride_A_layer + pid_h * stride_A_head).to(
            tl.float32
        )
    )

    dt_bias = tl.load(
        dt_bias_ptr
        + pid_l * stride_dt_bias_layer
        + pid_h * stride_dt_bias_head
        + offs_k * stride_dt_bias_dim,
        mask=mask_k,
        other=0.0,
    ).to(tl.float32)
    final_decay = tl.full([BK], 1.0, tl.float32)
    final_correction = tl.zeros([BV, BK], tl.float32)
    boundary_decay = tl.full([BK], 1.0, tl.float32)
    boundary_correction = tl.zeros([BV, BK], tl.float32)

    for reverse_offset in range(commit_len):
        token_offset = commit_len - reverse_offset - 1
        correction_ptr = (
            correction_cache_ptr + token_offset * stride_correction_cache_pos
        )
        kg_ptr = kg_cache_ptr + token_offset * stride_kg_cache_pos
        k = tl.load(
            kg_ptr + offs_k * stride_kg_cache_dim,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        correction = tl.load(
            correction_ptr + offs_v * stride_correction_cache_dim,
            mask=mask_v,
            other=0.0,
        ).to(tl.float32)
        raw_g = tl.load(
            kg_ptr + (K + offs_k) * stride_kg_cache_dim,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        normalized_k = k * tl.rsqrt(tl.sum(k * k) + 1e-6)
        gate = _kda_gate(
            raw_g,
            dt_bias,
            A,
            lower_bound,
            USE_LOWER_BOUND,
        )
        update = correction[:, None] * normalized_k[None, :]
        decay = tl.exp(gate)
        final_correction += update * final_decay[None, :]
        final_decay *= decay
        if ALIGN_MODE:
            before_boundary = token_offset < boundary_replay_count
            boundary_correction += tl.where(
                before_boundary,
                update * boundary_decay[None, :],
                0.0,
            )
            boundary_decay *= tl.where(before_boundary, decay, 1.0)

    state = initial_state * final_decay[None, :] + final_correction
    if ALIGN_MODE:
        boundary_ptrs = (
            state_ptr
            + boundary_state_idx * state_block_stride
            + pid_h * stride_state_head
            + offs_v[:, None] * stride_state_v
            + offs_k[None, :] * stride_state_k
        )
        boundary_state = initial_state * boundary_decay[None, :] + boundary_correction
        tl.store(
            boundary_ptrs,
            boundary_state,
            mask=mask_state & (boundary_state_idx > null_block_id),
        )

    final_ptrs = (
        state_ptr
        + final_state_idx * state_block_stride
        + pid_h * stride_state_head
        + offs_v[:, None] * stride_state_v
        + offs_k[None, :] * stride_state_k
    )
    tl.store(final_ptrs, state, mask=mask_state)


def kda_replayssm_spec_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    checkpoint_state: torch.Tensor,
    correction_cache: torch.Tensor,
    kg_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    spec_query_len: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Verify a KDA speculative window without modifying its checkpoint."""
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("KDA ReplaySSM q must have shape [1, tokens, heads, dim]")
    _, total_tokens, num_heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if k.shape != q.shape or v.shape != (1, total_tokens, num_heads, value_dim):
        raise ValueError("KDA ReplaySSM q, k, and v shapes are incompatible")
    if raw_g.shape != q.shape or raw_beta.shape != (1, total_tokens, num_heads):
        raise ValueError("KDA ReplaySSM gate or beta shape is incompatible")
    if any(tensor.stride()[2:] != (key_dim, 1) for tensor in (q, k, raw_g)):
        raise ValueError("KDA ReplaySSM q, k, and gate heads must be contiguous")
    if v.stride()[2:] != (value_dim, 1) or raw_beta.stride(2) != 1:
        raise ValueError("KDA ReplaySSM v and beta heads must be contiguous")
    num_blocks = checkpoint_state.shape[0]
    if checkpoint_state.shape[1:] != (
        num_heads,
        value_dim,
        key_dim,
    ):
        raise ValueError("KDA ReplaySSM checkpoint shape is incompatible")
    expected_correction_shape = (
        num_blocks,
        num_heads,
        spec_query_len,
        value_dim,
    )
    if correction_cache.shape != expected_correction_shape:
        raise ValueError(
            f"KDA ReplaySSM correction buffer needs shape {expected_correction_shape}"
        )
    expected_kg_shape = (num_blocks, num_heads, spec_query_len, 2 * key_dim)
    if kg_cache.shape != expected_kg_shape:
        raise ValueError(
            f"KDA ReplaySSM key/gate buffer needs shape {expected_kg_shape}"
        )
    if correction_cache.dtype != torch.float32:
        raise ValueError("KDA ReplaySSM correction buffer must use float32")
    if kg_cache.dtype != k.dtype:
        raise ValueError("KDA ReplaySSM key/gate buffer must match activation dtype")
    if A_log.shape != (num_heads,) or dt_bias.numel() != num_heads * key_dim:
        raise ValueError("KDA ReplaySSM gate parameters are incompatible")
    if not A_log.is_contiguous() or not dt_bias.is_contiguous():
        raise ValueError("KDA ReplaySSM gate parameters must be contiguous")
    batch = state_indices.shape[0]
    if query_start_loc.shape[0] != batch + 1:
        raise ValueError("KDA ReplaySSM query metadata is incompatible")
    if total_tokens > batch * spec_query_len:
        raise ValueError(
            "KDA ReplaySSM speculative decode input exceeds its activation capacity"
        )
    if out is None:
        out = torch.empty_like(v)
    if out.shape != v.shape:
        raise ValueError("KDA ReplaySSM output shape is incompatible")
    if out.stride()[2:] != (value_dim, 1):
        raise ValueError("KDA ReplaySSM output heads must be contiguous")
    device = q.device
    if any(
        tensor.device != device
        for tensor in (
            k,
            v,
            raw_g,
            raw_beta,
            A_log,
            dt_bias,
            checkpoint_state,
            correction_cache,
            kg_cache,
            query_start_loc,
            state_indices,
            out,
        )
    ):
        raise ValueError("KDA ReplaySSM inputs must be on the same device")
    if total_tokens == 0:
        return out

    block_k = triton.next_power_of_2(key_dim)
    block_v = min(triton.next_power_of_2(value_dim), 32)
    grid = (triton.cdiv(value_dim, block_v), batch, num_heads)
    with torch.accelerator.device_index(device.index):
        _kda_replayssm_verify_kernel[grid](
            q,
            k,
            v,
            raw_g,
            raw_beta,
            A_log,
            dt_bias,
            checkpoint_state,
            correction_cache,
            kg_cache,
            out,
            query_start_loc,
            state_indices,
            lower_bound or 0.0,
            NULL_BLOCK_ID,
            q.stride(1),
            k.stride(1),
            v.stride(1),
            raw_g.stride(1),
            raw_beta.stride(1),
            checkpoint_state.stride(0),
            checkpoint_state.stride(1),
            checkpoint_state.stride(2),
            checkpoint_state.stride(3),
            correction_cache.stride(0),
            correction_cache.stride(1),
            correction_cache.stride(2),
            correction_cache.stride(3),
            kg_cache.stride(0),
            kg_cache.stride(1),
            kg_cache.stride(2),
            kg_cache.stride(3),
            out.stride(1),
            query_start_loc.stride(0),
            state_indices.stride(0),
            K=key_dim,
            V=value_dim,
            BK=block_k,
            BV=block_v,
            SPEC_QUERY_LEN=spec_query_len,
            USE_LOWER_BOUND=lower_bound is not None,
            num_warps=4,
            num_stages=2,
        )
    return out


@dataclass
class KDAReplaySSMSpecCommitContext:
    conv_states: tuple[torch.Tensor, ...]
    conv_state_base_addrs: torch.Tensor
    conv_state_block_strides: torch.Tensor
    conv_state_dim_strides: torch.Tensor
    conv_state_token_strides: torch.Tensor
    conv_history_len: int
    checkpoints: tuple[torch.Tensor, ...]
    state_base_addrs: torch.Tensor
    state_block_strides: torch.Tensor
    correction_caches: tuple[torch.Tensor, ...]
    correction_cache_base_addrs: torch.Tensor
    correction_cache_block_strides: torch.Tensor
    kg_caches: tuple[torch.Tensor, ...]
    kg_cache_base_addrs: torch.Tensor
    kg_cache_block_strides: torch.Tensor
    commit_lens: torch.Tensor
    final_state_indices: torch.Tensor
    boundary_state_indices: torch.Tensor
    boundary_replay_counts: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    lower_bound: float | None
    spec_query_len: int

    @classmethod
    def create(
        cls,
        layers: Sequence[Any],
        *,
        spec_query_len: int,
        max_num_reqs: int,
    ) -> "KDAReplaySSMSpecCommitContext":
        if not layers:
            raise ValueError("KDA ReplaySSM commit requires at least one layer")
        if any(len(layer.kv_cache) != 4 for layer in layers):
            raise ValueError(
                "KDA ReplaySSM pages must contain conv, state, correction, and key/gate"
            )

        conv_states = [layer.kv_cache[0] for layer in layers]
        if not is_conv_state_dim_first():
            conv_states = [state.transpose(-1, -2) for state in conv_states]
        checkpoints = [layer.kv_cache[1] for layer in layers]
        correction_caches = [layer.kv_cache[2] for layer in layers]
        kg_caches = [layer.kv_cache[3] for layer in layers]
        A_log = [layer.A_log for layer in layers]
        dt_bias = [
            layer.dt_bias.view(layer.local_num_heads, layer.head_dim)
            for layer in layers
        ]
        lower_bounds = {layer.gate_lower_bound for layer in layers}
        if len(lower_bounds) != 1:
            raise ValueError("KDA ReplaySSM layers need matching gate bounds")

        state_ref = checkpoints[0]
        if state_ref.ndim != 4:
            raise ValueError("KDA ReplaySSM checkpoint must be four-dimensional")
        num_blocks, num_heads, value_dim, key_dim = state_ref.shape
        for state in checkpoints:
            if (
                state.shape != state_ref.shape
                or state.dtype != state_ref.dtype
                or state.device != state_ref.device
                or state.stride()[1:] != state_ref.stride()[1:]
            ):
                raise ValueError("KDA ReplaySSM layers need matching checkpoint layout")
        expected_correction_shape = (
            num_blocks,
            num_heads,
            spec_query_len,
            value_dim,
        )
        correction_ref = correction_caches[0]
        for correction_cache in correction_caches:
            if (
                correction_cache.shape != expected_correction_shape
                or correction_cache.dtype != torch.float32
                or correction_cache.device != state_ref.device
                or correction_cache.stride()[1:] != correction_ref.stride()[1:]
            ):
                raise ValueError(
                    "KDA ReplaySSM correction buffers need float32 shape "
                    f"{expected_correction_shape}"
                )
        expected_kg_shape = (num_blocks, num_heads, spec_query_len, 2 * key_dim)
        kg_ref = kg_caches[0]
        for kg_cache in kg_caches:
            if (
                kg_cache.shape != expected_kg_shape
                or kg_cache.dtype != kg_ref.dtype
                or kg_cache.device != state_ref.device
                or kg_cache.stride()[1:] != kg_ref.stride()[1:]
            ):
                raise ValueError(
                    f"KDA ReplaySSM key/gate buffers need shape {expected_kg_shape}"
                )
        if any(param.shape != (num_heads,) for param in A_log):
            raise ValueError("KDA ReplaySSM A_log shape is incompatible")
        if any(param.shape != (num_heads, key_dim) for param in dt_bias):
            raise ValueError("KDA ReplaySSM dt_bias shape is incompatible")

        conv_ref = conv_states[0]
        if conv_ref.ndim != 3:
            raise ValueError("KDA ReplaySSM conv state must be three-dimensional")
        conv_dim, conv_state_len = conv_ref.shape[1:]
        conv_history_len = conv_state_len - spec_query_len + 1
        if conv_history_len <= 0:
            raise ValueError("KDA ReplaySSM conv state is shorter than its window")
        for conv_state in conv_states:
            if (
                conv_state.shape != conv_ref.shape
                or conv_state.dtype != conv_ref.dtype
                or conv_state.device != state_ref.device
                or conv_state.shape[0] != num_blocks
            ):
                raise ValueError("KDA ReplaySSM layers need matching conv state")

        device = state_ref.device

        def _base_addrs(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
            return torch.tensor(
                [tensor.data_ptr() for tensor in tensors],
                dtype=torch.int64,
                device=device,
            )

        def _block_strides(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
            return torch.tensor(
                [tensor.stride(0) for tensor in tensors],
                dtype=torch.int64,
                device=device,
            )

        return cls(
            conv_states=tuple(conv_states),
            conv_state_base_addrs=_base_addrs(conv_states),
            conv_state_block_strides=_block_strides(conv_states),
            conv_state_dim_strides=torch.tensor(
                [state.stride(1) for state in conv_states],
                dtype=torch.int64,
                device=device,
            ),
            conv_state_token_strides=torch.tensor(
                [state.stride(2) for state in conv_states],
                dtype=torch.int64,
                device=device,
            ),
            conv_history_len=conv_history_len,
            checkpoints=tuple(checkpoints),
            state_base_addrs=_base_addrs(checkpoints),
            state_block_strides=_block_strides(checkpoints),
            correction_caches=tuple(correction_caches),
            correction_cache_base_addrs=_base_addrs(correction_caches),
            correction_cache_block_strides=_block_strides(correction_caches),
            kg_caches=tuple(kg_caches),
            kg_cache_base_addrs=_base_addrs(kg_caches),
            kg_cache_block_strides=_block_strides(kg_caches),
            commit_lens=torch.empty(max_num_reqs, dtype=torch.int32, device=device),
            final_state_indices=torch.empty(
                max_num_reqs, dtype=torch.int32, device=device
            ),
            boundary_state_indices=torch.empty(
                max_num_reqs, dtype=torch.int32, device=device
            ),
            boundary_replay_counts=torch.empty(
                max_num_reqs, dtype=torch.int32, device=device
            ),
            A_log=torch.stack(tuple(A_log)).contiguous(),
            dt_bias=torch.stack(tuple(dt_bias)).contiguous(),
            lower_bound=lower_bounds.pop(),
            spec_query_len=spec_query_len,
        )

    def commit(
        self,
        num_accepted_tokens: torch.Tensor,
        state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        request_indices: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        num_computed_tokens: torch.Tensor | None = None,
        mamba_block_size: int | None = None,
    ) -> None:
        """Fold accepted KDA and convolution inputs into every layer."""
        batch = state_indices.shape[0]
        if batch == 0:
            return
        if batch > self.commit_lens.shape[0]:
            raise ValueError("KDA ReplaySSM commit batch exceeds its plan capacity")
        if query_start_loc.shape[0] != batch + 1:
            raise ValueError("KDA ReplaySSM commit metadata is incompatible")
        if request_indices is not None and request_indices.shape[0] < batch:
            raise ValueError("KDA ReplaySSM request mapping is too short")
        align_args = (block_table, num_computed_tokens, mamba_block_size)
        if any(arg is not None for arg in align_args) and any(
            arg is None for arg in align_args
        ):
            raise ValueError("KDA ReplaySSM align metadata is incomplete")
        if mamba_block_size is not None and mamba_block_size < self.spec_query_len:
            raise ValueError(
                "KDA ReplaySSM align block size must cover one speculative window"
            )
        if block_table is not None and block_table.ndim != 2:
            raise ValueError("KDA ReplaySSM block table must be two-dimensional")
        device = self.checkpoints[0].device
        if (
            any(
                tensor.device != device
                for tensor in (
                    num_accepted_tokens,
                    state_indices,
                    query_start_loc,
                )
            )
            or (request_indices is not None and request_indices.device != device)
            or (block_table is not None and block_table.device != device)
            or (
                num_computed_tokens is not None and num_computed_tokens.device != device
            )
        ):
            raise ValueError("KDA ReplaySSM commit inputs must be on the same device")

        block_table_stride = (0, 0) if block_table is None else block_table.stride()
        num_computed_stride = (
            0 if num_computed_tokens is None else num_computed_tokens.stride(0)
        )

        num_layers = len(self.checkpoints)
        conv_ref = self.conv_states[0]
        conv_dim = conv_ref.shape[1]
        block_history = triton.next_power_of_2(self.conv_history_len)
        with torch.accelerator.device_index(device.index):
            _prepare_commit_plan_kernel[(batch,)](
                num_accepted_tokens,
                request_indices,
                state_indices,
                query_start_loc,
                block_table,
                num_computed_tokens,
                self.commit_lens,
                self.final_state_indices,
                self.boundary_state_indices,
                self.boundary_replay_counts,
                NULL_BLOCK_ID,
                mamba_block_size or 1,
                block_table.shape[1] if block_table is not None else 1,
                num_accepted_tokens.stride(0),
                request_indices.stride(0) if request_indices is not None else 0,
                state_indices.stride(0),
                query_start_loc.stride(0),
                block_table_stride[0],
                block_table_stride[1],
                num_computed_stride,
                SPEC_QUERY_LEN=self.spec_query_len,
                num_warps=1,
            )
            _compact_conv_state_kernel[(triton.cdiv(conv_dim, 256), batch, num_layers)](
                conv_ref,
                self.conv_state_base_addrs,
                self.conv_state_block_strides,
                self.conv_state_dim_strides,
                self.conv_state_token_strides,
                state_indices,
                self.commit_lens,
                self.final_state_indices,
                self.boundary_state_indices,
                self.boundary_replay_counts,
                NULL_BLOCK_ID,
                conv_dim,
                self.conv_history_len,
                state_indices.stride(0),
                BLOCK_D=256,
                BLOCK_HISTORY=block_history,
                ALIGN_MODE=block_table is not None,
                num_warps=4,
            )

        state_ref = self.checkpoints[0]
        _, num_heads, value_dim, key_dim = state_ref.shape
        block_k = triton.next_power_of_2(key_dim)
        block_v = min(triton.next_power_of_2(value_dim), 32)
        grid = (
            triton.cdiv(value_dim, block_v),
            batch,
            num_layers * num_heads,
        )
        with torch.accelerator.device_index(device.index):
            _commit_kda_state_kernel[grid](
                state_ref,
                self.state_base_addrs,
                self.state_block_strides,
                self.correction_caches[0],
                self.correction_cache_base_addrs,
                self.correction_cache_block_strides,
                self.kg_caches[0],
                self.kg_cache_base_addrs,
                self.kg_cache_block_strides,
                self.A_log,
                self.dt_bias,
                state_indices,
                self.commit_lens,
                self.final_state_indices,
                self.boundary_state_indices,
                self.boundary_replay_counts,
                self.lower_bound or 0.0,
                NULL_BLOCK_ID,
                state_ref.stride(1),
                state_ref.stride(2),
                state_ref.stride(3),
                self.correction_caches[0].stride(1),
                self.correction_caches[0].stride(2),
                self.correction_caches[0].stride(3),
                self.kg_caches[0].stride(1),
                self.kg_caches[0].stride(2),
                self.kg_caches[0].stride(3),
                self.A_log.stride(0),
                self.A_log.stride(1),
                self.dt_bias.stride(0),
                self.dt_bias.stride(1),
                self.dt_bias.stride(2),
                state_indices.stride(0),
                K=key_dim,
                V=value_dim,
                BK=block_k,
                BV=block_v,
                NUM_HEADS=num_heads,
                USE_LOWER_BOUND=self.lower_bound is not None,
                ALIGN_MODE=block_table is not None,
                num_warps=4,
                num_stages=2,
            )


__all__ = ["KDAReplaySSMSpecCommitContext", "kda_replayssm_spec_decode"]
