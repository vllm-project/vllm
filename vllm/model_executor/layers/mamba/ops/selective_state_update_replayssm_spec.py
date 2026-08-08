# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vllm.model_executor.layers.mamba.ops.mamba_ssm import softplus
from vllm.model_executor.layers.mamba.ops.replayssm_config import get_replayssm_config
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


@triton.heuristics(
    {"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])}
)
@triton.jit
def _scatter_and_precompute_kernel(
    x_ptr,
    dt_ptr,
    B_ptr,
    C_ptr,
    x_cache_ptr,
    dt_cache_ptr,
    B_cache_ptr,
    bc_pre_ptr,
    state_batch_indices_ptr,
    query_start_loc_ptr,
    null_block_id,
    ngroups,
    nheads,
    dim,
    dstate,
    stride_x_tok,
    stride_x_head,
    stride_x_dim,
    stride_dt_tok,
    stride_dt_head,
    stride_B_tok,
    stride_B_group,
    stride_B_dstate,
    stride_C_tok,
    stride_C_group,
    stride_C_dstate,
    stride_x_cache_batch,
    stride_x_cache_head,
    stride_x_cache_pos,
    stride_x_cache_dim,
    stride_dt_cache_batch,
    stride_dt_cache_head,
    stride_dt_cache_pos,
    stride_B_cache_batch,
    stride_B_cache_group,
    stride_B_cache_pos,
    stride_B_cache_dstate,
    stride_bc_pre_batch,
    stride_bc_pre_group,
    stride_bc_pre_k,
    stride_bc_pre_q,
    stride_state_indices_batch,
    RATIO: tl.constexpr,
    RATIO_P: tl.constexpr,
    NCX: tl.constexpr,
    BLOCK_CX: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    BLOCK_SIZE_SPEC: tl.constexpr,
    BLOCK_HL: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)
    state_batch_idx = tl.load(
        state_batch_indices_ptr + pid_b * stride_state_indices_batch
    ).to(tl.int64)
    if state_batch_idx == null_block_id:
        return

    bos = tl.load(query_start_loc_ptr + pid_b).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + pid_b + 1).to(tl.int64)
    query_len = (eos - bos).to(tl.int32)
    offs_q = tl.arange(0, BLOCK_SIZE_SPEC)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    q_mask = (offs_q < query_len) & (offs_q < SPEC_QUERY_LEN)
    n_mask = offs_n < dstate

    B_base = B_ptr + bos * stride_B_tok + pid_g * stride_B_group
    C_base = C_ptr + bos * stride_C_tok + pid_g * stride_C_group
    B_block = tl.load(
        B_base + offs_q[:, None] * stride_B_tok + offs_n[None, :] * stride_B_dstate,
        mask=q_mask[:, None] & n_mask[None, :],
        other=0.0,
    )
    C_block = tl.load(
        C_base + offs_q[:, None] * stride_C_tok + offs_n[None, :] * stride_C_dstate,
        mask=q_mask[:, None] & n_mask[None, :],
        other=0.0,
    )
    B_cache_base = (
        B_cache_ptr
        + state_batch_idx * stride_B_cache_batch
        + pid_g * stride_B_cache_group
    )
    tl.store(
        B_cache_base
        + offs_q[:, None] * stride_B_cache_pos
        + offs_n[None, :] * stride_B_cache_dstate,
        B_block,
        mask=q_mask[:, None] & n_mask[None, :],
    )

    x_base = x_ptr + bos * stride_x_tok
    x_cache_base = x_cache_ptr + state_batch_idx * stride_x_cache_batch
    for i in tl.static_range(NCX):
        offs_cx = i * BLOCK_CX + tl.arange(0, BLOCK_CX)
        cx_mask = offs_cx < RATIO_P
        global_head = pid_g * RATIO + offs_cx // dim
        offs_m = offs_cx % dim
        x_block = tl.load(
            x_base
            + offs_q[:, None] * stride_x_tok
            + global_head[None, :] * stride_x_head
            + offs_m[None, :] * stride_x_dim,
            mask=q_mask[:, None] & cx_mask[None, :],
            other=0.0,
        )
        tl.store(
            x_cache_base
            + global_head[None, :] * stride_x_cache_head
            + offs_q[:, None] * stride_x_cache_pos
            + offs_m[None, :] * stride_x_cache_dim,
            x_block,
            mask=q_mask[:, None] & cx_mask[None, :],
        )

    offs_h = tl.arange(0, BLOCK_HL)
    head_mask = offs_h < RATIO
    global_head = pid_g * RATIO + offs_h
    dt_block = tl.load(
        dt_ptr
        + bos * stride_dt_tok
        + offs_q[:, None] * stride_dt_tok
        + global_head[None, :] * stride_dt_head,
        mask=q_mask[:, None] & head_mask[None, :],
        other=0.0,
    )
    tl.store(
        dt_cache_ptr
        + state_batch_idx * stride_dt_cache_batch
        + global_head[None, :] * stride_dt_cache_head
        + offs_q[:, None] * stride_dt_cache_pos,
        dt_block,
        mask=q_mask[:, None] & head_mask[None, :],
    )

    bc = tl.dot(
        B_block.to(x_ptr.dtype.element_ty),
        tl.trans(C_block.to(x_ptr.dtype.element_ty)),
        input_precision="tf32x3",
    ).to(tl.float32)
    tl.store(
        bc_pre_ptr
        + pid_b * stride_bc_pre_batch
        + pid_g * stride_bc_pre_group
        + offs_q[:, None] * stride_bc_pre_k
        + offs_q[None, :] * stride_bc_pre_q,
        bc,
        mask=q_mask[:, None] & q_mask[None, :],
    )


@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.jit
def _verify_kernel(
    state_ptr,
    x_cache_ptr,
    dt_cache_ptr,
    C_ptr,
    bc_pre_ptr,
    D_ptr,
    z_ptr,
    dt_bias_ptr,
    A_ptr,
    out_ptr,
    state_batch_indices_ptr,
    query_start_loc_ptr,
    null_block_id,
    nheads_ngroups_ratio,
    dim,
    dstate,
    stride_state_batch,
    stride_state_head,
    stride_state_dim,
    stride_state_dstate,
    stride_x_cache_batch,
    stride_x_cache_head,
    stride_x_cache_pos,
    stride_x_cache_dim,
    stride_dt_cache_batch,
    stride_dt_cache_head,
    stride_dt_cache_pos,
    stride_C_tok,
    stride_C_dstate,
    stride_bc_pre_batch,
    stride_bc_pre_group,
    stride_bc_pre_k,
    stride_bc_pre_q,
    stride_D_head,
    stride_D_dim,
    stride_z_tok,
    stride_z_head,
    stride_z_dim,
    stride_dt_bias_head,
    stride_A_head,
    stride_out_tok,
    stride_out_head,
    stride_out_dim,
    stride_state_indices_batch,
    DT_SOFTPLUS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    BLOCK_SIZE_SPEC: tl.constexpr,
    DSTATE_TILE: tl.constexpr,
    NDS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    state_batch_idx = tl.load(
        state_batch_indices_ptr + pid_b * stride_state_indices_batch
    ).to(tl.int64)
    if state_batch_idx == null_block_id:
        return

    bos = tl.load(query_start_loc_ptr + pid_b).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + pid_b + 1).to(tl.int64)
    query_len = (eos - bos).to(tl.int32)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_SPEC)
    offs_q = tl.arange(0, BLOCK_SIZE_SPEC)
    offs_nt = tl.arange(0, DSTATE_TILE)
    m_mask = offs_m < dim
    k_mask = (offs_k < query_len) & (offs_k < SPEC_QUERY_LEN)
    q_mask = (offs_q < query_len) & (offs_q < SPEC_QUERY_LEN)

    state_ptr += state_batch_idx * stride_state_batch + pid_h * stride_state_head
    x_cache_ptr += state_batch_idx * stride_x_cache_batch + pid_h * stride_x_cache_head
    dt_cache_ptr += (
        state_batch_idx * stride_dt_cache_batch + pid_h * stride_dt_cache_head
    )
    group = pid_h // nheads_ngroups_ratio
    C_ptr += bos * stride_C_tok + group * dstate * stride_C_dstate
    bc_pre_ptr += pid_b * stride_bc_pre_batch + group * stride_bc_pre_group
    if HAS_D:
        D_ptr += pid_h * stride_D_head
    if HAS_Z:
        z_ptr += bos * stride_z_tok + pid_h * stride_z_head
    if HAS_DT_BIAS:
        dt_bias_ptr += pid_h * stride_dt_bias_head
    A_ptr += pid_h * stride_A_head
    out_ptr += bos * stride_out_tok + pid_h * stride_out_head

    A_val = tl.load(A_ptr).to(tl.float32)
    dt_bias_val = tl.load(dt_bias_ptr).to(tl.float32) if HAS_DT_BIAS else 0.0
    dt = tl.load(
        dt_cache_ptr + offs_k * stride_dt_cache_pos,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)
    if HAS_DT_BIAS:
        dt = tl.where(k_mask, dt + dt_bias_val, 0.0)
    if DT_SOFTPLUS:
        dt = tl.where(k_mask, tl.where(dt <= 20.0, softplus(dt), dt), 0.0)
    dt_cum = tl.cumsum(dt, axis=0)
    checkpoint_decay = tl.where(q_mask, tl.exp(tl.minimum(A_val * dt_cum, 0.0)), 0.0)

    bc = tl.load(
        bc_pre_ptr
        + offs_k[:, None] * stride_bc_pre_k
        + offs_q[None, :] * stride_bc_pre_q,
        mask=k_mask[:, None] & q_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    causal = k_mask[:, None] & q_mask[None, :] & (offs_k[:, None] <= offs_q[None, :])
    decay = tl.exp(tl.minimum(A_val * (dt_cum[None, :] - dt_cum[:, None]), 0.0))
    factor = tl.where(causal, bc * dt[:, None] * decay, 0.0)
    x = tl.load(
        x_cache_ptr
        + offs_m[:, None] * stride_x_cache_dim
        + offs_k[None, :] * stride_x_cache_pos,
        mask=m_mask[:, None] & k_mask[None, :],
        other=0.0,
    ).to(x_cache_ptr.dtype.element_ty)
    recurrence_out = tl.dot(
        x, factor.to(x_cache_ptr.dtype.element_ty), input_precision="tf32x3"
    ).to(tl.float32)

    checkpoint_out = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_SPEC], dtype=tl.float32)
    for i in tl.static_range(NDS):
        offs_n = i * DSTATE_TILE + offs_nt
        n_mask = offs_n < dstate
        state = tl.load(
            state_ptr
            + offs_m[:, None] * stride_state_dim
            + offs_n[None, :] * stride_state_dstate,
            mask=m_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        C = tl.load(
            C_ptr + offs_q[:, None] * stride_C_tok + offs_n[None, :] * stride_C_dstate,
            mask=q_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        if x_cache_ptr.dtype.element_ty == tl.float32:
            checkpoint_out += tl.dot(state, tl.trans(C), input_precision="tf32x3").to(
                tl.float32
            )
        else:
            checkpoint_out += tl.dot(state, tl.trans(C), input_precision="tf32").to(
                tl.float32
            )
    out = tl.trans(checkpoint_out * checkpoint_decay[None, :] + recurrence_out)

    if HAS_D:
        D = tl.load(D_ptr + offs_m * stride_D_dim, mask=m_mask, other=0.0).to(
            tl.float32
        )
        x_query = tl.load(
            x_cache_ptr
            + offs_q[:, None] * stride_x_cache_pos
            + offs_m[None, :] * stride_x_cache_dim,
            mask=q_mask[:, None] & m_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        out += x_query * D[None, :]
    if HAS_Z:
        z = tl.load(
            z_ptr + offs_q[:, None] * stride_z_tok + offs_m[None, :] * stride_z_dim,
            mask=q_mask[:, None] & m_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        out *= z * tl.sigmoid(z)
    tl.store(
        out_ptr + offs_q[:, None] * stride_out_tok + offs_m[None, :] * stride_out_dim,
        out,
        mask=q_mask[:, None] & m_mask[None, :],
    )


@triton.heuristics(
    {"HAS_FORCE_COMMIT": lambda args: args["force_commit_ptr"] is not None}
)
@triton.jit
def _compact_conv_state_kernel(
    conv_state_ref_ptr,
    conv_state_base_addrs_ptr,
    conv_state_block_strides_ptr,
    conv_state_dim_strides_ptr,
    conv_state_token_strides_ptr,
    num_accepted_ptr,
    force_commit_ptr,
    state_batch_indices_ptr,
    query_start_loc_ptr,
    null_block_id,
    conv_dim,
    conv_history_len,
    stride_num_accepted_batch,
    stride_force_commit_batch,
    stride_state_indices_batch,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_HISTORY: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    HAS_FORCE_COMMIT: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_l = tl.program_id(2)
    state_batch_idx = tl.load(
        state_batch_indices_ptr + pid_b * stride_state_indices_batch
    ).to(tl.int64)
    if state_batch_idx == null_block_id:
        return

    bos = tl.load(query_start_loc_ptr + pid_b).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + pid_b + 1).to(tl.int64)
    query_len = (eos - bos).to(tl.int32)
    num_accepted = tl.load(num_accepted_ptr + pid_b * stride_num_accepted_batch).to(
        tl.int32
    )
    if HAS_FORCE_COMMIT:
        force_commit = tl.load(force_commit_ptr + pid_b * stride_force_commit_batch).to(
            tl.int1
        )
        commit_len = tl.where(force_commit, query_len, num_accepted)
    else:
        commit_len = num_accepted
    commit_len = tl.minimum(tl.maximum(commit_len, 0), query_len)
    commit_len = tl.minimum(commit_len, SPEC_QUERY_LEN)
    if commit_len == 0:
        return

    conv_state_base_addr = tl.load(conv_state_base_addrs_ptr + pid_l)
    stride_conv_state_block = tl.load(conv_state_block_strides_ptr + pid_l)
    stride_conv_state_dim = tl.load(conv_state_dim_strides_ptr + pid_l)
    stride_conv_state_token = tl.load(conv_state_token_strides_ptr + pid_l)
    conv_state_ptr = conv_state_base_addr.to(
        tl.pointer_type(conv_state_ref_ptr.dtype.element_ty)
    )
    conv_state_ptr += state_batch_idx * stride_conv_state_block

    offs_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    offs_s = tl.arange(0, BLOCK_SIZE_HISTORY)
    mask = (offs_d[:, None] < conv_dim) & (offs_s[None, :] < conv_history_len)
    values = tl.load(
        conv_state_ptr
        + offs_d[:, None] * stride_conv_state_dim
        + (commit_len - 1 + offs_s[None, :]) * stride_conv_state_token,
        mask=mask,
    )
    tl.store(
        conv_state_ptr
        + offs_d[:, None] * stride_conv_state_dim
        + offs_s[None, :] * stride_conv_state_token,
        values,
        mask=mask,
    )


@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics(
    {"HAS_FORCE_COMMIT": lambda args: args["force_commit_ptr"] is not None}
)
@triton.jit
def _commit_kernel(
    state_ref_ptr,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    x_cache_ref_ptr,
    x_cache_base_addrs_ptr,
    x_cache_block_strides_ptr,
    dt_cache_ref_ptr,
    dt_cache_base_addrs_ptr,
    dt_cache_block_strides_ptr,
    B_cache_ref_ptr,
    B_cache_base_addrs_ptr,
    B_cache_block_strides_ptr,
    A_ptr,
    dt_bias_ptr,
    num_accepted_ptr,
    force_commit_ptr,
    state_batch_indices_ptr,
    query_start_loc_ptr,
    null_block_id,
    nheads_ngroups_ratio,
    dim,
    dstate,
    stride_state_head,
    stride_state_dim,
    stride_state_dstate,
    stride_x_cache_head,
    stride_x_cache_pos,
    stride_x_cache_dim,
    stride_dt_cache_head,
    stride_dt_cache_pos,
    stride_B_cache_group,
    stride_B_cache_pos,
    stride_B_cache_dstate,
    stride_A_layer,
    stride_A_head,
    stride_dt_bias_layer,
    stride_dt_bias_head,
    stride_num_accepted_batch,
    stride_force_commit_batch,
    stride_state_indices_batch,
    DT_SOFTPLUS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    BLOCK_SIZE_SPEC: tl.constexpr,
    DSTATE_TILE: tl.constexpr,
    NDS: tl.constexpr,
    NHEADS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_FORCE_COMMIT: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_lh = tl.program_id(2)
    pid_l = pid_lh // NHEADS
    pid_h = pid_lh % NHEADS
    state_batch_idx = tl.load(
        state_batch_indices_ptr + pid_b * stride_state_indices_batch
    ).to(tl.int64)
    if state_batch_idx == null_block_id:
        return

    bos = tl.load(query_start_loc_ptr + pid_b).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + pid_b + 1).to(tl.int64)
    query_len = (eos - bos).to(tl.int32)
    num_accepted = tl.load(num_accepted_ptr + pid_b * stride_num_accepted_batch).to(
        tl.int32
    )
    if HAS_FORCE_COMMIT:
        force_commit = tl.load(force_commit_ptr + pid_b * stride_force_commit_batch).to(
            tl.int1
        )
        commit_len = tl.where(force_commit, query_len, num_accepted)
    else:
        commit_len = num_accepted
    commit_len = tl.minimum(tl.maximum(commit_len, 0), query_len)
    commit_len = tl.minimum(commit_len, SPEC_QUERY_LEN)
    if commit_len == 0:
        return

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_SPEC)
    offs_nt = tl.arange(0, DSTATE_TILE)
    m_mask = offs_m < dim
    k_mask = offs_k < commit_len

    state_base_addr = tl.load(state_base_addrs_ptr + pid_l)
    stride_state_batch = tl.load(state_block_strides_ptr + pid_l)
    state_ptr = state_base_addr.to(tl.pointer_type(state_ref_ptr.dtype.element_ty))
    state_ptr += state_batch_idx * stride_state_batch + pid_h * stride_state_head
    x_cache_base_addr = tl.load(x_cache_base_addrs_ptr + pid_l)
    stride_x_cache_block = tl.load(x_cache_block_strides_ptr + pid_l)
    x_cache_ptr = x_cache_base_addr.to(
        tl.pointer_type(x_cache_ref_ptr.dtype.element_ty)
    )
    x_cache_ptr += state_batch_idx * stride_x_cache_block + pid_h * stride_x_cache_head
    dt_cache_base_addr = tl.load(dt_cache_base_addrs_ptr + pid_l)
    stride_dt_cache_block = tl.load(dt_cache_block_strides_ptr + pid_l)
    dt_cache_ptr = dt_cache_base_addr.to(
        tl.pointer_type(dt_cache_ref_ptr.dtype.element_ty)
    )
    dt_cache_ptr += (
        state_batch_idx * stride_dt_cache_block + pid_h * stride_dt_cache_head
    )
    B_cache_base_addr = tl.load(B_cache_base_addrs_ptr + pid_l)
    stride_B_cache_block = tl.load(B_cache_block_strides_ptr + pid_l)
    B_cache_ptr = B_cache_base_addr.to(
        tl.pointer_type(B_cache_ref_ptr.dtype.element_ty)
    )
    B_cache_ptr += (
        state_batch_idx * stride_B_cache_block
        + (pid_h // nheads_ngroups_ratio) * stride_B_cache_group
    )
    A_ptr += pid_l * stride_A_layer + pid_h * stride_A_head
    if HAS_DT_BIAS:
        dt_bias_ptr += pid_l * stride_dt_bias_layer + pid_h * stride_dt_bias_head

    A_val = tl.load(A_ptr).to(tl.float32)
    dt_bias_val = tl.load(dt_bias_ptr).to(tl.float32) if HAS_DT_BIAS else 0.0
    dt = tl.load(
        dt_cache_ptr + offs_k * stride_dt_cache_pos,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)
    if HAS_DT_BIAS:
        dt = tl.where(k_mask, dt + dt_bias_val, 0.0)
    if DT_SOFTPLUS:
        dt = tl.where(k_mask, tl.where(dt <= 20.0, softplus(dt), dt), 0.0)
    dt_cum = tl.cumsum(dt, axis=0)
    dt_total = tl.sum(dt, axis=0)
    state_decay = tl.exp(tl.minimum(A_val * dt_total, 0.0))
    input_scale = tl.where(
        k_mask,
        dt * tl.exp(tl.minimum(A_val * (dt_total - dt_cum), 0.0)),
        0.0,
    )
    x = tl.load(
        x_cache_ptr
        + offs_m[:, None] * stride_x_cache_dim
        + offs_k[None, :] * stride_x_cache_pos,
        mask=m_mask[:, None] & k_mask[None, :],
        other=0.0,
    ).to(x_cache_ptr.dtype.element_ty)

    for i in tl.static_range(NDS):
        offs_n = i * DSTATE_TILE + offs_nt
        n_mask = offs_n < dstate
        B = tl.load(
            B_cache_ptr
            + offs_k[:, None] * stride_B_cache_pos
            + offs_n[None, :] * stride_B_cache_dstate,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        scaled_B = (B.to(tl.float32) * input_scale[:, None]).to(
            x_cache_ptr.dtype.element_ty
        )
        delta = tl.dot(x, scaled_B, input_precision="tf32x3").to(tl.float32)
        state_ptrs = (
            state_ptr
            + offs_m[:, None] * stride_state_dim
            + offs_n[None, :] * stride_state_dstate
        )
        state = tl.load(
            state_ptrs,
            mask=m_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        new_state = state.to(tl.float32) * state_decay + delta
        tl.store(
            state_ptrs,
            new_state.to(state.dtype),
            mask=m_mask[:, None] & n_mask[None, :],
        )


def selective_state_update_replayssm_spec(
    state_checkpoint: torch.Tensor,
    x_cache: torch.Tensor,
    dt_cache: torch.Tensor,
    B_cache: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_batch_indices: torch.Tensor,
    spec_query_len: int,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = True,
    out: torch.Tensor | None = None,
    bc_pre: torch.Tensor | None = None,
    null_block_id: int = NULL_BLOCK_ID,
) -> torch.Tensor:
    """Verify one speculative window without modifying the checkpoint.

    The activation buffers hold only the current window. After sampling,
    :meth:`ReplaySSMSpecCommitContext.commit` folds its accepted prefix into the
    checkpoint, so no cursor or pending acceptance survives into the next step.
    """
    num_blocks, nheads, dim, n_state = state_checkpoint.shape
    total_tokens = x.shape[0]
    ngroups = B.shape[1]
    dstate = B.shape[2]
    assert n_state == dstate
    batch = state_batch_indices.shape[0]
    assert x_cache.shape == (num_blocks, nheads, spec_query_len, dim)
    assert dt_cache.shape == (num_blocks, nheads, spec_query_len)
    assert B_cache.shape == (num_blocks, ngroups, spec_query_len, dstate)
    assert x.shape == (total_tokens, nheads, dim)
    assert dt.shape == (total_tokens, nheads)
    assert B.shape == C.shape == (total_tokens, ngroups, dstate)
    assert nheads % ngroups == 0
    assert A.shape == (nheads, dim, dstate)
    assert A.stride(-1) == 0 and A.stride(-2) == 0
    assert query_start_loc.shape[0] == batch + 1
    if total_tokens > batch * spec_query_len:
        raise ValueError(
            "ReplaySSM speculative decode input exceeds its activation capacity"
        )

    if out is None:
        out = torch.empty(total_tokens, nheads, dim, device=x.device, dtype=x.dtype)
    if total_tokens == 0:
        return out

    block_spec = max(16, triton.next_power_of_2(spec_query_len))
    block_dstate = triton.next_power_of_2(dstate)
    bsm, num_warps, dstate_tile, num_stages = get_replayssm_config(
        "mamba2_spec_verify", dstate=dstate, base_block=spec_query_len
    )
    dstate_tile = max(16, min(dstate_tile, block_dstate))
    nds = triton.cdiv(block_dstate, dstate_tile)

    if bc_pre is None:
        bc_pre = torch.empty(
            batch,
            ngroups,
            spec_query_len,
            block_spec,
            device=x.device,
            dtype=torch.float32,
        )
    else:
        assert (
            bc_pre.shape[0] >= batch
            and bc_pre.shape[1] == ngroups
            and bc_pre.shape[2] >= spec_query_len
            and bc_pre.shape[3] >= block_spec
        ), (
            f"bc_pre shape {tuple(bc_pre.shape)} incompatible with "
            f"(batch={batch}, ngroups={ngroups}, query={spec_query_len}, "
            f"block_spec={block_spec})"
        )

    ratio = nheads // ngroups
    ratio_p = ratio * dim
    block_cx = 256
    ncx = triton.cdiv(ratio_p, block_cx)
    block_hl = max(1, triton.next_power_of_2(ratio))
    state_indices_stride = state_batch_indices.stride(0)
    with torch.accelerator.device_index(x.device.index):
        _scatter_and_precompute_kernel[(batch, ngroups)](
            x,
            dt,
            B,
            C,
            x_cache,
            dt_cache,
            B_cache,
            bc_pre,
            state_batch_indices,
            query_start_loc,
            null_block_id,
            ngroups,
            nheads,
            dim,
            dstate,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            dt.stride(0),
            dt.stride(1),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            x_cache.stride(0),
            x_cache.stride(1),
            x_cache.stride(2),
            x_cache.stride(3),
            dt_cache.stride(0),
            dt_cache.stride(1),
            dt_cache.stride(2),
            B_cache.stride(0),
            B_cache.stride(1),
            B_cache.stride(2),
            B_cache.stride(3),
            bc_pre.stride(0),
            bc_pre.stride(1),
            bc_pre.stride(2),
            bc_pre.stride(3),
            state_indices_stride,
            RATIO=ratio,
            RATIO_P=ratio_p,
            NCX=ncx,
            BLOCK_CX=block_cx,
            SPEC_QUERY_LEN=spec_query_len,
            BLOCK_SIZE_SPEC=block_spec,
            BLOCK_HL=block_hl,
            num_warps=4,
        )

    z_strides = (z.stride(0), z.stride(1), z.stride(2)) if z is not None else (0, 0, 0)
    grid = lambda meta: (triton.cdiv(dim, meta["BLOCK_SIZE_M"]), batch, nheads)
    with torch.accelerator.device_index(state_checkpoint.device.index):
        _verify_kernel[grid](
            state_checkpoint,
            x_cache,
            dt_cache,
            C,
            bc_pre,
            D,
            z,
            dt_bias,
            A,
            out,
            state_batch_indices,
            query_start_loc,
            null_block_id,
            ratio,
            dim,
            dstate,
            state_checkpoint.stride(0),
            state_checkpoint.stride(1),
            state_checkpoint.stride(2),
            state_checkpoint.stride(3),
            x_cache.stride(0),
            x_cache.stride(1),
            x_cache.stride(2),
            x_cache.stride(3),
            dt_cache.stride(0),
            dt_cache.stride(1),
            dt_cache.stride(2),
            C.stride(0),
            C.stride(2),
            bc_pre.stride(0),
            bc_pre.stride(1),
            bc_pre.stride(2),
            bc_pre.stride(3),
            D.stride(0) if D is not None else 0,
            D.stride(1) if D is not None else 0,
            z_strides[0],
            z_strides[1],
            z_strides[2],
            dt_bias.stride(0) if dt_bias is not None else 0,
            A.stride(0),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            state_indices_stride,
            DT_SOFTPLUS=dt_softplus,
            BLOCK_SIZE_M=bsm,
            SPEC_QUERY_LEN=spec_query_len,
            BLOCK_SIZE_SPEC=block_spec,
            DSTATE_TILE=dstate_tile,
            NDS=nds,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out


@dataclass
class ReplaySSMSpecCommitContext:
    """Block-keyed activation buffers and immutable group commit metadata."""

    conv_states: tuple[torch.Tensor, ...]
    conv_state_base_addrs: torch.Tensor
    conv_state_block_strides: torch.Tensor
    conv_state_dim_strides: torch.Tensor
    conv_state_token_strides: torch.Tensor
    conv_history_len: int
    state_checkpoints: tuple[torch.Tensor, ...]
    state_base_addrs: torch.Tensor
    state_block_strides: torch.Tensor
    x_caches: tuple[torch.Tensor, ...]
    x_cache_base_addrs: torch.Tensor
    x_cache_block_strides: torch.Tensor
    dt_caches: tuple[torch.Tensor, ...]
    dt_cache_base_addrs: torch.Tensor
    dt_cache_block_strides: torch.Tensor
    B_caches: tuple[torch.Tensor, ...]
    B_cache_base_addrs: torch.Tensor
    B_cache_block_strides: torch.Tensor
    A: torch.Tensor
    dt_bias: torch.Tensor
    spec_query_len: int

    @classmethod
    def create(
        cls,
        conv_states: Sequence[torch.Tensor],
        state_checkpoints: Sequence[torch.Tensor],
        x_caches: Sequence[torch.Tensor],
        dt_caches: Sequence[torch.Tensor],
        B_caches: Sequence[torch.Tensor],
        A: Sequence[torch.Tensor],
        dt_bias: Sequence[torch.Tensor],
        *,
        ngroups: int,
        spec_query_len: int,
    ) -> "ReplaySSMSpecCommitContext":
        if not state_checkpoints:
            raise ValueError("ReplaySSM commit requires at least one layer")
        num_layers = len(state_checkpoints)
        if not all(
            len(values) == num_layers
            for values in (conv_states, x_caches, dt_caches, B_caches, A, dt_bias)
        ):
            raise ValueError("ReplaySSM layer state and parameter counts must match")

        state_ref = state_checkpoints[0]
        if state_ref.ndim != 4:
            raise ValueError(
                "ReplaySSM checkpoint must have shape [blocks, heads, dim, state]"
            )
        nheads, dim, dstate = state_ref.shape[1:]
        if nheads % ngroups != 0:
            raise ValueError("ReplaySSM heads must divide evenly across groups")
        for state in state_checkpoints:
            if (
                state.shape != state_ref.shape
                or state.dtype != state_ref.dtype
                or state.device != state_ref.device
                or state.stride()[1:] != state_ref.stride()[1:]
            ):
                raise ValueError(
                    "ReplaySSM layers in one attention group need matching state "
                    "shape, dtype, device, and inner strides"
                )
        if any(param.shape != (nheads,) for param in (*A, *dt_bias)):
            raise ValueError("ReplaySSM A and dt_bias must contain one value per head")
        if any(param.device != state_ref.device for param in (*A, *dt_bias)):
            raise ValueError("ReplaySSM state and parameters must share a device")
        if any(param.dtype != A[0].dtype for param in A) or any(
            param.dtype != dt_bias[0].dtype for param in dt_bias
        ):
            raise ValueError("ReplaySSM parameters must use one dtype across layers")

        conv_ref = conv_states[0]
        if conv_ref.ndim != 3:
            raise ValueError(
                "ReplaySSM conv state must have shape [blocks, dim, state]"
            )
        conv_dim, conv_state_len = conv_ref.shape[1:]
        conv_history_len = conv_state_len - spec_query_len + 1
        if conv_history_len <= 0:
            raise ValueError("ReplaySSM conv state is shorter than its verify window")
        for layer_idx, conv_state in enumerate(conv_states):
            if (
                conv_state.shape != conv_ref.shape
                or conv_state.dtype != conv_ref.dtype
                or conv_state.device != conv_ref.device
            ):
                raise ValueError(
                    "ReplaySSM layers in one attention group need matching conv "
                    "state shape, dtype, and device"
                )
            if conv_state.shape[0] != state_checkpoints[layer_idx].shape[0]:
                raise ValueError("ReplaySSM conv and SSM block counts must match")

        num_blocks = state_ref.shape[0]
        expected_x_shape = (num_blocks, nheads, spec_query_len, dim)
        expected_dt_shape = (num_blocks, nheads, spec_query_len)
        expected_B_shape = (num_blocks, ngroups, spec_query_len, dstate)
        cache_specs = (
            ("x", x_caches, expected_x_shape),
            ("dt", dt_caches, expected_dt_shape),
            ("B", B_caches, expected_B_shape),
        )
        for name, caches, expected_shape in cache_specs:
            cache_ref = caches[0]
            for cache in caches:
                if (
                    cache.shape != expected_shape
                    or cache.dtype != cache_ref.dtype
                    or cache.device != state_ref.device
                    or cache.stride()[1:] != cache_ref.stride()[1:]
                ):
                    raise ValueError(
                        f"ReplaySSM {name} buffers need shape {expected_shape} "
                        "and matching dtype, device, and inner strides"
                    )
        if x_caches[0].dtype != B_caches[0].dtype:
            raise ValueError("ReplaySSM x and B buffers must use the same dtype")

        device = state_ref.device
        return cls(
            conv_states=tuple(conv_states),
            conv_state_base_addrs=torch.tensor(
                [state.data_ptr() for state in conv_states],
                dtype=torch.int64,
                device=device,
            ),
            conv_state_block_strides=torch.tensor(
                [state.stride(0) for state in conv_states],
                dtype=torch.int64,
                device=device,
            ),
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
            state_checkpoints=tuple(state_checkpoints),
            state_base_addrs=torch.tensor(
                [state.data_ptr() for state in state_checkpoints],
                dtype=torch.int64,
                device=device,
            ),
            state_block_strides=torch.tensor(
                [state.stride(0) for state in state_checkpoints],
                dtype=torch.int64,
                device=device,
            ),
            x_caches=tuple(x_caches),
            x_cache_base_addrs=torch.tensor(
                [cache.data_ptr() for cache in x_caches],
                dtype=torch.int64,
                device=device,
            ),
            x_cache_block_strides=torch.tensor(
                [cache.stride(0) for cache in x_caches],
                dtype=torch.int64,
                device=device,
            ),
            dt_caches=tuple(dt_caches),
            dt_cache_base_addrs=torch.tensor(
                [cache.data_ptr() for cache in dt_caches],
                dtype=torch.int64,
                device=device,
            ),
            dt_cache_block_strides=torch.tensor(
                [cache.stride(0) for cache in dt_caches],
                dtype=torch.int64,
                device=device,
            ),
            B_caches=tuple(B_caches),
            B_cache_base_addrs=torch.tensor(
                [cache.data_ptr() for cache in B_caches],
                dtype=torch.int64,
                device=device,
            ),
            B_cache_block_strides=torch.tensor(
                [cache.stride(0) for cache in B_caches],
                dtype=torch.int64,
                device=device,
            ),
            A=torch.stack(tuple(A)).contiguous(),
            dt_bias=torch.stack(tuple(dt_bias)).contiguous(),
            spec_query_len=spec_query_len,
        )

    def commit(
        self,
        num_accepted_tokens: torch.Tensor,
        state_batch_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        force_commit: torch.Tensor | None = None,
        null_block_id: int = NULL_BLOCK_ID,
    ) -> None:
        """Materialize each row's accepted recurrent state for every layer."""
        batch = state_batch_indices.shape[0]
        if batch == 0:
            return
        state_ref = self.state_checkpoints[0]
        x_cache_ref = self.x_caches[0]
        dt_cache_ref = self.dt_caches[0]
        B_cache_ref = self.B_caches[0]
        num_layers = len(self.state_checkpoints)
        nheads, dim, dstate = state_ref.shape[1:]
        ngroups = B_cache_ref.shape[1]
        assert self.A.shape == self.dt_bias.shape == (num_layers, nheads)
        assert num_accepted_tokens.shape[0] >= batch
        assert query_start_loc.shape[0] == batch + 1
        if force_commit is not None:
            assert force_commit.shape[0] >= batch

        conv_ref = self.conv_states[0]
        conv_dim = conv_ref.shape[1]
        block_history = triton.next_power_of_2(self.conv_history_len)
        with torch.accelerator.device_index(conv_ref.device.index):
            _compact_conv_state_kernel[(triton.cdiv(conv_dim, 256), batch, num_layers)](
                conv_ref,
                self.conv_state_base_addrs,
                self.conv_state_block_strides,
                self.conv_state_dim_strides,
                self.conv_state_token_strides,
                num_accepted_tokens,
                force_commit,
                state_batch_indices,
                query_start_loc,
                null_block_id,
                conv_dim,
                self.conv_history_len,
                num_accepted_tokens.stride(0),
                force_commit.stride(0) if force_commit is not None else 0,
                state_batch_indices.stride(0),
                BLOCK_SIZE_D=256,
                BLOCK_SIZE_HISTORY=block_history,
                SPEC_QUERY_LEN=self.spec_query_len,
                num_warps=4,
            )

        block_spec = max(16, triton.next_power_of_2(self.spec_query_len))
        block_dstate = triton.next_power_of_2(dstate)
        bsm, num_warps, dstate_tile, num_stages = get_replayssm_config(
            "mamba2_spec_commit", dstate=dstate, base_block=self.spec_query_len
        )
        dstate_tile = max(16, min(dstate_tile, block_dstate))
        nds = triton.cdiv(block_dstate, dstate_tile)
        ratio = nheads // ngroups
        grid = lambda meta: (
            triton.cdiv(dim, meta["BLOCK_SIZE_M"]),
            batch,
            num_layers * nheads,
        )
        with torch.accelerator.device_index(state_ref.device.index):
            _commit_kernel[grid](
                state_ref,
                self.state_base_addrs,
                self.state_block_strides,
                x_cache_ref,
                self.x_cache_base_addrs,
                self.x_cache_block_strides,
                dt_cache_ref,
                self.dt_cache_base_addrs,
                self.dt_cache_block_strides,
                B_cache_ref,
                self.B_cache_base_addrs,
                self.B_cache_block_strides,
                self.A,
                self.dt_bias,
                num_accepted_tokens,
                force_commit,
                state_batch_indices,
                query_start_loc,
                null_block_id,
                ratio,
                dim,
                dstate,
                state_ref.stride(1),
                state_ref.stride(2),
                state_ref.stride(3),
                x_cache_ref.stride(1),
                x_cache_ref.stride(2),
                x_cache_ref.stride(3),
                dt_cache_ref.stride(1),
                dt_cache_ref.stride(2),
                B_cache_ref.stride(1),
                B_cache_ref.stride(2),
                B_cache_ref.stride(3),
                self.A.stride(0),
                self.A.stride(1),
                self.dt_bias.stride(0),
                self.dt_bias.stride(1),
                num_accepted_tokens.stride(0),
                force_commit.stride(0) if force_commit is not None else 0,
                state_batch_indices.stride(0),
                DT_SOFTPLUS=True,
                BLOCK_SIZE_M=bsm,
                SPEC_QUERY_LEN=self.spec_query_len,
                BLOCK_SIZE_SPEC=block_spec,
                DSTATE_TILE=dstate_tile,
                NDS=nds,
                NHEADS=nheads,
                num_warps=num_warps,
                num_stages=num_stages,
            )


__all__ = ["ReplaySSMSpecCommitContext", "selective_state_update_replayssm_spec"]
