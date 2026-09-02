# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 RecoverSSM speculative verify and accepted-state recovery."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    TritonWarmupTensor,
    VllmTritonJitKernel,
    kernel_launcher,
    triton_scalar_specialization_rep,
)
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv, next_power_of_2
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
def _kda_recurrent_step(
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
def _kda_recoverssm_verify_kernel(
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
        updated_state, correction = _kda_recurrent_step(
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
    boundary_recovery_lens_ptr,
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
    boundary_recovery_len = 0
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
        boundary_recovery_len = next_boundary - num_computed
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
        boundary_recovery_lens_ptr + spec_idx,
        tl.where(valid, boundary_recovery_len, 0),
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
    boundary_recovery_lens_ptr,
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
    boundary_recovery_len = tl.load(boundary_recovery_lens_ptr + pid_b)

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
            + (boundary_recovery_len - 1 + offs_h[None, :]) * token_stride,
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
    boundary_recovery_lens_ptr,
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
    boundary_recovery_len = tl.load(boundary_recovery_lens_ptr + pid_b)

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
            before_boundary = token_offset < boundary_recovery_len
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


class KimiK3RecoverSSMVerifyKernel(
    VllmTritonJitKernel["KimiK3RecoverSSMVerifyKernel.CompileKey"]
):
    """JIT owner for RecoverSSM speculative verification."""

    kernel = staticmethod(_kda_recoverssm_verify_kernel)

    @dataclass(frozen=True)
    class CompileKey:
        io_dtype: torch.dtype
        state_dtype: torch.dtype
        a_log_dtype: torch.dtype
        dt_bias_dtype: torch.dtype
        num_heads: int
        key_dim: int
        value_dim: int
        block_k: int
        block_v: int
        spec_query_len: int
        stride_q_token: int
        stride_k_token: int
        stride_v_token: int
        stride_g_token: int
        stride_beta_token: int
        stride_out_token: int
        stride_state_block: int
        stride_correction_block: int
        stride_kg_block: int
        stride_state_indices: int
        use_lower_bound: bool

    def dispatch(  # type: ignore[override]
        self,
        *,
        io_dtype: torch.dtype,
        state_dtype: torch.dtype,
        a_log_dtype: torch.dtype,
        dt_bias_dtype: torch.dtype,
        num_heads: int,
        key_dim: int,
        value_dim: int,
        spec_query_len: int,
        stride_q_token: int,
        stride_k_token: int,
        stride_v_token: int,
        stride_g_token: int,
        stride_beta_token: int,
        stride_out_token: int,
        stride_state_block: int,
        stride_correction_block: int,
        stride_kg_block: int,
        stride_state_indices: int,
        use_lower_bound: bool,
    ) -> CompileKey:
        block_k = next_power_of_2(key_dim)
        block_v = min(next_power_of_2(value_dim), 32)
        return self.CompileKey(
            io_dtype=io_dtype,
            state_dtype=state_dtype,
            a_log_dtype=a_log_dtype,
            dt_bias_dtype=dt_bias_dtype,
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=value_dim,
            block_k=block_k,
            block_v=block_v,
            spec_query_len=spec_query_len,
            stride_q_token=stride_q_token,
            stride_k_token=stride_k_token,
            stride_v_token=stride_v_token,
            stride_g_token=stride_g_token,
            stride_beta_token=triton_scalar_specialization_rep(stride_beta_token),
            stride_out_token=stride_out_token,
            stride_state_block=stride_state_block,
            stride_correction_block=stride_correction_block,
            stride_kg_block=stride_kg_block,
            stride_state_indices=triton_scalar_specialization_rep(
                stride_state_indices
            ),
            use_lower_bound=use_lower_bound,
        )

    def get_warmup_keys(
        self,
        *,
        io_dtype: torch.dtype,
        state_dtype: torch.dtype,
        a_log_dtype: torch.dtype,
        dt_bias_dtype: torch.dtype,
        num_heads: int,
        head_dim: int,
        spec_query_len: int,
        stride_q_token: int,
        stride_k_token: int,
        stride_v_token: int,
        stride_g_token: int,
        stride_beta_token: int | tuple[int, ...],
        stride_out_token: int,
        stride_state_block: int,
        stride_correction_block: int,
        stride_kg_block: int,
        stride_state_indices: int | tuple[int, ...],
        use_lower_bound: bool,
    ) -> list[CompileKey]:
        return self._trace_dispatch(self.dispatch)(
            io_dtype=io_dtype,
            state_dtype=state_dtype,
            a_log_dtype=a_log_dtype,
            dt_bias_dtype=dt_bias_dtype,
            num_heads=num_heads,
            key_dim=head_dim,
            value_dim=head_dim,
            spec_query_len=spec_query_len,
            stride_q_token=stride_q_token,
            stride_k_token=stride_k_token,
            stride_v_token=stride_v_token,
            stride_g_token=stride_g_token,
            stride_beta_token=stride_beta_token,
            stride_out_token=stride_out_token,
            stride_state_block=stride_state_block,
            stride_correction_block=stride_correction_block,
            stride_kg_block=stride_kg_block,
            stride_state_indices=stride_state_indices,
            use_lower_bound=use_lower_bound,
        )

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        h = compile_key.num_heads
        k = compile_key.key_dim
        v = compile_key.value_dim
        spec = compile_key.spec_query_len
        io = compile_key.io_dtype
        t = spec
        nb = 2
        q_row = compile_key.stride_q_token
        k_row = compile_key.stride_k_token
        v_row = compile_key.stride_v_token
        g_row = compile_key.stride_g_token
        out_row = compile_key.stride_out_token
        return dict(
            q=TritonWarmupTensor(
                io, shape=(1, t, h, k), strides=(t * q_row, q_row, k, 1)
            ),
            k=TritonWarmupTensor(
                io, shape=(1, t, h, k), strides=(t * k_row, k_row, k, 1)
            ),
            v=TritonWarmupTensor(
                io, shape=(1, t, h, v), strides=(t * v_row, v_row, v, 1)
            ),
            raw_g=TritonWarmupTensor(
                io, shape=(1, t, h, k), strides=(t * g_row, g_row, k, 1)
            ),
            raw_beta=TritonWarmupTensor(
                io,
                shape=(1, t, h),
                strides=(
                    t * compile_key.stride_beta_token,
                    compile_key.stride_beta_token,
                    1,
                ),
            ),
            A_log=TritonWarmupTensor(compile_key.a_log_dtype, shape=(h,)),
            dt_bias=TritonWarmupTensor(compile_key.dt_bias_dtype, shape=(h * k,)),
            state=TritonWarmupTensor(
                compile_key.state_dtype,
                shape=(nb, h, v, k),
                strides=(compile_key.stride_state_block, v * k, k, 1),
            ),
            correction_cache=TritonWarmupTensor(
                torch.float32,
                shape=(nb, h, spec, v),
                strides=(compile_key.stride_correction_block, spec * v, v, 1),
            ),
            kg_cache=TritonWarmupTensor(
                io,
                shape=(nb, h, spec, 2 * k),
                strides=(compile_key.stride_kg_block, spec * 2 * k, 2 * k, 1),
            ),
            out=TritonWarmupTensor(
                io, shape=(1, t, h, v), strides=(t * out_row, out_row, v, 1)
            ),
            query_start_loc=TritonWarmupTensor(torch.int32, shape=(2,)),
            state_indices=TritonWarmupTensor(
                torch.int32,
                shape=(1,),
                strides=(compile_key.stride_state_indices,),
            ),
            lower_bound=-1.0 if compile_key.use_lower_bound else None,
            spec_query_len=spec,
        )

    @kernel_launcher
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        raw_g: torch.Tensor,
        raw_beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state: torch.Tensor,
        correction_cache: torch.Tensor,
        kg_cache: torch.Tensor,
        out: torch.Tensor,
        query_start_loc: torch.Tensor,
        state_indices: torch.Tensor,
        *,
        lower_bound: float | None,
        spec_query_len: int,
    ) -> LaunchSpec:
        batch = state_indices.shape[0]
        num_heads = q.shape[2]
        key_dim = q.shape[3]
        value_dim = v.shape[3]
        use_lower_bound = lower_bound is not None
        # Reproduce HEAD's launch geometry inline; ``dispatch`` remains an
        # independent compile-key enumeration so the JIT monitor can verify it.
        block_k = next_power_of_2(key_dim)
        block_v = min(next_power_of_2(value_dim), 32)
        grid = (cdiv(value_dim, block_v), batch, num_heads)
        return grid, dict(
            lower_bound=lower_bound if lower_bound is not None else 0.0,
            null_block_id=NULL_BLOCK_ID,
            stride_q_token=q.stride(1),
            stride_k_token=k.stride(1),
            stride_v_token=v.stride(1),
            stride_g_token=raw_g.stride(1),
            stride_beta_token=raw_beta.stride(1),
            stride_state_block=state.stride(0),
            stride_state_head=state.stride(1),
            stride_state_v=state.stride(2),
            stride_state_k=state.stride(3),
            stride_correction_block=correction_cache.stride(0),
            stride_correction_head=correction_cache.stride(1),
            stride_correction_pos=correction_cache.stride(2),
            stride_correction_dim=correction_cache.stride(3),
            stride_kg_block=kg_cache.stride(0),
            stride_kg_head=kg_cache.stride(1),
            stride_kg_pos=kg_cache.stride(2),
            stride_kg_dim=kg_cache.stride(3),
            stride_out_token=out.stride(1),
            stride_query_start_loc=query_start_loc.stride(0),
            stride_state_indices=state_indices.stride(0),
            K=key_dim,
            V=value_dim,
            BK=block_k,
            BV=block_v,
            SPEC_QUERY_LEN=spec_query_len,
            USE_LOWER_BOUND=use_lower_bound,
            num_warps=4,
            num_stages=2,
        )


def kda_recoverssm_verify(
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
        raise ValueError("KDA RecoverSSM q must have shape [1, tokens, heads, dim]")
    _, total_tokens, num_heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if k.shape != q.shape or v.shape != (1, total_tokens, num_heads, value_dim):
        raise ValueError("KDA RecoverSSM q, k, and v shapes are incompatible")
    if raw_g.shape != q.shape or raw_beta.shape != (1, total_tokens, num_heads):
        raise ValueError("KDA RecoverSSM gate or beta shape is incompatible")
    if any(tensor.stride()[2:] != (key_dim, 1) for tensor in (q, k, raw_g)):
        raise ValueError("KDA RecoverSSM q, k, and gate heads must be contiguous")
    if v.stride()[2:] != (value_dim, 1) or raw_beta.stride(2) != 1:
        raise ValueError("KDA RecoverSSM v and beta heads must be contiguous")
    num_blocks = checkpoint_state.shape[0]
    if checkpoint_state.shape[1:] != (
        num_heads,
        value_dim,
        key_dim,
    ):
        raise ValueError("KDA RecoverSSM checkpoint shape is incompatible")
    expected_correction_shape = (
        num_blocks,
        num_heads,
        spec_query_len,
        value_dim,
    )
    if correction_cache.shape != expected_correction_shape:
        raise ValueError(
            f"KDA RecoverSSM correction buffer needs shape {expected_correction_shape}"
        )
    expected_kg_shape = (num_blocks, num_heads, spec_query_len, 2 * key_dim)
    if kg_cache.shape != expected_kg_shape:
        raise ValueError(
            f"KDA RecoverSSM key/gate buffer needs shape {expected_kg_shape}"
        )
    if correction_cache.dtype != torch.float32:
        raise ValueError("KDA RecoverSSM correction buffer must use float32")
    if kg_cache.dtype != k.dtype:
        raise ValueError("KDA RecoverSSM key/gate buffer must match activation dtype")
    if A_log.shape != (num_heads,) or dt_bias.numel() != num_heads * key_dim:
        raise ValueError("KDA RecoverSSM gate parameters are incompatible")
    if not A_log.is_contiguous() or not dt_bias.is_contiguous():
        raise ValueError("KDA RecoverSSM gate parameters must be contiguous")
    batch = state_indices.shape[0]
    if query_start_loc.shape[0] != batch + 1:
        raise ValueError("KDA RecoverSSM query metadata is incompatible")
    if total_tokens > batch * spec_query_len:
        raise ValueError(
            "KDA RecoverSSM speculative decode input exceeds its activation capacity"
        )
    if out is None:
        out = torch.empty_like(v)
    if out.shape != v.shape:
        raise ValueError("KDA RecoverSSM output shape is incompatible")
    if out.stride()[2:] != (value_dim, 1):
        raise ValueError("KDA RecoverSSM output heads must be contiguous")
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
        raise ValueError("KDA RecoverSSM inputs must be on the same device")
    if total_tokens == 0:
        return out

    _RECOVERSSM_VERIFY_KERNEL(
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
        lower_bound=lower_bound,
        spec_query_len=spec_query_len,
    )
    return out


class KimiK3RecoverSSMPrepareCommitPlanKernel(
    VllmTritonJitKernel["KimiK3RecoverSSMPrepareCommitPlanKernel.CompileKey"]
):
    """JIT owner for RecoverSSM commit planning."""

    kernel = staticmethod(_prepare_commit_plan_kernel)

    @dataclass(frozen=True)
    class CompileKey:
        spec_query_len: int
        has_request_indices: bool
        align_mode: bool
        mamba_block_size: int
        block_table_width: int
        stride_state_indices: int

    def dispatch(  # type: ignore[override]
        self,
        *,
        spec_query_len: int,
        has_request_indices: bool,
        align_mode: bool,
        mamba_block_size: int,
        block_table_width: int,
        stride_state_indices: int,
    ) -> CompileKey:
        return self.CompileKey(
            spec_query_len=spec_query_len,
            has_request_indices=has_request_indices,
            align_mode=align_mode,
            mamba_block_size=mamba_block_size,
            block_table_width=block_table_width,
            stride_state_indices=triton_scalar_specialization_rep(
                stride_state_indices
            ),
        )

    def get_warmup_keys(
        self,
        *,
        spec_query_len: int,
        align_mode: bool,
        mamba_block_size: int,
        block_table_width: int,
        stride_state_indices: int | tuple[int, ...],
    ) -> list[CompileKey]:
        return self._trace_dispatch(self.dispatch)(
            spec_query_len=spec_query_len,
            has_request_indices=(False, True),
            align_mode=align_mode,
            mamba_block_size=mamba_block_size,
            block_table_width=block_table_width,
            stride_state_indices=stride_state_indices,
        )

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        n = 2
        width = compile_key.block_table_width
        request_indices = None
        if compile_key.has_request_indices:
            request_indices = TritonWarmupTensor(torch.int32, shape=(n,))
        block_table = None
        num_computed = None
        if compile_key.align_mode:
            block_table = TritonWarmupTensor(
                torch.int32, shape=(n, width), strides=(width, 1)
            )
            num_computed = TritonWarmupTensor(torch.int32, shape=(n,))
        return dict(
            num_accepted=TritonWarmupTensor(torch.int32, shape=(n,)),
            request_indices=request_indices,
            state_indices=TritonWarmupTensor(
                torch.int32,
                shape=(n,),
                strides=(compile_key.stride_state_indices,),
            ),
            query_start_loc=TritonWarmupTensor(torch.int32, shape=(n + 1,)),
            block_table=block_table,
            num_computed=num_computed,
            commit_lens=TritonWarmupTensor(torch.int32, shape=(n,)),
            final_state_indices=TritonWarmupTensor(torch.int32, shape=(n,)),
            boundary_state_indices=TritonWarmupTensor(torch.int32, shape=(n,)),
            boundary_recovery_lens=TritonWarmupTensor(torch.int32, shape=(n,)),
            spec_query_len=compile_key.spec_query_len,
            mamba_block_size=compile_key.mamba_block_size,
        )

    @kernel_launcher
    def __call__(
        self,
        num_accepted: torch.Tensor,
        request_indices: torch.Tensor | None,
        state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        block_table: torch.Tensor | None,
        num_computed: torch.Tensor | None,
        commit_lens: torch.Tensor,
        final_state_indices: torch.Tensor,
        boundary_state_indices: torch.Tensor,
        boundary_recovery_lens: torch.Tensor,
        *,
        spec_query_len: int,
        mamba_block_size: int,
    ) -> LaunchSpec:
        batch = state_indices.shape[0]
        has_request_indices = request_indices is not None
        align_mode = block_table is not None
        block_table_width = block_table.shape[1] if block_table is not None else 1
        block_table_stride = (0, 0) if block_table is None else block_table.stride()
        num_computed_stride = 0 if num_computed is None else num_computed.stride(0)
        grid = (batch,)
        return grid, dict(
            null_block_id=NULL_BLOCK_ID,
            mamba_block_size=mamba_block_size,
            block_table_width=block_table_width,
            stride_num_accepted=num_accepted.stride(0),
            stride_request_indices=(
                request_indices.stride(0) if request_indices is not None else 0
            ),
            stride_state_indices=state_indices.stride(0),
            stride_query_start_loc=query_start_loc.stride(0),
            stride_block_table_row=block_table_stride[0],
            stride_block_table_col=block_table_stride[1],
            stride_num_computed=num_computed_stride,
            SPEC_QUERY_LEN=spec_query_len,
            num_warps=1,
        )


class KimiK3RecoverSSMCompactConvStateKernel(
    VllmTritonJitKernel["KimiK3RecoverSSMCompactConvStateKernel.CompileKey"]
):
    """JIT owner for RecoverSSM convolution-state compaction."""

    kernel = staticmethod(_compact_conv_state_kernel)

    @dataclass(frozen=True)
    class CompileKey:
        conv_state_dtype: torch.dtype
        conv_dim: int
        conv_history_len: int
        block_history: int
        block_d: int
        align_mode: bool
        stride_state_indices: int

    def dispatch(  # type: ignore[override]
        self,
        *,
        conv_state_dtype: torch.dtype,
        conv_dim: int,
        conv_history_len: int,
        align_mode: bool,
        stride_state_indices: int,
    ) -> CompileKey:
        block_history = next_power_of_2(conv_history_len)
        block_d = 256
        return self.CompileKey(
            conv_state_dtype=conv_state_dtype,
            conv_dim=conv_dim,
            conv_history_len=conv_history_len,
            block_history=block_history,
            block_d=block_d,
            align_mode=align_mode,
            stride_state_indices=triton_scalar_specialization_rep(
                stride_state_indices
            ),
        )

    def get_warmup_keys(
        self,
        *,
        conv_state_dtype: torch.dtype,
        conv_dim: int,
        conv_history_len: int,
        align_mode: bool,
        stride_state_indices: int | tuple[int, ...],
    ) -> list[CompileKey]:
        return self._trace_dispatch(self.dispatch)(
            conv_state_dtype=conv_state_dtype,
            conv_dim=conv_dim,
            conv_history_len=conv_history_len,
            align_mode=align_mode,
            stride_state_indices=stride_state_indices,
        )

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        n = 2
        nl = 2
        return dict(
            conv_state_ref=TritonWarmupTensor(
                compile_key.conv_state_dtype, shape=(1,)
            ),
            conv_state_base_addrs=TritonWarmupTensor(torch.int64, shape=(nl,)),
            conv_state_block_strides=TritonWarmupTensor(torch.int64, shape=(nl,)),
            conv_state_dim_strides=TritonWarmupTensor(torch.int64, shape=(nl,)),
            conv_state_token_strides=TritonWarmupTensor(torch.int64, shape=(nl,)),
            state_indices=TritonWarmupTensor(
                torch.int32,
                shape=(n,),
                strides=(compile_key.stride_state_indices,),
            ),
            commit_lens=TritonWarmupTensor(torch.int32, shape=(n,)),
            final_state_indices=TritonWarmupTensor(torch.int32, shape=(n,)),
            boundary_state_indices=TritonWarmupTensor(torch.int32, shape=(n,)),
            boundary_recovery_lens=TritonWarmupTensor(torch.int32, shape=(n,)),
            conv_dim=compile_key.conv_dim,
            conv_history_len=compile_key.conv_history_len,
            align_mode=compile_key.align_mode,
        )

    @kernel_launcher
    def __call__(
        self,
        conv_state_ref: torch.Tensor,
        conv_state_base_addrs: torch.Tensor,
        conv_state_block_strides: torch.Tensor,
        conv_state_dim_strides: torch.Tensor,
        conv_state_token_strides: torch.Tensor,
        state_indices: torch.Tensor,
        commit_lens: torch.Tensor,
        final_state_indices: torch.Tensor,
        boundary_state_indices: torch.Tensor,
        boundary_recovery_lens: torch.Tensor,
        *,
        conv_dim: int,
        conv_history_len: int,
        align_mode: bool,
    ) -> LaunchSpec:
        batch = state_indices.shape[0]
        num_layers = conv_state_base_addrs.shape[0]
        block_history = next_power_of_2(conv_history_len)
        grid = (cdiv(conv_dim, 256), batch, num_layers)
        return grid, dict(
            null_block_id=NULL_BLOCK_ID,
            conv_dim=conv_dim,
            conv_history_len=conv_history_len,
            stride_state_indices=state_indices.stride(0),
            BLOCK_D=256,
            BLOCK_HISTORY=block_history,
            ALIGN_MODE=align_mode,
            num_warps=4,
        )


class KimiK3RecoverSSMCommitKdaStateKernel(
    VllmTritonJitKernel["KimiK3RecoverSSMCommitKdaStateKernel.CompileKey"]
):
    """JIT owner for RecoverSSM KDA-state commits."""

    kernel = staticmethod(_commit_kda_state_kernel)

    @dataclass(frozen=True)
    class CompileKey:
        state_dtype: torch.dtype
        kg_dtype: torch.dtype
        a_log_dtype: torch.dtype
        dt_bias_dtype: torch.dtype
        num_heads: int
        key_dim: int
        value_dim: int
        block_k: int
        block_v: int
        spec_query_len: int
        use_lower_bound: bool
        align_mode: bool
        stride_state_indices: int

    def dispatch(  # type: ignore[override]
        self,
        *,
        state_dtype: torch.dtype,
        kg_dtype: torch.dtype,
        a_log_dtype: torch.dtype,
        dt_bias_dtype: torch.dtype,
        num_heads: int,
        key_dim: int,
        value_dim: int,
        spec_query_len: int,
        use_lower_bound: bool,
        align_mode: bool,
        stride_state_indices: int,
    ) -> CompileKey:
        block_k = next_power_of_2(key_dim)
        block_v = min(next_power_of_2(value_dim), 32)
        return self.CompileKey(
            state_dtype=state_dtype,
            kg_dtype=kg_dtype,
            a_log_dtype=a_log_dtype,
            dt_bias_dtype=dt_bias_dtype,
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=value_dim,
            block_k=block_k,
            block_v=block_v,
            spec_query_len=spec_query_len,
            use_lower_bound=use_lower_bound,
            align_mode=align_mode,
            stride_state_indices=triton_scalar_specialization_rep(
                stride_state_indices
            ),
        )

    def get_warmup_keys(
        self,
        *,
        state_dtype: torch.dtype,
        kg_dtype: torch.dtype,
        a_log_dtype: torch.dtype,
        dt_bias_dtype: torch.dtype,
        num_heads: int,
        head_dim: int,
        spec_query_len: int,
        use_lower_bound: bool,
        align_mode: bool,
        stride_state_indices: int | tuple[int, ...],
    ) -> list[CompileKey]:
        return self._trace_dispatch(self.dispatch)(
            state_dtype=state_dtype,
            kg_dtype=kg_dtype,
            a_log_dtype=a_log_dtype,
            dt_bias_dtype=dt_bias_dtype,
            num_heads=num_heads,
            key_dim=head_dim,
            value_dim=head_dim,
            spec_query_len=spec_query_len,
            use_lower_bound=use_lower_bound,
            align_mode=align_mode,
            stride_state_indices=stride_state_indices,
        )

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        h = compile_key.num_heads
        k = compile_key.key_dim
        v = compile_key.value_dim
        spec = compile_key.spec_query_len
        nb = 2
        nl = 2
        n = 2
        return dict(
            state_ref=TritonWarmupTensor(
                compile_key.state_dtype, shape=(nb, h, v, k)
            ),
            state_base_addrs=TritonWarmupTensor(torch.int64, shape=(nl,)),
            state_block_strides=TritonWarmupTensor(torch.int64, shape=(nl,)),
            correction_cache_ref=TritonWarmupTensor(
                torch.float32, shape=(nb, h, spec, v)
            ),
            correction_cache_base_addrs=TritonWarmupTensor(
                torch.int64, shape=(nl,)
            ),
            correction_cache_block_strides=TritonWarmupTensor(
                torch.int64, shape=(nl,)
            ),
            kg_cache_ref=TritonWarmupTensor(
                compile_key.kg_dtype, shape=(nb, h, spec, 2 * k)
            ),
            kg_cache_base_addrs=TritonWarmupTensor(torch.int64, shape=(nl,)),
            kg_cache_block_strides=TritonWarmupTensor(torch.int64, shape=(nl,)),
            A_log=TritonWarmupTensor(compile_key.a_log_dtype, shape=(nl, h)),
            dt_bias=TritonWarmupTensor(compile_key.dt_bias_dtype, shape=(nl, h, k)),
            state_indices=TritonWarmupTensor(
                torch.int32,
                shape=(n,),
                strides=(compile_key.stride_state_indices,),
            ),
            commit_lens=TritonWarmupTensor(torch.int32, shape=(n,)),
            final_state_indices=TritonWarmupTensor(torch.int32, shape=(n,)),
            boundary_state_indices=TritonWarmupTensor(torch.int32, shape=(n,)),
            boundary_recovery_lens=TritonWarmupTensor(torch.int32, shape=(n,)),
            lower_bound=-1.0 if compile_key.use_lower_bound else None,
            align_mode=compile_key.align_mode,
        )

    @kernel_launcher
    def __call__(
        self,
        state_ref: torch.Tensor,
        state_base_addrs: torch.Tensor,
        state_block_strides: torch.Tensor,
        correction_cache_ref: torch.Tensor,
        correction_cache_base_addrs: torch.Tensor,
        correction_cache_block_strides: torch.Tensor,
        kg_cache_ref: torch.Tensor,
        kg_cache_base_addrs: torch.Tensor,
        kg_cache_block_strides: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state_indices: torch.Tensor,
        commit_lens: torch.Tensor,
        final_state_indices: torch.Tensor,
        boundary_state_indices: torch.Tensor,
        boundary_recovery_lens: torch.Tensor,
        *,
        lower_bound: float | None,
        align_mode: bool,
    ) -> LaunchSpec:
        batch = state_indices.shape[0]
        num_layers = state_base_addrs.shape[0]
        num_heads = state_ref.shape[1]
        value_dim = state_ref.shape[2]
        key_dim = state_ref.shape[3]
        use_lower_bound = lower_bound is not None
        # Reproduce HEAD's launch geometry inline; ``dispatch`` remains an
        # independent compile-key enumeration so the JIT monitor can verify it.
        block_k = next_power_of_2(key_dim)
        block_v = min(next_power_of_2(value_dim), 32)
        grid = (
            cdiv(value_dim, block_v),
            batch,
            num_layers * num_heads,
        )
        return grid, dict(
            lower_bound=lower_bound if lower_bound is not None else 0.0,
            null_block_id=NULL_BLOCK_ID,
            stride_state_head=state_ref.stride(1),
            stride_state_v=state_ref.stride(2),
            stride_state_k=state_ref.stride(3),
            stride_correction_cache_head=correction_cache_ref.stride(1),
            stride_correction_cache_pos=correction_cache_ref.stride(2),
            stride_correction_cache_dim=correction_cache_ref.stride(3),
            stride_kg_cache_head=kg_cache_ref.stride(1),
            stride_kg_cache_pos=kg_cache_ref.stride(2),
            stride_kg_cache_dim=kg_cache_ref.stride(3),
            stride_A_layer=A_log.stride(0),
            stride_A_head=A_log.stride(1),
            stride_dt_bias_layer=dt_bias.stride(0),
            stride_dt_bias_head=dt_bias.stride(1),
            stride_dt_bias_dim=dt_bias.stride(2),
            stride_state_indices=state_indices.stride(0),
            K=key_dim,
            V=value_dim,
            BK=block_k,
            BV=block_v,
            NUM_HEADS=num_heads,
            USE_LOWER_BOUND=use_lower_bound,
            ALIGN_MODE=align_mode,
            num_warps=4,
            num_stages=2,
        )


@dataclass
class KDARecoverSSMCommitContext:
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
    boundary_recovery_lens: torch.Tensor
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
    ) -> "KDARecoverSSMCommitContext":
        if not layers:
            raise ValueError("KDA RecoverSSM commit requires at least one layer")
        if any(len(layer.kv_cache) != 4 for layer in layers):
            raise ValueError(
                "KDA RecoverSSM pages must contain conv, state, correction, "
                "and key/gate"
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
            raise ValueError("KDA RecoverSSM layers need matching gate bounds")

        state_ref = checkpoints[0]
        if state_ref.ndim != 4:
            raise ValueError("KDA RecoverSSM checkpoint must be four-dimensional")
        num_blocks, num_heads, value_dim, key_dim = state_ref.shape
        for state in checkpoints:
            if (
                state.shape != state_ref.shape
                or state.dtype != state_ref.dtype
                or state.device != state_ref.device
                or state.stride()[1:] != state_ref.stride()[1:]
            ):
                raise ValueError(
                    "KDA RecoverSSM layers need matching checkpoint layout"
                )
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
                    "KDA RecoverSSM correction buffers need float32 shape "
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
                    f"KDA RecoverSSM key/gate buffers need shape {expected_kg_shape}"
                )
        if any(param.shape != (num_heads,) for param in A_log):
            raise ValueError("KDA RecoverSSM A_log shape is incompatible")
        if any(param.shape != (num_heads, key_dim) for param in dt_bias):
            raise ValueError("KDA RecoverSSM dt_bias shape is incompatible")

        conv_ref = conv_states[0]
        if conv_ref.ndim != 3:
            raise ValueError("KDA RecoverSSM conv state must be three-dimensional")
        conv_dim, conv_state_len = conv_ref.shape[1:]
        conv_history_len = conv_state_len - spec_query_len + 1
        if conv_history_len <= 0:
            raise ValueError("KDA RecoverSSM conv state is shorter than its window")
        for conv_state in conv_states:
            if (
                conv_state.shape != conv_ref.shape
                or conv_state.dtype != conv_ref.dtype
                or conv_state.device != state_ref.device
                or conv_state.shape[0] != num_blocks
            ):
                raise ValueError("KDA RecoverSSM layers need matching conv state")

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
            boundary_recovery_lens=torch.empty(
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
            raise ValueError("KDA RecoverSSM commit batch exceeds its plan capacity")
        if query_start_loc.shape[0] != batch + 1:
            raise ValueError("KDA RecoverSSM commit metadata is incompatible")
        if request_indices is not None and request_indices.shape[0] < batch:
            raise ValueError("KDA RecoverSSM request mapping is too short")
        align_args = (block_table, num_computed_tokens, mamba_block_size)
        if any(arg is not None for arg in align_args) and any(
            arg is None for arg in align_args
        ):
            raise ValueError("KDA RecoverSSM align metadata is incomplete")
        if mamba_block_size is not None and mamba_block_size < self.spec_query_len:
            raise ValueError(
                "KDA RecoverSSM align block size must cover one speculative window"
            )
        if block_table is not None and block_table.ndim != 2:
            raise ValueError("KDA RecoverSSM block table must be two-dimensional")
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
            raise ValueError("KDA RecoverSSM commit inputs must be on the same device")

        conv_ref = self.conv_states[0]
        conv_dim = conv_ref.shape[1]
        align_mode = block_table is not None
        _PREPARE_COMMIT_PLAN_KERNEL(
            num_accepted_tokens,
            request_indices,
            state_indices,
            query_start_loc,
            block_table,
            num_computed_tokens,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            spec_query_len=self.spec_query_len,
            mamba_block_size=mamba_block_size or 1,
        )
        _COMPACT_CONV_STATE_KERNEL(
            conv_ref,
            self.conv_state_base_addrs,
            self.conv_state_block_strides,
            self.conv_state_dim_strides,
            self.conv_state_token_strides,
            state_indices,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            conv_dim=conv_dim,
            conv_history_len=self.conv_history_len,
            align_mode=align_mode,
        )
        _COMMIT_KDA_STATE_KERNEL(
            self.checkpoints[0],
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
            self.boundary_recovery_lens,
            lower_bound=self.lower_bound,
            align_mode=align_mode,
        )


_RECOVERSSM_VERIFY_KERNEL = KimiK3RecoverSSMVerifyKernel()
_PREPARE_COMMIT_PLAN_KERNEL = KimiK3RecoverSSMPrepareCommitPlanKernel()
_COMPACT_CONV_STATE_KERNEL = KimiK3RecoverSSMCompactConvStateKernel()
_COMMIT_KDA_STATE_KERNEL = KimiK3RecoverSSMCommitKdaStateKernel()


__all__ = ["KDARecoverSSMCommitContext", "kda_recoverssm_verify"]
