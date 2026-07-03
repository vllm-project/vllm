# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

import torch

from vllm.model_executor.layers.mamba.ops.mamba_ssm import softplus
from vllm.model_executor.layers.mamba.ops.replayssm_config import get_replayssm_config
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


# ======================================================================
# Fused scatter + precompute  (grid: (batch, ngroups))
#
# Scatters all conv_dim channels (x|B|C, partitioned by group) + dt of the
# fresh spec tokens into the circular post-conv / dt caches at
# ``(origin + write_pos + s) % buf``, and computes ``bc[k, s] = B_full[k] . C[s]``
# over the window (history B from the cache + fresh spec B, no read-back).
# ======================================================================
@triton.heuristics(
    {"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])}
)
@triton.jit
def _fused_scatter_precompute_kernel(
    conv_out_ptr,  # (total_tokens, conv_dim) packed channel-last conv output
    dt_spec_ptr,  # (total_tokens, nheads) packed raw dt
    post_conv_cache_ptr,  # (num_blocks, buf, conv_dim) circular paged
    dt_cache_ptr,  # (num_blocks, nheads, buf) circular paged
    write_pos_ptr,  # (num_state_slots,) block-keyed
    post_origin_ptr,  # (num_state_slots,) block-keyed
    bc_pre_ptr,  # (max_bs, ngroups, max_cache_len, block_spec) dense per-row scratch
    state_batch_indices_ptr,  # (batch,) physical block per dense decode row
    query_start_loc_ptr,  # (batch + 1,) packed token offsets
    null_block_id,
    batch,
    ngroups,
    nheads,
    dstate,
    d_inner,
    conv_dim,
    max_cache_len,
    stride_conv_out_tok,
    stride_conv_out_c,
    stride_dt_spec_tok,
    stride_dt_spec_h,
    stride_post_conv_cache_b,
    stride_post_conv_cache_pos,
    stride_post_conv_cache_c,
    stride_dt_cache_b,
    stride_dt_cache_h,
    stride_dt_cache_pos,
    stride_bc_pre_batch,
    stride_bc_pre_group,
    stride_bc_pre_pos,
    stride_bc_pre_spec,
    stride_state_indices_batch,
    RATIO: tl.constexpr,
    RATIO_P: tl.constexpr,
    NCX: tl.constexpr,
    BLOCK_CX: tl.constexpr,
    CACHE_BUF_LEN: tl.constexpr,
    BLOCK_SIZE_CACHE: tl.constexpr,
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
    spec_len = (eos - bos).to(tl.int32)
    write_pos = tl.load(write_pos_ptr + state_batch_idx).to(tl.int32)
    post_origin = tl.load(post_origin_ptr + state_batch_idx).to(tl.int32)

    offs_s = tl.arange(0, BLOCK_SIZE_SPEC)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    spec_valid = offs_s < spec_len
    nmask = offs_n < dstate
    phys_spec = (post_origin + write_pos + offs_s) & (CACHE_BUF_LEN - 1)

    b_c0 = d_inner + pid_g * dstate
    c_c0 = d_inner + ngroups * dstate + pid_g * dstate

    src_base = conv_out_ptr + bos * stride_conv_out_tok
    # fresh spec B / C  [S, N]
    B_spec = tl.load(
        src_base + (b_c0 + offs_n[None, :]) * stride_conv_out_c + offs_s[:, None] * stride_conv_out_tok,
        mask=spec_valid[:, None] & nmask[None, :],
        other=0.0,
    )
    C_spec = tl.load(
        src_base + (c_c0 + offs_n[None, :]) * stride_conv_out_c + offs_s[:, None] * stride_conv_out_tok,
        mask=spec_valid[:, None] & nmask[None, :],
        other=0.0,
    )
    cache_base = post_conv_cache_ptr + state_batch_idx * stride_post_conv_cache_b
    # scatter B (C is not cached; read fresh from conv_out)
    tl.store(
        cache_base + phys_spec[:, None] * stride_post_conv_cache_pos + (b_c0 + offs_n[None, :]) * stride_post_conv_cache_c,
        B_spec,
        mask=spec_valid[:, None] & nmask[None, :],
    )

    # scatter x channels owned by this group: [g*RATIO_P, (g+1)*RATIO_P)
    gx0 = pid_g * RATIO_P
    for i in tl.static_range(NCX):
        offs_cx = i * BLOCK_CX + tl.arange(0, BLOCK_CX)
        cxm = offs_cx < RATIO_P
        gx = gx0 + offs_cx
        xv = tl.load(
            src_base
            + gx[None, :] * stride_conv_out_c
            + offs_s[:, None] * stride_conv_out_tok,
            mask=spec_valid[:, None] & cxm[None, :],
            other=0.0,
        )
        tl.store(
            cache_base
            + phys_spec[:, None] * stride_post_conv_cache_pos
            + gx[None, :] * stride_post_conv_cache_c,
            xv,
            mask=spec_valid[:, None] & cxm[None, :],
        )

    # scatter dt for this group's heads
    offs_hl = tl.arange(0, BLOCK_HL)
    hlm = offs_hl < RATIO
    gh = pid_g * RATIO + offs_hl
    dt_base = dt_spec_ptr + bos * stride_dt_spec_tok
    dtv = tl.load(
        dt_base + offs_s[:, None] * stride_dt_spec_tok + gh[None, :] * stride_dt_spec_h,
        mask=spec_valid[:, None] & hlm[None, :],
        other=0.0,
    )
    dtc_base = dt_cache_ptr + state_batch_idx * stride_dt_cache_b
    tl.store(
        dtc_base
        + gh[None, :] * stride_dt_cache_h
        + phys_spec[:, None] * stride_dt_cache_pos,
        dtv,
        mask=spec_valid[:, None] & hlm[None, :],
    )

    # bc: history B from cache + fresh spec B  (no read-back of spec B)
    offs_k = tl.arange(0, BLOCK_SIZE_CACHE)
    hist_mask = offs_k < write_pos
    cache_valid = (offs_k < max_cache_len) & (offs_k < (write_pos + spec_len))
    spec_tok = (offs_k >= write_pos) & (offs_k < (write_pos + spec_len))
    spec_off = offs_k - write_pos
    phys_k = (post_origin + offs_k) & (CACHE_BUF_LEN - 1)
    B_hist = tl.load(
        cache_base + phys_k[:, None] * stride_post_conv_cache_pos + (b_c0 + offs_n[None, :]) * stride_post_conv_cache_c,
        mask=hist_mask[:, None] & nmask[None, :],
        other=0.0,
    )
    B_specrows = tl.load(
        src_base + (b_c0 + offs_n[None, :]) * stride_conv_out_c + spec_off[:, None] * stride_conv_out_tok,
        mask=spec_tok[:, None] & nmask[None, :],
        other=0.0,
    )
    B_full = tl.where(spec_tok[:, None], B_specrows, B_hist)
    B_full = tl.where(cache_valid[:, None], B_full.to(tl.float32), 0.0).to(
        conv_out_ptr.dtype.element_ty
    )
    bc = tl.dot(
        B_full,
        tl.trans(C_spec.to(conv_out_ptr.dtype.element_ty)),
        input_precision="tf32x3",
    ).to(tl.float32)
    bc_ptrs = (
        bc_pre_ptr
        + pid_b * stride_bc_pre_batch
        + pid_g * stride_bc_pre_group
        + offs_k[:, None] * stride_bc_pre_pos
        + offs_s[None, :] * stride_bc_pre_spec
    )
    tl.store(
        bc_ptrs,
        bc.to(bc_pre_ptr.dtype.element_ty),
        mask=cache_valid[:, None] & spec_valid[None, :],
    )


# ======================================================================
# Verify launch (non-flush rows): per-draft output from the fixed checkpoint
# S_0, no state write. dstate-tiled. Flush rows early-exit.
# ======================================================================
@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics({"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])})
@triton.jit
def _replayssm_spec_nf_kernel(
    state_ptr, x_cache_ptr, dt_cache_ptr, B_cache_ptr, C_src_ptr, bc_pre_ptr,
    D_ptr, z_ptr, dt_bias_ptr, A_ptr, out_ptr, is_flush_flags_ptr, write_pos_ptr,
    post_origin_ptr, state_batch_indices_ptr, query_start_loc_ptr, null_block_id,
    batch, nheads, dim, dstate, max_cache_len, nheads_ngroups_ratio,
    stride_state_batch, stride_state_head, stride_state_dim, stride_state_dstate,
    stride_x_cache_batch, stride_x_cache_head, stride_x_cache_dim, stride_x_cache_pos,
    stride_dt_cache_batch, stride_dt_cache_head, stride_dt_cache_pos,
    stride_B_cache_batch, stride_B_cache_group, stride_B_cache_dstate, stride_B_cache_pos,
    stride_C_src_tok, stride_C_src_c,
    stride_bc_pre_batch, stride_bc_pre_group, stride_bc_pre_pos, stride_bc_pre_spec,
    stride_D_head, stride_D_dim, stride_z_tok, stride_z_head, stride_z_dim,
    stride_dt_bias_head, stride_A_head, stride_out_tok, stride_out_head, stride_out_dim,
    stride_state_indices_batch,
    DT_SOFTPLUS: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_CACHE: tl.constexpr,
    BLOCK_SIZE_SPEC: tl.constexpr, HAS_DT_BIAS: tl.constexpr, HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr, CACHE_BUF_LEN: tl.constexpr, DSTATE_TILE: tl.constexpr,
    NDS: tl.constexpr, BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    state_batch_idx = tl.load(state_batch_indices_ptr + pid_b * stride_state_indices_batch).to(tl.int64)
    if state_batch_idx == null_block_id:
        return
    if tl.load(is_flush_flags_ptr + state_batch_idx) != 0:
        return
    bos = tl.load(query_start_loc_ptr + pid_b).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + pid_b + 1).to(tl.int64)
    spec_len = (eos - bos).to(tl.int32)
    write_pos = tl.load(write_pos_ptr + state_batch_idx).to(tl.int32)
    post_origin = tl.load(post_origin_ptr + state_batch_idx).to(tl.int32)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_CACHE)
    offs_s = tl.arange(0, BLOCK_SIZE_SPEC)
    offs_nt = tl.arange(0, DSTATE_TILE)
    spec_valid_mask = offs_s < spec_len
    hist_mask = offs_k < write_pos
    cache_valid_mask = (offs_k < max_cache_len) & (offs_k < (write_pos + spec_len))
    spec_token_mask = (offs_k >= write_pos) & (offs_k < (write_pos + spec_len))
    spec_cache_pos = write_pos + offs_s
    spec_prefix_mask = spec_valid_mask[:, None] & spec_token_mask[None, :] & (offs_k[None, :] <= spec_cache_pos[:, None])
    phys_k = (post_origin + offs_k) & (CACHE_BUF_LEN - 1)
    phys_spec = (post_origin + spec_cache_pos) & (CACHE_BUF_LEN - 1)

    state_ptr += state_batch_idx * stride_state_batch + pid_h * stride_state_head
    x_cache_ptr += state_batch_idx * stride_x_cache_batch + pid_h * stride_x_cache_head
    dt_cache_ptr += state_batch_idx * stride_dt_cache_batch + pid_h * stride_dt_cache_head
    C_src_ptr += bos * stride_C_src_tok + (pid_h // nheads_ngroups_ratio) * dstate * stride_C_src_c
    bc_pre_ptr += pid_b * stride_bc_pre_batch + (pid_h // nheads_ngroups_ratio) * stride_bc_pre_group
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

    # dt over the window (+ bias / softplus), then the per-draft decay weights.
    dt_blk = tl.load(dt_cache_ptr + phys_k * stride_dt_cache_pos, mask=cache_valid_mask, other=0.0).to(tl.float32)
    dt_blk = tl.where(cache_valid_mask, dt_blk, 0.0)
    if HAS_DT_BIAS:
        dt_blk = tl.where(cache_valid_mask, dt_blk + dt_bias_val, 0.0)
    if DT_SOFTPLUS:
        dt_blk = tl.where(cache_valid_mask, tl.where(dt_blk <= 20.0, softplus(dt_blk), dt_blk), 0.0)
    dt_cum = tl.cumsum(dt_blk, axis=0)
    hist_total = tl.sum(tl.where(hist_mask, dt_blk, 0.0), axis=0)
    spec_cum = tl.sum(tl.where(spec_prefix_mask, dt_blk[None, :], 0.0), axis=1)
    spec_cum = tl.where(spec_valid_mask, spec_cum, 0.0)
    spec_total = hist_total + spec_cum
    checkpoint_decay = tl.where(spec_valid_mask, tl.exp(tl.minimum(A_val * spec_total, 0.0)), 0.0)

    # Causal weighted sum over cached values: spec_contrib = x_cache @ factor.
    x_blk = tl.load(x_cache_ptr + phys_k[None, :] * stride_x_cache_pos + offs_m[:, None] * stride_x_cache_dim, mask=(offs_m[:, None] < dim) & cache_valid_mask[None, :], other=0.0)
    x_ty = x_blk.to(x_cache_ptr.dtype.element_ty)
    bc = tl.load(bc_pre_ptr + offs_k[:, None] * stride_bc_pre_pos + offs_s[None, :] * stride_bc_pre_spec, mask=cache_valid_mask[:, None] & spec_valid_mask[None, :], other=0.0).to(tl.float32)
    spec_scale = dt_blk[:, None] * tl.exp(tl.minimum(A_val * (spec_total[None, :] - dt_cum[:, None]), 0.0))
    causal = spec_valid_mask[None, :] & cache_valid_mask[:, None] & (offs_k[:, None] <= spec_cache_pos[None, :])
    factor = tl.where(causal, bc * spec_scale, 0.0)
    spec_contrib = tl.dot(x_ty, factor.to(x_cache_ptr.dtype.element_ty), input_precision="tf32x3").to(tl.float32)

    # Decayed checkpoint readout S_0 @ C, dstate-tiled. tf32x3 keeps fp32-act
    # parity; bf16 act uses single-pass tf32 (the flag is a no-op on bf16 inputs).
    checkpoint_out = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_SPEC], dtype=tl.float32)
    for i in tl.static_range(NDS):
        offs_n = i * DSTATE_TILE + offs_nt
        nmask = offs_n < dstate
        st = tl.load(state_ptr + offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate, mask=(offs_m[:, None] < dim) & nmask[None, :], other=0.0).to(tl.float32)
        c_mask = spec_valid_mask[:, None] & nmask[None, :]
        c_chunk = tl.load(C_src_ptr + offs_s[:, None] * stride_C_src_tok + offs_n[None, :] * stride_C_src_c, mask=c_mask, other=0.0).to(tl.float32)
        if x_cache_ptr.dtype.element_ty == tl.float32:
            checkpoint_out += tl.dot(st, tl.trans(c_chunk), input_precision="tf32x3").to(tl.float32)
        else:
            checkpoint_out += tl.dot(st, tl.trans(c_chunk), input_precision="tf32").to(tl.float32)
    checkpoint_out *= checkpoint_decay[None, :]
    out = tl.trans(checkpoint_out + spec_contrib)

    if HAS_D:
        x_spec_sm = tl.load(x_cache_ptr + offs_m[None, :] * stride_x_cache_dim + phys_spec[:, None] * stride_x_cache_pos, mask=spec_valid_mask[:, None] & (offs_m[None, :] < dim), other=0.0).to(tl.float32)
        D_val = tl.load(D_ptr + offs_m * stride_D_dim, mask=offs_m < dim, other=0.0).to(tl.float32)
        out += x_spec_sm * D_val[None, :]
    if HAS_Z:
        z_val = tl.load(z_ptr + offs_s[:, None] * stride_z_tok + offs_m[None, :] * stride_z_dim, mask=spec_valid_mask[:, None] & (offs_m[None, :] < dim), other=0.0).to(tl.float32)
        out *= z_val * tl.sigmoid(z_val)
    out = tl.where(spec_valid_mask[:, None], out, 0.0)
    tl.store(out_ptr + offs_s[:, None] * stride_out_tok + offs_m[None, :] * stride_out_dim, out, mask=spec_valid_mask[:, None] & (offs_m[None, :] < dim))


# ======================================================================
# Flush launch (flush rows): reconstruct the committed-history state S_1, store
# it as the checkpoint, and read the output via S_1 + the intra-spec window.
# Non-flush rows early-exit. dstate-tiled.
# ======================================================================
@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics({"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])})
@triton.jit
def _replayssm_spec_fl_kernel(
    state_ptr, x_cache_ptr, dt_cache_ptr, B_cache_ptr, C_src_ptr, bc_pre_ptr,
    D_ptr, z_ptr, dt_bias_ptr, A_ptr, out_ptr, is_flush_flags_ptr, write_pos_ptr,
    post_origin_ptr, state_batch_indices_ptr, query_start_loc_ptr, null_block_id,
    batch, nheads, dim, dstate, max_cache_len, nheads_ngroups_ratio,
    stride_state_batch, stride_state_head, stride_state_dim, stride_state_dstate,
    stride_x_cache_batch, stride_x_cache_head, stride_x_cache_dim, stride_x_cache_pos,
    stride_dt_cache_batch, stride_dt_cache_head, stride_dt_cache_pos,
    stride_B_cache_batch, stride_B_cache_group, stride_B_cache_dstate, stride_B_cache_pos,
    stride_C_src_tok, stride_C_src_c,
    stride_bc_pre_batch, stride_bc_pre_group, stride_bc_pre_pos, stride_bc_pre_spec,
    stride_D_head, stride_D_dim, stride_z_tok, stride_z_head, stride_z_dim,
    stride_dt_bias_head, stride_A_head, stride_out_tok, stride_out_head, stride_out_dim,
    stride_state_indices_batch,
    DT_SOFTPLUS: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_CACHE: tl.constexpr,
    BLOCK_SIZE_SPEC: tl.constexpr, HAS_DT_BIAS: tl.constexpr, HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr, CACHE_BUF_LEN: tl.constexpr, DSTATE_TILE: tl.constexpr,
    NDS: tl.constexpr, BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    state_batch_idx = tl.load(state_batch_indices_ptr + pid_b * stride_state_indices_batch).to(tl.int64)
    if state_batch_idx == null_block_id:
        return
    if tl.load(is_flush_flags_ptr + state_batch_idx) == 0:
        return
    bos = tl.load(query_start_loc_ptr + pid_b).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + pid_b + 1).to(tl.int64)
    spec_len = (eos - bos).to(tl.int32)
    write_pos = tl.load(write_pos_ptr + state_batch_idx).to(tl.int32)
    post_origin = tl.load(post_origin_ptr + state_batch_idx).to(tl.int32)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_CACHE)
    offs_s = tl.arange(0, BLOCK_SIZE_SPEC)
    offs_nt = tl.arange(0, DSTATE_TILE)
    spec_valid_mask = offs_s < spec_len
    spec_cache_pos = write_pos + offs_s
    phys_spec = (post_origin + spec_cache_pos) & (CACHE_BUF_LEN - 1)
    hist_mask = offs_k < write_pos
    phys_h = (post_origin + offs_k) & (CACHE_BUF_LEN - 1)

    state_ptr += state_batch_idx * stride_state_batch + pid_h * stride_state_head
    x_cache_ptr += state_batch_idx * stride_x_cache_batch + pid_h * stride_x_cache_head
    dt_cache_ptr += state_batch_idx * stride_dt_cache_batch + pid_h * stride_dt_cache_head
    B_cache_ptr += state_batch_idx * stride_B_cache_batch + (pid_h // nheads_ngroups_ratio) * stride_B_cache_group
    C_src_ptr += bos * stride_C_src_tok + (pid_h // nheads_ngroups_ratio) * dstate * stride_C_src_c
    bc_pre_ptr += pid_b * stride_bc_pre_batch + (pid_h // nheads_ngroups_ratio) * stride_bc_pre_group
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

    # History decay (for the S_1 reconstruction) and spec-prefix decay (output).
    dt_h = tl.load(dt_cache_ptr + phys_h * stride_dt_cache_pos, mask=hist_mask, other=0.0).to(tl.float32)
    dt_h = tl.where(hist_mask, dt_h, 0.0)
    if HAS_DT_BIAS:
        dt_h = tl.where(hist_mask, dt_h + dt_bias_val, 0.0)
    if DT_SOFTPLUS:
        dt_h = tl.where(hist_mask, tl.where(dt_h <= 20.0, softplus(dt_h), dt_h), 0.0)
    hist_cum = tl.cumsum(dt_h, axis=0)
    hist_total = tl.sum(dt_h, axis=0)
    hist_decay = tl.exp(tl.minimum(A_val * hist_total, 0.0))
    hist_scale = tl.where(hist_mask, dt_h * tl.exp(tl.minimum(A_val * (hist_total - hist_cum), 0.0)), 0.0)
    dt_s = tl.load(dt_cache_ptr + phys_spec * stride_dt_cache_pos, mask=spec_valid_mask, other=0.0).to(tl.float32)
    dt_s = tl.where(spec_valid_mask, dt_s, 0.0)
    if HAS_DT_BIAS:
        dt_s = tl.where(spec_valid_mask, dt_s + dt_bias_val, 0.0)
    if DT_SOFTPLUS:
        dt_s = tl.where(spec_valid_mask, tl.where(dt_s <= 20.0, softplus(dt_s), dt_s), 0.0)
    spec_cum = tl.cumsum(dt_s, axis=0)
    spec_decay = tl.where(spec_valid_mask, tl.exp(tl.minimum(A_val * spec_cum, 0.0)), 0.0)
    x_hist = tl.load(x_cache_ptr + phys_h[None, :] * stride_x_cache_pos + offs_m[:, None] * stride_x_cache_dim, mask=(offs_m[:, None] < dim) & hist_mask[None, :], other=0.0)
    x_hist_ty = x_hist.to(x_cache_ptr.dtype.element_ty)

    # Reconstruct S_1 = S_0 * hist_decay + (x_hist @ scaled B_hist), store it, and
    # accumulate the checkpoint readout S_1 @ C. dstate-tiled.
    checkpoint_out = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_SPEC], dtype=tl.float32)
    for i in tl.static_range(NDS):
        offs_n = i * DSTATE_TILE + offs_nt
        nmask = offs_n < dstate
        B_block = tl.load(B_cache_ptr + phys_h[:, None] * stride_B_cache_pos + offs_n[None, :] * stride_B_cache_dstate, mask=hist_mask[:, None] & nmask[None, :], other=0.0)
        B_hist_scaled = (tl.where(hist_mask[:, None], B_block.to(tl.float32), 0.0) * hist_scale[:, None]).to(x_cache_ptr.dtype.element_ty)
        delta = tl.dot(x_hist_ty, B_hist_scaled, input_precision="tf32x3").to(tl.float32)
        st_ptrs = state_ptr + offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
        st = tl.load(st_ptrs, mask=(offs_m[:, None] < dim) & nmask[None, :], other=0.0)
        S1 = st.to(tl.float32) * hist_decay + delta
        if write_pos > 0:
            tl.store(st_ptrs, S1.to(st.dtype), mask=(offs_m[:, None] < dim) & nmask[None, :])
        c_mask = spec_valid_mask[:, None] & nmask[None, :]
        c_chunk = tl.load(C_src_ptr + offs_s[:, None] * stride_C_src_tok + offs_n[None, :] * stride_C_src_c, mask=c_mask, other=0.0).to(tl.float32)
        if x_cache_ptr.dtype.element_ty == tl.float32:
            checkpoint_out += tl.dot(S1, tl.trans(c_chunk), input_precision="tf32x3").to(tl.float32)
        else:
            checkpoint_out += tl.dot(S1, tl.trans(c_chunk), input_precision="tf32").to(tl.float32)
    checkpoint_out *= spec_decay[None, :]

    # Intra-spec window contribution: intra = x_spec @ factor_intra (causal T x T).
    bc_spec = tl.load(bc_pre_ptr + (write_pos + offs_k)[:, None] * stride_bc_pre_pos + offs_s[None, :] * stride_bc_pre_spec, mask=(offs_k[:, None] < spec_len) & spec_valid_mask[None, :], other=0.0).to(tl.float32)
    dt_k = tl.load(dt_cache_ptr + ((post_origin + write_pos + offs_k) & (CACHE_BUF_LEN - 1)) * stride_dt_cache_pos, mask=offs_k < spec_len, other=0.0).to(tl.float32)
    dt_k = tl.where(offs_k < spec_len, dt_k, 0.0)
    if HAS_DT_BIAS:
        dt_k = tl.where(offs_k < spec_len, dt_k + dt_bias_val, 0.0)
    if DT_SOFTPLUS:
        dt_k = tl.where(offs_k < spec_len, tl.where(dt_k <= 20.0, softplus(dt_k), dt_k), 0.0)
    speccum_k = tl.cumsum(dt_k, axis=0)
    causal = (offs_k[:, None] < spec_len) & spec_valid_mask[None, :] & (offs_k[:, None] <= offs_s[None, :])
    decay_ks = tl.exp(tl.minimum(A_val * (spec_cum[None, :] - speccum_k[:, None]), 0.0))
    factor_intra = tl.where(causal, bc_spec * dt_k[:, None] * decay_ks, 0.0)
    x_src = tl.load(x_cache_ptr + ((post_origin + write_pos + offs_k)[None, :] & (CACHE_BUF_LEN - 1)) * stride_x_cache_pos + offs_m[:, None] * stride_x_cache_dim, mask=(offs_m[:, None] < dim) & (offs_k[None, :] < spec_len), other=0.0).to(x_cache_ptr.dtype.element_ty)
    intra = tl.dot(x_src, factor_intra.to(x_cache_ptr.dtype.element_ty), input_precision="tf32x3").to(tl.float32)
    out = tl.trans(checkpoint_out + intra)

    if HAS_D:
        x_spec_sm = tl.load(x_cache_ptr + offs_m[None, :] * stride_x_cache_dim + phys_spec[:, None] * stride_x_cache_pos, mask=spec_valid_mask[:, None] & (offs_m[None, :] < dim), other=0.0).to(tl.float32)
        D_val = tl.load(D_ptr + offs_m * stride_D_dim, mask=offs_m < dim, other=0.0).to(tl.float32)
        out += x_spec_sm * D_val[None, :]
    if HAS_Z:
        z_val = tl.load(z_ptr + offs_s[:, None] * stride_z_tok + offs_m[None, :] * stride_z_dim, mask=spec_valid_mask[:, None] & (offs_m[None, :] < dim), other=0.0).to(tl.float32)
        out *= z_val * tl.sigmoid(z_val)
    out = tl.where(spec_valid_mask[:, None], out, 0.0)
    tl.store(out_ptr + offs_s[:, None] * stride_out_tok + offs_m[None, :] * stride_out_dim, out, mask=spec_valid_mask[:, None] & (offs_m[None, :] < dim))


@triton.jit
def _advance_write_pos_origin_kernel(
    write_pos_ptr,
    post_origin_ptr,
    is_flush_ptr,
    num_accepted_ptr,
    state_batch_indices_ptr,
    null_block_id,
    batch,
    stride_state_indices_batch,
    MAX_CACHE_LEN: tl.constexpr,
    MAX_SPEC_LEN: tl.constexpr,
    CACHE_BUF_LEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    row_mask = offs < batch
    state_batch_idx = tl.load(state_batch_indices_ptr + offs * stride_state_indices_batch, mask=row_mask, other=null_block_id).to(
        tl.int64
    )
    valid = row_mask & (state_batch_idx != null_block_id)
    write_pos = tl.load(
        write_pos_ptr + state_batch_idx, mask=valid, other=0
    ).to(tl.int32)
    post_origin = tl.load(
        post_origin_ptr + state_batch_idx, mask=valid, other=0
    ).to(tl.int32)
    is_flush_cur = tl.load(
        is_flush_ptr + state_batch_idx, mask=valid, other=0
    ).to(tl.int32)
    num_accepted = tl.load(num_accepted_ptr + offs, mask=valid, other=0).to(tl.int32)
    total_commit = tl.where(valid, num_accepted, 0).to(tl.int32)
    flush_now = (total_commit > 0) & (is_flush_cur != 0)
    new_origin = tl.where(
        flush_now, (post_origin + write_pos) & (CACHE_BUF_LEN - 1), post_origin
    ).to(tl.int32)
    new_wp = tl.where(
        total_commit <= 0,
        write_pos,
        tl.where(is_flush_cur != 0, total_commit, write_pos + total_commit),
    ).to(tl.int32)
    # Early-flush one window early so every verify step satisfies
    # write_pos + spec_len <= max_cache_len (the spec window never overflows).
    next_is_flush = ((new_wp + 2 * MAX_SPEC_LEN) > MAX_CACHE_LEN).to(tl.int8)
    tl.store(post_origin_ptr + state_batch_idx, new_origin, mask=valid)
    tl.store(write_pos_ptr + state_batch_idx, new_wp, mask=valid)
    tl.store(is_flush_ptr + state_batch_idx, next_is_flush, mask=valid)


@triton.jit
def _reset_replayssm_spec_cursors_kernel(
    write_pos_ptr,
    post_origin_ptr,
    is_flush_ptr,
    first_decode_ptr,  # (batch,) int8 mask
    state_batch_indices_ptr,
    null_block_id,
    batch,
    stride_state_indices_batch,
    INIT_IS_FLUSH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    row_mask = offs < batch
    state_batch_idx = tl.load(state_batch_indices_ptr + offs * stride_state_indices_batch, mask=row_mask, other=null_block_id).to(
        tl.int64
    )
    first = tl.load(first_decode_ptr + offs, mask=row_mask, other=0).to(tl.int32)
    do_reset = row_mask & (state_batch_idx != null_block_id) & (first != 0)
    tl.store(
        write_pos_ptr + state_batch_idx,
        tl.zeros_like(state_batch_idx).to(tl.int32),
        mask=do_reset,
    )
    tl.store(
        post_origin_ptr + state_batch_idx,
        tl.zeros_like(state_batch_idx).to(tl.int32),
        mask=do_reset,
    )
    tl.store(
        is_flush_ptr + state_batch_idx,
        (tl.zeros_like(state_batch_idx) + INIT_IS_FLUSH).to(tl.int8),
        mask=do_reset,
    )


def selective_state_update_replayssm_spec(
    state_checkpoint: torch.Tensor,  # (num_blocks, H, P, N) checkpoint (flush updates in place)
    post_conv_cache: torch.Tensor,  # (num_blocks, cache_buf_len, conv_dim) circular
    dt_cache: torch.Tensor,  # (num_blocks, H, cache_buf_len) circular
    conv_out: torch.Tensor,  # (total_tokens, conv_dim) packed channel-last post-conv
    dt_spec: torch.Tensor,  # (total_tokens, H) packed raw dt
    A: torch.Tensor,  # (H, P, N) TIE_HDIM (A.stride(-1)==A.stride(-2)==0)
    write_pos: torch.Tensor,  # (num_state_slots,) int32 block-keyed cursor
    post_conv_state_pos: torch.Tensor,  # (num_state_slots,) int32 circular origin
    is_flush: torch.Tensor,  # (num_state_slots,) int8 block-keyed flag
    query_start_loc: torch.Tensor,  # (batch + 1,) int32 packed offsets
    state_batch_indices: torch.Tensor,  # (batch,) int32 physical block per row
    max_cache_len: int,  # logical flush threshold L = B + max_spec_len
    max_spec_len: int,
    d_inner: int,
    ngroups: int,
    dstate: int,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = True,
    out: torch.Tensor | None = None,
    bc_pre: torch.Tensor | None = None,
    null_block_id: int = NULL_BLOCK_ID,
) -> torch.Tensor:
    """One Mamba2 speculative verify step on the paged CIRCULAR post-conv cache.

    Hybrid conv variant: ``conv_out`` is the post-conv output of vLLM's
    ``causal_conv1d_update`` (packed channel-last ``[total_tokens, conv_dim]``).
    ``max_cache_len`` is the logical flush threshold L = B + max_spec_len; the
    physical pow2 buffer (= ``post_conv_cache.shape[1]`` = ``next_pow2(L)``) wraps
    the ring, while the history/cache tile is ``next_pow2(L - max_spec_len)``.
    Fuses scatter + bc-precompute in one ``(batch, ngroups)`` launch, then runs
    two dedicated launches (verify + flush) with device-side row routing. Cursors
    are block-keyed and advanced once per step by ``commit_replayssm_spec``.
    """
    num_blocks, nheads, dim, n_state = state_checkpoint.shape
    assert n_state == dstate
    total_tokens, conv_dim = conv_out.shape
    buf = post_conv_cache.shape[1]
    cache_buf_len = buf
    assert cache_buf_len & (cache_buf_len - 1) == 0, "cache_buf_len must be a power of two"
    assert d_inner == nheads * dim
    cache_conv_dim = d_inner + ngroups * dstate  # x|B only (no C)
    assert post_conv_cache.shape == (num_blocks, buf, cache_conv_dim)
    assert dt_cache.shape == (num_blocks, nheads, buf)
    assert dt_spec.shape == (total_tokens, nheads)
    assert A.shape == (nheads, dim, dstate) and A.stride(-1) == 0 and A.stride(-2) == 0
    batch = state_batch_indices.shape[0]
    assert query_start_loc.shape[0] == batch + 1

    if out is None:
        out = torch.empty(total_tokens, nheads, dim, device=conv_out.device, dtype=conv_out.dtype)
    if total_tokens == 0:
        return out

    L = max_cache_len
    base_block = max(1, L - max_spec_len)
    main_block = max(16, triton.next_power_of_2(base_block))  # history/cache tile
    block_spec = max(1, triton.next_power_of_2(max_spec_len))
    pre_block = max(16, triton.next_power_of_2(L))  # precompute window tile
    block_dstate = triton.next_power_of_2(dstate)

    bsm_v, nw_v, dt_v, ns_v = get_replayssm_config(
        "mamba2_spec_verify", dstate=dstate, base_block=base_block, max_spec_len=max_spec_len
    )
    bsm_f, nw_f, dt_f, ns_f = get_replayssm_config(
        "mamba2_spec_flush", dstate=dstate, base_block=base_block, max_spec_len=max_spec_len
    )
    dt_v = max(16, min(dt_v, block_dstate)); nds_v = triton.cdiv(block_dstate, dt_v)
    dt_f = max(16, min(dt_f, block_dstate)); nds_f = triton.cdiv(block_dstate, dt_f)

    if bc_pre is None:
        bc_pre = torch.empty(batch, ngroups, buf, block_spec, device=conv_out.device, dtype=conv_out.dtype)
    sis = state_batch_indices.stride(0)

    # --- fused scatter + precompute (full-window bc over [0, L)) ---
    ratio = nheads // ngroups
    ratio_p = ratio * dim
    BLOCK_CX = 256
    NCX = triton.cdiv(ratio_p, BLOCK_CX)
    block_hl = max(1, triton.next_power_of_2(ratio))
    with torch.cuda.device(conv_out.device.index):
        _fused_scatter_precompute_kernel[(batch, ngroups)](
            conv_out, dt_spec, post_conv_cache, dt_cache, write_pos, post_conv_state_pos,
            bc_pre, state_batch_indices, query_start_loc, null_block_id, batch, ngroups,
            nheads, dstate, d_inner, conv_dim, L,
            conv_out.stride(0), conv_out.stride(1), dt_spec.stride(0), dt_spec.stride(1),
            post_conv_cache.stride(0), post_conv_cache.stride(1), post_conv_cache.stride(2),
            dt_cache.stride(0), dt_cache.stride(1), dt_cache.stride(2),
            bc_pre.stride(0), bc_pre.stride(1), bc_pre.stride(2), bc_pre.stride(3), sis,
            RATIO=ratio, RATIO_P=ratio_p, NCX=NCX, BLOCK_CX=BLOCK_CX, CACHE_BUF_LEN=cache_buf_len,
            BLOCK_SIZE_CACHE=pre_block, BLOCK_SIZE_SPEC=block_spec, BLOCK_HL=block_hl, num_warps=4,
        )

    # views into the paged circular post-conv cache: x | B | C on the channel axis
    x_view = post_conv_cache[:, :, :d_inner].view(num_blocks, buf, nheads, dim).permute(0, 2, 1, 3)
    B_view = post_conv_cache[:, :, d_inner : d_inner + ngroups * dstate].view(num_blocks, buf, ngroups, dstate).permute(0, 2, 1, 3)
    # C is not cached; the kernels read it fresh from this conv_out slice.
    C_src = conv_out[:, d_inner + ngroups * dstate :]
    z_strides = (z.stride(0), z.stride(1), z.stride(2)) if z is not None else (0, 0, 0)

    def _args(bsm):
        grid = lambda META: (triton.cdiv(dim, META["BLOCK_SIZE_M"]), batch, nheads)
        return grid, (
            state_checkpoint, x_view, dt_cache, B_view, C_src, bc_pre, D, z, dt_bias, A,
            out, is_flush, write_pos, post_conv_state_pos, state_batch_indices,
            query_start_loc, null_block_id, batch, nheads, dim, dstate, L, ratio,
            state_checkpoint.stride(0), state_checkpoint.stride(1), state_checkpoint.stride(2), state_checkpoint.stride(3),
            x_view.stride(0), x_view.stride(1), x_view.stride(3), x_view.stride(2),
            dt_cache.stride(0), dt_cache.stride(1), dt_cache.stride(2),
            B_view.stride(0), B_view.stride(1), B_view.stride(3), B_view.stride(2),
            C_src.stride(0), C_src.stride(1),
            bc_pre.stride(0), bc_pre.stride(1), bc_pre.stride(2), bc_pre.stride(3),
            D.stride(0) if D is not None else 0, D.stride(1) if D is not None else 0,
            z_strides[0], z_strides[1], z_strides[2], dt_bias.stride(0) if dt_bias is not None else 0,
            A.stride(0), out.stride(0), out.stride(1), out.stride(2), sis, dt_softplus, bsm,
        )

    with torch.cuda.device(state_checkpoint.device.index):
        grid, base = _args(bsm_v)
        _replayssm_spec_nf_kernel[grid](
            *base, main_block, block_spec,
            CACHE_BUF_LEN=cache_buf_len, DSTATE_TILE=dt_v, NDS=nds_v, num_warps=nw_v, num_stages=ns_v,
        )
        grid, base = _args(bsm_f)
        _replayssm_spec_fl_kernel[grid](
            *base, main_block, block_spec,
            CACHE_BUF_LEN=cache_buf_len, DSTATE_TILE=dt_f, NDS=nds_f, num_warps=nw_f, num_stages=ns_f,
        )
    return out


def commit_replayssm_spec(
    write_pos: torch.Tensor,
    post_conv_state_pos: torch.Tensor,
    is_flush: torch.Tensor,
    num_accepted_tokens: torch.Tensor,  # (batch,) int32, INCLUDES bonus (min 1)
    state_batch_indices: torch.Tensor,  # (batch,) int32
    max_cache_len: int,  # logical flush threshold L
    max_spec_len: int,
    cache_buf_len: int | None = None,  # physical pow2 buffer next_pow2(L)
    null_block_id: int = NULL_BLOCK_ID,
) -> None:
    """CUDA-graph-safe block-keyed commit. Advances ``write_pos`` and the circular
    origin (flush = O(1) bump) per decode row, and precomputes next-step
    ``is_flush``. Maps vLLM ``num_accepted_tokens`` (incl. bonus) to the commit."""
    batch = state_batch_indices.shape[0]
    if cache_buf_len is None:
        cache_buf_len = max(1, triton.next_power_of_2(max_cache_len))
    BLOCK = max(1, triton.next_power_of_2(batch))
    with torch.cuda.device(write_pos.device.index):
        _advance_write_pos_origin_kernel[(1,)](
            write_pos,
            post_conv_state_pos,
            is_flush,
            num_accepted_tokens,
            state_batch_indices,
            null_block_id,
            batch,
            state_batch_indices.stride(0),
            MAX_CACHE_LEN=max_cache_len,
            MAX_SPEC_LEN=max_spec_len,
            CACHE_BUF_LEN=cache_buf_len,
            BLOCK_SIZE=BLOCK,
            num_warps=1,
        )


def reset_replayssm_spec_cursors(
    write_pos: torch.Tensor,
    post_conv_state_pos: torch.Tensor,
    is_flush: torch.Tensor,
    first_decode_mask: torch.Tensor,  # (batch,) int8
    state_batch_indices: torch.Tensor,  # (batch,) int32
    max_cache_len: int,  # logical flush threshold L
    max_spec_len: int,
    null_block_id: int = NULL_BLOCK_ID,
) -> None:
    """Prefill->decode reset for first-decode rows (block-keyed). Seeds the
    initial ``is_flush`` to match the steady-state early-flush cadence."""
    batch = state_batch_indices.shape[0]
    BLOCK = max(1, triton.next_power_of_2(batch))
    init_is_flush = 1 if 2 * max_spec_len > max_cache_len else 0
    with torch.cuda.device(write_pos.device.index):
        _reset_replayssm_spec_cursors_kernel[(1,)](
            write_pos,
            post_conv_state_pos,
            is_flush,
            first_decode_mask,
            state_batch_indices,
            null_block_id,
            batch,
            state_batch_indices.stride(0),
            INIT_IS_FLUSH=init_is_flush,
            BLOCK_SIZE=BLOCK,
            num_warps=1,
        )


__all__ = [
    "selective_state_update_replayssm_spec",
    "commit_replayssm_spec",
    "reset_replayssm_spec_cursors",
]
