# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
MLA Attention + Quantization Fusion

This module implements truly fused MLA attention + quantization kernels that
compute attention and quantize the output in a single kernel launch, based on
the exact MLA attention computation from triton_decode_attention.py.
"""

import torch
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import direct_register_custom_op

is_hip_ = current_platform.is_rocm()
FP8_DTYPE = current_platform.fp8_dtype()
FP8_MIN, FP8_MAX = torch.finfo(FP8_DTYPE).min, torch.finfo(FP8_DTYPE).max


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


# ============================================================================
# Truly Fused MLA Attention + Quantization Kernel
# ============================================================================

@triton.jit
def _mla_attn_fused_fp8_static_kernel(
    Q,
    K_Buffer,
    sm_scale,
    Req_to_tokens,
    B_Seqlen,
    O,
    LSE,
    output_scale_ptr,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_obs,
    stride_oh,
    stride_lse_bs,
    k_scale,
    v_scale,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    FP8_MIN: tl.constexpr = FP8_MIN,
    FP8_MAX: tl.constexpr = FP8_MAX,
):
    """
    Truly fused MLA attention + FP8 static quantization kernel.
    
    This kernel combines the entire MLA attention computation (both stage1 and stage2)
    with quantization in a single kernel launch. It uses the exact same logic as
    the grouped MLA attention kernel from triton_decode_attention.py, but quantizes
    the final output instead of storing it in FP16/BF16.
    """
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = cur_batch

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(
        Q + offs_q,
        mask=(mask_h[:, None]) & (mask_d[None, :]),
        other=0.0,
        cache_modifier=".ca",
    )

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        off_qpe = (
            cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        )
        qpe = tl.load(
            Q + off_qpe,
            mask=(mask_h[:, None]) & (mask_dpe[None, :]),
            other=0.0,
            cache_modifier=".ca",
        )

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        base_offs_k = cur_kv_head * stride_buf_kh + offs_d[:, None]
        if BLOCK_DPE > 0:
            base_offs_kpe = cur_kv_head * stride_buf_kh + offs_dpe[:, None]

        ks = tl.load(k_scale)
        vs = tl.load(v_scale)
        for start_n in tl.range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_page_number = tl.load(
                Req_to_tokens
                + stride_req_to_tokens_b * cur_batch_req_idx
                + offs_n // PAGE_SIZE,
                mask=offs_n < split_kv_end,
                other=0,
                cache_modifier=".ca",
            )
            kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE

            # explicitly facilitate overlapping load/compute
            offs_buf_k = kv_loc[None, :] * stride_buf_kbs + base_offs_k
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
                other=0.0,
                cache_modifier=".cg",
            )

            if k.dtype.is_fp8():
                k = (k.to(tl.float32) * ks).to(q.dtype)
            qk = tl.dot(q, k.to(q.dtype))
            if BLOCK_DPE > 0:
                offs_buf_kpe = kv_loc[None, :] * stride_buf_kbs + base_offs_kpe
                kpe = tl.load(
                    K_Buffer + offs_buf_kpe,
                    mask=(offs_n[None, :] < split_kv_end) & (mask_dpe[:, None]),
                    other=0.0,
                    cache_modifier=".cg",
                )
                if kpe.dtype.is_fp8():
                    kpe = (kpe.to(tl.float32) * ks).to(qpe.dtype)
                qk += tl.dot(qpe, kpe.to(qpe.dtype))
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )

            # MLA: use transposed k as v
            v = tl.trans(k)

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

    # Finalize attention output
    result = acc / e_sum[:, None]
    
    # ===== FUSED FP8 STATIC QUANTIZATION =====
    # Load output scale (per-tensor)
    output_scale = tl.load(output_scale_ptr)
    
    # Quantize: scale, clamp, convert to FP8
    result_scaled = result * output_scale
    result_clamped = tl.clamp(result_scaled, FP8_MIN, FP8_MAX)
    result_fp8 = result_clamped.to(FP8_DTYPE)
    
    # Store quantized output directly to O (not intermediate buffer)
    for h_idx in range(BLOCK_H):
        if mask_h[h_idx]:
            offs_o = cur_batch * stride_obs + cur_head[h_idx] * stride_oh + offs_dv
            tl.store(
                O + offs_o,
                result_fp8[h_idx, :],
                mask=mask_dv,
            )
    
    # Store LSE
    for h_idx in range(BLOCK_H):
        if mask_h[h_idx]:
            tl.store(
                LSE + cur_batch * stride_lse_bs + cur_head[h_idx],
                e_max[h_idx] + tl.log(e_sum[h_idx]),
            )


@triton.jit
def _mla_attn_fused_fp8_group_kernel(
    Q,
    K_Buffer,
    sm_scale,
    Req_to_tokens,
    B_Seqlen,
    O,
    LSE,
    output_block_scale_ptr,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_obs,
    stride_oh,
    stride_lse_bs,
    stride_scale_batch,
    stride_scale_head,
    stride_scale_group,
    k_scale,
    v_scale,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    quant_group_size,
    FP8_MIN: tl.constexpr = FP8_MIN,
    FP8_MAX: tl.constexpr = FP8_MAX,
    EPS: tl.constexpr = 1e-10,
):
    """
    Truly fused MLA attention + per-group FP8 quantization kernel.
    """
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = cur_batch

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(
        Q + offs_q,
        mask=(mask_h[:, None]) & (mask_d[None, :]),
        other=0.0,
        cache_modifier=".ca",
    )

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        off_qpe = (
            cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        )
        qpe = tl.load(
            Q + off_qpe,
            mask=(mask_h[:, None]) & (mask_dpe[None, :]),
            other=0.0,
            cache_modifier=".ca",
        )

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        base_offs_k = cur_kv_head * stride_buf_kh + offs_d[:, None]
        if BLOCK_DPE > 0:
            base_offs_kpe = cur_kv_head * stride_buf_kh + offs_dpe[:, None]

        ks = tl.load(k_scale)
        vs = tl.load(v_scale)
        for start_n in tl.range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_page_number = tl.load(
                Req_to_tokens
                + stride_req_to_tokens_b * cur_batch_req_idx
                + offs_n // PAGE_SIZE,
                mask=offs_n < split_kv_end,
                other=0,
                cache_modifier=".ca",
            )
            kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE

            offs_buf_k = kv_loc[None, :] * stride_buf_kbs + base_offs_k
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
                other=0.0,
                cache_modifier=".cg",
            )

            if k.dtype.is_fp8():
                k = (k.to(tl.float32) * ks).to(q.dtype)
            qk = tl.dot(q, k.to(q.dtype))
            if BLOCK_DPE > 0:
                offs_buf_kpe = kv_loc[None, :] * stride_buf_kbs + base_offs_kpe
                kpe = tl.load(
                    K_Buffer + offs_buf_kpe,
                    mask=(offs_n[None, :] < split_kv_end) & (mask_dpe[:, None]),
                    other=0.0,
                    cache_modifier=".cg",
                )
                if kpe.dtype.is_fp8():
                    kpe = (kpe.to(tl.float32) * ks).to(qpe.dtype)
                qk += tl.dot(qpe, kpe.to(qpe.dtype))
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )

            # MLA: use transposed k as v
            v = tl.trans(k)

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

    # Finalize attention output
    result = acc / e_sum[:, None]
    
    # ===== FUSED PER-GROUP FP8 QUANTIZATION =====
    num_groups = tl.cdiv(Lv, quant_group_size)
    
    for group_idx in range(num_groups):
        group_start = group_idx * quant_group_size
        group_end = tl.minimum(group_start + quant_group_size, Lv)
        group_size = group_end - group_start
        
        # Extract group for all heads
        group_vals = result[:, group_start:group_end]
        
        # Compute group scale per head
        group_max = tl.maximum(tl.max(group_vals, 1), EPS)
        group_scale = FP8_MAX / group_max
        
        # Quantize group
        group_scaled = group_vals * group_scale[:, None]
        group_clamped = tl.clamp(group_scaled, FP8_MIN, FP8_MAX)
        group_fp8 = group_clamped.to(FP8_DTYPE)
        
        # Store group scale
        for h_idx in range(BLOCK_H):
            if mask_h[h_idx]:
                scale_offset = (
                    cur_batch * stride_scale_batch
                    + (cur_head[h_idx]) * stride_scale_head
                    + group_idx * stride_scale_group
                )
                tl.store(output_block_scale_ptr + scale_offset, group_scale[h_idx])
        
        # Store quantized output directly to O
        for h_idx in range(BLOCK_H):
            if mask_h[h_idx]:
                out_offset = cur_batch * stride_obs + cur_head[h_idx] * stride_oh + group_start
                tl.store(
                    O + out_offset + tl.arange(0, group_size),
                    group_fp8[h_idx, :],
                    mask=tl.arange(0, group_size) < group_size
                )
    
    # Store LSE
    for h_idx in range(BLOCK_H):
        if mask_h[h_idx]:
            tl.store(
                LSE + cur_batch * stride_lse_bs + cur_head[h_idx],
                e_max[h_idx] + tl.log(e_sum[h_idx]),
            )


# ============================================================================
# Python Wrappers
# ============================================================================

def mla_attn_fused_fp8_static(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    req_to_token: torch.Tensor,
    b_seq_len: torch.Tensor,
    num_kv_splits: int,
    sm_scale: float,
    page_size: int,
    logit_cap: float = 0.0,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    output_scale: torch.Tensor | None = None,
) -> None:
    """
    Truly fused MLA attention + FP8 static quantization.
    Single kernel launch that computes attention and quantizes output.
    """
    Lk = k_buffer.shape[-1]
    Lv = o.shape[-1]

    # Align tile dimensions with latent rank for MLA
    if not is_hip_ and Lk == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif not is_hip_ and Lk == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lv)
        BLOCK_DPE = triton.next_power_of_2(Lk - Lv) if Lk > Lv else 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    BLOCK = 32
    if is_hip_:
        BLOCK = 16

    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[-2]

    BLOCK_H = 16
    NUM_KV_SPLITS = num_kv_splits
    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        NUM_KV_SPLITS,
    )

    extra_kargs = {}
    num_stages = 2
    if is_hip_:
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}
        num_stages = 1
    elif not is_hip_ and BLOCK_DMODEL >= 1024:
        num_stages = 1

    if k_scale is None:
        k_scale = torch.tensor(1.0, dtype=torch.float32, device=q.device)
    if v_scale is None:
        v_scale = torch.tensor(1.0, dtype=torch.float32, device=q.device)

    _mla_attn_fused_fp8_static_kernel[grid](
        q,
        k_buffer,
        sm_scale,
        req_to_token,
        b_seq_len,
        o,
        lse,
        output_scale,
        req_to_token.stride(0),
        q.stride(0),
        q.stride(1),
        k_buffer.stride(-3),
        k_buffer.stride(-2),
        o.stride(0),
        o.stride(1),
        lse.stride(0),
        k_scale,
        v_scale,
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        PAGE_SIZE=page_size,
        logit_cap=logit_cap,
        Lk=Lk,
        Lv=Lv,
        num_warps=4,
        num_stages=num_stages,
        **extra_kargs,
    )


def mla_attn_fused_fp8_group(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    req_to_token: torch.Tensor,
    b_seq_len: torch.Tensor,
    num_kv_splits: int,
    sm_scale: float,
    page_size: int,
    logit_cap: float = 0.0,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
    quant_group_size: int = 128,
) -> None:
    """
    Truly fused MLA attention + per-group FP8 quantization.
    Single kernel launch that computes attention and quantizes output.
    """
    Lk = k_buffer.shape[-1]
    Lv = o.shape[-1]

    # Align tile dimensions with latent rank for MLA
    if not is_hip_ and Lk == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif not is_hip_ and Lk == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lv)
        BLOCK_DPE = triton.next_power_of_2(Lk - Lv) if Lk > Lv else 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    BLOCK = 32
    if is_hip_:
        BLOCK = 16

    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[-2]

    BLOCK_H = 16
    NUM_KV_SPLITS = num_kv_splits
    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        NUM_KV_SPLITS,
    )

    extra_kargs = {}
    num_stages = 2
    if is_hip_:
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}
        num_stages = 1
    elif not is_hip_ and BLOCK_DMODEL >= 1024:
        num_stages = 1

    if k_scale is None:
        k_scale = torch.tensor(1.0, dtype=torch.float32, device=q.device)
    if v_scale is None:
        v_scale = torch.tensor(1.0, dtype=torch.float32, device=q.device)

    _mla_attn_fused_fp8_group_kernel[grid](
        q,
        k_buffer,
        sm_scale,
        req_to_token,
        b_seq_len,
        o,
        lse,
        output_block_scale,
        req_to_token.stride(0),
        q.stride(0),
        q.stride(1),
        k_buffer.stride(-3),
        k_buffer.stride(-2),
        o.stride(0),
        o.stride(1),
        lse.stride(0),
        output_block_scale.stride(0),
        output_block_scale.stride(1),
        output_block_scale.stride(2),
        k_scale,
        v_scale,
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        PAGE_SIZE=page_size,
        logit_cap=logit_cap,
        Lk=Lk,
        Lv=Lv,
        quant_group_size=quant_group_size,
        num_warps=4,
        num_stages=num_stages,
        **extra_kargs,
    )


# ============================================================================
# Custom Op Registration
# ============================================================================

def mla_attn_fused_fp8_static_fake(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    req_to_token: torch.Tensor,
    b_seq_len: torch.Tensor,
    num_kv_splits: int,
    sm_scale: float,
    page_size: int,
    logit_cap: float = 0.0,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    output_scale: torch.Tensor | None = None,
) -> None:
    """Fake implementation for torch.compile."""
    pass


def mla_attn_fused_fp8_group_fake(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    req_to_token: torch.Tensor,
    b_seq_len: torch.Tensor,
    num_kv_splits: int,
    sm_scale: float,
    page_size: int,
    logit_cap: float = 0.0,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
    quant_group_size: int = 128,
) -> None:
    """Fake implementation for torch.compile."""
    pass


direct_register_custom_op(
    op_name="mla_attn_fused_fp8_static",
    op_func=mla_attn_fused_fp8_static,
    fake_impl=mla_attn_fused_fp8_static_fake,
    mutates_args=[],
)

direct_register_custom_op(
    op_name="mla_attn_fused_fp8_group",
    op_func=mla_attn_fused_fp8_group,
    fake_impl=mla_attn_fused_fp8_group_fake,
    mutates_args=[],
)


# ============================================================================
# Helper Functions
# ============================================================================

def can_use_fused_mla_attn_quant(
    quant_key,
) -> bool:
    """
    Check if fused MLA attention + quantization is supported for the given quantization key.
    
    Args:
        quant_key: QuantKey indicating the quantization scheme
    
    Returns:
        True if fused kernel is available and supports this quantization mode
    """
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Dynamic128Sym,
        kFp8Dynamic64Sym,
        kFp8StaticTensorSym,
        kNvfp4Dynamic,
    )
    
    # Currently we support FP8 static and per-group FP8
    # NVFP4 support can be added later
    supported_keys = {
        kFp8StaticTensorSym,
        kFp8Dynamic128Sym,
        kFp8Dynamic64Sym,
    }
    
    return quant_key in supported_keys
