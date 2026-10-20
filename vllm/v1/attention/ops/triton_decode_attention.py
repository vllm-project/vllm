# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/sgl-project/sglang/blob/9f635ea50de920aa507f486daafba26a5b837574/python/sglang/srt/layers/attention/triton_ops/decode_attention.py
# which was originally adapted from
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage1.py
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage2.py

# Changes:
# - Add support for page size >= 1.

# Copyright 2025 vLLM Team
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Memory-efficient attention for decoding.
It supports page size >= 1.
"""

import logging

import torch
from packaging import version

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

is_hip_ = current_platform.is_rocm()

logger = logging.getLogger(__name__)

# Only print the following warnings when triton version < 3.2.0.
# The issue won't affect performance or accuracy.
if version.parse(triton.__version__) < version.parse("3.2.0"):
    logger.warning(
        "The following error message 'operation scheduled before its operands' "
        "can be ignored."
    )


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


def _page_stride(buf, page_size):
    # Stride between pages. 4D buffers have a page dim; 3D buffers pack pages
    # along the token dim, so split it out first. Read the real stride (a
    # cross-layer view has gaps), don't assume PAGE_SIZE * token stride.
    if buf.ndim == 3:
        buf = buf.unflatten(-3, (-1, page_size))
    return buf.stride(-4)


@triton.jit
def _fwd_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    Req_to_tokens,
    B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kpbs,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vpbs,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    k_scale,
    v_scale,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = cur_batch

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    q = tl.load(Q + off_q, mask=mask_d, other=0.0)

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        ks = tl.load(k_scale)
        vs = tl.load(v_scale)
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_page_number = tl.load(
                Req_to_tokens
                + stride_req_to_tokens_b * cur_batch_req_idx
                + offs_n // PAGE_SIZE,
                mask=offs_n < split_kv_end,
                other=0,
            ).to(tl.int64)  # page_number * page stride overflows int32
            kv_in_page = offs_n % PAGE_SIZE
            offs_buf_k = (
                (kv_page_number * stride_buf_kpbs + kv_in_page * stride_buf_kbs)[
                    :, None
                ]
                + cur_kv_head * stride_buf_kh
                + offs_d[None, :]
            )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[:, None] < split_kv_end) & (mask_d[None, :]),
                other=0.0,
            )
            if k.dtype.is_fp8():
                k = (k.to(tl.float32) * ks).to(q.dtype)
            qk = tl.sum(q[None, :] * k, 1)
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))

            offs_buf_v = (
                (kv_page_number * stride_buf_vpbs + kv_in_page * stride_buf_vbs)[
                    :, None
                ]
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )
            if v.dtype.is_fp8():
                v = (v.to(tl.float32) * vs).to(q.dtype)

            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            acc *= re_scale
            acc += tl.sum(p[:, None] * v, 0)

            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch.to(tl.int64) * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum,
            mask=(mask_dv),
        )

        offs_mid_o_1 = (
            cur_batch.to(tl.int64) * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + Lv
        )

        tl.store(
            Att_Out + offs_mid_o_1,
            e_max + tl.log(e_sum),
        )


def _decode_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    Req_to_tokens,
    B_Seqlen,
    num_kv_splits,
    sm_scale,
    page_size,
    logit_cap,
    k_scale,
    v_scale,
):
    BLOCK = 64 if not is_hip_ else 8

    NUM_KV_SPLITS = num_kv_splits
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    batch, head_num = q.shape[0], q.shape[1]

    grid = (batch, head_num, NUM_KV_SPLITS)
    kv_group_num = q.shape[1] // k_buffer.shape[-2]

    num_warps = 4
    if kv_group_num != 1:
        num_warps = 1 if is_hip_ else 2

    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DV = triton.next_power_of_2(Lv)

    _fwd_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        Req_to_tokens,
        B_Seqlen,
        att_out,
        Req_to_tokens.stride(0),
        q.stride(0),
        q.stride(1),
        _page_stride(k_buffer, page_size),
        k_buffer.stride(-3),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        k_buffer.stride(-2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        _page_stride(v_buffer, page_size),
        v_buffer.stride(-3),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        v_buffer.stride(-2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        k_scale,
        v_scale,
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        PAGE_SIZE=page_size,
        logit_cap=logit_cap,
        num_warps=num_warps,
        num_stages=2,
        Lk=Lk,
        Lv=Lv,
    )


@triton.jit
def _fwd_grouped_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    Req_to_tokens,
    B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kpbs,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vpbs,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
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
    IS_MLA: tl.constexpr = False,
):
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
        base_offs_v = cur_kv_head * stride_buf_vh + offs_dv[None, :]
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
            ).to(tl.int64)  # page_number * page stride overflows int32
            kv_off_k = (
                kv_page_number * stride_buf_kpbs + (offs_n % PAGE_SIZE) * stride_buf_kbs
            )

            # explicitly facilitate overlapping load/compute
            offs_buf_k = kv_off_k[None, :] + base_offs_k
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
                offs_buf_kpe = kv_off_k[None, :] + base_offs_kpe
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

            if not IS_MLA:
                kv_off_v = (
                    kv_page_number * stride_buf_vpbs
                    + (offs_n % PAGE_SIZE) * stride_buf_vbs
                )
                offs_buf_v = kv_off_v[:, None] + base_offs_v
                v = tl.load(
                    V_Buffer + offs_buf_v,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                    other=0.0,
                )
                if v.dtype.is_fp8():
                    v = (v.to(tl.float32) * vs).to(q.dtype)
            else:
                # MLA uses a single c_kv.
                # loading the same c_kv to interpret it as v is not necessary.
                # transpose the existing c_kv (aka k) for the dot product.
                v = tl.trans(k)

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch.to(tl.int64) * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv[None, :]
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        offs_mid_o_1 = (
            cur_batch.to(tl.int64) * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + Lv
        )

        tl.store(
            Att_Out + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


def _decode_grouped_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    Req_to_tokens,
    B_Seqlen,
    num_kv_splits,
    sm_scale,
    page_size,
    logit_cap,
    k_scale,
    v_scale,
    is_mla=False,
):
    # with is_mla there is only a single c_kv in smem.
    # could increase BLOCK or num_stages.
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    # Align tile dimensions with latent rank for MLA to avoid shape mismatch.
    if is_mla:
        if not is_hip_ and Lk == 576:
            BLOCK_DMODEL = 512
            BLOCK_DPE = 64
        elif not is_hip_ and Lk == 288:
            BLOCK_DMODEL = 256
            BLOCK_DPE = 32
        else:
            BLOCK_DMODEL = triton.next_power_of_2(Lv)
            BLOCK_DPE = triton.next_power_of_2(Lk - Lv) if Lk > Lv else 0
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
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
        # https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html#mi300x-triton-kernel-performance-optimization
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}
        num_stages = 1
    elif not is_hip_ and BLOCK_DMODEL >= 1024:
        # Avoid shared memory overflow on NVIDIA when BLOCK_DMODEL is large
        # like non-MLA D_QK=576, BLOCK_DMODEL=1024, BLOCK_H=16
        # exceeds 101376 bytes limit
        num_stages = 1

    _fwd_grouped_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        Req_to_tokens,
        B_Seqlen,
        att_out,
        Req_to_tokens.stride(0),
        q.stride(0),
        q.stride(1),
        _page_stride(k_buffer, page_size),
        k_buffer.stride(-3),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        k_buffer.stride(-2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        _page_stride(v_buffer, page_size),
        v_buffer.stride(-3),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        v_buffer.stride(-2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
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
        num_warps=4,
        num_stages=num_stages,
        Lk=Lk,
        Lv=Lv,
        IS_MLA=is_mla,
        **extra_kargs,
    )


@triton.jit
def _fwd_kernel_stage2(
    Mid_O,
    o,
    lse,
    B_Seqlen,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    stride_lse_bs,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
    OUTPUT_FP16: tl.constexpr = 0,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch.to(tl.int64) * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch.to(tl.int64) * stride_mid_ob + cur_head * stride_mid_oh + Lv

    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(
                Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0
            )
            tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    result = acc / e_sum
    if OUTPUT_FP16:
        result = result.to(tl.float16)
    tl.store(
        o + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        result,
        mask=mask_d,
    )
    lse_val = e_max + tl.log(e_sum)
    tl.store(
        lse + cur_batch * stride_lse_bs + cur_head,
        lse_val,
    )


def _decode_softmax_reducev_fwd(
    logits,
    q,
    o,
    lse,
    v_buffer,
    b_seq_len,
    num_kv_splits,
):
    batch, head_num = q.shape[0], q.shape[1]
    Lv = v_buffer.shape[-1]
    BLOCK_DV = triton.next_power_of_2(Lv)

    NUM_KV_SPLITS = num_kv_splits

    extra_kargs = {}
    if is_hip_:
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}

    grid = (batch, head_num)
    _fwd_kernel_stage2[grid](
        logits,
        o,
        lse,
        b_seq_len,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        o.stride(0),
        o.stride(1),
        lse.stride(0),
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        num_warps=4,
        num_stages=2,
        **extra_kargs,
    )


def decode_attention_fwd_normal(
    q,
    k_buffer,
    v_buffer,
    o,
    lse,
    req_to_token,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    page_size,
    logit_cap=0.0,
    k_scale=None,
    v_scale=None,
):
    _decode_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        req_to_token,
        b_seq_len,
        num_kv_splits,
        sm_scale,
        page_size,
        logit_cap,
        k_scale,
        v_scale,
    )
    _decode_softmax_reducev_fwd(
        attn_logits, q, o, lse, v_buffer, b_seq_len, num_kv_splits
    )


def decode_attention_fwd_grouped(
    q,
    k_buffer,
    v_buffer,
    o,
    lse,
    req_to_token,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    page_size,
    logit_cap=0.0,
    k_scale=None,
    v_scale=None,
    is_mla=False,
):
    _decode_grouped_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        req_to_token,
        b_seq_len,
        num_kv_splits,
        sm_scale,
        page_size,
        logit_cap,
        k_scale,
        v_scale,
        is_mla=is_mla,
    )
    _decode_softmax_reducev_fwd(
        attn_logits, q, o, lse, v_buffer, b_seq_len, num_kv_splits
    )


def decode_attention_fwd(
    q,
    k_buffer,
    v_buffer,
    o,
    lse,
    req_to_token,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    page_size=1,
    logit_cap=0.0,
    k_scale=None,
    v_scale=None,
    is_mla=False,
):
    assert num_kv_splits == attn_logits.shape[2]

    if k_scale is None:
        k_scale = torch.tensor(1.0, dtype=torch.float32, device=q.device)
    if v_scale is None:
        v_scale = torch.tensor(1.0, dtype=torch.float32, device=q.device)

    kv_group_num = q.shape[1] // v_buffer.shape[-2]

    if kv_group_num == 1:
        # MHA
        decode_attention_fwd_normal(
            q,
            k_buffer,
            v_buffer,
            o,
            lse,
            req_to_token,
            b_seq_len,
            attn_logits,
            num_kv_splits,
            sm_scale,
            page_size,
            logit_cap,
            k_scale,
            v_scale,
        )
    else:
        # GQA/MQA/MLA
        decode_attention_fwd_grouped(
            q,
            k_buffer,
            v_buffer,
            o,
            lse,
            req_to_token,
            b_seq_len,
            attn_logits,
            num_kv_splits,
            sm_scale,
            page_size,
            logit_cap,
            k_scale,
            v_scale,
            is_mla=is_mla,
        )


# ---------------------------------------------------------------------------
# Folded causal verify (MLA, single latent KV head).
#
# One stage-1 program owns (request, row block, split). The rows of a request
# are the flattened (verify token t, head h) pairs r = t * H + h, so a program
# loads each KV tile of the request once for BLOCK_H rows and applies the
# causal limit lim_r = min(L - qlen + 1 + t_r, L_real) per row. Splits
# partition [0, max_r lim_r) identically in both stages. Partials and the LSE
# are kept in the log2 domain inside the workspace; the final LSE is natural.
# With a single split, stage 1 writes o / lse directly and stage 2 is skipped.
# ---------------------------------------------------------------------------

_LOG2E = 1.4426950408889634
_LN2 = 0.6931471805599453


@triton.jit
def _verify_split_len(
    L_split,
    NUM_KV_SPLITS,
    SPLIT_ALIGN: tl.constexpr,
    MIN_SPLIT_LEN: tl.constexpr,
):
    # NUM_KV_SPLITS is a static upper bound (sized for cudagraphs); short
    # requests use fewer, longer splits so tiny splits do not dominate.
    per = tl.maximum(tl.cdiv(L_split, NUM_KV_SPLITS), MIN_SPLIT_LEN)
    return tl.cdiv(per, SPLIT_ALIGN) * SPLIT_ALIGN


@triton.jit
def _verify_tile_off(
    pt,
    start_n,
    split_end,
    stride_buf_kpbs,
    stride_buf_kbs,
    BLOCK_N: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    # Row offsets of keys [start_n, start_n + BLOCK_N); masked keys read
    # nothing, so a tile past split_end is free.
    offs_n = start_n + tl.arange(0, BLOCK_N)
    mask_n = offs_n < split_end
    kv_page_number = tl.load(pt + offs_n // PAGE_SIZE, mask=mask_n, other=0).to(
        tl.int64
    )  # page_number * page stride overflows int32
    return kv_page_number * stride_buf_kpbs + (offs_n % PAGE_SIZE) * stride_buf_kbs


@triton.jit
def _verify_load_tile_qsplit(
    K_Buffer,
    kv_off,
    mask_n,
    offs_d0,
    offs_d,
    offs_dpe,
    HALF: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
):
    # [HALF, N] K halves (QK operands), [N, 2 * HALF] V (PV operand, loaded
    # in its own layout) and the [DPE, N] rope part.
    k0 = tl.load(
        K_Buffer + kv_off[None, :] + offs_d0[:, None], mask=mask_n[None, :], other=0.0
    )
    k1 = tl.load(
        K_Buffer + kv_off[None, :] + HALF + offs_d0[:, None],
        mask=mask_n[None, :],
        other=0.0,
    )
    v = tl.load(
        K_Buffer + kv_off[:, None] + offs_d[None, :], mask=mask_n[:, None], other=0.0
    )
    if BLOCK_DPE > 0:
        kpe = tl.load(
            K_Buffer + kv_off[None, :] + offs_dpe[:, None],
            mask=mask_n[None, :],
            other=0.0,
        )
    else:
        kpe = k0
    return k0, k1, v, kpe


@triton.jit
def _fwd_verify_kernel_stage1(
    Q,
    K_Buffer,
    sm_scale_log2,
    Req_to_tokens,
    B_Seqlen,
    Real_Seqlen,
    Att_Out,
    Out,
    Lse,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kpbs,
    stride_buf_kbs,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    stride_lse_bs,
    stride_lse_h,
    k_scale,
    BATCH,
    QLEN: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HAS_REAL: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS,
    SPLIT_ALIGN: tl.constexpr,
    MIN_SPLIT_LEN: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    Lv: tl.constexpr,
    QSPLIT: tl.constexpr,
    SINGLE: tl.constexpr,
):
    # 1D grid ordered (split, request, row block), row blocks fastest: the row
    # blocks of one request run back to back and share its KV tiles through
    # L2. The row-block count is constexpr so the decode needs no runtime
    # division by it.
    NUM_ROW_BLOCKS: tl.constexpr = (QLEN * NUM_HEADS + BLOCK_H - 1) // BLOCK_H
    pid = tl.program_id(0)
    row_block = pid % NUM_ROW_BLOCKS
    group = pid // NUM_ROW_BLOCKS
    cur_batch = group % BATCH
    split_kv_id = group // BATCH

    rows = row_block * BLOCK_H + tl.arange(0, BLOCK_H)
    row_valid = rows < QLEN * NUM_HEADS
    tok = rows // NUM_HEADS
    head = rows % NUM_HEADS

    seq_len = tl.load(B_Seqlen + cur_batch)
    L_cap = seq_len
    if HAS_REAL:
        L_cap = tl.minimum(seq_len, tl.load(Real_Seqlen + cur_batch))
    L_cap = tl.maximum(L_cap, 0)
    lim = tl.minimum(seq_len - QLEN + 1 + tok, L_cap)
    lim = tl.where(row_valid, tl.maximum(lim, 0), 0)

    per_split = _verify_split_len(L_cap, NUM_KV_SPLITS, SPLIT_ALIGN, MIN_SPLIT_LEN)
    split_start = per_split * split_kv_id
    split_end = tl.minimum(split_start + per_split, tl.max(lim, 0))

    # SINGLE (num_kv_splits == 1): o / lse are written here, rows without keys
    # get o = 0, lse = -inf, and stage 2 is not launched.
    if (split_end > split_start) | SINGLE:
        q_row = cur_batch.to(tl.int64) * QLEN + tok
        offs_d = tl.arange(0, BLOCK_DMODEL)
        offs_q = q_row[:, None] * stride_qbs + head[:, None] * stride_qh
        HALF: tl.constexpr = BLOCK_DMODEL // 2
        offs_d0 = tl.arange(0, HALF)
        if QSPLIT:
            # Two half-width Q tiles: halves the LDS scratch of the Q layout
            # conversion at large BLOCK_H.
            q = tl.load(
                Q + offs_q + offs_d0[None, :], mask=row_valid[:, None], other=0.0
            )
            q1 = tl.load(
                Q + offs_q + HALF + offs_d0[None, :],
                mask=row_valid[:, None],
                other=0.0,
            )
        else:
            q = tl.load(
                Q + offs_q + offs_d[None, :],
                mask=row_valid[:, None],
                other=0.0,
            )
        if BLOCK_DPE > 0:
            offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
            qpe = tl.load(
                Q + offs_q + offs_dpe[None, :],
                mask=row_valid[:, None],
                other=0.0,
            )
        else:
            offs_dpe = offs_d0
        row_end = tl.minimum(lim, split_end)

        e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
        e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc = tl.zeros([BLOCK_H, BLOCK_DMODEL], dtype=tl.float32)
        ks = tl.load(k_scale)
        pt = Req_to_tokens + stride_req_to_tokens_b * cur_batch.to(tl.int64)
        if QSPLIT:
            # Software pipeline: the next K/V tile is loaded before this
            # tile's math and the block table is read two tiles ahead, so
            # global latency overlaps the MFMAs (one wave per SIMD here).
            nkv_off = _verify_tile_off(
                pt,
                split_start,
                split_end,
                stride_buf_kpbs,
                stride_buf_kbs,
                BLOCK_N,
                PAGE_SIZE,
            )
            nmask = split_start + tl.arange(0, BLOCK_N) < split_end
            nk0, nk1, nv, nkpe = _verify_load_tile_qsplit(
                K_Buffer, nkv_off, nmask, offs_d0, offs_d, offs_dpe, HALF, BLOCK_DPE
            )
            nkv_off = _verify_tile_off(
                pt,
                split_start + BLOCK_N,
                split_end,
                stride_buf_kpbs,
                stride_buf_kbs,
                BLOCK_N,
                PAGE_SIZE,
            )
            for start_n in tl.range(split_start, split_end, BLOCK_N):
                offs_n = start_n + tl.arange(0, BLOCK_N)
                k0 = nk0
                k1 = nk1
                v = nv
                kpe = nkpe
                nmask = offs_n + BLOCK_N < split_end
                nk0, nk1, nv, nkpe = _verify_load_tile_qsplit(
                    K_Buffer,
                    nkv_off,
                    nmask,
                    offs_d0,
                    offs_d,
                    offs_dpe,
                    HALF,
                    BLOCK_DPE,
                )
                nkv_off = _verify_tile_off(
                    pt,
                    start_n + 2 * BLOCK_N,
                    split_end,
                    stride_buf_kpbs,
                    stride_buf_kbs,
                    BLOCK_N,
                    PAGE_SIZE,
                )
                if v.dtype.is_fp8():
                    k0 = k0.to(tl.float32) * ks
                    k1 = k1.to(tl.float32) * ks
                    v = v.to(tl.float32) * ks
                    kpe = kpe.to(tl.float32) * ks
                qk = tl.dot(q, k0.to(q.dtype))
                qk = tl.dot(q1, k1.to(q.dtype), qk)
                if BLOCK_DPE > 0:
                    qk = tl.dot(qpe, kpe.to(qpe.dtype), qk)
                qk *= sm_scale_log2
                qk = tl.where(offs_n[None, :] < row_end[:, None], qk, float("-inf"))

                n_e_max = tl.maximum(tl.max(qk, 1), e_max)
                safe_max = tl.where(n_e_max == float("-inf"), 0.0, n_e_max)
                re_scale = tl.exp2(e_max - safe_max)
                p = tl.exp2(qk - safe_max[:, None])
                acc = acc * re_scale[:, None] + tl.dot(p.to(q.dtype), v.to(q.dtype))
                e_sum = e_sum * re_scale + tl.sum(p, 1)
                e_max = n_e_max
        else:
            for start_n in tl.range(split_start, split_end, BLOCK_N):
                offs_n = start_n + tl.arange(0, BLOCK_N)
                mask_n = offs_n < split_end
                kv_off = _verify_tile_off(
                    pt,
                    start_n,
                    split_end,
                    stride_buf_kpbs,
                    stride_buf_kbs,
                    BLOCK_N,
                    PAGE_SIZE,
                )
                k = tl.load(
                    K_Buffer + kv_off[None, :] + offs_d[:, None],
                    mask=mask_n[None, :],
                    other=0.0,
                )
                if k.dtype.is_fp8():
                    k = (k.to(tl.float32) * ks).to(q.dtype)
                k = k.to(q.dtype)
                qk = tl.dot(q, k)
                # MLA reads V from the latent K tile.
                v = tl.trans(k)
                if BLOCK_DPE > 0:
                    kpe = tl.load(
                        K_Buffer + kv_off[None, :] + offs_dpe[:, None],
                        mask=mask_n[None, :],
                        other=0.0,
                    )
                    if kpe.dtype.is_fp8():
                        kpe = (kpe.to(tl.float32) * ks).to(qpe.dtype)
                    qk += tl.dot(qpe, kpe.to(qpe.dtype))
                qk *= sm_scale_log2
                qk = tl.where(offs_n[None, :] < row_end[:, None], qk, float("-inf"))

                n_e_max = tl.maximum(tl.max(qk, 1), e_max)
                safe_max = tl.where(n_e_max == float("-inf"), 0.0, n_e_max)
                re_scale = tl.exp2(e_max - safe_max)
                p = tl.exp2(qk - safe_max[:, None])
                acc = acc * re_scale[:, None] + tl.dot(p.to(q.dtype), v)
                e_sum = e_sum * re_scale + tl.sum(p, 1)
                e_max = n_e_max

        has = e_sum > 0.0
        denom = tl.where(has, e_sum, 1.0)
        if SINGLE:
            tl.store(
                Out
                + q_row[:, None] * stride_obs
                + head[:, None] * stride_oh
                + offs_d[None, :],
                (acc / denom[:, None]).to(Out.dtype.element_ty),
                mask=row_valid[:, None],
            )
            tl.store(
                Lse + q_row * stride_lse_bs + head * stride_lse_h,
                tl.where(
                    has, (e_max + tl.log2(denom)) * 0.6931471805599453, float("-inf")
                ),
                mask=row_valid,
            )
        else:
            offs_mid = (
                q_row * stride_mid_ob
                + head * stride_mid_oh
                + split_kv_id * stride_mid_os
            )
            tl.store(
                Att_Out + offs_mid[:, None] + offs_d[None, :],
                acc / denom[:, None],
                mask=row_valid[:, None],
            )
            # The LSE is computed after the tile store: the reverse order
            # measured ~6% slower on gfx942 (16,8,32,16384).
            tl.store(
                Att_Out + offs_mid + Lv,
                tl.where(has, e_max + tl.log2(denom), float("-inf")),
                mask=row_valid,
            )


@triton.jit
def _fwd_verify_kernel_stage2(
    Mid_O,
    o,
    lse,
    B_Seqlen,
    Real_Seqlen,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    stride_lse_bs,
    stride_lse_h,
    QLEN: tl.constexpr,
    HAS_REAL: tl.constexpr,
    NUM_KV_SPLITS,
    SPLIT_ALIGN: tl.constexpr,
    MIN_SPLIT_LEN: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    # One (row, head) per program; BLOCK_S splits are combined per step so
    # the loads of a step are independent (long split lists stay cheap).
    q_row = tl.program_id(0)
    head = tl.program_id(1)
    cur_batch = q_row // QLEN
    tok = q_row % QLEN

    seq_len = tl.load(B_Seqlen + cur_batch)
    L_cap = seq_len
    if HAS_REAL:
        L_cap = tl.minimum(seq_len, tl.load(Real_Seqlen + cur_batch))
    L_cap = tl.maximum(L_cap, 0)
    lim = tl.maximum(tl.minimum(seq_len - QLEN + 1 + tok, L_cap), 0)
    per_split = _verify_split_len(L_cap, NUM_KV_SPLITS, SPLIT_ALIGN, MIN_SPLIT_LEN)
    num_active = tl.cdiv(lim, per_split)

    offs_d = tl.arange(0, BLOCK_DV)
    offs_s = tl.arange(0, BLOCK_S)
    base = q_row.to(tl.int64) * stride_mid_ob + head * stride_mid_oh

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)
    for s0 in range(0, num_active, BLOCK_S):
        sid = s0 + offs_s
        mask_s = sid < num_active
        off = base + sid * stride_mid_os
        tlogic = tl.load(Mid_O + off + Lv, mask=mask_s, other=float("-inf"))
        tv = tl.load(
            Mid_O + off[:, None] + offs_d[None, :], mask=mask_s[:, None], other=0.0
        )
        n_e_max = tl.maximum(tl.max(tlogic, 0), e_max)
        old_scale = tl.exp2(e_max - n_e_max)
        w = tl.exp2(tlogic - n_e_max)
        acc = acc * old_scale + tl.sum(w[:, None] * tv, 0)
        e_sum = e_sum * old_scale + tl.sum(w, 0)
        e_max = n_e_max

    has = e_sum > 0.0
    denom = tl.where(has, e_sum, 1.0)
    tl.store(
        o + q_row.to(tl.int64) * stride_obs + head * stride_oh + offs_d,
        (acc / denom).to(o.dtype.element_ty),
    )
    tl.store(
        lse + q_row.to(tl.int64) * stride_lse_bs + head * stride_lse_h,
        tl.where(has, (e_max + tl.log2(denom)) * 0.6931471805599453, float("-inf")),
    )


# Default launch config; overridable per call for tuning.
VERIFY_DEFAULT_CONFIG = dict(
    BLOCK_H=32,
    BLOCK_N=16,
    num_warps=4,
    num_stages=1,
    waves_per_eu=1,
    matrix_instr_nonkdim=16,
    kpack=2,
    SPLIT_ALIGN=16,
    MIN_SPLIT_LEN=256,
    QSPLIT=False,
    S2_BLOCK_S=16,
    s2_num_warps=4,
)


def verify_row_blocks(qlen: int, num_heads: int, block_h: int) -> int:
    return triton.cdiv(qlen * num_heads, block_h)


def verify_num_kv_splits(
    max_seq_len: int,
    batch: int,
    qlen: int,
    num_heads: int,
    sm_count: int,
    block_h: int = VERIFY_DEFAULT_CONFIG["BLOCK_H"],
    occupancy: float = 1.0,
    min_work_per_split: int = 64,
    pow2: bool = False,
    tail_aware: bool = True,
    max_splits: int = 8,
    tail_tol: float = 0.05,
) -> int:
    """Static upper bound on KV splits (safe to bake into a cudagraph).

    The kernel splits each request's own length into
    max(cdiv(L, splits), MIN_SPLIT_LEN) chunks, so this only caps the split
    count. ``occupancy`` is the number of resident stage-1 programs per CU.

    tail_aware (default): if one split of every request fits in a single
    dispatch round, use as many splits as fit in that round. Otherwise pick
    n <= max_splits minimising rounds(n) / n (the idle tail of the last
    round), taking the fewest splits within ``tail_tol`` of the best.
    tail_aware=False: about ``occupancy`` programs per CU (ceil; ``pow2``
    rounds up to a power of two).
    """
    programs_per_split = max(1, batch) * verify_row_blocks(qlen, num_heads, block_h)
    ideal = triton.cdiv(max(1, max_seq_len), max(1, min_work_per_split))
    slots = max(1, int(occupancy * sm_count))
    if tail_aware:
        if programs_per_split <= slots:
            n = slots // programs_per_split
        else:
            cost = [
                (triton.cdiv(n * programs_per_split, slots) / n, n)
                for n in range(1, max(1, min(ideal, max_splits)) + 1)
            ]
            best = min(c for c, _ in cost)
            n = min(n for c, n in cost if c <= best * (1.0 + tail_tol))
    else:
        n = triton.cdiv(slots, programs_per_split)
        if pow2:
            n = triton.next_power_of_2(n)
    return max(1, min(ideal, n, sm_count * 2))


def decode_attention_fwd_verify(
    q,
    k_buffer,
    v_buffer,
    o,
    lse,
    req_to_token,
    seq_lens,
    real_seq_lens,
    qlen,
    attn_logits,
    num_kv_splits,
    sm_scale,
    page_size,
    k_scale=None,
    config=None,
):
    """Causal MLA verify over a per-request block table.

    Args:
        q: [B * qlen, H, Lk] queries, verify tokens of a request contiguous.
        k_buffer: paged latent cache, [num_pages, page_size, 1, Lk] (or 3D
            [num_pages * page_size, 1, Lk]).
        v_buffer: k_buffer[..., :Lv] (V is read from K; only its width is used).
        o: [B * qlen, H, Lv] output.
        lse: [B * qlen, H] fp32 natural-log LSE (-inf for rows with no key).
        req_to_token: [B, max_pages] per-request block table (not repeated).
        seq_lens: [B] int32 kernel KV lengths L.
        real_seq_lens: [B] int32 or None; row limit is
            max(min(L - qlen + 1 + t, L_real), 0).
        qlen: verify tokens per request.
        attn_logits: fp32 workspace with >= B * qlen * H * num_kv_splits *
            (Lv + 1) elements, laid out [B * qlen, H, num_kv_splits, Lv + 1].
        num_kv_splits: static cap on KV splits per request (see
            verify_num_kv_splits); shorter requests use fewer, longer splits.
        sm_scale: softmax scale.
        page_size: tokens per KV cache page.
        k_scale: scalar tensor dequantizing an FP8 cache; ignored for bf16.
        config: optional launch-config overrides on select_verify_config().

    """
    cfg = select_verify_config(q.shape[0] // qlen, qlen, q.shape[1], page_size)
    if config:
        cfg.update(config)
    cfg["MIN_SPLIT_LEN"] = max(1, cfg["MIN_SPLIT_LEN"])
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]
    assert k_buffer.shape[-2] == 1, "verify path expects a single latent KV head"
    num_rows, num_heads = q.shape[0], q.shape[1]
    assert num_rows % qlen == 0
    batch = num_rows // qlen
    assert attn_logits.shape[-2] == num_kv_splits
    if k_scale is None:
        k_scale = torch.ones((), dtype=torch.float32, device=q.device)
    BLOCK_DMODEL = triton.next_power_of_2(Lv)
    BLOCK_DPE = triton.next_power_of_2(Lk - Lv) if Lk > Lv else 0
    assert Lv == BLOCK_DMODEL and Lk == BLOCK_DMODEL + BLOCK_DPE
    has_real = real_seq_lens is not None
    real = real_seq_lens if has_real else seq_lens

    hip_kw = {}
    if is_hip_:
        hip_kw = dict(
            waves_per_eu=cfg["waves_per_eu"],
            matrix_instr_nonkdim=cfg["matrix_instr_nonkdim"],
            kpack=cfg["kpack"],
        )
    grid = (verify_row_blocks(qlen, num_heads, cfg["BLOCK_H"]) * batch * num_kv_splits,)
    _fwd_verify_kernel_stage1[grid](
        q,
        k_buffer,
        sm_scale * _LOG2E,
        req_to_token,
        seq_lens,
        real,
        attn_logits,
        o,
        lse,
        req_to_token.stride(0),
        q.stride(0),
        q.stride(1),
        _page_stride(k_buffer, page_size),
        k_buffer.stride(-3),
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        o.stride(0),
        o.stride(1),
        lse.stride(0),
        lse.stride(1),
        k_scale,
        batch,
        QLEN=qlen,
        NUM_HEADS=num_heads,
        HAS_REAL=has_real,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_N=cfg["BLOCK_N"],
        BLOCK_H=cfg["BLOCK_H"],
        NUM_KV_SPLITS=num_kv_splits,
        SPLIT_ALIGN=cfg["SPLIT_ALIGN"],
        MIN_SPLIT_LEN=cfg["MIN_SPLIT_LEN"],
        PAGE_SIZE=page_size,
        Lv=Lv,
        QSPLIT=bool(cfg["QSPLIT"]),
        SINGLE=num_kv_splits == 1,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
        **hip_kw,
    )
    if num_kv_splits == 1:
        return  # stage 1 wrote o / lse
    _fwd_verify_kernel_stage2[(num_rows, num_heads)](
        attn_logits,
        o,
        lse,
        seq_lens,
        real,
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        o.stride(0),
        o.stride(1),
        lse.stride(0),
        lse.stride(1),
        QLEN=qlen,
        HAS_REAL=has_real,
        NUM_KV_SPLITS=num_kv_splits,
        SPLIT_ALIGN=cfg["SPLIT_ALIGN"],
        MIN_SPLIT_LEN=cfg["MIN_SPLIT_LEN"],
        BLOCK_S=cfg["S2_BLOCK_S"],
        BLOCK_DV=triton.next_power_of_2(Lv),
        Lv=Lv,
        num_warps=cfg["s2_num_warps"],
        num_stages=1,
    )


_VERIFY_SMALL_CONFIG = dict(
    BLOCK_H=32, BLOCK_N=32, num_warps=4, kpack=1, MIN_SPLIT_LEN=128, occupancy=2.0
)
# 64-row tiles, Q and the QK operand split in two 256-wide halves (QSPLIT,
# which also enables the K/V prefetch), 32 keys per step on 4 warps. 8 warps
# cap VGPRs at 256 and spill or idle MFMA rows here.
_VERIFY_LARGE_CONFIG = dict(
    BLOCK_H=64,
    BLOCK_N=32,
    num_warps=4,
    QSPLIT=True,
    MIN_SPLIT_LEN=128,
    occupancy=1.0,
)
# Many heads use the large config (name kept for callers / tests).
_VERIFY_WIDE_CONFIG = _VERIFY_LARGE_CONFIG


def select_verify_config(batch: int, qlen: int, num_heads: int, page_size: int):
    """Launch config from MI300X (gfx942) sweeps, bf16, page 16.

    Few rows (B * qlen * H <= 640): 32-row tiles with more KV splits fill the
    GPU. Otherwise 64-row tiles (Q split in two 256-wide halves, next K/V
    tile prefetched) halve the KV re-reads per request.
    """
    cfg = dict(VERIFY_DEFAULT_CONFIG)
    if batch * qlen * num_heads <= 640:
        cfg.update(_VERIFY_SMALL_CONFIG)
    else:
        cfg.update(_VERIFY_LARGE_CONFIG)
    return cfg
