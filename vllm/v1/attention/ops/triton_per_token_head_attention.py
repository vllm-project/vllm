# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dedicated split-KV attention for per-token-head quantized KV cache.

Modelled on TurboQuant's decode kernel:
  - stage1: per-query split-KV tiled attention with inline fp8/int8 dequant
  - stage2: log-sum-exp reduction (reused ``_fwd_kernel_stage2``)

Covers pure decode (q_len=1) and small continuation prefill (q_len ≤ threshold)
via precomputed ``q_to_klen`` — each query gets its own causal K length, which
is the synthetic-seq-lens trick from TurboQuant. Pre-allocated buffers on the
layer avoid per-forward allocation and keep grid dims constant for CUDA graphs.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_decode_attention import _fwd_kernel_stage2


@triton.jit
def _pth_attn_stage1(
    Q_ptr,
    K_ptr,
    V_ptr,
    K_scale_ptr,
    V_scale_ptr,
    Block_table_ptr,
    Q_to_req_ptr,
    Q_to_klen_ptr,
    Mid_o_ptr,
    stride_q_tok,
    stride_q_h,
    stride_kc_blk,
    stride_kc_slot,
    stride_kc_head,
    stride_vc_blk,
    stride_vc_slot,
    stride_vc_head,
    stride_ks_blk,
    stride_ks_slot,
    stride_ks_head,
    stride_vs_blk,
    stride_vs_slot,
    stride_vs_head,
    stride_bt_r,
    stride_mid_q,
    stride_mid_h,
    stride_mid_s,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    ATTN_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    q_id = tl.program_id(0)
    h_id = tl.program_id(1)
    sid = tl.program_id(2)

    kv_head = h_id // KV_GROUP_SIZE

    k_len = tl.load(Q_to_klen_ptr + q_id)
    if k_len <= 0:
        return

    split_len = tl.cdiv(k_len, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, k_len)
    if split_start >= split_end:
        return

    req_id = tl.load(Q_to_req_ptr + q_id)

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    kv_range = tl.arange(0, BLOCK_KV)

    q_base = q_id * stride_q_tok + h_id * stride_q_h
    q_vec = tl.load(Q_ptr + q_base + d_offs, mask=d_mask, other=0.0).to(tl.float32)

    m_prev = -float("inf")
    l_prev = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    bt_base = req_id * stride_bt_r

    for start_n in range(split_start, split_end, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < split_end

        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        block_nums = tl.load(
            Block_table_ptr + bt_base + page_idx, mask=kv_mask, other=0
        )

        # K
        k_bases = (
            block_nums * stride_kc_blk
            + page_off * stride_kc_slot
            + kv_head * stride_kc_head
        )
        k_addrs = k_bases[:, None] + d_offs[None, :]
        k_raw = tl.load(
            K_ptr + k_addrs,
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0,
        )
        k_f = k_raw.to(tl.float32)

        k_sc_addrs = (
            block_nums * stride_ks_blk
            + page_off * stride_ks_slot
            + kv_head * stride_ks_head
        )
        k_scales = tl.load(K_scale_ptr + k_sc_addrs, mask=kv_mask, other=1.0)

        dots = tl.sum(tl.where(d_mask[None, :], q_vec[None, :] * k_f, 0.0), axis=1)
        scores = dots * k_scales * ATTN_SCALE
        scores = tl.where(kv_mask, scores, -float("inf"))

        # online softmax
        n_e_max = tl.maximum(tl.max(scores, 0), m_prev)
        re_scale = tl.exp(m_prev - n_e_max)
        p = tl.exp(scores - n_e_max)

        # V
        v_bases = (
            block_nums * stride_vc_blk
            + page_off * stride_vc_slot
            + kv_head * stride_vc_head
        )
        v_addrs = v_bases[:, None] + d_offs[None, :]
        v_raw = tl.load(
            V_ptr + v_addrs,
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0,
        )
        v_f = v_raw.to(tl.float32)

        v_sc_addrs = (
            block_nums * stride_vs_blk
            + page_off * stride_vs_slot
            + kv_head * stride_vs_head
        )
        v_scales = tl.load(V_scale_ptr + v_sc_addrs, mask=kv_mask, other=1.0)

        pv_w = p * v_scales
        acc = acc * re_scale + tl.sum(pv_w[:, None] * v_f, 0)
        l_prev = l_prev * re_scale + tl.sum(p, 0)
        m_prev = n_e_max

    out_base = q_id * stride_mid_q + h_id * stride_mid_h + sid * stride_mid_s
    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    tl.store(Mid_o_ptr + out_base + d_offs, acc / safe_l, mask=d_mask)
    lse = m_prev + tl.log(safe_l)
    tl.store(Mid_o_ptr + out_base + HEAD_DIM, lse)


# ------------------------------------------------------------------ #
#  CPU-side query maps (vectorized, no per-request Python loop)       #
# ------------------------------------------------------------------ #


def _build_q_maps(
    query_start_loc_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_lens_i32 = (query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]).to(torch.int32)
    num_reqs = q_lens_i32.shape[0]
    total_q = int(query_start_loc_cpu[-1].item())

    if num_reqs == total_q:
        q_to_req = torch.arange(num_reqs, dtype=torch.int32)
        q_to_klen = seq_lens_cpu.to(torch.int32)
    else:
        qsl_i32 = query_start_loc_cpu[:-1].to(torch.int32)
        seq_lens_i32 = seq_lens_cpu.to(torch.int32)
        q_to_req = torch.repeat_interleave(
            torch.arange(num_reqs, dtype=torch.int32), q_lens_i32
        )
        cached_len_per_req = seq_lens_i32 - q_lens_i32
        pos_in_req = torch.arange(total_q, dtype=torch.int32) - qsl_i32[q_to_req.long()]
        q_to_klen = cached_len_per_req[q_to_req.long()] + pos_in_req + 1

    return (
        q_to_req.to(device, non_blocking=True),
        q_to_klen.to(device, non_blocking=True),
    )


# ------------------------------------------------------------------ #
#  Launcher (TQ-style pre-allocated buffers + constant splits)        #
# ------------------------------------------------------------------ #


def triton_per_token_head_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    block_table: torch.Tensor,
    q_to_req: torch.Tensor,
    q_to_klen: torch.Tensor,
    scale: float,
    max_num_kv_splits: int = 32,
    block_kv: int = 16,
    output: torch.Tensor | None = None,
    mid_o_buf: torch.Tensor | None = None,
    output_buf: torch.Tensor | None = None,
    lse_buf: torch.Tensor | None = None,
    buf_holder: Any = None,
) -> torch.Tensor:
    """Per-token-head split-KV attention.

    ``q_to_req`` and ``q_to_klen`` are precomputed int32 GPU tensors of
    shape ``(total_q,)`` — typically slices of persistent buffers owned by
    the metadata builder so their pointers stay stable under CUDA graph
    capture/replay.
    """
    total_q, Hq, D = query.shape
    Hk = key_cache.shape[2]
    kv_group = Hq // Hk

    device = query.device
    block_size = key_cache.shape[1]
    BLOCK_D = triton.next_power_of_2(D)
    NUM_KV_SPLITS = max_num_kv_splits

    # Pre-allocated buffer reuse (TQ pattern: allocate once, slice by batch)
    if mid_o_buf is not None and mid_o_buf.shape[0] >= total_q:
        mid_o = mid_o_buf[:total_q, :Hq, :NUM_KV_SPLITS, :]
    else:
        mid_o = torch.empty(
            (total_q, Hq, NUM_KV_SPLITS, D + 1),
            dtype=torch.float32,
            device=device,
        )
        if buf_holder is not None:
            buf_holder._pth_mid_o_buf = mid_o

    if output is not None:
        out = output.view(total_q, Hq, D) if output.ndim != 3 else output
    elif output_buf is not None and output_buf.shape[0] >= total_q:
        out = output_buf[:total_q, :Hq, :D]
    else:
        out = torch.empty((total_q, Hq, D), dtype=query.dtype, device=device)
        if buf_holder is not None:
            buf_holder._pth_output_buf = out

    if lse_buf is not None and lse_buf.shape[0] >= total_q:
        lse = lse_buf[:total_q, :Hq]
    else:
        lse = torch.empty((total_q, Hq), dtype=torch.float32, device=device)
        if buf_holder is not None:
            buf_holder._pth_lse_buf = lse

    # Stage 1
    _pth_attn_stage1[(total_q, Hq, NUM_KV_SPLITS)](
        query,
        key_cache,
        value_cache,
        k_scale_cache,
        v_scale_cache,
        block_table,
        q_to_req,
        q_to_klen,
        mid_o,
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        k_scale_cache.stride(0),
        k_scale_cache.stride(1),
        k_scale_cache.stride(2),
        v_scale_cache.stride(0),
        v_scale_cache.stride(1),
        v_scale_cache.stride(2),
        block_table.stride(0),
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        HEAD_DIM=D,
        BLOCK_SIZE=block_size,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        KV_GROUP_SIZE=kv_group,
        ATTN_SCALE=scale,
        BLOCK_D=BLOCK_D,
        BLOCK_KV=block_kv,
        num_warps=4,
        num_stages=2,
    )

    # Stage 2
    _fwd_kernel_stage2[(total_q, Hq)](
        mid_o,
        out,
        lse,
        q_to_klen,
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        out.stride(0),
        out.stride(1),
        lse.stride(0),
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        BLOCK_DV=BLOCK_D,
        Lv=D,
    )

    return out
