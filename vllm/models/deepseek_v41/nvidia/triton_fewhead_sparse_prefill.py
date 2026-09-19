# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM90 DSV4.1 few-head BF16 sparse MLA prefill.

FlashMLA `flash_mla_sparse_fwd` on Hopper tiles WGMMA with B_H=64, so TP8's
8 local heads are zero-padded to 64. Decode FP8 still needs that width. Prefill
does not: this kernel runs native h_q (8 or 16) and is faster on H20 than the
padded FlashMLA path.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import LOG2E, tl, triton


@triton.jit
def _bf16_fewhead_sparse_fwd(
    q_ptr,
    kv_ptr,
    indices_ptr,
    out_ptr,
    sink_ptr,
    topk_len_ptr,
    seq_kv,
    h_q,
    stride_q_s,
    stride_q_h,
    stride_kv_s,
    stride_idx_s,
    stride_o_s,
    stride_o_h,
    sm_scale,
    index_topk: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_SINK: tl.constexpr,
    HAS_TOPK_LEN: tl.constexpr,
):
    token = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < h_q
    offs_d = tl.arange(0, BLOCK_D)

    q = tl.load(
        q_ptr + token * stride_q_s + offs_h[:, None] * stride_q_h + offs_d[None, :],
        mask=mask_h[:, None],
        other=0.0,
    )

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - 1.0e30
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    valid_len = tl.load(topk_len_ptr + token) if HAS_TOPK_LEN else index_topk

    for start in range(0, index_topk, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        mask_n = (offs_n < index_topk) & (offs_n < valid_len)
        idx = tl.load(
            indices_ptr + token * stride_idx_s + offs_n,
            mask=mask_n,
            other=-1,
        )
        mask_kv = (idx >= 0) & (idx < seq_kv) & mask_n
        k = tl.load(
            kv_ptr + idx[None, :] * stride_kv_s + offs_d[:, None],
            mask=mask_kv[None, :],
            other=0.0,
        )
        qk = tl.dot(q, k.to(q.dtype), out_dtype=tl.float32)
        qk *= sm_scale
        qk = tl.where(mask_h[:, None] & mask_kv[None, :], qk, -1.0e30)

        v = tl.load(
            kv_ptr + idx[:, None] * stride_kv_s + offs_d[None, :],
            mask=mask_kv[:, None],
            other=0.0,
        )
        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp2(e_max - n_e_max)
        p = tl.exp2(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc += tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    if HAS_SINK:
        sink = tl.load(sink_ptr + offs_h, mask=mask_h, other=-1.0e30)
        n_e_max = tl.maximum(e_max, sink)
        re_scale = tl.exp2(e_max - n_e_max)
        acc *= re_scale[:, None]
        e_sum = e_sum * re_scale + tl.exp2(sink - n_e_max)

    e_sum = tl.maximum(e_sum, 1.0e-30)
    acc /= e_sum[:, None]
    tl.store(
        out_ptr + token * stride_o_s + offs_h[:, None] * stride_o_h + offs_d[None, :],
        acc.to(tl.bfloat16),
        mask=mask_h[:, None],
    )


def fewhead_sparse_mla_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run native-head BF16 sparse MLA prefill.

    Args:
        q: ``[s_q, h_q, d]``, ``h_q`` in ``{1..16}``, ``d == 512``, bf16.
        kv: ``[s_kv, 1, d]`` or ``[s_kv, d]``.
        indices: ``[s_q, 1, topk]`` or ``[s_q, topk]``, invalid slots = -1.
        sm_scale: Softmax scale (typically ``1/sqrt(d)``).
        attn_sink: Optional ``[h_q]`` fp32 extra logits, same units as scaled QK.
        topk_length: Optional ``[s_q]`` int32 valid prefix of ``indices``.
        out: Optional ``[s_q, h_q, d]`` buffer.

    Returns:
        Attention output ``[s_q, h_q, d]``.

    """
    if kv.ndim == 3:
        kv = kv.squeeze(1)
    if indices.ndim == 3:
        indices = indices.squeeze(1)
    s_q, h_q, d = q.shape
    topk = indices.shape[-1]
    if out is None:
        out = torch.empty_like(q)
    has_sink = attn_sink is not None
    has_topk_len = topk_length is not None
    if attn_sink is None:
        attn_sink = q.new_empty((1,), dtype=torch.float32)
    else:
        attn_sink = (
            attn_sink.to(dtype=torch.float32, device=q.device).contiguous() * LOG2E
        )
    if topk_length is None:
        topk_length = q.new_empty((1,), dtype=torch.int32)
    else:
        topk_length = topk_length.to(dtype=torch.int32, device=q.device).contiguous()
    dummy = q.new_empty((1,), dtype=torch.float32)
    _bf16_fewhead_sparse_fwd[(s_q,)](
        q,
        kv,
        indices,
        out,
        attn_sink if has_sink else dummy,
        topk_length,
        kv.shape[0],
        h_q,
        q.stride(0),
        q.stride(1),
        kv.stride(0),
        indices.stride(0),
        out.stride(0),
        out.stride(1),
        sm_scale * LOG2E,
        index_topk=topk,
        BLOCK_H=16,
        BLOCK_N=32,
        BLOCK_D=d,
        HAS_SINK=has_sink,
        HAS_TOPK_LEN=has_topk_len,
        num_warps=4,
        num_stages=2,
    )
    return out
