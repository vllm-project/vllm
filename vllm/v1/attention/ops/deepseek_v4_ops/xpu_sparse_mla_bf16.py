# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# pyright: reportUnknownVariableType=none, reportUnknownMemberType=none, reportUnknownArgumentType=none, reportUnknownParameterType=none, reportMissingParameterType=none, reportCallIssue=none, reportUnreachable=none, reportPrivateImportUsage=none, reportUnusedCallResult=none
"""bf16 Triton C4 sparse prefill over a contiguous KV workspace.

This kernel targets the DeepSeek V4 main MLA sparse prefill path on XPU.
It performs top-k indexed FlashAttention over a contiguous KV workspace with
an online softmax update per ``(token, head)`` program.

"""

import torch

from vllm.triton_utils import tl, triton

HEAD_DIM = 512
BLOCK_K = 64


@triton.jit
def _xpu_sparse_mla_bf16_kernel(
    q_ptr,
    kv_ptr,
    topk_indices_ptr,
    topk_lens_ptr,
    out_ptr,
    softmax_scale,
    q_stride_t,
    q_stride_h,
    kv_stride_s,
    topk_stride_t,
    out_stride_t,
    out_stride_h,
    K_MAX: tl.constexpr,
    D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)

    d_off = tl.arange(0, D)
    q = tl.load(q_ptr + t * q_stride_t + h * q_stride_h + d_off).to(tl.float32)
    topk_len = tl.load(topk_lens_ptr + t)

    m_i = tl.full((), float("-inf"), tl.float32)
    d_i = tl.zeros((), tl.float32)
    num_i = tl.zeros((D,), tl.float32)

    for k_start in tl.range(0, K_MAX, BLOCK_K):
        k_off = k_start + tl.arange(0, BLOCK_K)
        valid = k_off < topk_len
        slot = tl.load(
            topk_indices_ptr + t * topk_stride_t + k_off,
            mask=valid,
            other=-1,
        ).to(tl.int32)
        valid = valid & (slot >= 0)
        valid_count = tl.sum(valid.to(tl.int32), axis=0)
        block_has_values = valid_count > 0

        k = tl.load(
            kv_ptr + slot[:, None] * kv_stride_s + d_off[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        logits = tl.sum(k * q[None, :], axis=1) * softmax_scale
        logits = tl.where(valid, logits, float("-inf"))

        m_block = tl.max(logits, axis=0)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(logits - m_new)
        p = tl.where(valid, p, 0.0)

        d_next = d_i * alpha + tl.sum(p, axis=0)
        num_next = num_i * alpha + tl.sum(p[:, None] * k, axis=0)
        d_i = tl.where(block_has_values, d_next, d_i)
        num_i = tl.where(block_has_values, num_next, num_i)
        m_i = tl.where(block_has_values, m_new, m_i)

    out = tl.where(d_i > 0, num_i / d_i, tl.zeros_like(num_i))
    tl.store(out_ptr + t * out_stride_t + h * out_stride_h + d_off, out.to(tl.bfloat16))


def xpu_sparse_mla_bf16(
    q: torch.Tensor,
    kv_workspace: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_lens: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor,
) -> None:
    assert q.dtype == torch.bfloat16
    assert kv_workspace.dtype == torch.bfloat16
    assert out.dtype == torch.bfloat16
    assert topk_indices.dtype == torch.int32
    assert topk_lens.dtype == torch.int32
    assert q.dim() == 3
    assert kv_workspace.dim() == 2
    assert topk_indices.dim() == 2
    assert topk_lens.dim() == 1
    assert q.shape[-1] == HEAD_DIM
    assert kv_workspace.shape[-1] == HEAD_DIM
    assert out.shape == q.shape
    assert topk_indices.shape[0] == q.shape[0]
    assert topk_lens.shape == (q.shape[0],)

    T, H, D = q.shape
    k_max = topk_indices.shape[1]

    grid = (T, H)
    _xpu_sparse_mla_bf16_kernel[grid](
        q,
        kv_workspace,
        topk_indices,
        topk_lens,
        out,
        float(softmax_scale),
        q.stride(0),
        q.stride(1),
        kv_workspace.stride(0),
        topk_indices.stride(0),
        out.stride(0),
        out.stride(1),
        K_MAX=k_max,
        D=D,
        BLOCK_K=BLOCK_K,
        num_warps=16,
    )


__all__ = ["xpu_sparse_mla_bf16"]
