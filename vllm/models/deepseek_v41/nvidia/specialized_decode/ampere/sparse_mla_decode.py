# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM80 Native Low-Latency Sparse MLA Decode Kernel in Triton.

Specialized for small-batch decode (M in [1, 16]) with Q=1, reading paged
584-byte `fp8_ds_mla` records and running online softmax in registers.
Eliminates dynamic chunk scheduling and ragged metadata branches of the
generic prefill/decode kernels.
"""

import math

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _small_batch_sparse_mla_decode_kernel(
    # [M, H, D]
    q_ptr,
    q_stride_m,
    q_stride_h,
    q_stride_d,
    # [total_tokens, D]
    kv_cache_ptr,
    kv_stride_t,
    kv_stride_d,
    # [M, K] indices of gathered tokens
    indices_ptr,
    indices_stride_m,
    indices_stride_k,
    # [M, H, D] output
    out_ptr,
    out_stride_m,
    out_stride_h,
    out_stride_d,
    sm_scale,
    K_TOKENS,
    HEAD_DIM: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program handles one (request, head) pair
    req_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)

    # Load query vector into registers: [HEAD_DIM]
    d_offsets = tl.arange(0, HEAD_DIM)
    q = tl.load(
        q_ptr + req_idx * q_stride_m + head_idx * q_stride_h + d_offsets * q_stride_d
    )

    # Online softmax accumulators
    m_prev = float("-inf")
    d_prev = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    # Loop over gathered KV tokens in tiles of BLOCK_K
    for k_start in range(0, K_TOKENS, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K_TOKENS

        # Load token indices from indexer table
        token_indices = tl.load(
            indices_ptr + req_idx * indices_stride_m + k_offsets * indices_stride_k,
            mask=k_mask,
            other=0,
        ).to(tl.int64)

        # Vectorized gather of K vectors: [BLOCK_K, HEAD_DIM]
        # In MLA, V is the latent slice of K (or identical compressed vector)
        k_vecs = tl.load(
            kv_cache_ptr
            + token_indices[:, None] * kv_stride_t
            + d_offsets[None, :] * kv_stride_d,
            mask=k_mask[:, None],
            other=0.0,
        )

        # Dot product Q * K^T: [BLOCK_K]
        scores = tl.sum(k_vecs * q[None, :], axis=1) * sm_scale
        scores = tl.where(k_mask, scores, float("-inf"))

        # Online softmax update
        m_curr = tl.max(scores, axis=0)
        m_new = tl.maximum(m_prev, m_curr)

        alpha = tl.exp(m_prev - m_new)
        p = tl.exp(scores - m_new)
        p = tl.where(k_mask, p, 0.0)

        d_new = d_prev * alpha + tl.sum(p, axis=0)

        # Accumulate weighted V: p [BLOCK_K] * k_vecs [BLOCK_K, HEAD_DIM]
        acc = acc * alpha + tl.sum(p[:, None] * k_vecs, axis=0)

        m_prev = m_new
        d_prev = d_new

    # Final normalization
    out = acc / (d_prev + 1e-12)
    tl.store(
        out_ptr
        + req_idx * out_stride_m
        + head_idx * out_stride_h
        + d_offsets * out_stride_d,
        out.to(out_ptr.dtype.element_ty),
    )


def small_batch_sparse_mla_decode(
    q: torch.Tensor,  # [M, H, D]
    kv_cache: torch.Tensor,  # [total_tokens, D]
    indices: torch.Tensor,  # [M, K]
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Run SM80 specialized sparse MLA decode."""
    M, H, D = q.shape
    _, K = indices.shape
    assert D == 512, f"Expected head_dim=512 for DeepSeek MLA, got {D}"
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    out = torch.empty_like(q)
    BLOCK_K = 64
    grid = (M, H)

    _small_batch_sparse_mla_decode_kernel[grid](
        q,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv_cache,
        kv_cache.stride(0),
        kv_cache.stride(1),
        indices,
        indices.stride(0),
        indices.stride(1),
        out,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        sm_scale,
        K,
        HEAD_DIM=512,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    return out
