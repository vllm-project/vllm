# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax M3 CuteDSL sparse prefill attention wrapper."""

import torch

from .interface import sparse_atten_func
from .sm100.prepare_k2q_csr import build_k2q_csr_with_schedule_sm100

SPARSE_BLOCK_SIZE = 128


def minimax_m3_sparse_attn_cutedsl(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    seq_lens: torch.Tensor,
    max_query_len: int,
    max_seq_len: int,
    num_kv_heads: int,
    sm_scale: float,
    out: torch.Tensor,
    *,
    total_kv_blocks: int,
) -> None:
    """Run CuteDSL sparse attention directly over vLLM's paged KV cache."""
    if kv_cache.shape[2] != SPARSE_BLOCK_SIZE:
        raise ValueError("MiniMax M3 CuteDSL path requires block_size == 128")
    topk = topk_idx.shape[-1]
    k_cache = kv_cache[:, 0]
    v_cache = kv_cache[:, 1]
    k2q_row_ptr, k2q_q_indices, schedule = build_k2q_csr_with_schedule_sm100(
        topk_idx,
        cu_seqlens_q,
        cu_seqlens_k,
        blk_kv=SPARSE_BLOCK_SIZE,
        max_seqlen_k=max_seq_len,
        max_seqlen_q=max_query_len,
        total_rows=total_kv_blocks,
        qhead_per_kv=q.shape[1] // num_kv_heads,
    )
    sparse_atten_func(
        q,
        k_cache,
        v_cache,
        k2q_row_ptr,
        k2q_q_indices,
        topK=topk,
        blk_kv=SPARSE_BLOCK_SIZE,
        causal=True,
        softmax_scale=sm_scale,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_query_len,
        page_table=block_table,
        seqused_k=seq_lens,
        schedule=schedule,
        out=out,
    )
