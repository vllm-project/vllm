# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Tuple
import warnings

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False

_WARNED_TRITON_FALLBACK = False


def _warn_triton_fallback(msg: str) -> None:
    global _WARNED_TRITON_FALLBACK
    if _WARNED_TRITON_FALLBACK:
        return
    warnings.warn(msg, stacklevel=2)
    _WARNED_TRITON_FALLBACK = True


if _HAS_TRITON:

    @triton.jit
    def moba_block_repr_kernel(
        k_cache_ptr,
        block_repr_ptr,
        mapping_ptr,
        num_mappings,
        block_size,
        num_heads,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_h = tl.program_id(axis=1)
        if pid_m >= num_mappings or pid_h >= num_heads:
            return

        src_block = tl.load(mapping_ptr + pid_m * 2)
        dst_block = tl.load(mapping_ptr + pid_m * 2 + 1)
        src_block = tl.cast(src_block, tl.int64)
        dst_block = tl.cast(dst_block, tl.int64)

        base_src = k_cache_ptr + src_block * block_size * num_heads * HEAD_DIM + pid_h * HEAD_DIM
        base_dst = block_repr_ptr + dst_block * num_heads * HEAD_DIM + pid_h * HEAD_DIM

        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
        dim_idx = tl.arange(0, HEAD_DIM)
        token_idx = tl.arange(0, BLOCK_SIZE)
        for i in range(0, block_size, BLOCK_SIZE):
            offsets = (i + token_idx) * num_heads * HEAD_DIM
            k_indices = base_src + offsets[:, None] + dim_idx[None, :]
            k = tl.load(k_indices)
            acc += tl.sum(k, axis=0)

        acc = acc / block_size
        tl.store(base_dst + dim_idx, acc)


    @triton.jit
    def compute_block_scores_kernel(
        block_indices_ptr,
        k_repr_ptr,
        query_ptr,
        query_start_loc_ptr,
        seq_lens_ptr,
        scores_ptr,
        block_size,
        batch_size,
        max_blocks,
        num_kv_heads,
        num_heads,
        head_size,
        heads_per_kv,
        block_indices_stride,
        k_repr_stride_0,
        k_repr_stride_1,
        query_stride_0,
        query_stride_1,
        scores_stride,
        BLOCK_K: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        block_idx = tl.program_id(1)
        if batch_idx >= batch_size or block_idx >= max_blocks:
            return

        query_start = tl.load(query_start_loc_ptr + batch_idx)
        query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
        num_query_tokens = query_end - query_start
        seq_len = tl.load(seq_lens_ptr + batch_idx)
        kv_len = seq_len - num_query_tokens
        num_full_blocks = kv_len // block_size

        out_ptr = scores_ptr + batch_idx * scores_stride + block_idx
        if block_idx >= num_full_blocks:
            tl.store(out_ptr, float("-inf"))
            return
        if num_full_blocks > 0 and (block_idx == 0 or block_idx == num_full_blocks - 1):
            tl.store(out_ptr, float("inf"))
            return

        block_offset = batch_idx * block_indices_stride + block_idx
        block_id = tl.load(block_indices_ptr + block_offset)
        token_idx = query_end - 1

        total_score = 0.0
        d_idx = tl.arange(0, BLOCK_K)
        d_mask = d_idx < head_size
        for kv_h in range(0, num_kv_heads):
            k_base = block_id * k_repr_stride_0 + kv_h * k_repr_stride_1
            k = tl.load(k_repr_ptr + k_base + d_idx, mask=d_mask, other=0.0)

            qh_start = kv_h * heads_per_kv
            qh_end = (kv_h + 1) * heads_per_kv
            for q_h in range(qh_start, qh_end):
                q_base = token_idx * query_stride_0 + q_h * query_stride_1
                q = tl.load(query_ptr + q_base + d_idx, mask=d_mask, other=0.0)
                total_score += tl.sum(k * q)

        tl.store(out_ptr, total_score)


def kv_repr_gen(
    kv_cache: torch.Tensor,
    block_repr: torch.Tensor,
    mapping: torch.Tensor,
    num_mappings: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    if num_mappings <= 0:
        return block_repr

    mapping_slice = mapping.narrow(0, 0, num_mappings)
    if (
        _HAS_TRITON
        and kv_cache.is_cuda
        and block_repr.is_cuda
        and mapping_slice.is_cuda
    ):
        try:
            grid = (num_mappings, num_kv_heads)
            moba_block_repr_kernel[grid](
                kv_cache,
                block_repr,
                mapping_slice,
                num_mappings,
                block_size,
                num_kv_heads,
                BLOCK_SIZE=block_size,
                HEAD_DIM=head_dim,
            )
            return block_repr
        except Exception:
            _warn_triton_fallback(
                "Triton kv_repr_gen failed; falling back to torch implementation."
            )

    if mapping_slice.device.type != "cpu":
        mapping_slice = mapping_slice.cpu()
    key_cache = kv_cache[0]
    for src_block, dst_block in mapping_slice.tolist():
        block_repr[dst_block].copy_(
            key_cache[src_block].to(torch.float32).mean(dim=0).to(block_repr.dtype),
            non_blocking=False,
        )
    return block_repr


def sparse_kv_selection(
    block_table: torch.Tensor,
    batch_size: int,
    block_size: int,
    max_num_blocks_this_batch: int,
    seq_lens: torch.Tensor,
    k_repr: torch.Tensor,
    query: torch.Tensor,
    query_start_loc: torch.Tensor,
    top_k: int,
    scores: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if batch_size <= 0:
        return (
            torch.empty((0, top_k), dtype=torch.int64, device=block_table.device),
            torch.empty((0, top_k), dtype=scores.dtype, device=scores.device),
        )

    num_kv_heads = int(k_repr.shape[1])
    head_size = int(k_repr.shape[2])
    num_heads = int(query.shape[1])
    heads_per_kv = max(1, num_heads // max(1, num_kv_heads))

    if _HAS_TRITON and all(
        t.is_cuda for t in (block_table, seq_lens, k_repr, query, query_start_loc, scores)
    ):
        try:
            grid = (batch_size, max_num_blocks_this_batch)
            compute_block_scores_kernel[grid](
                block_table,
                k_repr,
                query,
                query_start_loc,
                seq_lens,
                scores,
                block_size,
                batch_size,
                max_num_blocks_this_batch,
                num_kv_heads,
                num_heads,
                head_size,
                heads_per_kv,
                block_table.stride(0),
                k_repr.stride(0),
                k_repr.stride(1),
                query.stride(0),
                query.stride(1),
                scores.stride(0),
                BLOCK_K=triton.next_power_of_2(head_size),
            )
            topk_scores, topk_choices = torch.topk(scores, top_k, dim=1, sorted=False)
            return topk_choices, topk_scores
        except Exception:
            _warn_triton_fallback(
                "Triton sparse_kv_selection failed; falling back to torch implementation."
            )

    # Torch fallback path.
    scores.fill_(float("-inf"))
    for req_idx in range(batch_size):
        q_start = int(query_start_loc[req_idx])
        q_end = int(query_start_loc[req_idx + 1])
        q_len = q_end - q_start
        seq_len = int(seq_lens[req_idx])
        kv_len = seq_len - q_len
        num_full_blocks = kv_len // block_size
        if num_full_blocks <= 0:
            continue

        scores[req_idx, 0] = float("inf")
        scores[req_idx, num_full_blocks - 1] = float("inf")
        q_last = query[q_end - 1]

        for logical_block_id in range(1, max(0, num_full_blocks - 1)):
            cpu_block_id = int(block_table[req_idx, logical_block_id])
            if cpu_block_id < 0:
                continue
            repr_block = k_repr[cpu_block_id]  # [num_kv_heads, head_size]
            total = torch.tensor(0.0, dtype=torch.float32, device=scores.device)
            for kv_h in range(num_kv_heads):
                q_start_h = kv_h * heads_per_kv
                q_end_h = min((kv_h + 1) * heads_per_kv, num_heads)
                if q_start_h >= q_end_h:
                    continue
                q_slice = q_last[q_start_h:q_end_h]  # [heads_per_kv, head_size]
                # Sum over query heads for each kv head.
                total += torch.sum(q_slice * repr_block[kv_h], dtype=torch.float32)
            scores[req_idx, logical_block_id] = total.to(dtype=scores.dtype)

    topk_scores, topk_choices = torch.topk(scores, top_k, dim=1, sorted=False)
    return topk_choices, topk_scores

