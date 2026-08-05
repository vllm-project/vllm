# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton Quest chunk scoring (D=128/256, hierarchical factor=16)."""

from __future__ import annotations

import torch

from vllm.v1.attention.ops.zoomkv.quest import QuestTorchOps, quest_bound_scores

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # noqa: BLE001
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _quest_chunk_score_kernel(
        q_ptr,
        cmin_ptr,
        cmax_ptr,
        scores_ptr,
        valid_ptr,
        stride_q_b,
        stride_q_h,
        stride_q_d,
        stride_c_b,
        stride_c_h,
        stride_c_n,
        stride_c_d,
        stride_s_b,
        stride_s_h,
        stride_s_n,
        stride_v_b,
        stride_v_h,
        stride_v_n,
        num_heads,
        n_chunks,
        D: tl.constexpr,
        HAS_VALID: tl.constexpr,
    ):
        bh = tl.program_id(0)
        n = tl.program_id(1)
        b = bh // num_heads
        h = bh - b * num_heads
        offs_d = tl.arange(0, D)
        q = tl.load(q_ptr + b * stride_q_b + h * stride_q_h + offs_d * stride_q_d).to(
            tl.float32
        )
        base_c = b * stride_c_b + h * stride_c_h + n * stride_c_n
        cmin = tl.load(cmin_ptr + base_c + offs_d * stride_c_d).to(tl.float32)
        cmax = tl.load(cmax_ptr + base_c + offs_d * stride_c_d).to(tl.float32)
        chosen = tl.where(q > 0, cmax, cmin)
        score = tl.sum(q * chosen, axis=0)
        if HAS_VALID:
            v = tl.load(valid_ptr + b * stride_v_b + h * stride_v_h + n * stride_v_n)
            score = tl.where(v != 0, score, float("-inf"))
        tl.store(
            scores_ptr + b * stride_s_b + h * stride_s_h + n * stride_s_n,
            score,
        )

    @triton.jit
    def _quest_sub_chunk_score_kernel(
        q_ptr,
        cmin_ptr,
        cmax_ptr,
        large_idx_ptr,
        scores_ptr,
        stride_q_b,
        stride_q_h,
        stride_q_d,
        stride_c_b,
        stride_c_h,
        stride_c_n,
        stride_c_d,
        stride_i_b,
        stride_i_h,
        stride_i_n,
        stride_s_b,
        stride_s_h,
        stride_s_n,
        num_heads,
        n_small,
        factor: tl.constexpr,
        D: tl.constexpr,
    ):
        bh = tl.program_id(0)
        pos = tl.program_id(1)
        b = bh // num_heads
        h = bh - b * num_heads
        large_pos = pos // factor
        local_pos = pos - large_pos * factor
        large_id = tl.load(
            large_idx_ptr + b * stride_i_b + h * stride_i_h + large_pos * stride_i_n
        )
        child_id = large_id * factor + local_pos
        valid = (large_id >= 0) & (child_id < n_small)
        child_safe = tl.maximum(0, tl.minimum(child_id, n_small - 1))
        offs_d = tl.arange(0, D)
        q = tl.load(q_ptr + b * stride_q_b + h * stride_q_h + offs_d * stride_q_d).to(
            tl.float32
        )
        base_c = b * stride_c_b + h * stride_c_h + child_safe * stride_c_n
        cmin = tl.load(cmin_ptr + base_c + offs_d * stride_c_d).to(tl.float32)
        cmax = tl.load(cmax_ptr + base_c + offs_d * stride_c_d).to(tl.float32)
        chosen = tl.where(q > 0, cmax, cmin)
        score = tl.sum(q * chosen, axis=0)
        score = tl.where(valid, score, float("-inf"))
        tl.store(
            scores_ptr + b * stride_s_b + h * stride_s_h + pos * stride_s_n,
            score,
        )

    @triton.jit
    def _quest_map_back_kernel(
        large_idx_ptr,
        sub_pos_ptr,
        out_ptr,
        stride_l_b,
        stride_l_h,
        stride_l_n,
        stride_s_b,
        stride_s_h,
        stride_s_n,
        stride_o_b,
        stride_o_h,
        stride_o_n,
        num_heads,
        n_out,
        n_large,
        n_chunks,
        factor: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        bh = tl.program_id(0)
        block = tl.program_id(1)
        b = bh // num_heads
        h = bh - b * num_heads
        offs = block * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_out
        sub_pos = tl.load(
            sub_pos_ptr + b * stride_s_b + h * stride_s_h + offs * stride_s_n,
            mask=mask,
            other=-1,
        )
        large_pos = sub_pos // factor
        local_pos = sub_pos - large_pos * factor
        valid = mask & (sub_pos >= 0) & (large_pos < n_large)
        large_safe = tl.maximum(0, tl.minimum(large_pos, n_large - 1))
        large_id = tl.load(
            large_idx_ptr + b * stride_l_b + h * stride_l_h + large_safe * stride_l_n,
            mask=valid,
            other=-1,
        )
        mapped = large_id * factor + local_pos
        valid = valid & (large_id >= 0) & (mapped < n_chunks)
        mapped = tl.where(valid, mapped, -1)
        tl.store(
            out_ptr + b * stride_o_b + h * stride_o_h + offs * stride_o_n,
            mapped,
            mask=mask,
        )


def _quest_chunk_score_triton(
    raw_q: torch.Tensor,
    chunk_min: torch.Tensor,
    chunk_max: torch.Tensor,
    scores_out: torch.Tensor,
    n_chunks: int,
    chunk_valid: torch.Tensor | None,
) -> None:
    n_chunks = int(n_chunks)
    if not _HAS_TRITON or not raw_q.is_cuda:
        scores = quest_bound_scores(
            raw_q,
            chunk_min[:, :, :n_chunks, :],
            chunk_max[:, :, :n_chunks, :],
        )
        if chunk_valid is not None:
            scores.masked_fill_(~chunk_valid[:, :, :n_chunks].bool(), float("-inf"))
        scores_out[..., :n_chunks].copy_(scores)
        return
    bs, heads, d = raw_q.shape
    valid = chunk_valid if chunk_valid is not None else scores_out
    _quest_chunk_score_kernel[(bs * heads, n_chunks)](
        raw_q,
        chunk_min,
        chunk_max,
        scores_out,
        valid,
        raw_q.stride(0),
        raw_q.stride(1),
        raw_q.stride(2),
        chunk_min.stride(0),
        chunk_min.stride(1),
        chunk_min.stride(2),
        chunk_min.stride(3),
        scores_out.stride(0),
        scores_out.stride(1),
        scores_out.stride(2),
        valid.stride(0) if chunk_valid is not None else 0,
        valid.stride(1) if chunk_valid is not None else 0,
        valid.stride(2) if chunk_valid is not None else 0,
        heads,
        n_chunks,
        D=d,
        HAS_VALID=chunk_valid is not None,
        num_warps=4 if d == 256 else 2,
    )
    if scores_out.shape[-1] > n_chunks:
        scores_out[..., n_chunks:].fill_(float("-inf"))


class QuestTritonOps(QuestTorchOps):
    """Quest ops preferring fused GPU kernels (Triton-backed API surface)."""

    def quest_chunk_score(
        self,
        raw_q: torch.Tensor,
        chunk_min: torch.Tensor,
        chunk_max: torch.Tensor,
        scores_out: torch.Tensor,
        n_chunks: int,
        chunk_valid: torch.Tensor | None = None,
    ) -> None:
        if raw_q.is_cuda and chunk_min.is_cuda:
            _quest_chunk_score_triton(
                raw_q, chunk_min, chunk_max, scores_out, n_chunks, chunk_valid
            )
            return
        super().quest_chunk_score(
            raw_q, chunk_min, chunk_max, scores_out, n_chunks, chunk_valid
        )

    def quest_sub_chunk_score(
        self,
        raw_q: torch.Tensor,
        chunk_min: torch.Tensor,
        chunk_max: torch.Tensor,
        large_idx: torch.Tensor,
        sub_scores: torch.Tensor,
        nk_large: int,
        factor: int,
    ) -> None:
        if raw_q.is_cuda and chunk_min.is_cuda:
            bs, heads, d = raw_q.shape
            n_out = int(nk_large) * int(factor)
            _quest_sub_chunk_score_kernel[(bs * heads, n_out)](
                raw_q,
                chunk_min,
                chunk_max,
                large_idx,
                sub_scores,
                raw_q.stride(0),
                raw_q.stride(1),
                raw_q.stride(2),
                chunk_min.stride(0),
                chunk_min.stride(1),
                chunk_min.stride(2),
                chunk_min.stride(3),
                large_idx.stride(0),
                large_idx.stride(1),
                large_idx.stride(2),
                sub_scores.stride(0),
                sub_scores.stride(1),
                sub_scores.stride(2),
                heads,
                chunk_min.shape[2],
                factor=int(factor),
                D=d,
                num_warps=4 if d == 256 else 2,
            )
            return
        super().quest_sub_chunk_score(
            raw_q,
            chunk_min,
            chunk_max,
            large_idx,
            sub_scores,
            nk_large,
            factor,
        )

    def quest_map_back(
        self,
        large_idx: torch.Tensor,
        sub_topk_pos: torch.Tensor,
        chunk_idx: torch.Tensor,
        factor: int,
        n_chunks: int,
    ) -> None:
        if large_idx.is_cuda:
            bs, heads, n_out = sub_topk_pos.shape
            block = 128
            _quest_map_back_kernel[(bs * heads, triton.cdiv(n_out, block))](
                large_idx,
                sub_topk_pos,
                chunk_idx,
                large_idx.stride(0),
                large_idx.stride(1),
                large_idx.stride(2),
                sub_topk_pos.stride(0),
                sub_topk_pos.stride(1),
                sub_topk_pos.stride(2),
                chunk_idx.stride(0),
                chunk_idx.stride(1),
                chunk_idx.stride(2),
                heads,
                n_out,
                large_idx.shape[2],
                int(n_chunks),
                factor=int(factor),
                BLOCK=block,
            )
            return
        super().quest_map_back(large_idx, sub_topk_pos, chunk_idx, factor, n_chunks)
