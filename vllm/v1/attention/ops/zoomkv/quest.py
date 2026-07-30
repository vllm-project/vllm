# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyTorch Quest stage-1 scoring for ZoomKV."""

from __future__ import annotations

import torch


def quest_bound_scores(
    raw_q: torch.Tensor,
    chunk_min: torch.Tensor,
    chunk_max: torch.Tensor,
) -> torch.Tensor:
    """Quest upper-bound scores.

    Args:
        raw_q: [bs, kv, D]
        chunk_min / chunk_max: [bs, kv, n_chunks, D]
    Returns:
        scores: [bs, kv, n_chunks] float32
    """
    q = raw_q.unsqueeze(2).to(torch.float32)
    cmin = chunk_min.to(torch.float32)
    cmax = chunk_max.to(torch.float32)
    chosen = torch.where(q > 0, cmax, cmin)
    return (q * chosen).sum(dim=-1)


class QuestTorchOps:
    """Quest hierarchical filtering ops (PyTorch)."""

    def quest_chunk_score(
        self,
        raw_q: torch.Tensor,
        chunk_min: torch.Tensor,
        chunk_max: torch.Tensor,
        scores_out: torch.Tensor,
        n_chunks: int,
        chunk_valid: torch.Tensor | None = None,
    ) -> None:
        n_chunks = int(n_chunks)
        scores = quest_bound_scores(
            raw_q,
            chunk_min[:, :, :n_chunks, :],
            chunk_max[:, :, :n_chunks, :],
        )
        if chunk_valid is not None:
            valid = chunk_valid
            if valid.dtype != torch.bool:
                valid = valid.bool()
            if valid.dim() == 1:
                valid = valid[:n_chunks].view(1, 1, n_chunks)
            elif valid.dim() == 2:
                valid = valid[:, :n_chunks].unsqueeze(0)
            else:
                valid = valid[:, :, :n_chunks]
            scores = scores.masked_fill(~valid, float("-inf"))
        scores_out[..., :n_chunks].copy_(scores)
        if scores_out.shape[-1] > n_chunks:
            scores_out[..., n_chunks:].fill_(float("-inf"))

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
        bs, kv, _ = raw_q.shape
        nk_large = int(nk_large)
        factor = int(factor)
        device = raw_q.device
        offsets = large_idx.to(torch.int64)[..., :nk_large].unsqueeze(
            -1
        ) * factor + torch.arange(factor, device=device, dtype=torch.int64).view(
            1, 1, 1, factor
        )
        flat_ids = offsets.reshape(bs, kv, nk_large * factor)
        n_small = chunk_min.shape[2]
        flat_ids = flat_ids.clamp(0, max(n_small - 1, 0))
        gather_min = torch.gather(
            chunk_min,
            2,
            flat_ids.unsqueeze(-1).expand(-1, -1, -1, chunk_min.shape[-1]),
        )
        gather_max = torch.gather(
            chunk_max,
            2,
            flat_ids.unsqueeze(-1).expand(-1, -1, -1, chunk_max.shape[-1]),
        )
        scores = quest_bound_scores(raw_q, gather_min, gather_max)
        sub_scores[..., : scores.shape[-1]].copy_(scores)
        if sub_scores.shape[-1] > scores.shape[-1]:
            sub_scores[..., scores.shape[-1] :].fill_(float("-inf"))

    def quest_map_back(
        self,
        large_idx: torch.Tensor,
        sub_topk_pos: torch.Tensor,
        chunk_idx: torch.Tensor,
        factor: int,
        n_chunks: int,
    ) -> None:
        factor = int(factor)
        n_chunks = int(n_chunks)
        large_sel = torch.div(sub_topk_pos, factor, rounding_mode="floor")
        local = torch.remainder(sub_topk_pos, factor)
        large_ids = torch.gather(large_idx.to(torch.int64), 2, large_sel.clamp(min=0))
        mapped = large_ids * factor + local
        if n_chunks > 0:
            mapped = mapped.clamp(0, n_chunks - 1)
        chunk_idx.copy_(mapped)
