# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ZoomKV hierarchical Quest + KIVI retrieval pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from vllm.v1.attention.ops.zoomkv.kernels import (
    chunk_density_scores,
    dense_mask_from_topk,
    get_quest_ops,
)
from vllm.v1.attention.ops.zoomkv.kivi_rerank import partial_chunk_kivi_qk
from vllm.v1.attention.ops.zoomkv.state import ZoomKVBlockSummary


@dataclass
class ZoomKVRuntimeConfig:
    sink_size: int = 64
    local_size: int = 256
    final_topk: int = 100
    quest_chunk: int = 16
    quest_large_chunk: int = 256
    quest_large_ratio: float = 0.8
    quest_small_ratio: float = 0.5
    dense_ratio: float = 0.4
    dense_topk: int = 16
    sparse_topk: int = 8
    full_attention_threshold: int = 2000
    dense_fallback: bool = False
    strict_kernels: bool = False
    enable_offload: bool = False
    per_query_head: bool = False

    @property
    def hq_factor(self) -> int:
        return max(1, self.quest_large_chunk // self.quest_chunk)


def _topk_3d(scores: torch.Tensor, k: int, strict: bool = False) -> torch.Tensor:
    from vllm.v1.attention.ops.zoomkv.kernels import float_topk_3d

    return float_topk_3d(scores, k, strict=strict)


def gqa_mean_query(query: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    """Average Q heads within each GQA group.

    Args:
        query: [num_tokens, num_q_heads, head_dim] (decode: num_tokens==1 per req)
               or [bs, num_q_heads, head_dim]
    Returns:
        [bs, num_kv_heads, head_dim]
    """
    if query.dim() == 3 and query.shape[0] != num_kv_heads:
        bs, hq, d = query.shape
        assert hq % num_kv_heads == 0, f"Hq={hq} not divisible by Hkv={num_kv_heads}"
        g = hq // num_kv_heads
        return query.view(bs, num_kv_heads, g, d).mean(dim=2)
    raise ValueError(f"Unexpected query shape {tuple(query.shape)}")


def gqa_max_query(query: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    """Reduce Q heads by selecting the per-dim max absolute query in each group.

    This keeps a representative direction per KV head without averaging away
    conflicting query-head signals.
    """
    if query.dim() == 3 and query.shape[0] != num_kv_heads:
        bs, hq, d = query.shape
        assert hq % num_kv_heads == 0, f"Hq={hq} not divisible by Hkv={num_kv_heads}"
        g = hq // num_kv_heads
        grouped = query.view(bs, num_kv_heads, g, d)
        # Pick the query head with the largest L2 norm inside each KV group.
        norms = grouped.float().norm(dim=-1)  # [bs, Hkv, G]
        idx = norms.argmax(dim=-1, keepdim=True)  # [bs, Hkv, 1]
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, -1, d)
        return torch.gather(grouped, 2, gather_idx).squeeze(2)
    raise ValueError(f"Unexpected query shape {tuple(query.shape)}")


def prepare_retrieval_query(
    query: torch.Tensor,
    num_kv_heads: int,
    per_query_head: bool = False,
) -> torch.Tensor:
    if per_query_head:
        return gqa_max_query(query, num_kv_heads)
    return gqa_mean_query(query, num_kv_heads)


class ZoomKVRetriever:
    def __init__(self, cfg: ZoomKVRuntimeConfig) -> None:
        self.cfg = cfg
        self.quest = get_quest_ops(prefer_triton=True, strict=cfg.strict_kernels)
        # Scratch-buffer cache to align with the reference implementation:
        # score/index tensors depend only on (n_chunks, n_large, kv_heads),
        # which stay constant across all layers of a decode step (and only
        # change when a new block completes ~every block_size tokens).  Reusing
        # them removes per-layer/per-step allocations in the retrieve hot path.
        self._scratch: dict[str, torch.Tensor] = {}

    def _scratch_buf(
        self,
        key: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        fill: float | None = None,
    ) -> torch.Tensor:
        buf = self._scratch.get(key)
        if (
            buf is None
            or tuple(buf.shape) != tuple(shape)
            or buf.dtype != dtype
            or buf.device != device
        ):
            buf = torch.empty(shape, dtype=dtype, device=device)
            self._scratch[key] = buf
        if fill is not None:
            buf.fill_(fill)
        return buf

    def should_use_dense(self, seq_len: int) -> bool:
        if self.cfg.dense_fallback:
            return True
        min_total = max(
            self.cfg.full_attention_threshold,
            self.cfg.sink_size + self.cfg.local_size + self.cfg.final_topk * 2,
        )
        return seq_len < min_total

    def retrieve_topk_from_block_summaries(
        self,
        raw_q: torch.Tensor,
        packed: torch.Tensor,
        cmin: torch.Tensor,
        cmax: torch.Tensor,
        centroid: torch.Tensor,
        valid: torch.Tensor,
        seq_len: int,
        block_size: int,
        start_b: int,
    ) -> torch.Tensor:
        """Run Quest+KIVI on pre-gathered CPU-slot or physical summaries."""
        cfg = self.cfg
        n_chunks = packed.shape[2]
        if n_chunks <= 0:
            return torch.full(
                (1, raw_q.shape[1], cfg.final_topk),
                -1,
                dtype=torch.int64,
                device=raw_q.device,
            )
        factor = cfg.hq_factor
        use_hq = n_chunks >= factor and cfg.quest_large_chunk > cfg.quest_chunk
        if use_hq:
            tmp = ZoomKVBlockSummary.__new__(ZoomKVBlockSummary)
            tmp.num_kv_heads = raw_q.shape[1]
            tmp.head_dim = raw_q.shape[2]
            tmp.blocks_per_parent = factor
            parent_min, parent_max, parent_valid = (
                ZoomKVBlockSummary.build_parent_minmax(
                    tmp, torch.empty(0), cmin, cmax, valid
                )
            )
            chunk_idx = self._hierarchical_quest(
                raw_q,
                cmin,
                cmax,
                parent_min,
                parent_max,
                parent_valid,
                n_chunks,
                factor,
            )
        else:
            chunk_idx = self._flat_quest(raw_q, cmin, cmax, valid, n_chunks)
        topk_local = self._cds_select(
            chunk_idx, packed, cmin, cmax, centroid, raw_q, block_size
        )
        ret_token_offset = start_b * block_size
        return torch.where(
            topk_local >= 0,
            topk_local + ret_token_offset,
            torch.full_like(topk_local, -1),
        )

    def retrieval_block_range(self, seq_len: int, block_size: int) -> tuple[int, int]:
        """Return [start_block, end_block) of retrieval-zone child chunks."""
        sink_blocks = self.cfg.sink_size // block_size
        local_tokens = min(self.cfg.local_size, max(0, seq_len - self.cfg.sink_size))
        local_start = max(self.cfg.sink_size, seq_len - local_tokens)
        # Only fully completed child chunks in the retrieval zone.
        ret_start = self.cfg.sink_size
        ret_end = (local_start // block_size) * block_size
        if ret_end <= ret_start:
            return sink_blocks, sink_blocks
        start_b = ret_start // block_size
        end_b = ret_end // block_size
        return start_b, end_b

    def retrieve_topk_tokens(
        self,
        raw_q: torch.Tensor,
        block_summary: ZoomKVBlockSummary,
        physical_block_ids: torch.Tensor,
        seq_len: int,
        cache_key: tuple | None = None,
    ) -> torch.Tensor:
        """Run Quest + KIVI and return logical token indices in full-seq coords.

        Args:
            raw_q: [1, kv_heads, D] GQA-averaged query
            physical_block_ids: [n_ret_blocks] physical ids for retrieval zone
                in logical order (block i corresponds to tokens
                [ret_start + i*bs, ...))
            seq_len: full sequence length
        Returns:
            topk_logical: [1, kv_heads, final_topk] token indices into the
            full sequence (absolute positions).  Invalid slots are -1.
        """
        cfg = self.cfg
        block_size = block_summary.block_size
        start_b, end_b = self.retrieval_block_range(seq_len, block_size)
        n_ret = end_b - start_b
        if n_ret <= 0 or physical_block_ids.numel() == 0:
            return torch.full(
                (1, raw_q.shape[1], cfg.final_topk),
                -1,
                dtype=torch.int64,
                device=raw_q.device,
            )

        ids = physical_block_ids[:n_ret]
        if cache_key is not None:
            (
                packed,
                cmin,
                cmax,
                centroid,
                valid,
                parent_min,
                parent_max,
                parent_valid,
            ) = block_summary.cached_request_block_summaries(ids, cache_key)
        else:
            packed, cmin, cmax, centroid, valid = (
                block_summary.gather_request_block_summaries(ids)
            )
            parent_min = parent_max = parent_valid = None
        n_chunks = packed.shape[2]
        factor = cfg.hq_factor
        use_hq = n_chunks >= factor and cfg.quest_large_chunk > cfg.quest_chunk

        if use_hq:
            if parent_min is None:
                parent_min, parent_max, parent_valid = (
                    block_summary.build_parent_minmax(ids, cmin, cmax, valid)
                )
            chunk_idx = self._hierarchical_quest(
                raw_q,
                cmin,
                cmax,
                parent_min,
                parent_max,
                parent_valid,
                n_chunks,
                factor,
            )
        else:
            chunk_idx = self._flat_quest(raw_q, cmin, cmax, valid, n_chunks)

        topk_local = self._cds_select(
            chunk_idx, packed, cmin, cmax, centroid, raw_q, block_size
        )
        # Map retrieval-zone-local token ids → absolute sequence positions.
        ret_token_offset = start_b * block_size
        abs_idx = torch.where(
            topk_local >= 0,
            topk_local + ret_token_offset,
            torch.full_like(topk_local, -1),
        )
        return abs_idx

    def _flat_quest(
        self,
        raw_q: torch.Tensor,
        cmin: torch.Tensor,
        cmax: torch.Tensor,
        valid: torch.Tensor,
        n_chunks: int,
    ) -> torch.Tensor:
        cfg = self.cfg
        scores = self._scratch_buf(
            "flat_scores",
            (1, raw_q.shape[1], n_chunks),
            torch.float32,
            raw_q.device,
            fill=float("-inf"),
        )
        self.quest.quest_chunk_score(raw_q, cmin, cmax, scores, n_chunks, valid)
        # Candidate budget ~ ratio of chunks (aligned with ZoomKV defaults).
        target = max(1, int(math.ceil(n_chunks * cfg.quest_small_ratio)))
        nk = min(n_chunks, target)
        return _topk_3d(scores, nk, strict=cfg.strict_kernels)

    def _hierarchical_quest(
        self,
        raw_q: torch.Tensor,
        cmin: torch.Tensor,
        cmax: torch.Tensor,
        parent_min: torch.Tensor,
        parent_max: torch.Tensor,
        parent_valid: torch.Tensor,
        n_chunks: int,
        factor: int,
    ) -> torch.Tensor:
        cfg = self.cfg
        n_large = parent_min.shape[2]
        nk_large = max(1, int(math.ceil(n_large * cfg.quest_large_ratio)))
        nk_large = min(nk_large, n_large)
        large_scores = self._scratch_buf(
            "large_scores",
            (1, raw_q.shape[1], n_large),
            torch.float32,
            raw_q.device,
            fill=float("-inf"),
        )
        self.quest.quest_chunk_score(
            raw_q, parent_min, parent_max, large_scores, n_large, parent_valid
        )
        large_idx = _topk_3d(large_scores, nk_large, strict=cfg.strict_kernels)

        sub_scores = self._scratch_buf(
            "sub_scores",
            (1, raw_q.shape[1], nk_large * factor),
            torch.float32,
            raw_q.device,
            fill=float("-inf"),
        )
        self.quest.quest_sub_chunk_score(
            raw_q, cmin, cmax, large_idx, sub_scores, nk_large, factor
        )
        nk_small = max(1, int(math.ceil(nk_large * factor * cfg.quest_small_ratio)))
        nk_small = min(nk_small, nk_large * factor)
        sub_pos = _topk_3d(sub_scores, nk_small, strict=cfg.strict_kernels)
        chunk_idx = self._scratch_buf(
            "chunk_idx",
            (1, raw_q.shape[1], nk_small),
            torch.int64,
            raw_q.device,
        )
        self.quest.quest_map_back(large_idx, sub_pos, chunk_idx, factor, n_chunks)
        return chunk_idx

    def _cds_select(
        self,
        chunk_idx: torch.Tensor,
        packed: torch.Tensor,
        cmin: torch.Tensor,
        cmax: torch.Tensor,
        centroid: torch.Tensor,
        raw_q: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        cfg = self.cfg
        nk = chunk_idx.shape[2]
        # Density via centroid @ q
        density = chunk_density_scores(
            chunk_idx, centroid, raw_q, strict=cfg.strict_kernels
        )
        n_dense = max(1, int(nk * cfg.dense_ratio))
        n_dense = min(n_dense, nk)
        dense_pos = _topk_3d(density, n_dense, strict=cfg.strict_kernels)
        dense_mask = dense_mask_from_topk(dense_pos, nk, strict=cfg.strict_kernels)

        dense_topk = max(1, min(cfg.dense_topk, block_size))
        sparse_topk = max(1, min(cfg.sparse_topk, block_size))
        out_scores, out_indices = partial_chunk_kivi_qk(
            chunk_idx,
            dense_mask,
            packed,
            cmin,
            cmax,
            raw_q.to(cmin.dtype),
            group_size=block_size,
            dense_topk=dense_topk,
            sparse_topk=sparse_topk,
            strict=cfg.strict_kernels,
        )
        actual_topk = min(cfg.final_topk, out_scores.shape[-1])
        top_pos = out_scores.topk(actual_topk, dim=-1, largest=True).indices
        selected = torch.gather(out_indices, -1, top_pos)
        if actual_topk < cfg.final_topk:
            pad = torch.full(
                (
                    selected.shape[0],
                    selected.shape[1],
                    cfg.final_topk - actual_topk,
                ),
                -1,
                dtype=torch.int64,
                device=raw_q.device,
            )
            selected = torch.cat([selected, pad], dim=-1)
        return selected
