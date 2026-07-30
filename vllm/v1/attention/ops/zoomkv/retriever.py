# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ZoomKV hierarchical Quest + KIVI retrieval pipeline."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import torch

from vllm.v1.attention.ops.zoomkv import stage_timer as _zt
from vllm.v1.attention.ops.zoomkv.kernels import (
    chunk_density_scores,
    density_score_physical,
    dense_mask_from_topk,
    direct_physical_retrieval_available,
    float_topk_values_3d,
    get_quest_ops,
    kivi_physical,
    quest_chunk_score_physical,
    quest_parent_score_physical,
    quest_sub_score_physical,
)
from vllm.v1.attention.ops.zoomkv.kivi_rerank import partial_chunk_kivi_qk
from vllm.v1.attention.ops.zoomkv.state import ZoomKVBlockSummary

_RETRIEVE_DETAIL_TIMER = (
    os.environ.get("VLLM_ZOOMKV_RETRIEVE_STAGE_TIMER", "0") == "1"
)


class _NoopStage:
    __slots__ = ()

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> None:
        return None


_NOOP_STAGE = _NoopStage()


def _retrieve_stage(name: str) -> _zt.Stage | _NoopStage:
    if _RETRIEVE_DETAIL_TIMER:
        return _zt.Stage(f"retrieve.{name}")
    return _NOOP_STAGE


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


@dataclass(frozen=True)
class ZoomKVRetrievalResult:
    topk: torch.Tensor
    context_fully_valid: bool
    used_direct_physical: bool


def _topk_3d(scores: torch.Tensor, k: int, strict: bool = False) -> torch.Tensor:
    from vllm.v1.attention.ops.zoomkv.kernels import float_topk_3d

    return float_topk_3d(scores, k, strict=strict)


def gqa_mean_query(query: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    """Average Q heads within each GQA group.

    Args:
        query: [bs, num_q_heads, head_dim] (decode: bs==num_reqs, q_len packed)
    Returns:
        [bs, num_kv_heads, head_dim]
    """
    if query.dim() != 3:
        raise ValueError(f"Unexpected query shape {tuple(query.shape)}")
    bs, hq, d = query.shape
    if hq % num_kv_heads != 0:
        raise ValueError(f"Hq={hq} not divisible by Hkv={num_kv_heads}")
    g = hq // num_kv_heads
    return query.view(bs, num_kv_heads, g, d).mean(dim=2)


def gqa_max_query(query: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    """Reduce Q heads by selecting the per-dim max absolute query in each group.

    This keeps a representative direction per KV head without averaging away
    conflicting query-head signals.
    """
    if query.dim() != 3:
        raise ValueError(f"Unexpected query shape {tuple(query.shape)}")
    bs, hq, d = query.shape
    if hq % num_kv_heads != 0:
        raise ValueError(f"Hq={hq} not divisible by Hkv={num_kv_heads}")
    g = hq // num_kv_heads
    grouped = query.view(bs, num_kv_heads, g, d)
    # Pick the query head with the largest L2 norm inside each KV group.
    norms = grouped.float().norm(dim=-1)  # [bs, Hkv, G]
    idx = norms.argmax(dim=-1, keepdim=True)  # [bs, Hkv, 1]
    gather_idx = idx.unsqueeze(-1).expand(-1, -1, -1, d)
    return torch.gather(grouped, 2, gather_idx).squeeze(2)


def prepare_retrieval_query(
    query: torch.Tensor,
    num_kv_heads: int,
    per_query_head: bool = False,
) -> torch.Tensor:
    if per_query_head:
        out = gqa_max_query(query, num_kv_heads)
    else:
        out = gqa_mean_query(query, num_kv_heads)
    # Direct physical kernels require a dense [B,H,D] layout.
    return out.contiguous() if not out.is_contiguous() else out


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
        # Set by retrieve_* when sink+local+topk are guaranteed fully valid,
        # so the attention backend can skip host-side ``_ctx_valid.all()`` sync.
        self.last_context_fully_valid: bool = False
        self._last_topk_fully_filled: bool = False

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
        batch = raw_q.shape[0]
        n_chunks = packed.shape[2]
        if n_chunks <= 0:
            return torch.full(
                (batch, raw_q.shape[1], cfg.final_topk),
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
                (raw_q.shape[0], raw_q.shape[1], cfg.final_topk),
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
            packed, cmin, cmax, centroid, valid = (block_summary.gather_request_block_summaries(ids))
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

    def retrieve_topk_tokens_batch(
        self,
        raw_q: torch.Tensor,
        block_summary: ZoomKVBlockSummary,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Compatibility wrapper returning only Top-K logical token ids."""
        return self.retrieve_topk_tokens_batch_result(
            raw_q,
            block_summary,
            block_table,
            seq_lens,
            summaries_guaranteed_valid=False,
        ).topk

    def retrieve_topk_tokens_batch_result(
        self,
        raw_q: torch.Tensor,
        block_summary: ZoomKVBlockSummary,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        summaries_guaranteed_valid: bool,
    ) -> ZoomKVRetrievalResult:
        """Batched Quest + KIVI over a decode batch of requests.

        Args:
            raw_q: [B, kv_heads, D]
            block_table: [B, max_blocks] physical block ids
            seq_lens: [B] full sequence lengths (CPU or GPU)
        Returns:
            Explicit Top-K/direct-path validity result. The caller may use
            unmasked FlashAttention only when ``context_fully_valid`` is true.
        """
        cfg = self.cfg
        batch = raw_q.shape[0]
        block_size = block_summary.block_size
        device = raw_q.device
        self.last_context_fully_valid = False
        self._last_topk_fully_filled = False
        with _retrieve_stage("metadata"):
            if isinstance(seq_lens, torch.Tensor):
                seq_list = [int(x) for x in seq_lens.tolist()]
            else:
                seq_list = [int(x) for x in seq_lens]

            ranges = [
                self.retrieval_block_range(seq_len, block_size)
                for seq_len in seq_list
            ]
        start_b = ranges[0][0]
        # sink_size is constant, so start_b is identical for every request.
        if any(s != start_b for s, _ in ranges):
            raise RuntimeError("ZoomKV batched retrieve requires uniform sink_size")
        n_rets = [max(0, e - s) for s, e in ranges]
        max_n = max(n_rets) if n_rets else 0
        # Long-context sparse decode always fills sink+local to full width.
        sink_local_full = all(
            int(s) >= cfg.sink_size + cfg.local_size for s in seq_list
        )
        if max_n <= 0:
            topk = torch.full(
                (batch, raw_q.shape[1], cfg.final_topk),
                -1,
                dtype=torch.int64,
                device=device,
            )
            return ZoomKVRetrievalResult(
                topk=topk,
                context_fully_valid=False,
                used_direct_physical=False,
            )

        # Quest / KIVI candidate budgets scale with n_chunks. Padding shorter
        # requests to max_n would change those budgets vs the serial path, so
        # run direct physical retrieval per request when widths differ.
        # This preserves exact budgets without materializing summaries.
        if len(set(n_rets)) != 1:
            bt = block_table.to(device=device)
            direct_slices = [
                bt[i : i + 1, s:e].contiguous()
                for i, (s, e) in enumerate(ranges)
            ]
            can_direct = all(
                n > 0
                and self._can_use_direct_physical(
                    raw_q[i : i + 1], block_summary, direct_slices[i]
                )
                for i, n in enumerate(n_rets)
            )
            if can_direct:
                topk = torch.empty(
                    batch,
                    raw_q.shape[1],
                    cfg.final_topk,
                    dtype=torch.int64,
                    device=device,
                )
                all_widths_filled = True
                for i, ((start, _), n_ret) in enumerate(zip(ranges, n_rets)):
                    topk_i = self._retrieve_topk_physical(
                        raw_q[i : i + 1],
                        block_summary,
                        direct_slices[i],
                        n_ret,
                        start * block_size,
                    )
                    all_widths_filled = (
                        all_widths_filled and self._last_topk_fully_filled
                    )
                    topk[i : i + 1].copy_(topk_i)
                fully_valid = (
                    summaries_guaranteed_valid
                    and sink_local_full
                    and all_widths_filled
                )
                self.last_context_fully_valid = fully_valid
                return ZoomKVRetrievalResult(
                    topk=topk,
                    context_fully_valid=fully_valid,
                    used_direct_physical=True,
                )

            # Generic compatibility fallback for unsupported dtype/layout.
            outs = []
            for i, ((s, e), seq_len) in enumerate(zip(ranges, seq_list)):
                phys = (
                    bt[i, s:e].to(torch.int64)
                    if e > s
                    else torch.empty(0, dtype=torch.int64, device=device)
                )
                outs.append(
                    self.retrieve_topk_tokens(
                        raw_q[i : i + 1], block_summary, phys, seq_len
                    )
                )
            return ZoomKVRetrievalResult(
                topk=torch.cat(outs, dim=0),
                context_fully_valid=False,
                used_direct_physical=False,
            )

        with _retrieve_stage("physical_ids"):
            # Uniform retrieval widths can use the block-table slice directly.
            # This is the hot serving path and avoids per-layer zeros+copy.
            bt = block_table[:batch, start_b : start_b + max_n]
            if not bt.is_contiguous() or bt.stride(-1) != 1:
                bt = bt.contiguous()
            use_direct = self._can_use_direct_physical(raw_q, block_summary, bt)
        if use_direct:
            topk = self._retrieve_topk_physical(
                raw_q,
                block_summary,
                bt,
                max_n,
                start_b * block_size,
            )
            # Direct physical CDS fills final_topk without -1 padding when
            # the KIVI candidate pool is wide enough; combined with full
            # sink+local this guarantees a fully-valid sparse context.
            self.last_context_fully_valid = (summaries_guaranteed_valid and sink_local_full and self._last_topk_fully_filled)
            return ZoomKVRetrievalResult(
                topk=topk,
                context_fully_valid=self.last_context_fully_valid,
                used_direct_physical=True,
            )
        with _retrieve_stage("physical_ids_materialize"):
            phys_ids = bt.to(device=device, dtype=torch.int64)
            chunk_valid = None

        with _retrieve_stage("summary_gather"):
            packed, cmin, cmax, centroid, valid = (
                block_summary.gather_batch_block_summaries(
                    phys_ids,
                    chunk_valid,
                    assume_valid_ids=True,
                )
            )
        n_chunks = packed.shape[2]
        factor = cfg.hq_factor
        use_hq = n_chunks >= factor and cfg.quest_large_chunk > cfg.quest_chunk
        if use_hq:
            with _retrieve_stage("parent_minmax"):
                parent_min, parent_max, parent_valid = (
                    block_summary.build_parent_minmax(phys_ids, cmin, cmax, valid)
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

        with _retrieve_stage("cds_select"):
            topk_local = self._cds_select(
                chunk_idx, packed, cmin, cmax, centroid, raw_q, block_size
            )
        ret_token_offset = start_b * block_size
        topk_abs = self._scratch_buf(
            "topk_abs",
            tuple(topk_local.shape),
            torch.int64,
            raw_q.device,
        )
        topk_abs.copy_(topk_local)
        valid_topk = topk_abs >= 0
        topk_abs.add_(ret_token_offset)
        topk_abs.masked_fill_(~valid_topk, -1)
        # Prefer the direct physical path for fully-valid; avoid a host sync
        # here just to advertise the guarantee on the slower materialize path.
        return ZoomKVRetrievalResult(
            topk=topk_abs,
            context_fully_valid=False,
            used_direct_physical=False,
        )

    @staticmethod
    def _can_use_direct_physical(
        raw_q: torch.Tensor,
        block_summary: ZoomKVBlockSummary,
        physical_ids: torch.Tensor,
    ) -> bool:
        """Keep direct kernels behind an exact-layout compatibility gate."""
        return (
            direct_physical_retrieval_available()
            and raw_q.is_cuda
            and raw_q.is_contiguous()
            and raw_q.dtype == torch.bfloat16
            and raw_q.shape[-1] in (128, 256)
            and physical_ids.is_cuda
            and physical_ids.device == raw_q.device
            and physical_ids.dtype == torch.int32
            and physical_ids.stride(-1) == 1
            and block_summary.block_size == 16
            and block_summary.dtype == torch.bfloat16
            and block_summary.chunk_min.device == raw_q.device
        )

    def _retrieve_topk_physical(
        self,
        raw_q: torch.Tensor,
        block_summary: ZoomKVBlockSummary,
        physical_ids: torch.Tensor,
        n_chunks: int,
        token_offset: int,
    ) -> torch.Tensor:
        """Quest+CDS+KIVI directly over global physical summary pools."""
        cfg = self.cfg
        factor = cfg.hq_factor
        use_hq = n_chunks >= factor and cfg.quest_large_chunk > cfg.quest_chunk
        if use_hq:
            chunk_idx = self._hierarchical_quest_physical(
                raw_q, block_summary, physical_ids, n_chunks, factor
            )
        else:
            chunk_idx = self._flat_quest_physical(
                raw_q, block_summary, physical_ids, n_chunks
            )
        return self._cds_select_physical(
            chunk_idx,
            raw_q,
            block_summary,
            physical_ids,
            n_chunks,
            token_offset,
        )

    def _flat_quest_physical(
        self,
        raw_q: torch.Tensor,
        block_summary: ZoomKVBlockSummary,
        physical_ids: torch.Tensor,
        n_chunks: int,
    ) -> torch.Tensor:
        cfg = self.cfg
        scores = self._scratch_buf(
            "flat_scores",
            (raw_q.shape[0], raw_q.shape[1], n_chunks),
            torch.float32,
            raw_q.device,
        )
        with _retrieve_stage("quest_flat_score_direct"):
            quest_chunk_score_physical(
                raw_q,
                physical_ids,
                block_summary.chunk_min,
                block_summary.chunk_max,
                block_summary.valid,
                scores,
                n_chunks,
            )
        nk = min(
            n_chunks,
            max(1, int(math.ceil(n_chunks * cfg.quest_small_ratio))),
        )
        with _retrieve_stage("quest_flat_topk"):
            return _topk_3d(scores, nk, strict=cfg.strict_kernels)

    def _hierarchical_quest_physical(
        self,
        raw_q: torch.Tensor,
        block_summary: ZoomKVBlockSummary,
        physical_ids: torch.Tensor,
        n_chunks: int,
        factor: int,
    ) -> torch.Tensor:
        cfg = self.cfg
        batch, kv_heads = raw_q.shape[:2]
        n_large = n_chunks // factor
        nk_large = min(
            n_large,
            max(1, int(math.ceil(n_large * cfg.quest_large_ratio))),
        )
        large_scores = self._scratch_buf(
            "large_scores",
            (batch, kv_heads, n_large),
            torch.float32,
            raw_q.device,
        )
        with _retrieve_stage("quest_parent_score_direct"):
            quest_parent_score_physical(
                raw_q,
                physical_ids,
                block_summary.chunk_min,
                block_summary.chunk_max,
                block_summary.valid,
                large_scores,
                n_chunks,
                factor,
            )
        with _retrieve_stage("quest_large_topk"):
            large_idx = _topk_3d(
                large_scores, nk_large, strict=cfg.strict_kernels
            )

        sub_scores = self._scratch_buf(
            "sub_scores",
            (batch, kv_heads, nk_large * factor),
            torch.float32,
            raw_q.device,
        )
        with _retrieve_stage("quest_sub_score_direct"):
            quest_sub_score_physical(
                raw_q,
                physical_ids,
                block_summary.chunk_min,
                block_summary.chunk_max,
                block_summary.valid,
                large_idx,
                sub_scores,
                nk_large,
                factor,
                n_chunks,
            )
        nk_small = min(
            nk_large * factor,
            max(
                1,
                int(
                    math.ceil(
                        nk_large * factor * cfg.quest_small_ratio
                    )
                ),
            ),
        )
        with _retrieve_stage("quest_sub_topk"):
            sub_pos = _topk_3d(
                sub_scores, nk_small, strict=cfg.strict_kernels
            )
        chunk_idx = self._scratch_buf(
            "chunk_idx",
            (batch, kv_heads, nk_small),
            torch.int64,
            raw_q.device,
        )
        with _retrieve_stage("quest_map_back"):
            self.quest.quest_map_back(
                large_idx, sub_pos, chunk_idx, factor, n_chunks
            )
        return chunk_idx

    def _cds_select_physical(
        self,
        chunk_idx: torch.Tensor,
        raw_q: torch.Tensor,
        block_summary: ZoomKVBlockSummary,
        physical_ids: torch.Tensor,
        n_chunks: int,
        token_offset: int,
    ) -> torch.Tensor:
        cfg = self.cfg
        batch, kv_heads = raw_q.shape[:2]
        nk = chunk_idx.shape[2]
        density = self._scratch_buf(
            "density_scores",
            (batch, kv_heads, nk),
            torch.float32,
            raw_q.device,
        )
        with _retrieve_stage("cds_density_direct"):
            density_score_physical(
                chunk_idx,
                physical_ids,
                block_summary.centroid,
                block_summary.valid,
                raw_q,
                density,
                n_chunks,
            )
        n_dense = min(nk, max(1, int(nk * cfg.dense_ratio)))
        with _retrieve_stage("cds_density_topk"):
            dense_pos = _topk_3d(
                density, n_dense, strict=cfg.strict_kernels
            )
        dense_mask = self._scratch_buf(
            "dense_mask",
            (batch, kv_heads, nk),
            torch.bool,
            raw_q.device,
        )
        dense_mask_from_topk(
            dense_pos,
            nk,
            out=dense_mask,
            strict=cfg.strict_kernels,
        )

        dense_topk = max(1, min(cfg.dense_topk, block_summary.block_size))
        sparse_topk = max(1, min(cfg.sparse_topk, block_summary.block_size))
        out_width = nk * block_summary.block_size
        out_scores = self._scratch_buf(
            "kivi_scores",
            (batch, kv_heads, out_width),
            torch.float32,
            raw_q.device,
        )
        out_indices = self._scratch_buf(
            "kivi_indices",
            (batch, kv_heads, out_width),
            torch.int64,
            raw_q.device,
        )
        with _retrieve_stage("kivi_qk_direct"):
            kivi_physical(
                chunk_idx,
                dense_mask,
                physical_ids,
                block_summary.packed,
                block_summary.chunk_min,
                block_summary.chunk_max,
                block_summary.valid,
                raw_q,
                dense_topk,
                sparse_topk,
                token_offset,
                out_scores,
                out_indices,
            )
        actual_topk = min(cfg.final_topk, out_width)
        with _retrieve_stage("final_topk_direct"):
            selected = float_topk_values_3d(
                out_scores,
                out_indices,
                actual_topk,
                strict=cfg.strict_kernels,
            )
        if actual_topk == cfg.final_topk:
            # final_topk slots are filled from KIVI candidates; no -1 padding.
            self._last_topk_fully_filled = True
            return selected
        self._last_topk_fully_filled = False
        padded = self._scratch_buf(
            "selected_direct",
            (batch, kv_heads, cfg.final_topk),
            torch.int64,
            raw_q.device,
            fill=-1,
        )
        padded[..., :actual_topk].copy_(selected)
        return padded

    def _flat_quest(
        self,
        raw_q: torch.Tensor,
        cmin: torch.Tensor,
        cmax: torch.Tensor,
        valid: torch.Tensor,
        n_chunks: int,
    ) -> torch.Tensor:
        cfg = self.cfg
        batch = raw_q.shape[0]
        scores = self._scratch_buf(
            "flat_scores",
            (batch, raw_q.shape[1], n_chunks),
            torch.float32,
            raw_q.device,
        )
        with _retrieve_stage("quest_flat_score"):
            self.quest.quest_chunk_score(raw_q, cmin, cmax, scores, n_chunks, valid)
        # Candidate budget ~ ratio of chunks (aligned with ZoomKV defaults).
        target = max(1, int(math.ceil(n_chunks * cfg.quest_small_ratio)))
        nk = min(n_chunks, target)
        with _retrieve_stage("quest_flat_topk"):
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
        batch = raw_q.shape[0]
        n_large = parent_min.shape[2]
        nk_large = max(1, int(math.ceil(n_large * cfg.quest_large_ratio)))
        nk_large = min(nk_large, n_large)
        large_scores = self._scratch_buf(
            "large_scores",
            (batch, raw_q.shape[1], n_large),
            torch.float32,
            raw_q.device,
        )
        with _retrieve_stage("quest_large_score"):
            self.quest.quest_chunk_score(
                raw_q, parent_min, parent_max, large_scores, n_large, parent_valid
            )
        with _retrieve_stage("quest_large_topk"):
            large_idx = _topk_3d(large_scores, nk_large, strict=cfg.strict_kernels)

        sub_scores = self._scratch_buf(
            "sub_scores",
            (batch, raw_q.shape[1], nk_large * factor),
            torch.float32,
            raw_q.device,
        )
        with _retrieve_stage("quest_sub_score"):
            self.quest.quest_sub_chunk_score(
                raw_q, cmin, cmax, large_idx, sub_scores, nk_large, factor
            )
        nk_small = max(1, int(math.ceil(nk_large * factor * cfg.quest_small_ratio)))
        nk_small = min(nk_small, nk_large * factor)
        with _retrieve_stage("quest_sub_topk"):
            sub_pos = _topk_3d(sub_scores, nk_small, strict=cfg.strict_kernels)
        chunk_idx = self._scratch_buf(
            "chunk_idx",
            (batch, raw_q.shape[1], nk_small),
            torch.int64,
            raw_q.device,
        )
        with _retrieve_stage("quest_map_back"):
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
        batch = raw_q.shape[0]
        kv_heads = raw_q.shape[1]
        nk = chunk_idx.shape[2]
        # Density via centroid @ q
        density = self._scratch_buf(
            "density_scores",
            (batch, kv_heads, nk),
            torch.float32,
            raw_q.device,
        )
        density = chunk_density_scores(
            chunk_idx,
            centroid,
            raw_q,
            out=density,
            strict=cfg.strict_kernels,
        )
        n_dense = max(1, int(nk * cfg.dense_ratio))
        n_dense = min(n_dense, nk)
        with _retrieve_stage("cds_density_topk"):
            dense_pos = _topk_3d(density, n_dense, strict=cfg.strict_kernels)
        dense_mask = self._scratch_buf(
            "dense_mask",
            (batch, kv_heads, nk),
            torch.bool,
            raw_q.device,
        )
        dense_mask = dense_mask_from_topk(
            dense_pos,
            nk,
            out=dense_mask,
            strict=cfg.strict_kernels,
        )

        dense_topk = max(1, min(cfg.dense_topk, block_size))
        sparse_topk = max(1, min(cfg.sparse_topk, block_size))
        out_width = nk * block_size
        out_scores = self._scratch_buf(
            "kivi_scores",
            (batch, kv_heads, out_width),
            torch.float32,
            raw_q.device,
        )
        out_indices = self._scratch_buf(
            "kivi_indices",
            (batch, kv_heads, out_width),
            torch.int64,
            raw_q.device,
        )
        with _retrieve_stage("kivi_qk"):
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
                out_scores=out_scores,
                out_indices=out_indices,
                strict=cfg.strict_kernels,
            )
        actual_topk = min(cfg.final_topk, out_scores.shape[-1])
        # Prefer fused CUDA radix top-k when available; otherwise torch.topk.
        with _retrieve_stage("final_topk"):
            top_pos = _topk_3d(out_scores, actual_topk, strict=False)
        selected = self._scratch_buf(
            "selected",
            (batch, kv_heads, cfg.final_topk),
            torch.int64,
            raw_q.device,
        )
        torch.gather(out_indices, -1, top_pos, out=selected[..., :actual_topk])
        if actual_topk < cfg.final_topk:
            selected[..., actual_topk:].fill_(-1)
        return selected
