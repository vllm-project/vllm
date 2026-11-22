# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""index_score stage: decode-time index page scores, served by upstream
vLLM's `minimax_m3_index_decode_score_cutedsl`
(`vllm/models/minimax_m3/nvidia/ops/index_decode_score.py`).

Upstream reuse, no port: that CuTe DSL kernel accepts an FP8 E4M3 index query
and an FP8 E4M3 index-K cache ([pages, 128, D], unscaled e4m3, as written by
upstream's `MiniMaxM3IndexerCache` for `indexer_kv_dtype="fp8"`) and
writes fp32 page scores by strides, so it serves the fp8 indexer path of the
specialized decode as is. Its grid is (batch, split_k = 256) and it does not
spin, so it needs no residency check.

NVFP4 index caches are rejected (`_dtypes`); the upstream NVFP4 index path
is a different kernel and a different cache layout.
"""

from __future__ import annotations

from typing import Any

from . import _dtypes

PAGE = 128
# Rows of the largest launch: the grid is per request, so rows only bound the executor's
# largest step (8 requests x qlen 8).
MAX_ROWS = 64
# CTAs per request of the scorer grid (batch, split_k); the scores do not depend on it.
# Upstream's wrapper uses 256, which the steps of up to SPLIT_MIN_ROWS rows (one request
# at K = 7) keep. With two or more requests the grid is batch-fastest, so a long
# request's CTAs are every batch-th block and land on a fraction of the SMs; a larger
# split spreads them (768 measured faster for a long request next to a short one).
UPSTREAM_SPLIT_K = 256
SPLIT_MIN_ROWS = 8
SPLIT_K_MULTI = 768


def split_k_for(rows: int) -> int:
    """The scorer split of a step of `rows` rows."""
    return UPSTREAM_SPLIT_K if rows <= SPLIT_MIN_ROWS else SPLIT_K_MULTI


class StockIndexScoreStage:
    """`__call__(idx_q, index_cache, block_table, seq_lens, score_out, *,
    decode_query_len)`

    idx_q        e4m3 [batch * qlen, H_idx, D]   (H_idx per rank; H_idx * max_qlen <= 32)
    index_cache  e4m3 [pages, 128, D]
    block_table  int32 [batch, max_pages]
    seq_lens     int32 [batch]
    score_out    float32 [H_idx, batch * qlen, >= max_pages] (may be a transposed view of
                 upstream `unified_scores[:nd]`)
    """  # noqa: E501

    stage = "index_score"

    def __init__(
        self,
        *,
        num_index_heads: int,
        max_decode_query_len: int,
        indexer_kv_dtype: str = "fp8",
        max_rows: int = MAX_ROWS,
    ):
        # Rejects nvfp4 (and anything without a kernel here) with a clear error.
        self.fmt = _dtypes.indexer_cache_format(indexer_kv_dtype)
        if num_index_heads * max_decode_query_len > 32:
            raise ValueError(
                f"index_score: {num_index_heads} index heads x qlen "
                f"{max_decode_query_len} > 32 "
                "(the upstream CuTe DSL decode-score tile)"
            )
        self.num_index_heads = int(num_index_heads)
        self.max_decode_query_len = int(max_decode_query_len)
        self.max_rows = int(max_rows)
        self._fn: Any = None
        self._kernels: dict[
            int, Any
        ] = {}  # split_k -> compiled kernel (not upstream's 256)

    def build(self):
        from vllm.models.minimax_m3.nvidia.ops.index_decode_score import (
            minimax_m3_index_decode_score_cutedsl,
        )

        self._fn = minimax_m3_index_decode_score_cutedsl
        import torch

        from vllm.models.minimax_m3.nvidia.ops.index_decode_score import (
            _TORCH_TO_CUTE_DTYPE,
            IndexDecodeScoreKernel,
        )

        # Compile every split the row counts use now, not inside a capture.
        for sk in {split_k_for(r) for r in range(1, self.max_rows + 1)} - {
            UPSTREAM_SPLIT_K
        }:
            self._kernels[sk] = IndexDecodeScoreKernel.compile(
                _TORCH_TO_CUTE_DTYPE[torch.float8_e4m3fn],
                self.num_index_heads,
                self.max_decode_query_len,
                sk,
                128,
            )
        return self

    def __call__(
        self,
        idx_q,
        index_cache,
        block_table,
        seq_lens,
        score_out,
        *,
        decode_query_len: int,
    ):
        import torch

        if self._fn is None:
            self.build()
        if idx_q.shape[0] > self.max_rows:
            raise ValueError(f"index_score: {idx_q.shape[0]} rows > {self.max_rows}")
        if idx_q.dtype != torch.float8_e4m3fn:
            raise TypeError(f"index_score: fp8 index query expected, got {idx_q.dtype}")
        _dtypes.check_cache_tensor("indexer K", index_cache, self.fmt)
        if index_cache.dtype != torch.float8_e4m3fn:
            index_cache = index_cache.view(torch.float8_e4m3fn)
        if index_cache.dim() != 3 or index_cache.shape[1] != PAGE:
            raise ValueError(
                "index_score: index cache [pages, 128, D] expected, got "
                f"{tuple(index_cache.shape)}"
            )
        kernel = self._kernels.get(split_k_for(idx_q.shape[0]))
        if kernel is not None:
            if idx_q.shape[0] != block_table.shape[0] * int(decode_query_len):
                raise ValueError("index_score: rows != batch x decode_query_len")
            kernel(idx_q, index_cache, block_table, score_out, seq_lens)
            return score_out
        return self._fn(
            idx_q,
            index_cache,
            block_table,
            seq_lens,
            0,  # max_seq_len: unused by the CuTe DSL kernel
            0,  # init_blocks: unused (forced pages are page_topk's job)
            0,  # local_blocks: unused
            1,  # num_kv_heads: unused
            int(decode_query_len),
            self.max_decode_query_len,
            score_out,
        )


__all__ = ["StockIndexScoreStage"]
