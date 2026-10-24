# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER FlyDSL kernels for the Qwen4Exp QSA indexer and sparse GQA.

Opt-in with ``VLLM_ROCM_QSA_FLYDSL=1``. K1 replaces paged MQA scoring plus
top-k and writes the same compressed block ids, so the Triton expand kernel
and the MTP top-k reuse are unchanged. K2 replaces the sparse paged GQA and
reads the K|V cache views in place. A shape either kernel cannot serve stays
on the Triton path.
"""

from __future__ import annotations

import os
from functools import cache
from types import ModuleType

import torch

from vllm.logger import init_logger

from .qsa import _LOGITS_WORKSPACE_BYTES, expand_qsa_block_indices_cuda

logger = init_logger(__name__)

_ENABLED = os.getenv("VLLM_ROCM_QSA_FLYDSL", "0") == "1"


@cache
def _flydsl_qsa() -> ModuleType | None:
    if not _ENABLED:
        return None
    from aiter.ops.flydsl import qsa

    logger.info_once("Using AITER FlyDSL QSA kernels where the shape is served.")
    return qsa


def flydsl_select_paged_tokens(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    out: torch.Tensor | None,
) -> torch.Tensor | None:
    """``qsa_select_paged_tokens`` with FlyDSL K1 scoring and top-k.

    Returns None when K1 cannot serve these tensors.
    """
    qsa = _flydsl_qsa()
    if qsa is None:
        return None
    reason = qsa.qsa_k1_selection_serves(
        token_topk, compress_ratio
    ) or qsa.qsa_k1_serves(q, k_cache, page_table)
    if reason is not None:
        logger.warning_once("FlyDSL QSA K1 skipped: %s", reason)
        return None
    logger.info_once("Using AITER FlyDSL QSA K1 indexer.")

    rows = q.shape[0]
    output_width = token_topk + compress_ratio - 1
    if out is None:
        out = torch.empty((rows, output_width), dtype=torch.int32, device=q.device)
    if out.shape != (rows, output_width):
        raise ValueError("QSA selection output has an invalid shape")
    if not rows:
        return out

    positions = query_positions.to(torch.int32)
    # Long rows score into an fp32 [rows, columns] buffer, so cap it as the
    # Triton path does.
    columns = page_table.shape[1] * k_cache.shape[1]
    rows_per_chunk = max(1, _LOGITS_WORKSPACE_BYTES // max(columns * 4, 1))
    block_topk = token_topk // compress_ratio
    blocks_buffer = torch.empty(
        (min(rows, rows_per_chunk), block_topk), dtype=torch.int32, device=q.device
    )
    for row_start in range(0, rows, rows_per_chunk):
        rows_slice = slice(row_start, min(row_start + rows_per_chunk, rows))
        blocks = blocks_buffer[: rows_slice.stop - row_start]
        qsa.qsa_k1_block_ids(
            q[rows_slice],
            k_cache,
            page_table,
            token_to_req[rows_slice],
            positions[rows_slice],
            sequence_lengths,
            out=blocks,
            heads=(int(q.shape[1]),),
        )
        expand_qsa_block_indices_cuda(
            blocks,
            query_positions[rows_slice],
            sequence_lengths,
            token_to_req[rows_slice],
            compress_ratio,
            token_topk,
            out[rows_slice],
        )
    return out


def flydsl_sparse_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor,
) -> bool:
    """Run FlyDSL K2 into ``out``. Returns False when K2 cannot serve it."""
    qsa = _flydsl_qsa()
    if qsa is None:
        return False
    if not out.is_contiguous() or not logical_indices.is_contiguous():
        logger.warning_once("FlyDSL QSA K2 skipped: strided output or indices.")
        return False
    reason = qsa.qsa_k2_serves(q, k_cache, v_cache, logical_indices, block_table)
    if reason is not None:
        logger.warning_once("FlyDSL QSA K2 skipped: %s", reason)
        return False
    logger.info_once("Using AITER FlyDSL QSA K2 sparse GQA.")
    qsa.qsa_k2(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        out=out,
    )
    return True
