# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import zlib
from collections.abc import Sequence

import torch

from vllm.pooling_params import LateInteractionParams, PoolingParams

try:
    from vllm.v1.pool.flash_maxsim.flash_maxsim_varlen import (
        flash_maxsim_packed,
        pack_docs,
    )
    _HAS_FLASH_MAXSIM = not os.environ.get("VLLM_FORCE_VANILLA_MAXSIM")
except ImportError:
    _HAS_FLASH_MAXSIM = False

LATE_INTERACTION_MODE_CACHE_QUERY = "cache_query"
LATE_INTERACTION_MODE_SCORE_DOC = "score_doc"


def get_late_interaction_engine_index(
    pooling_params: PoolingParams | None,
    num_engines: int,
) -> int | None:
    if pooling_params is None or pooling_params.late_interaction_params is None:
        return None

    late_interaction_params = pooling_params.late_interaction_params
    mode = late_interaction_params.mode
    if mode not in (
        LATE_INTERACTION_MODE_CACHE_QUERY,
        LATE_INTERACTION_MODE_SCORE_DOC,
    ):
        return None

    query_key = late_interaction_params.query_key
    if not isinstance(query_key, str) or not query_key:
        return None

    # query embeddings are cached in process-local worker memory,
    # pin requests sharing the same query key to the same engine.
    return zlib.crc32(query_key.encode("utf-8")) % num_engines


def build_late_interaction_query_params(
    query_key: str,
    query_uses: int,
) -> LateInteractionParams:
    return LateInteractionParams(
        mode=LATE_INTERACTION_MODE_CACHE_QUERY,
        query_key=query_key,
        query_uses=max(1, int(query_uses)),
    )


def build_late_interaction_doc_params(
    query_key: str,
) -> LateInteractionParams:
    return LateInteractionParams(
        mode=LATE_INTERACTION_MODE_SCORE_DOC,
        query_key=query_key,
    )


def _vanilla_compute_maxsim_score_batched(
    q_embs: Sequence[torch.Tensor],
    d_embs: Sequence[torch.Tensor],
    max_batch_size: int = 64,
    max_score_matrix_elements: int = 64_000_000,
) -> list[torch.Tensor]:
    """Vanilla MaxSim via padded bmm. Used as fallback when Triton is
    unavailable."""
    if len(q_embs) != len(d_embs):
        raise ValueError("q_embs and d_embs must have the same length")

    num_pairs = len(q_embs)
    if num_pairs == 0:
        return []

    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be greater than 0")
    if max_score_matrix_elements <= 0:
        raise ValueError("max_score_matrix_elements must be greater than 0")

    for q_emb, d_emb in zip(q_embs, d_embs):
        if q_emb.ndim != 2 or d_emb.ndim != 2:
            raise ValueError("Each embedding tensor must be 2-D")
        if q_emb.shape[1] != d_emb.shape[1]:
            raise ValueError("Query and document embeddings must have same dim")
        if q_emb.device != d_emb.device:
            raise ValueError("Query and document embeddings must be on same device")

    scores: list[torch.Tensor] = []
    start = 0
    while start < num_pairs:
        end = min(start + max_batch_size, num_pairs)
        max_q = max(int(x.shape[0]) for x in q_embs[start:end])
        max_d = max(int(x.shape[0]) for x in d_embs[start:end])

        # keep score matrix bounded to avoid oversized allocations.
        while (
            end - start > 1
            and (end - start) * max_q * max_d > max_score_matrix_elements
        ):
            end -= 1
            max_q = max(int(x.shape[0]) for x in q_embs[start:end])
            max_d = max(int(x.shape[0]) for x in d_embs[start:end])

        batch_q = q_embs[start:end]
        batch_d = d_embs[start:end]
        batch_size = end - start
        device = batch_q[0].device
        dim = int(batch_q[0].shape[1])

        q_batch = torch.zeros(
            (batch_size, max_q, dim), dtype=torch.float32, device=device
        )
        d_batch = torch.zeros(
            (batch_size, max_d, dim), dtype=torch.float32, device=device
        )
        q_mask = torch.zeros((batch_size, max_q), dtype=torch.bool, device=device)
        d_mask = torch.zeros((batch_size, max_d), dtype=torch.bool, device=device)

        # copy to padded tensors
        for i, (q_emb, d_emb) in enumerate(zip(batch_q, batch_d)):
            q_len = int(q_emb.shape[0])
            d_len = int(d_emb.shape[0])
            q_batch[i, :q_len] = q_emb.to(device=device, dtype=torch.float32)
            d_batch[i, :d_len] = d_emb.to(device=device, dtype=torch.float32)
            q_mask[i, :q_len] = True
            d_mask[i, :d_len] = True

        token_scores = torch.bmm(q_batch, d_batch.transpose(1, 2))
        token_scores.masked_fill_(~d_mask.unsqueeze(1), float("-inf"))
        max_per_query = token_scores.amax(dim=-1)
        max_per_query.masked_fill_(~q_mask, 0.0)
        batch_scores = max_per_query.sum(dim=-1)
        scores.extend(batch_scores.unbind(0))
        start = end

    return scores


def compute_maxsim_score_batched(
    q_embs: Sequence[torch.Tensor],
    d_embs: Sequence[torch.Tensor],
    max_batch_size: int = 64,
    max_score_matrix_elements: int = 64_000_000,
) -> list[torch.Tensor]:
    """Compute MaxSim for multiple query/doc pairs.

    Uses fused flash-maxsim Triton kernels when available (22x faster,
    O(1) memory). Falls back to vanilla padded bmm otherwise.
    """
    if len(q_embs) != len(d_embs):
        raise ValueError("q_embs and d_embs must have the same length")
    if len(q_embs) == 0:
        return []

    # Fall back to vanilla when flash-maxsim is unavailable, when
    # embedding dim < 16 (Triton tl.dot requires K >= 16), or when
    # tensors are on CPU (Triton requires CUDA).
    if not _HAS_FLASH_MAXSIM or (
        q_embs[0].shape[1] < 16 or not q_embs[0].is_cuda
    ):
        return _vanilla_compute_maxsim_score_batched(
            q_embs, d_embs, max_batch_size, max_score_matrix_elements,
        )

    # Group by query identity — queries sharing the same cached tensor
    # (same data_ptr) can be scored together with shared_docs=True.
    groups: dict[int, tuple[torch.Tensor, list[int], list[torch.Tensor]]] = {}
    for i, (q, d) in enumerate(zip(q_embs, d_embs)):
        key = q.data_ptr()
        if key not in groups:
            groups[key] = (q, [], [])
        groups[key][1].append(i)
        groups[key][2].append(d)

    results: list[torch.Tensor | None] = [None] * len(q_embs)

    for _, (query, indices, docs) in groups.items():
        # Pack docs contiguously — single torch.cat, no padding, no
        # Python-loop copy. flash_maxsim_packed reads the packed layout
        # directly, skipping padding tokens entirely.
        D_packed, cu_seqlens, max_ld = pack_docs(docs)
        scores = flash_maxsim_packed(query, D_packed, cu_seqlens, max_ld)

        for j, idx in enumerate(indices):
            results[idx] = scores[j]

    return results  # type: ignore[return-value]
