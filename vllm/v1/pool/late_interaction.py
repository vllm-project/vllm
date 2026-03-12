# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import zlib
from collections.abc import Sequence

import torch

from vllm.pooling_params import LateInteractionParams, PoolingParams

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


def compute_maxsim_score(
    q_emb: torch.Tensor,
    d_emb: torch.Tensor,
) -> torch.Tensor:
    # compute in float32 for numerical stability
    token_scores = torch.matmul(q_emb.float(), d_emb.float().T)
    return token_scores.amax(dim=-1).sum()


def compute_maxsim_scores(
    q_embs: Sequence[torch.Tensor],
    d_embs: Sequence[torch.Tensor],
    max_batch_size: int = 64,
    max_score_matrix_elements: int = 64_000_000,
) -> list[torch.Tensor]:
    """Compute MaxSim for multiple query/doc pairs in mini-batches."""
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
