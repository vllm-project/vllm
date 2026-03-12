# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import zlib

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
