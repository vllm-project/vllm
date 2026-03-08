# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import zlib
from typing import Any

import torch

from vllm.pooling_params import PoolingParams

LATE_INTERACTION_MODE_KEY = "late_interaction_mode"
LATE_INTERACTION_QUERY_KEY = "late_interaction_query_key"
LATE_INTERACTION_QUERY_USES_KEY = "late_interaction_query_uses"

LATE_INTERACTION_MODE_CACHE_QUERY = "cache_query"
LATE_INTERACTION_MODE_SCORE_DOC = "score_doc"


def get_late_interaction_engine_index(
    pooling_params: PoolingParams | None,
    num_engines: int,
) -> int | None:
    if pooling_params is None or pooling_params.extra_kwargs is None:
        return None

    extra_kwargs = pooling_params.extra_kwargs
    mode = extra_kwargs.get(LATE_INTERACTION_MODE_KEY)
    if mode not in (
        LATE_INTERACTION_MODE_CACHE_QUERY,
        LATE_INTERACTION_MODE_SCORE_DOC,
    ):
        return None

    query_key = extra_kwargs.get(LATE_INTERACTION_QUERY_KEY)
    if not isinstance(query_key, str) or not query_key:
        return None

    # query embeddings are cached in process-local worker memory,
    # pin requests sharing the same query key to the same engine.
    return zlib.crc32(query_key.encode("utf-8")) % num_engines


def build_late_interaction_query_kwargs(
    query_key: str,
    query_uses: int,
    base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kwargs = {} if base is None else dict(base)
    kwargs[LATE_INTERACTION_MODE_KEY] = LATE_INTERACTION_MODE_CACHE_QUERY
    kwargs[LATE_INTERACTION_QUERY_KEY] = query_key
    kwargs[LATE_INTERACTION_QUERY_USES_KEY] = max(1, int(query_uses))
    return kwargs


def build_late_interaction_doc_kwargs(
    query_key: str,
    base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kwargs = {} if base is None else dict(base)
    kwargs[LATE_INTERACTION_MODE_KEY] = LATE_INTERACTION_MODE_SCORE_DOC
    kwargs[LATE_INTERACTION_QUERY_KEY] = query_key
    return kwargs


def compute_maxsim_score(
    q_emb: torch.Tensor,
    d_emb: torch.Tensor,
) -> torch.Tensor:
    # compute in float32 for numerical stability
    token_scores = torch.matmul(q_emb.float(), d_emb.float().T)
    return token_scores.amax(dim=-1).sum()
