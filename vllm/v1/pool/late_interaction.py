# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

LATE_INTERACTION_MODE_KEY = "late_interaction_mode"
LATE_INTERACTION_QUERY_KEY = "late_interaction_query_key"
LATE_INTERACTION_QUERY_USES_KEY = "late_interaction_query_uses"

LATE_INTERACTION_MODE_CACHE_QUERY = "cache_query"
LATE_INTERACTION_MODE_SCORE_DOC = "score_doc"


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
