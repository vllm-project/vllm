# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import cast

import torch

from .typing import (
    ScoreContentPartParam,
    ScoreData,
    ScoreInput,
    ScoreInputs,
    ScoringData,
)


def compute_maxsim_score(q_emb: torch.Tensor, d_emb: torch.Tensor) -> torch.Tensor:
    """
    Compute ColBERT MaxSim score.

    Args:
        q_emb: Query token embeddings [query_len, dim]
        d_emb: Document token embeddings [doc_len, dim]

    Returns:
        MaxSim score (sum over query tokens of max similarity to any doc token)
    """
    # [query_len, doc_len]
    token_scores = torch.matmul(q_emb, d_emb.T)
    # Max over document tokens, sum over query tokens
    return token_scores.amax(dim=-1).sum()


def _validate_mm_score_input(
    data: list[ScoreInput],
    is_multimodal_model: bool,
    architecture: str,
) -> list[ScoreData]:
    out: list[ScoreData] = []
    for d in data:
        if isinstance(d, str):
            out.append(d)
        else:
            if not is_multimodal_model:
                raise ValueError(f"MultiModalParam is not supported for {architecture}")
            content = cast(list[ScoreContentPartParam], d.get("content", []))
            out.append(content)
    return out


def _validate_score_input_lens(
    data_1: list[ScoreData],
    data_2: list[ScoreData],
):
    len_1 = len(data_1)
    len_2 = len(data_2)

    if len_1 > 1 and len_1 != len_2:
        raise ValueError("Input lengths must be either 1:1, 1:N or N:N")
    if len_1 == 0:
        raise ValueError("At least one text element must be given")
    if len_2 == 0:
        raise ValueError("At least one text_pair element must be given")


def validate_score_input(
    data_1: ScoreInputs,
    data_2: ScoreInputs,
    is_multimodal_model: bool,
    architecture: str,
) -> ScoringData:
    if not isinstance(data_1, list):
        data_1 = [data_1]

    if not isinstance(data_2, list):
        data_2 = [data_2]

    score_input_1 = _validate_mm_score_input(data_1, is_multimodal_model, architecture)
    score_input_2 = _validate_mm_score_input(data_2, is_multimodal_model, architecture)
    _validate_score_input_lens(score_input_1, score_input_2)
    return ScoringData(data_1=score_input_1, data_2=score_input_2)
