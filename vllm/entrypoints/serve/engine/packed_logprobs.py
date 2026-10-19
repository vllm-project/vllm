# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed top-k wire format shared by generate and completion responses."""

import numpy as np
import pybase64
from pydantic import BaseModel

from vllm.exceptions import GenerationError
from vllm.logprobs import FlatLogprobs, PromptLogprobs, SampleLogprobs


class PackedTopK(BaseModel):
    """Engine slots 1..k as base64 little-endian int32/float32 arrays.

    Both arrays have shape [num_positions, k], in engine candidate order.
    Scores are not clamped or renormalized. Slot 0 (the sampled token) is
    excluded; the sampled token can still occur among the top-k candidates.
    """

    num_positions: int
    k: int
    token_ids: str
    logprobs: str


def pack_top_k_logprobs(
    container: SampleLogprobs | PromptLogprobs | None,
    num_tokens: int,
    k: int,
) -> tuple[bytes, bytes]:
    rows = container.rows() if isinstance(container, FlatLogprobs) else None
    if rows is None or k < 1:
        raise GenerationError("Top-k logprobs are unavailable for this request")
    token_ids, logprobs, _ = rows
    if len(token_ids) != num_tokens or (num_tokens and token_ids.shape[1] != k + 1):
        raise GenerationError("Top-k logprobs are unavailable for this request")
    top_ids = np.ascontiguousarray(token_ids[:, 1:])
    top_logprobs = np.ascontiguousarray(logprobs[:, 1:])
    return (
        pybase64.b64encode(top_ids.tobytes()),
        pybase64.b64encode(top_logprobs.tobytes()),
    )
