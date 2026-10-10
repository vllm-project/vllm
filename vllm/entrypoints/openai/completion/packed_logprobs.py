# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pack completion candidates directly from FlatLogprobs, without Logprob objects."""

from collections.abc import Sequence

import numpy as np
import pybase64

from vllm.entrypoints.openai.completion.protocol import (
    CompletionLogProbs,
    PackedTopK,
)
from vllm.logprobs import FlatLogprobs, PromptLogprobs, SampleLogprobs


def create_packed_completion_logprobs(
    token_ids: Sequence[int],
    logprobs: SampleLogprobs | PromptLogprobs | None,
    k: int | None,
    initial_text_offset: int = 0,
) -> CompletionLogProbs:
    if not isinstance(logprobs, FlatLogprobs) or k is None or k < 1:
        raise ValueError("Packed completion scores require FlatLogprobs")
    # FlatLogprobs retains the sampler's slots: sampled token, then k
    # candidates, including a duplicate when the sampled token is in top-k.
    starts = np.asarray(logprobs.start_indices, dtype=np.int64)
    ends = np.asarray(logprobs.end_indices, dtype=np.int64)
    if len(starts) != len(token_ids) or np.any(ends - starts != k + 1):
        raise ValueError("Packed completion scores require sampled + k slots")
    ids = np.asarray(logprobs.token_ids, dtype="<i4")
    values = np.asarray(logprobs.logprobs, dtype="<f4")
    if not np.array_equal(ids[starts], token_ids):
        raise ValueError("Engine logprob rows must begin with the sampled token")
    indices = starts[:, None] + np.arange(1, k + 1)
    head_ids = ids[indices]
    head_values = values[indices]
    tokens = [f"token_id:{token_id}" for token_id in token_ids]
    offsets = (
        initial_text_offset + np.cumsum([0] + [len(token) for token in tokens[:-1]])
    ).tolist()
    return CompletionLogProbs(
        tokens=tokens,
        token_logprobs=np.maximum(values[starts], -9999.0).tolist(),
        text_offset=offsets if tokens else [],
        top_k=PackedTopK(
            num_positions=len(token_ids),
            k=k,
            token_ids=pybase64.b64encode(head_ids.tobytes()).decode("ascii"),
            logprobs=pybase64.b64encode(head_values.tobytes()).decode("ascii"),
        ),
    )
