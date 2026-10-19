# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pack completion candidates directly from FlatLogprobs, without Logprob objects."""

from collections.abc import Sequence

import numpy as np

from vllm.entrypoints.openai.completion.protocol import (
    CompletionLogProbs,
)
from vllm.entrypoints.serve.engine.packed_logprobs import (
    PackedTopK,
    pack_top_k_logprobs,
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
    head_ids, head_values = pack_top_k_logprobs(logprobs, len(token_ids), k)
    rows = logprobs.rows()
    assert rows is not None  # Validated by the shared packer.
    ids, values, _ = rows
    if len(token_ids) and not np.array_equal(ids[:, 0], token_ids):
        raise ValueError("Engine logprob rows must begin with the sampled token")
    sampled = np.maximum(values[:, 0], -9999.0).tolist() if len(token_ids) else []
    tokens = [f"token_id:{token_id}" for token_id in token_ids]
    offsets = (
        initial_text_offset + np.cumsum([0] + [len(token) for token in tokens[:-1]])
    ).tolist()
    return CompletionLogProbs(
        tokens=tokens,
        token_logprobs=sampled,
        text_offset=offsets if tokens else [],
        top_k=PackedTopK(
            num_positions=len(token_ids),
            k=k,
            token_ids=head_ids.decode("ascii"),
            logprobs=head_values.decode("ascii"),
        ),
    )
