# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whole-batch sequence parallelism of DSA (sparse MLA) layers.

Tensor parallelism shards the query heads, so every rank runs the indexer and
the sparse attention for every token. With
`attention_config.sparse_mla_sequence_parallel`, the TP ranks instead split
each step's tokens zigzag (`zigzag_shards`), decodes included, and each rank
scores and attends its own tokens with all heads over the replicated KV cache.
"""

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

import vllm.envs as envs
from vllm.models.common.ops.sequence_parallel import zigzag_chunk_size
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def use_sparse_mla_sequence_parallel(vllm_config: "VllmConfig") -> bool:
    """Whether the DSA layers split the step's tokens over the TP group.

    Raises:
        ValueError: If the configuration does not support it.

    """
    if not vllm_config.attention_config.sparse_mla_sequence_parallel:
        return False
    parallel_config = vllm_config.parallel_config
    if parallel_config.tensor_parallel_size == 1:
        return False
    speculative_config = vllm_config.speculative_config
    unsupported = {
        "a non-CUDA platform": not current_platform.is_cuda(),
        "pipeline parallelism": parallel_config.pipeline_parallel_size > 1,
        # The MoE layers' data-parallel dispatch sizes the ranks' rows without
        # the zigzag padding.
        "data parallelism": parallel_config.data_parallel_size > 1,
        "context parallelism": parallel_config.prefill_context_parallel_size > 1
        or parallel_config.decode_context_parallel_size > 1,
        "micro-batching": parallel_config.use_ubatching,
        "LoRA": vllm_config.lora_config is not None,
        "drafters that read auxiliary hidden states": speculative_config is not None
        and speculative_config.uses_aux_hidden_states,
        "adaptive verification": speculative_config is not None
        and speculative_config.enable_adaptive_verification,
        "batch invariance": envs.VLLM_BATCH_INVARIANT,
        "HiSparse": vllm_config.attention_config.hisparse_config is not None,
    }
    for feature, used in unsupported.items():
        if used:
            raise ValueError(
                f"sparse_mla_sequence_parallel does not support {feature}."
            )
    return True


class ZigzagLocalBatch(NamedTuple):
    """A TP rank's zigzag rows as a batch of their own.

    Each local request is a run of consecutive rows of one request, or a single
    row with `per_token`. Its tokens are the last ones of its own sequence, so
    it attends the context through its last row, which the KV cache holds once
    every rank has written its rows.
    """

    req_indices: np.ndarray
    """The request of each local request."""
    tails: np.ndarray
    """How many of its request's rows follow each local request."""
    query_start_loc: np.ndarray
    num_tokens: int
    """Rows of real tokens; the rank's other rows pad the batch."""


def zigzag_local_batch(
    query_start_loc: np.ndarray,
    num_tokens: int,
    num_tokens_padded: int,
    world_size: int,
    rank: int,
    per_token: bool,
) -> ZigzagLocalBatch:
    """Split this rank's rows of a step into local requests.

    Args:
        query_start_loc: The step's requests' query start offsets.
        num_tokens: The step's real tokens.
        num_tokens_padded: The rows the model runs, which `zigzag_shards`
            splits.
        world_size: TP world size.
        rank: TP rank.
        per_token: Give each row its own request, so that a uniform decode
            batch keeps one local layout per CUDA graph size.

    """
    chunk = zigzag_chunk_size(num_tokens_padded, world_size)
    rows = np.concatenate(
        [
            np.arange(c * chunk, min((c + 1) * chunk, num_tokens))
            for c in (rank, 2 * world_size - 1 - rank)
        ]
    )
    reqs = np.searchsorted(query_start_loc[1:], rows, side="right")
    starts = np.ones(len(rows), dtype=bool)
    if not per_token:
        starts[1:] = (reqs[1:] != reqs[:-1]) | (rows[1:] != rows[:-1] + 1)
    first = np.flatnonzero(starts)
    last = np.append(first[1:], len(rows)) - 1
    req_indices = reqs[first]
    return ZigzagLocalBatch(
        req_indices=req_indices,
        tails=query_start_loc[req_indices + 1] - 1 - rows[last],
        query_start_loc=np.append(first, len(rows)).astype(np.int32),
        num_tokens=len(rows),
    )
