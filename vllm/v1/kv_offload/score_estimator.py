# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Attention importance score estimation for KV cache eviction.

Estimates per-block attention importance from hidden state magnitudes
without modifying attention kernels. The intuition is that KV blocks
contributing more to the model's output will produce higher-magnitude
hidden states at their corresponding positions.

This is Option B from the design doc: approximate attention importance
via output magnitude per block (cheap, no kernel changes).

Architecture:
  Worker side: compute_block_scores_from_hidden_states()
    -> returns req_id -> list[float] (per-block scores in block order)
    -> attached to KVConnectorOutput.attention_block_scores

  Scheduler side: map_scores_to_block_hashes()
    -> maps positional scores to block_hashes using request metadata
    -> feeds to OffloadingManager.update_attention_scores()
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

from vllm.v1.core.kv_cache_utils import BlockHash

if TYPE_CHECKING:
    pass


def compute_block_scores_from_hidden_states(
    hidden_states: torch.Tensor,
    num_scheduled_tokens: dict[str, int],
    block_size: int,
) -> dict[str, list[float]]:
    """
    Compute per-block importance scores from hidden states (WORKER SIDE).

    Computes L2 norm of hidden states averaged over each block's tokens.
    Returns scores indexed by block position, not block_hash (worker
    doesn't know block_hashes).

    Args:
        hidden_states: Model output [total_tokens, hidden_size].
        num_scheduled_tokens: req_id -> number of tokens scheduled.
        block_size: Tokens per KV cache block.

    Returns:
        req_id -> list of per-block scores (in block order).
    """
    if hidden_states is None or hidden_states.dim() < 2:
        return {}

    # Compute per-token norms on GPU, move to CPU once
    # Using float32 for numerical stability
    with torch.no_grad():
        token_norms = torch.norm(
            hidden_states.float(), dim=-1
        ).cpu()

    scores_per_request: dict[str, list[float]] = {}
    token_offset = 0

    for req_id, num_tokens in num_scheduled_tokens.items():
        if num_tokens <= 0:
            continue

        num_blocks = (num_tokens + block_size - 1) // block_size
        block_scores: list[float] = []

        for block_idx in range(num_blocks):
            block_start = block_idx * block_size
            block_end = min(block_start + block_size, num_tokens)

            global_start = token_offset + block_start
            global_end = token_offset + block_end

            if global_end > token_norms.shape[0]:
                break

            block_norms = token_norms[global_start:global_end]
            avg_norm = float(block_norms.mean())
            block_scores.append(avg_norm)

        if block_scores:
            scores_per_request[req_id] = block_scores

        token_offset += num_tokens

    return scores_per_request


def map_scores_to_block_hashes(
    per_request_scores: dict[str, list[float]],
    request_block_hashes: dict[str, Iterable[BlockHash]],
    block_size_factor: int = 1,
) -> dict[BlockHash, float]:
    """
    Map positional block scores to block_hashes (SCHEDULER SIDE).

    The scheduler knows block_hashes for each request. This function
    maps the worker-computed positional scores to the correct hashes.

    Args:
        per_request_scores: req_id -> list of per-block scores from worker.
        request_block_hashes: req_id -> iterable of block_hashes.
        block_size_factor: ratio of offloaded_block_size / gpu_block_size.
            When > 1, multiple GPU blocks map to one offloaded block.

    Returns:
        block_hash -> score mapping for the OffloadingManager.
    """
    aggregated: dict[BlockHash, float] = {}

    for req_id, scores in per_request_scores.items():
        block_hashes = request_block_hashes.get(req_id)
        if block_hashes is None:
            continue

        hashes_list = list(block_hashes)

        for i, score in enumerate(scores):
            # Map GPU block index to offloaded block index
            offloaded_idx = i // block_size_factor
            if offloaded_idx >= len(hashes_list):
                break

            block_hash = hashes_list[offloaded_idx]
            # Take max score across GPU blocks within one offloaded block
            existing = aggregated.get(block_hash, 0.0)
            aggregated[block_hash] = max(existing, score)

    return aggregated
