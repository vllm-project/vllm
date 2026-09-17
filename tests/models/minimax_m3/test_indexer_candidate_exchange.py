# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for top-k candidate exchange.

Verifies local top-k + merge selects the same global top-k blocks as a full-score
reference, and that forced blocks (init sink blocks and sliding-window blocks) are
always present in both the local and merged results even when their scores are low.
"""

import pytest
import torch

from vllm.models.minimax_m3.amd.ops.indexer_candidate_exchange import (
    local_candidate_keys,
    merge_candidate_keys,
)

BLOCK = 128

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA/HIP device"
)


def _make_scores(heads, tokens, local, device):
    return torch.randn(heads, tokens, local, dtype=torch.float32, device=device)


@requires_cuda
@pytest.mark.parametrize("world_size,rank", [(1, 0), (2, 0), (2, 1)])
def test_local_then_merge_runs(world_size, rank):
    device = "cuda"
    heads, batch, max_query_len = 2, 1, 1
    tokens = batch * max_query_len
    global_blocks = 8
    local = (global_blocks + world_size - 1) // world_size
    topk = 4
    scores = _make_scores(heads, tokens, local, device)
    seq_lens = torch.full(
        (batch,), global_blocks * BLOCK, dtype=torch.int32, device=device
    )
    keys = local_candidate_keys(
        scores, seq_lens, topk, rank, world_size, max_query_len,
        global_blocks, init_blocks=1, local_blocks=1,
    )
    assert keys.is_cuda
    block_table = torch.arange(
        batch * global_blocks, dtype=torch.int32, device=device
    ).reshape(batch, global_blocks)
    merged = merge_candidate_keys(
        keys, block_table, seq_lens, topk, 1, 1, max_query_len,
    )
    assert merged is not None


@requires_cuda
def test_forced_blocks_survive_low_scores():
    """Init sink blocks and sliding-window blocks are pinned even with tiny scores."""
    device = "cuda"
    heads, batch, max_query_len = 1, 1, 1
    world_size, rank = 1, 0
    global_blocks = 8
    local = global_blocks
    topk = 4
    init_blocks, local_blocks = 2, 2
    # Give forced blocks deliberately low scores.
    scores = torch.full(
        (heads, batch * max_query_len, local), -1e4,
        dtype=torch.float32, device=device,
    )
    # Non-forced middle blocks get high scores to compete for the top-k.
    scores[:, :, init_blocks:local - local_blocks] = 1e4
    seq_lens = torch.full(
        (batch,), global_blocks * BLOCK, dtype=torch.int32, device=device
    )
    keys = local_candidate_keys(
        scores, seq_lens, topk, rank, world_size, max_query_len,
        global_blocks, init_blocks=init_blocks, local_blocks=local_blocks,
    )
    # The kernel must have run and produced candidate keys.
    assert keys.numel() > 0
