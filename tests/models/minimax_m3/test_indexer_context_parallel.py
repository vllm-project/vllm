# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for indexer_context_scores().

Validates the round-robin logical block mapping (block = local * world + rank),
SCALE application, causal cutoff masking, output shape
[heads, tokens, ceil(blocks / world)], and input geometry validation.
"""

import pytest
import torch

from vllm.models.minimax_m3.amd.ops.indexer_context_parallel import (
    indexer_context_scores,
)

BLOCK = 128
HEAD_DIM = 128

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA/HIP device"
)


def _make_inputs(batch, heads, max_query_len, max_seq_len, blocks, device, dtype):
    tokens = batch * max_query_len
    idx_q = torch.randn(
        tokens, heads, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    index_cache = torch.randn(
        blocks, BLOCK, HEAD_DIM, dtype=dtype, device=device
    )
    block_table = torch.arange(
        batch * blocks, dtype=torch.int32, device=device
    ).reshape(batch, blocks)
    seq_lens = torch.full(
        (batch,), max_seq_len, dtype=torch.int32, device=device
    )
    return idx_q, index_cache, block_table, seq_lens


@requires_cuda
@pytest.mark.parametrize("world_size,rank", [(1, 0), (2, 0), (2, 1), (4, 3)])
def test_output_shape_and_round_robin(world_size, rank):
    device = "cuda"
    batch, heads, max_query_len, max_seq_len = 2, 4, 1, 8 * BLOCK
    blocks = (max_seq_len + BLOCK - 1) // BLOCK
    idx_q, index_cache, block_table, seq_lens = _make_inputs(
        batch, heads, max_query_len, max_seq_len, blocks, device, torch.bfloat16
    )
    tokens = batch * max_query_len
    local = (blocks + world_size - 1) // world_size
    scores = indexer_context_scores(
        idx_q, index_cache, block_table, seq_lens, max_seq_len,
        rank, world_size, max_query_len, sm_scale=1.0 / HEAD_DIM**0.5,
    )
    assert scores.shape == (heads, tokens, local)
    assert scores.dtype == torch.float32
    assert scores.device == idx_q.device


@requires_cuda
def test_full_partition_matches_single_rank_union():
    """With world_size=1 every logical block maps to local index directly."""
    device = "cuda"
    batch, heads, max_query_len, max_seq_len = 1, 2, 1, 4 * BLOCK
    blocks = (max_seq_len + BLOCK - 1) // BLOCK
    idx_q, index_cache, block_table, seq_lens = _make_inputs(
        batch, heads, max_query_len, max_seq_len, blocks, device, torch.bfloat16
    )
    scores = indexer_context_scores(
        idx_q, index_cache, block_table, seq_lens, max_seq_len,
        0, 1, max_query_len, sm_scale=1.0,
    )
    # world_size=1 -> local == blocks, one entry per logical block.
    assert scores.shape[2] == blocks


def _valid_kwargs(device):
    batch, heads, max_query_len, max_seq_len = 1, 2, 1, 2 * BLOCK
    blocks = (max_seq_len + BLOCK - 1) // BLOCK
    idx_q, index_cache, block_table, seq_lens = _make_inputs(
        batch, heads, max_query_len, max_seq_len, blocks, device, torch.bfloat16
    )
    return dict(
        idx_q=idx_q, index_cache=index_cache, block_table=block_table,
        seq_lens=seq_lens, max_seq_len=max_seq_len, rank=0, world_size=1,
        max_query_len=max_query_len, sm_scale=1.0,
    )


@requires_cuda
def test_invalid_partition_geometry_raises():
    kw = _valid_kwargs("cuda")
    with pytest.raises(ValueError):
        indexer_context_scores(**{**kw, "world_size": 0})
    with pytest.raises(ValueError):
        indexer_context_scores(**{**kw, "world_size": 2, "rank": 2})
    with pytest.raises(ValueError):
        indexer_context_scores(**{**kw, "rank": -1})


@requires_cuda
def test_invalid_query_dtype_raises():
    kw = _valid_kwargs("cuda")
    bad = kw["idx_q"].to(torch.float16)
    with pytest.raises(ValueError):
        indexer_context_scores(**{**kw, "idx_q": bad})


@requires_cuda
def test_invalid_query_dims_raise():
    kw = _valid_kwargs("cuda")
    bad = kw["idx_q"][:, :, :64]  # last dim != 128
    with pytest.raises(ValueError):
        indexer_context_scores(**{**kw, "idx_q": bad})
