# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness and capture-safety tests for MiniMax-M3 CP indexer scoring."""

import pytest
import torch

from vllm.platforms import current_platform

if not (current_platform.is_cuda() or current_platform.is_rocm()):
    pytest.skip(
        "MiniMax M3 CP indexer kernels require CUDA or ROCm.",
        allow_module_level=True,
    )

from vllm.models.minimax_m3.amd.ops.indexer_context_parallel import (  # noqa: E402
    indexer_context_scores,
)

DEVICE = "cuda"
HEAD_DIM = 128
BLOCK = 128
LOG2E = 1.4426950409


def _reference_shard(
    idx_q: torch.Tensor,
    cache: torch.Tensor,
    table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    rank: int,
    world_size: int,
    query_len: int,
    sm_scale: float,
) -> torch.Tensor:
    tokens, heads, _ = idx_q.shape
    batch = seq_lens.numel()
    blocks = (max_seq_len + BLOCK - 1) // BLOCK
    local = (blocks + world_size - 1) // world_size
    scores = torch.full(
        (heads, tokens, local), float("-inf"), device=idx_q.device, dtype=torch.float32
    )
    scale = sm_scale * LOG2E
    qf = idx_q.float()
    kf = cache.float()
    for req in range(batch):
        length = int(seq_lens[req].item())
        for loc in range(local):
            block = loc * world_size + rank
            if block >= blocks or block * BLOCK >= length:
                continue
            page = int(table[req, block].item())
            k = kf[page]
            for t in range(query_len):
                row = req * query_len + t
                cutoff = length - query_len + t + 1
                q = qf[row]
                dots = (k @ q.T) * scale
                pos = torch.arange(BLOCK, device=idx_q.device)
                dots = torch.where(
                    (block * BLOCK + pos)[:, None] < cutoff, dots, float("-inf")
                )
                scores[:, row, loc] = dots.amax(dim=0)
    return scores


@pytest.mark.parametrize("rank", [0, 1])
@torch.inference_mode()
def test_indexer_context_scores_matches_reference(rank: int):
    torch.manual_seed(0)
    world_size = 2
    batch, heads, query_len = 2, 2, 1
    seq_lens = torch.tensor([130, 250], device=DEVICE, dtype=torch.int32)
    max_seq_len = int(seq_lens.max().item())
    blocks = (max_seq_len + BLOCK - 1) // BLOCK
    pages = batch * blocks
    table = torch.arange(pages, device=DEVICE, dtype=torch.int32).reshape(
        batch, blocks
    )
    cache = torch.randn(pages, BLOCK, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    idx_q = torch.randn(
        batch * query_len, heads, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16
    )
    sm_scale = HEAD_DIM**-0.5
    got = indexer_context_scores(
        idx_q,
        cache,
        table,
        seq_lens,
        max_seq_len,
        rank,
        world_size,
        query_len,
        sm_scale,
    )
    ref = _reference_shard(
        idx_q,
        cache,
        table,
        seq_lens,
        max_seq_len,
        rank,
        world_size,
        query_len,
        sm_scale,
    )
    assert got.shape == ref.shape
    finite = torch.isfinite(ref)
    assert torch.allclose(got[finite], ref[finite], rtol=2e-2, atol=2e-2)
    assert torch.isneginf(got[~finite]).all()


@torch.inference_mode()
def test_indexer_context_scores_dummy_capture_pages_are_safe():
    """Profiling/FULL capture uses max_model_len metadata against a tiny cache."""
    torch.manual_seed(1)
    world_size, rank = 4, 0
    batch, heads, query_len = 8, 1, 1
    max_seq_len = 2048
    blocks = (max_seq_len + BLOCK - 1) // BLOCK
    seq_lens = torch.full((batch,), query_len, device=DEVICE, dtype=torch.int32)
    table = torch.zeros(batch, blocks, device=DEVICE, dtype=torch.int32)
    cache = torch.randn(1, BLOCK, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    idx_q = torch.randn(
        batch * query_len, heads, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16
    )
    out = torch.empty(
        (heads, batch * query_len, blocks),
        device=DEVICE,
        dtype=torch.float32,
    )
    scores = indexer_context_scores(
        idx_q,
        cache,
        table,
        seq_lens,
        max_seq_len,
        rank,
        world_size,
        query_len,
        HEAD_DIM**-0.5,
        out=out,
    )
    assert scores.data_ptr() == out.data_ptr()
    torch.cuda.synchronize()
