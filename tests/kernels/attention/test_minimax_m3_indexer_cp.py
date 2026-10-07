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
from vllm.models.minimax_m3.amd.ops.indexer_cp_exchange import (  # noqa: E402
    local_topk_keys,
    merge_topk_keys,
)

DEVICE = "cuda"
HEAD_DIM = 128
BLOCK = 128
LOG2E = 1.4426950409
# Marks score slots past a request's causal reach, which the kernel must skip.
SENTINEL = -12345.0


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference shard scores and the mask of slots the kernel must write.

    The kernel leaves slots past a request's causal reach untouched rather
    than filling them, so the mask is part of the contract: the top-k bounds
    its own scan the same way and never reads them.
    """
    tokens, heads, _ = idx_q.shape
    batch = seq_lens.numel()
    blocks = (max_seq_len + BLOCK - 1) // BLOCK
    local = (blocks + world_size - 1) // world_size
    scores = torch.full(
        (heads, tokens, local), float("-inf"), device=idx_q.device, dtype=torch.float32
    )
    written = torch.zeros((tokens, local), device=idx_q.device, dtype=torch.bool)
    scale = sm_scale * LOG2E
    qf = idx_q.float()
    kf = cache.float()
    for req in range(batch):
        length = int(seq_lens[req].item())
        for loc in range(local):
            block = loc * world_size + rank
            if block >= blocks or block * BLOCK >= length:
                continue
            written[req * query_len : (req + 1) * query_len, loc] = True
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
    return scores, written


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
    table = torch.arange(pages, device=DEVICE, dtype=torch.int32).reshape(batch, blocks)
    cache = torch.randn(pages, BLOCK, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    idx_q = torch.randn(
        batch * query_len, heads, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16
    )
    sm_scale = HEAD_DIM**-0.5
    ref, written = _reference_shard(
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
    out = torch.full_like(ref, SENTINEL)
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
        out=out,
    )
    assert got.shape == ref.shape
    hit = written.expand_as(ref)
    finite = hit & torch.isfinite(ref)
    assert torch.allclose(got[finite], ref[finite], rtol=2e-2, atol=2e-2)
    assert torch.isneginf(got[hit & ~finite]).all()
    assert (got[~hit] == SENTINEL).all()


@pytest.mark.parametrize("bound", [8192, 65536])
@torch.inference_mode()
def test_indexer_context_scores_independent_of_shape_bound(bound: int):
    """A grid sized for max_model_len scores a short batch identically.

    CUDA graphs bake the grid and ``CHUNK`` at capture, where the dummy batch
    is one token long. Shapes must therefore come from ``max_model_len``, and
    scoring must not depend on how far past the batch that bound reaches.
    """
    torch.manual_seed(2)
    world_size, rank, query_len = 4, 2, 1
    batch, heads = 3, 2
    seq_lens = torch.tensor([1000, 4000, 2500], device=DEVICE, dtype=torch.int32)
    tight = int(seq_lens.max().item())
    blocks = (bound + BLOCK - 1) // BLOCK
    table = torch.randint(0, 64, (batch, blocks), device=DEVICE, dtype=torch.int32)
    cache = torch.randn(64, BLOCK, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    idx_q = torch.randn(
        batch * query_len, heads, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16
    )
    sm_scale = HEAD_DIM**-0.5

    def score(seq_bound: int) -> torch.Tensor:
        cols = (seq_bound + BLOCK - 1) // BLOCK
        return indexer_context_scores(
            idx_q,
            cache,
            table[:, :cols].contiguous(),
            seq_lens,
            seq_bound,
            rank,
            world_size,
            query_len,
            sm_scale,
        )

    wide = score(bound)
    narrow = score(tight)
    _, written = _reference_shard(
        idx_q, cache, table, seq_lens, tight, rank, world_size, query_len, sm_scale
    )
    hit = written.expand(heads, -1, -1)
    assert torch.equal(wide[:, :, : narrow.shape[2]][hit], narrow[hit])


@torch.inference_mode()
def test_indexer_context_scores_unpopulated_block_table_is_safe():
    """Profiling runs score a max_model_len context off an empty block table.

    ``_dummy_run`` with ``force_attention`` leaves the block table unwritten
    while seq_lens claim the full context, so every page id the score loop
    reads is out of range for the cache it is handed.
    """
    torch.manual_seed(1)
    world_size, rank = 4, 0
    batch, heads, query_len = 8, 1, 1
    max_seq_len = 2048
    blocks = (max_seq_len + BLOCK - 1) // BLOCK
    seq_lens = torch.full((batch,), max_seq_len, device=DEVICE, dtype=torch.int32)
    table = torch.full((batch, blocks), 1 << 20, device=DEVICE, dtype=torch.int32)
    cache = torch.randn(1, BLOCK, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    idx_q = torch.randn(
        batch * query_len, heads, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16
    )
    out = torch.empty(
        (heads, batch * query_len, (blocks + world_size - 1) // world_size),
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
    torch.accelerator.synchronize()


def _force_scores(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    query_len: int,
    init_blocks: int,
    local_keep: int,
) -> torch.Tensor:
    heads, tokens, blocks = scores.shape
    forced = scores.clone()
    for row in range(tokens):
        req = row // query_len
        token = row % query_len
        causal_len = int(seq_lens[req].item()) - query_len + token + 1
        causal_blocks = (causal_len + 127) // 128
        local_start = max(0, causal_blocks - local_keep)
        blk = torch.arange(blocks, device=scores.device)
        valid = blk < causal_blocks
        forced[:, row] = torch.where(
            valid & (blk < init_blocks),
            torch.full_like(forced[:, row], 1e30),
            forced[:, row],
        )
        forced[:, row] = torch.where(
            valid & (blk >= local_start),
            torch.full_like(forced[:, row], 1e29),
            forced[:, row],
        )
        forced[:, row] = torch.where(
            valid, forced[:, row], torch.full_like(forced[:, row], -1e30)
        )
    return forced


def _lexsort_topk(score_row: torch.Tensor, n: int) -> torch.Tensor:
    """Higher score first, then higher block id (matches packed CP keys)."""
    score_cpu = score_row.detach().float().cpu()
    # Start in descending block-id order, then stable-sort on descending score
    # so ties keep the higher block id, which is how the packed key breaks them.
    order = torch.arange(score_cpu.numel() - 1, -1, -1)
    order = order[torch.argsort(-score_cpu[order], stable=True)]
    return order[:n]


@pytest.mark.parametrize("heads", [1, 4])
@torch.inference_mode()
def test_local_topk_merge_matches_full_topk(heads: int):
    """Sharded per-head local top-k + merge equals a full-row top-k per head.

    Every shard scores every head, as the CP decode does after gathering the
    index queries; the merge is exact only under that condition.
    """
    torch.manual_seed(2)
    world, query_len, topk = 4, 1, 16
    init_blocks, local_keep = 1, 2
    seq_lens = torch.tensor([128 * 20, 128 * 7], device=DEVICE, dtype=torch.int32)
    tokens = seq_lens.numel() * query_len
    blocks = int((seq_lens.max().item() + 127) // 128)
    scores = torch.randn(heads, tokens, blocks, device=DEVICE, dtype=torch.float32)
    forced = _force_scores(scores, seq_lens, query_len, init_blocks, local_keep)
    ref = torch.full((heads, tokens, topk), -1, device=DEVICE, dtype=torch.int32)
    for row in range(tokens):
        req = row // query_len
        token = row % query_len
        causal_len = int(seq_lens[req].item()) - query_len + token + 1
        n = min(topk, (causal_len + 127) // 128)
        for head in range(heads):
            ref[head, row, :n] = _lexsort_topk(forced[head, row], n).to(
                device=DEVICE, dtype=torch.int32
            )

    local = (blocks + world - 1) // world
    shard_keys = []
    for rank in range(world):
        shard = torch.full(
            (heads, tokens, local), -1e30, device=DEVICE, dtype=torch.float32
        )
        for loc in range(local):
            block = loc * world + rank
            if block < blocks:
                shard[:, :, loc] = scores[:, :, block]
        keys = torch.empty((heads, tokens, topk), device=DEVICE, dtype=torch.int64)
        local_topk_keys(
            shard,
            seq_lens,
            topk=topk,
            rank=rank,
            world=world,
            query_len=query_len,
            global_blocks=blocks,
            init_blocks=init_blocks,
            local_blocks=local_keep,
            out=keys,
        )
        shard_keys.append(keys)
    gathered = torch.stack(shard_keys, dim=0)
    got = torch.empty((heads, tokens, topk), device=DEVICE, dtype=torch.int32)
    merge_topk_keys(
        gathered,
        got,
        seq_lens,
        query_len=query_len,
        init_blocks=init_blocks,
        local_blocks=local_keep,
    )
    torch.accelerator.synchronize()
    assert torch.equal(got, ref)


@pytest.mark.parametrize(
    "total_heads,world,expected",
    [
        (4, 4, [(1, 0), (1, 1), (1, 2), (1, 3)]),
        (4, 2, [(2, 0), (2, 2)]),
        (4, 8, [(1, 0), (1, 0), (1, 1), (1, 1), (1, 2), (1, 2), (1, 3), (1, 3)]),
    ],
)
def test_cp_decode_owns_kv_sharded_heads(
    total_heads: int, world: int, expected: list[tuple[int, int]]
):
    """Each rank merges the index heads the fused QKV linear gives it."""
    from vllm.models.minimax_m3.amd.indexer_context_parallel import IndexerCPDecode

    for rank, (local_heads, offset) in enumerate(expected):
        dec = IndexerCPDecode(
            total_heads=total_heads,
            rank=rank,
            world_size=world,
            max_tokens=1,
            max_seq_len=BLOCK,
            topk=16,
            init_blocks=0,
            local_blocks=1,
            scale=1.0,
            group=None,  # type: ignore[arg-type]  # only used by forward()
            device=torch.device(DEVICE),
        )
        assert (dec.local_heads, dec.head_offset) == (local_heads, offset)
        assert dec._q_full.shape[1] == total_heads
        assert dec._scores.shape[0] == total_heads


@pytest.mark.parametrize("indexer_kv_dtype", ["auto", "bf16"])
def test_indexer_cp_config_accepts_bf16(indexer_kv_dtype: str):
    from vllm.config.attention import AttentionConfig

    cfg = AttentionConfig(minimax_m3_indexer_cp=True, indexer_kv_dtype=indexer_kv_dtype)
    assert cfg.minimax_m3_indexer_cp


@pytest.mark.parametrize("indexer_kv_dtype", ["fp8", "mxfp4", "nvfp4"])
def test_indexer_cp_config_rejects_non_bf16(indexer_kv_dtype: str):
    """CP scores the index cache with no scale; fp8 uses the AITER indexer."""
    from vllm.config.attention import AttentionConfig

    with pytest.raises(ValueError, match="minimax_m3_indexer_cp requires a bf16"):
        AttentionConfig(minimax_m3_indexer_cp=True, indexer_kv_dtype=indexer_kv_dtype)


def test_indexer_cp_config_default_off():
    from vllm.config.attention import AttentionConfig

    assert AttentionConfig().minimax_m3_indexer_cp is False
