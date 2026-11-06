# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ShardQ split of the ROCm FP8 sparse-indexer prefill (DeepSeek-V4.1).

One GPU plays every TP rank: each rank scores its zigzag tokens, which must
select what the replicated run selects for them, including a candidate
consumer whose candidate rows of other ranks hold junk; the exchange then puts
every row back in token order. Rows compare as sets: the top-k order within a
row depends on the batch shape."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)

HEADS, DIM, PAGE, TOPK = 32, 128, 64, 512
MAX_MODEL_LEN = 1 << 16
# More candidate positions than TOPK, so no row picks among masked ties.
CAND_COLS, CAND_BLOCK = 128, 8
PREFIX = "indexer"
# (cached context, new rows) per request, in two chunks of 2 + 1 requests.
REQUESTS = [(0, 1500), (400, 300), (0, 2600)]
CHUNKS = [(0, 2), (2, 3)]
ROWS = sum(new for _, new in REQUESTS)


@pytest.fixture(autouse=True)
def _aiter(monkeypatch):
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import mqa_logits_module
    from vllm.v1.worker.workspace import (
        init_workspace_manager,
        is_workspace_manager_initialized,
    )

    monkeypatch.setattr(rocm_aiter_ops, "_AITER_ENABLED", True)
    if mqa_logits_module() is None:
        pytest.skip("AITER's MQA logits module is not installed")
    if not is_workspace_manager_initialized():
        init_workspace_manager(torch.device("cuda"))


def _cache(ratio: int):
    """Queries, weights, a paged K cache, its block tables and key counts."""
    from aiter import dtypes

    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        indexer_k_quant_and_cache_triton,
    )

    torch.manual_seed(ratio)
    q = (torch.randn(ROWS, HEADS, DIM, device="cuda") * 0.5).to(dtypes.fp8)
    weights = torch.randn(ROWS, HEADS, device="cuda")
    keys = [(ctx + new) // ratio for ctx, new in REQUESTS]
    pages = [(n + PAGE - 1) // PAGE for n in keys]
    perm = torch.randperm(sum(pages), device="cuda").int().split(pages)
    tables = torch.zeros(len(REQUESTS), max(pages), dtype=torch.int32, device="cuda")
    slots = []
    for i, (n, row) in enumerate(zip(keys, perm)):
        tables[i, : len(row)] = row
        pos = torch.arange(n, device="cuda")
        slots.append(tables[i, pos // PAGE].long() * PAGE + pos % PAGE)
    kv = torch.zeros(sum(pages), PAGE, DIM + 4, dtype=torch.uint8, device="cuda")
    k = torch.randn(sum(keys), DIM, dtype=torch.bfloat16, device="cuda")
    indexer_k_quant_and_cache_triton(k, kv, torch.cat(slots), 128, "ue8m0")
    return ratio, q, weights, kv, tables, keys


def _chunks(cache, tokens: np.ndarray):
    """The prefill chunks of ``tokens`` (ascending ids into all the rows),
    numbered compactly, as the indexer builder chunks a ShardQ split."""
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerPrefillChunkMetadata,
    )

    ratio, _, _, _, tables, keys = cache
    starts = np.cumsum([0] + [new for _, new in REQUESTS])
    req = np.searchsorted(starts, tokens, side="right") - 1
    chunks, row = [], 0
    for lo, hi in CHUNKS:
        mine = (req >= lo) & (req < hi)
        cu = torch.tensor([0, *keys[lo:hi]], device="cuda").cumsum(0).int()
        r = req[mine]
        pos = torch.from_numpy(tokens[mine] - starts[r] + np.array(REQUESTS)[r, 0])
        ks = cu[torch.from_numpy(r - lo).cuda()]
        ke = ks + (pos.cuda() + 1) // ratio
        chunks.append(
            DeepseekV32IndexerPrefillChunkMetadata(
                block_table=tables[lo:hi],
                cu_seqlen_ks=ks.int(),
                cu_seqlen_ke=ke.int(),
                cu_seq_lens=cu,
                token_to_seq=torch.repeat_interleave(
                    torch.arange(hi - lo, dtype=torch.int32, device="cuda"),
                    torch.tensor(keys[lo:hi], device="cuda"),
                ),
                total_seq_lens=int(cu[-1]),
                max_seqlen_k=int((ke - ks).max()),
                token_start=row,
                token_end=row + int(mine.sum()),
                num_reqs=hi - lo,
            )
        )
        row += int(mine.sum())
    return chunks


def _run(monkeypatch, cache, split=None, tokens=None, cand=None, write=False):
    """One indexer layer over all the rows, or over a rank's ShardQ ``tokens``."""
    from vllm.config import CUDAGraphMode
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadata,
        DeepseekV32IndexerPrefillMetadata,
    )
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse

    ratio, q, weights, kv, _, _ = cache
    indices = None if tokens is None else torch.from_numpy(tokens).cuda()
    chunks = _chunks(cache, np.arange(ROWS) if tokens is None else tokens)
    metadata = DeepseekV32IndexerMetadata(
        seq_lens=torch.empty(0, device="cuda"),
        max_seq_len=0,
        slot_mapping=torch.empty(0, dtype=torch.int64, device="cuda"),
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=len(REQUESTS),
        num_prefill_tokens=ROWS,
        prefill=DeepseekV32IndexerPrefillMetadata(
            chunks, shardq_gather_indices=indices, shardq_split=split
        ),
    )
    context = SimpleNamespace(
        attn_metadata={PREFIX: metadata},
        no_compile_layers={PREFIX: SimpleNamespace(gather_shardq_topk=True)},
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )
    monkeypatch.setattr(rocm_aiter_mla_sparse, "get_forward_context", lambda: context)
    out = torch.full((ROWS, TOPK), -7, dtype=torch.int32, device="cuda")
    rocm_aiter_mla_sparse.rocm_aiter_sparse_attn_indexer(
        torch.empty(ROWS, 1, device="cuda"),
        PREFIX,
        kv,
        q,
        None,
        weights,
        128,
        "ue8m0",
        TOPK,
        DIM,
        MAX_MODEL_LEN,
        max(c.total_seq_lens for c in chunks),
        out,
        skip_k_cache_insert=True,
        compress_ratio=ratio,
        candidate_blocks=cand,
        candidate_block_size=CAND_BLOCK if cand is not None else 0,
        candidate_write=write,
    )
    return out


@pytest.mark.parametrize("world", [2, 4])
def test_shardq_split_matches_replicated(monkeypatch, world):
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse
    from vllm.v1.attention.ops.shardq import ShardQSplit, zigzag_split

    def sets(t):
        return t.sort(dim=-1).values

    two, one = _cache(2), _cache(1)
    ref_cand = torch.empty(ROWS, CAND_COLS, dtype=torch.int32, device="cuda")
    # The ratio-2 layer, then the candidate source and a consumer.
    ref = (
        _run(monkeypatch, two),
        _run(monkeypatch, one, cand=ref_cand, write=True),
        _run(monkeypatch, one, cand=ref_cand),
    )
    token_ids, counts = zigzag_split(np.array([new for _, new in REQUESTS]), world)
    split = ShardQSplit(0, token_ids, counts)
    for rank in range(world):
        start = counts[:rank].sum()
        tokens = token_ids[start : start + counts[rank]]
        mine = torch.from_numpy(tokens).cuda()
        gathered = []

        def all_gatherv(local, dim, sizes, gathered=gathered):
            gathered.append(local.clone())
            # The other ranks' rows, so the exchange can be checked whole.
            return ref[len(gathered) - 1][torch.from_numpy(token_ids).cuda()]

        group = SimpleNamespace(rank_in_group=rank, all_gatherv=all_gatherv)
        monkeypatch.setattr(rocm_aiter_mla_sparse, "get_tp_group", lambda g=group: g)
        # Rows of other ranks hold junk, never the replicated selection.
        cand = torch.randint(0, 400, ref_cand.shape, dtype=torch.int32, device="cuda")
        outs = (
            _run(monkeypatch, two, split, tokens),
            _run(monkeypatch, one, split, tokens, cand, write=True),
            _run(monkeypatch, one, split, tokens, cand),
        )
        assert len(gathered) == 3
        for local, out, want in zip(gathered, outs, ref):
            assert torch.equal(sets(local), sets(want[mine]))
            assert torch.equal(sets(out), sets(want))
        assert torch.equal(sets(cand[: len(mine)]), sets(ref_cand[mine]))
