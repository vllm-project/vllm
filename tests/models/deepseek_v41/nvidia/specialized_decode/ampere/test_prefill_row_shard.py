# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SM80/SM90 indexer's prefill row split: the ranks' balanced shares
cover every row, and each rank's narrowed chunks score exactly its share,
with the matching key bounds, while a chunk holding none of its rows keeps
an empty window (it still gathers the keys a continuation chunk reuses).
Host arithmetic only."""

import pytest
import torch

from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere.row_shard import (
    PrefillRowShard,
    RowShardedPrefillMetadata,
    balanced_prefill_row_shard,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerPrefillChunkMetadata,
)


def _chunk(start: int, end: int, skip_kv_gather: bool):
    rows = torch.arange(start, end)
    return DeepseekV32IndexerPrefillChunkMetadata(
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        cu_seqlen_ks=rows * 10,
        cu_seqlen_ke=rows * 10 + 7,
        cu_seq_lens=torch.zeros(2, dtype=torch.int32),
        token_to_seq=torch.zeros(end - start, dtype=torch.int32),
        total_seq_lens=0,
        max_seqlen_k=0,
        token_start=start,
        token_end=end,
        num_reqs=1,
        skip_kv_gather=skip_kv_gather,
    )


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_shares_cover_the_rows(world_size: int) -> None:
    seq_lens = torch.tensor([98304, 512, 8192])
    query_lens = torch.tensor([4096, 512, 1000])
    sizes = balanced_prefill_row_shard(seq_lens, query_lens, 4, world_size)
    assert sizes is not None and len(sizes) == world_size
    assert sum(sizes) == int(query_lens.sum()) and min(sizes) > 0
    # Fewer rows than ranks stay replicated.
    few = torch.tensor([3])
    assert balanced_prefill_row_shard(seq_lens[:1], few, 4, 8) is None


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_narrowed_chunks_score_each_share(world_size: int) -> None:
    num_decode_tokens = 12
    # One request's 900 rows in a first chunk and two continuation chunks,
    # then a second request's 100 rows.
    bounds = [(12, 412, False), (412, 812, True), (812, 912, True), (912, 1012, False)]
    prefill = RowShardedPrefillMetadata(
        [_chunk(*b) for b in bounds],
        row_shard_sizes=[1000 // world_size] * (world_size - 1)
        + [1000 - (world_size - 1) * (1000 // world_size)],
    )
    covered: list[int] = []
    for rank in range(world_size):
        shard = PrefillRowShard(num_decode_tokens, prefill.row_shard_sizes, rank)
        narrowed = shard.narrow_prefill(prefill)
        assert len(narrowed.chunks) == len(prefill.chunks)
        for chunk, full in zip(narrowed.chunks, prefill.chunks):
            assert chunk.skip_kv_gather == full.skip_kv_gather
            assert full.token_start <= chunk.token_start <= chunk.token_end
            assert chunk.token_end <= full.token_end
            rows = torch.arange(chunk.token_start, chunk.token_end)
            assert torch.equal(chunk.cu_seqlen_ks, rows * 10)
            assert torch.equal(chunk.cu_seqlen_ke, rows * 10 + 7)
            if chunk.token_end > chunk.token_start:
                assert shard.start <= chunk.token_start
                assert chunk.token_end <= shard.stop
            covered += rows.tolist()
    assert covered == list(range(num_decode_tokens, num_decode_tokens + 1000))
