# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ROCm MXFP4 indexer reads the paged cache in place, so each prefill
chunk is split back into its requests' rows. The kernel's sequences share one
row count, so a launch takes a run of consecutive requests with equal query
rows. A chunk is whole requests or one query slice of a long request, and both
have to come back with the right rows and the right block-table rows.
"""

import torch

from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerPrefillChunkMetadata,
)
from vllm.v1.attention.backends.mla.rocm_mxfp4_indexer import (
    native_decode,
    plan_prefill_chunks,
)


def _chunk(block_table, token_start, token_end, row_ends):
    ends = torch.tensor(row_ends, dtype=torch.int32)
    return DeepseekV32IndexerPrefillChunkMetadata(
        block_table=block_table,
        cu_seqlen_ks=torch.full_like(ends, 7),
        cu_seqlen_ke=ends + 7,
        cu_seq_lens=torch.zeros(1, dtype=torch.int32),
        token_to_seq=torch.zeros(1, dtype=torch.int32),
        total_seq_lens=1,
        token_start=token_start,
        token_end=token_end,
        num_reqs=block_table.shape[0],
    )


def test_chunks_split_into_requests_and_launches():
    # A decode (1 token), then prefills of 3, 3, 5 and 6 query tokens; the
    # last one is sliced into two chunks.
    query_start_loc = [0, 1, 4, 7, 12, 18]
    seq_lens = torch.tensor([40, 10, 12, 21, 64])
    block_table = torch.arange(20, dtype=torch.int32).view(5, 4)
    chunks = [
        _chunk(block_table[1:4], 1, 12, list(range(11))),
        _chunk(block_table[4:5], 12, 15, [29, 30, 30]),
        _chunk(block_table[4:5], 15, 18, [31, 31, 32]),
    ]
    context_lens = seq_lens.int() // 2
    plans = plan_prefill_chunks(
        chunks, query_start_loc, seq_lens, context_lens, 2, 8, 20.0
    )

    assert [p.requests for p in plans] == [
        [(0, 3, 0), (3, 6, 1), (6, 11, 2)],
        [(0, 3, 0)],
        [(0, 3, 0)],
    ]
    # the two 3-row requests share a launch, the 5-row one gets its own
    assert [p.launches for p in plans] == [
        [(0, 6, 0, 2), (6, 11, 2, 1)],
        [(0, 3, 0, 1)],
        [(0, 3, 0, 1)],
    ]
    # the logits are as wide as the longest compressed context in the chunk
    assert [p.width for p in plans] == [10, 32, 32]
    assert [p.context_lens.tolist() for p in plans] == [[5, 6, 10], [32], [32]]
    torch.testing.assert_close(plans[2].row_ends, torch.tensor([31, 31, 32]).int())
    torch.testing.assert_close(plans[2].block_ends, torch.tensor([4, 4, 4]).int())
    assert [p.use_gather for p in plans] == [False, True, True]
    assert all(p.gathers == [None] * len(p.requests) for p in plans)
    # Without the device query_start_loc (varlen off) every chunk launches per run.
    assert all(p.query_start_loc is None for p in plans)

    packed = plan_prefill_chunks(
        chunks,
        query_start_loc,
        seq_lens,
        context_lens,
        2,
        8,
        20.0,
        torch.tensor(query_start_loc, dtype=torch.int32),
    )
    # Only the ragged chunk packs, with offsets local to the chunk; the
    # single-request slices keep their one uniform launch.
    assert packed[0].query_start_loc.tolist() == [0, 3, 6, 11]
    assert packed[1].query_start_loc is None and packed[2].query_start_loc is None


def test_no_gather_without_candidates():
    block_table = torch.zeros(1, 2, dtype=torch.int32)
    (plan,) = plan_prefill_chunks(
        [_chunk(block_table, 0, 2, [3, 4])],
        [0, 2],
        torch.tensor([1000]),
        torch.tensor([1000], dtype=torch.int32),
        1,
        0,
        None,
    )
    assert plan.block_ends is None and not plan.use_gather


def test_decode_launches_native_on_uniform_steps():
    """A decode step goes to the kernel as next_n-row sequences only when its
    shape says so: each request has next_n rows, and cudagraph padding (query
    length 0) comes only after them. A FULL graph captured on a uniform batch
    then replays the same launch on a padded one."""
    rows = torch.zeros(8, dtype=torch.int32)
    row_block_table = torch.arange(32, dtype=torch.int32).view(8, 4)
    context_lens = torch.tensor([9, 4, 7, 0, 5], dtype=torch.int32)

    native = native_decode(rows, row_block_table, [2, 2, 2, 0], 2, context_lens)
    assert native is not None and native.next_n == 2
    assert native.context_lens.tolist() == [9, 4, 7, 0]
    # every request's first flattened row, as a view
    assert native.block_table.tolist() == row_block_table[::2].tolist()
    assert native.block_table.data_ptr() == row_block_table.data_ptr()

    assert native_decode(rows, row_block_table, [2, 0, 2, 2], 2, context_lens) is None
    # a ragged step, rows padded past the requests, and no speculation
    assert native_decode(rows[:5], row_block_table, [2, 1, 2], 2, context_lens) is None
    assert native_decode(rows, row_block_table, [2, 2, 2], 2, context_lens) is None
    assert native_decode(rows[:4], row_block_table, [1] * 4, 1, context_lens) is None
