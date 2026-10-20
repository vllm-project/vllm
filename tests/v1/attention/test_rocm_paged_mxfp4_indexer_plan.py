# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ROCm MXFP4 indexer reads the paged cache in place, so each prefill
chunk is split back into its requests' rows, which one launch finds through
the chunk's query_start_loc. A chunk is whole requests or one query slice of a
long request, and both have to come back with the right rows and the right
block-table rows.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm-only", allow_module_level=True)

from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerPrefillChunkMetadata,
)
from vllm.v1.attention.backends.mla.rocm_paged_mxfp4_indexer import (
    native_decode,
    plan_gather_launches,
    plan_prefill_chunks,
    split_prefill_chunks,
)
from vllm.v1.attention.ops.rocm_paged_mxfp4_indexer import rocm_mxfp4_consumer_rows


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


# A decode (1 token), then prefills of 3, 3, 5 and 6 query tokens; the last
# one is sliced into two chunks.
QUERY_START_LOC = [0, 1, 4, 7, 12, 18]
SEQ_LENS = torch.tensor([40, 10, 12, 21, 64])
BLOCK_TABLE = torch.arange(20, dtype=torch.int32).view(5, 4)
CHUNKS = [
    _chunk(BLOCK_TABLE[1:4], 1, 12, list(range(11))),
    _chunk(BLOCK_TABLE[4:5], 12, 15, [29, 30, 30]),
    _chunk(BLOCK_TABLE[4:5], 15, 18, [31, 31, 32]),
]


def _plans(min_gather_width, block=8):
    return plan_prefill_chunks(
        CHUNKS,
        QUERY_START_LOC,
        SEQ_LENS,
        SEQ_LENS.int() // 2,
        2,
        block,
        min_gather_width,
        torch.tensor(QUERY_START_LOC, dtype=torch.int32),
    )


def test_chunks_split_into_requests():
    plans = _plans(20.0)

    assert [p.requests for p in plans] == [
        [(0, 3, 0), (3, 6, 1), (6, 11, 2)],
        [(0, 3, 0)],
        [(0, 3, 0)],
    ]
    # every chunk launches once, with offsets local to the chunk
    assert [p.query_start_loc.tolist() for p in plans] == [
        [0, 3, 6, 11],
        [0, 3],
        [0, 3],
    ]
    # the logits are as wide as the longest compressed context in the chunk
    assert [p.width for p in plans] == [10, 32, 32]
    assert [p.context_lens.tolist() for p in plans] == [[5, 6, 10], [32], [32]]
    torch.testing.assert_close(plans[2].row_ends, torch.tensor([31, 31, 32]).int())
    torch.testing.assert_close(plans[2].block_ends, torch.tensor([4, 4, 4]).int())
    assert [p.use_gather for p in plans] == [False, True, True]
    assert [p.first_request for p in plans] == [1, 4, 4]
    # no candidate blocks, as in the model-neutral builder
    for plan in _plans(None, block=0):
        assert plan.block_ends is None and not plan.use_gather


def test_launches_follow_the_logits_budget(monkeypatch):
    """Each launch's fp32 logits fit in VLLM_SPARSE_INDEXER_MAX_LOGITS_MB. A dense
    chunk's logits are [rows, longest context in the chunk]: K is read in place,
    not gathered, so requests share a launch until their rows no longer fit, and
    a request that does not fit alone is split on its query rows. A consumer's
    logits are [rows, pool]."""
    budget = 4 * 1000 * 30
    contexts, queries = torch.tensor([1000, 600, 200]), torch.tensor([10, 10, 10])
    assert split_prefill_chunks(contexts, queries, budget, 2) == [
        (slice(2, 5), slice(0, 30))
    ]
    # the third request would take the chunk past the budget
    queries = torch.tensor([10, 10, 11])
    assert split_prefill_chunks(contexts, queries, budget) == [
        (slice(0, 2), slice(0, 20)),
        (slice(2, 3), slice(0, 11)),
    ]
    assert split_prefill_chunks(torch.tensor([1000]), torch.tensor([100]), budget) == [
        (slice(0, 1), slice(lo, min(lo + 30, 100))) for lo in range(0, 100, 30)
    ]
    # the default 512 MiB holds 128 rows of a 1M context, or 8192 of a 16K pool
    assert split_prefill_chunks(
        torch.tensor([1 << 20]), torch.tensor([200]), 512 << 20
    ) == [(slice(0, 1), slice(0, 128)), (slice(0, 1), slice(128, 200))]
    monkeypatch.setenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "512")
    assert rocm_mxfp4_consumer_rows(16384) == 8192


def test_gather_launches_rejoin_a_requests_rows():
    """The consumers' logits are [rows, pool] whatever the context, so a
    request's rows go back together across the chunks that cut them, and a
    launch never mixes requests."""
    # the first chunk is too narrow to gather, the last request's slices rejoin
    (joined,) = plan_gather_launches(CHUNKS, _plans(20.0), 100)
    assert (joined.token_start, joined.token_end) == (12, 18)
    # every chunk gathers: the first chunk's three requests launch apart
    launches = plan_gather_launches(CHUNKS, _plans(0.0), 100)
    assert [(g.token_start, g.token_end) for g in launches] == [
        (1, 4),
        (4, 7),
        (7, 12),
        (12, 18),
    ]
    assert plan_gather_launches(CHUNKS, _plans(None), 100) == []


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
