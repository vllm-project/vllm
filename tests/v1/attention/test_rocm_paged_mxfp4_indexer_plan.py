# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ROCm MXFP4 indexer reads the paged cache in place, so each prefill
chunk is split back into its requests' rows. The kernel's sequences share one
row count, so a launch takes a run of consecutive requests with equal query
rows. A chunk is whole requests or one query slice of a long request, and both
have to come back with the right rows and the right block-table rows.
"""

import types

import pytest
import torch

from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerPrefillChunkMetadata,
)
from vllm.v1.attention.backends.mla.rocm_paged_mxfp4_indexer import (
    native_decode,
    plan_gather_launches,
    plan_prefill_chunks,
    split_prefill_chunks,
)
from vllm.v1.attention.ops import rocm_paged_mxfp4_indexer as ops


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


def _plans(min_gather_width, query_start_loc_device=None):
    return plan_prefill_chunks(
        CHUNKS,
        QUERY_START_LOC,
        SEQ_LENS,
        SEQ_LENS.int() // 2,
        2,
        8,
        min_gather_width,
        query_start_loc_device,
    )


def test_chunks_split_into_requests_and_launches():
    plans = _plans(20.0)

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
    assert [p.first_request for p in plans] == [1, 4, 4]
    # Without the device query_start_loc (varlen off) every chunk launches per run.
    assert all(p.query_start_loc is None for p in plans)

    packed = _plans(20.0, torch.tensor(QUERY_START_LOC, dtype=torch.int32))
    # Only the ragged chunk packs, with offsets local to the chunk; the
    # single-request slices keep their one uniform launch.
    assert packed[0].query_start_loc.tolist() == [0, 3, 6, 11]
    assert packed[1].query_start_loc is None and packed[2].query_start_loc is None


def test_dense_split_budgets_the_widest_row():
    """A dense launch holds [rows, widest context] logits: the requests packed
    into a chunk do not add their contexts up, as a gathered K would, and a
    request too wide to launch whole is cut on its query rows."""
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
    # a logits tensor stays under 2 GiB whatever the budget
    (first, *_) = split_prefill_chunks(
        torch.tensor([1 << 20]), torch.tensor([1000]), 1 << 40
    )
    assert first[1] == slice(0, (2**31 - 1) // (4 << 20))


def test_consumer_launch_rows_follow_the_logits_budget(monkeypatch):
    """A consumer launch holds [rows, pool] fp32 logits whatever the context,
    so the logits budget alone sizes it, not how many of the step's rows
    gather."""
    monkeypatch.setenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "512")
    assert ops.rocm_mxfp4_consumer_rows(16384) == 8192
    monkeypatch.setenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "4096")
    assert ops.rocm_mxfp4_consumer_rows(16384) == (2**31 - 1) // (4 * 16384)


def test_gather_launches_rejoin_a_requests_rows():
    """The consumers' logits are [rows, pool] whatever the context, so their
    launches are not the dense split's: a request's rows go back together
    across the chunks that cut them, then out again at most max_rows a
    launch, never mixing requests."""
    plans = _plans(20.0)
    (joined,) = plan_gather_launches(CHUNKS, plans, 100)
    assert (joined.token_start, joined.token_end) == (12, 18)
    assert joined.row_ends.tolist() == [29, 30, 30, 31, 31, 32]
    assert joined.block_table.tolist() == BLOCK_TABLE[4:5].tolist()
    assert joined.context_len.tolist() == [32]

    split = plan_gather_launches(CHUNKS, plans, 4)
    assert [(g.token_start, g.token_end) for g in split] == [(12, 16), (16, 18)]
    assert [g.row_ends.tolist() for g in split] == [[29, 30, 30, 31], [31, 32]]

    # every chunk gathers: the first chunk's three requests launch apart
    launches = plan_gather_launches(CHUNKS, _plans(0.0), 100)
    assert [(g.token_start, g.token_end) for g in launches] == [
        (1, 4),
        (4, 7),
        (7, 12),
        (12, 18),
    ]
    assert [g.block_table[0, 0].item() for g in launches] == [4, 8, 12, 16]
    assert plan_gather_launches(CHUNKS, _plans(None), 100) == []


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


@pytest.mark.parametrize(
    "fmt, layout",
    [
        # today's cache_format, which implies the scale order and lane split
        ({"n_per_tile": 32, "d_per_tile": 16, "block_kv": 64}, (32, 16, 2)),
        ({"n_per_tile": 16, "d_per_tile": 16, "scale_lanes": 4}, (16, 16, 4)),
        # a key renamed, and a scale order the writer does not implement
        ({"n_per_tile": 32, "k_width": 16}, None),
        ({"n_per_tile": 32, "d_per_tile": 16, "scale_mode": 0}, None),
    ],
)
def test_cache_layout_from_aiter_fails_closed(monkeypatch, fmt, layout):
    """The K writer's page order is read from aiter's cache_format, an API
    that may move: what it reports either maps onto the order the writer
    implements or stops startup, never a silently misordered cache."""
    aiter = types.SimpleNamespace(cache_format=lambda *args: fmt)
    monkeypatch.setattr(ops, "_aiter", lambda: aiter)
    build = ops.rocm_paged_mxfp4_cache_layout.__wrapped__  # bypass the cache
    if layout is None:
        with pytest.raises(ValueError, match="cache_format"):
            build(32, 128, 128)
    else:
        found = build(32, 128, 128)
        assert (found.n_per_tile, found.d_per_tile, found.scale_lanes) == layout
