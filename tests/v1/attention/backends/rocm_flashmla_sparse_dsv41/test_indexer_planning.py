# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Planning in the ROCm paged MXFP4 indexer builder (DeepSeek V4.1's
compress-ratio > 1 and two-level candidate paths): how a step's rows become
aiter launches. Each planner is checked against the invariant it promises,
on host tensors (no GPU): every query row is planned exactly once, in order,
within the logits budget, and the candidate consumers' gathers rejoin a
request's rows across the chunks that cut it.

The kernels these plans drive are checked against a float32 reference in
``tests/kernels/attention/test_rocm_paged_mxfp4_indexer.py``; the whole path
runs in the engine in ``test_dispatch.py``.
"""

import random
from types import SimpleNamespace

import pytest
import torch

mxfp4 = pytest.importorskip("vllm.v1.attention.backends.mla.rocm_paged_mxfp4_indexer")


# ---- native_decode: next_n-row sequences for uniform decode steps ----


def _native(query_lens, next_n, rows=None):
    num_rows = sum(query_lens) if rows is None else rows
    row_lens = torch.arange(num_rows, dtype=torch.int32) + 100
    block_table = torch.arange(num_rows * 4, dtype=torch.int32).view(num_rows, 4)
    context = torch.arange(len(query_lens) + 3, dtype=torch.int32) + 7
    return (
        mxfp4.native_decode(row_lens, block_table, query_lens, next_n, context),
        block_table,
        context,
    )


def test_native_decode_uniform_step():
    native, block_table, context = _native([3, 3, 3, 3], 3)
    assert native is not None and native.next_n == 3
    torch.testing.assert_close(native.context_lens, context[:4])
    # Each request's first flattened row.
    torch.testing.assert_close(native.block_table, block_table[::3])


def test_native_decode_keeps_trailing_padding():
    """CUDA-graph padding requests (query length 0) keep their next_n rows,
    so a FULL graph captured on a uniform batch replays the same launch."""
    native, _, context = _native([2, 2, 0, 0], 2, rows=8)
    assert native is not None
    torch.testing.assert_close(native.context_lens, context[:4])


@pytest.mark.parametrize(
    ("query_lens", "next_n", "rows"),
    [
        ([1, 1, 1], 1, None),  # plain decode: no sequences to form
        ([3, 2, 3], 3, None),  # a short request (rejected drafts)
        ([3, 0, 3], 3, 9),  # an empty request that is not trailing
        ([3, 3], 3, 5),  # row count disagrees with the shape
    ],
    ids=["next_n-1", "ragged", "hole", "rows"],
)
def test_native_decode_refuses_other_steps(query_lens, next_n, rows):
    native, _, _ = _native(query_lens, next_n, rows)
    assert native is None


# ---- split_prefill_chunks: logits-budget chunking without a K gather ----


def _coverage(chunks, query_lens):
    rows = []
    for reqs, q in chunks:
        lens = query_lens[reqs]
        flat = [
            (r, i) for r, n in zip(range(reqs.start, reqs.stop), lens) for i in range(n)
        ]
        rows.extend(flat[q])
    return rows


@pytest.mark.parametrize("seed", range(20))
def test_split_prefill_chunks_covers_every_row_within_budget(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 9)
    context = torch.tensor([rng.choice([1, 64, 513, 4096, 65536]) for _ in range(n)])
    query = torch.tensor([rng.randint(1, 3000) for _ in range(n)])
    budget = rng.choice([1 << 16, 1 << 20, 1 << 24])
    chunks = mxfp4.split_prefill_chunks(context, query, budget)
    want = [(r, i) for r in range(n) for i in range(int(query[r]))]
    assert _coverage(chunks, query.tolist()) == want
    for reqs, q in chunks:
        width = int(context[reqs].max())
        rows = q.stop - q.start
        # A chunk fits the fp32 logits budget, unless one row alone does not.
        assert rows == 1 or rows * width * 4 <= budget, (reqs, q, width)


def test_split_prefill_chunks_request_offset():
    context, query = torch.tensor([10, 10]), torch.tensor([4, 4])
    chunks = mxfp4.split_prefill_chunks(context, query, 1 << 20, request_offset=5)
    assert chunks == [(slice(5, 7), slice(0, 8))]


# ---- plan_prefill_chunks / plan_gather_launches ----


def _chunk(token_start, token_end, num_reqs, ks, ke):
    return SimpleNamespace(
        token_start=token_start,
        token_end=token_end,
        num_reqs=num_reqs,
        cu_seqlen_ks=torch.tensor(ks, dtype=torch.int32),
        cu_seqlen_ke=torch.tensor(ke, dtype=torch.int32),
        block_table=torch.arange(num_reqs * 2, dtype=torch.int32).view(num_reqs, 2),
    )


def test_plan_prefill_chunks_splits_requests():
    # Two requests of 3 and 5 query rows, the second cut after its 2nd row.
    query_start_loc = [0, 3, 8]
    seq_lens = torch.tensor([40, 100])
    ratio, block = 2, 8
    context = seq_lens // ratio
    chunks = [
        _chunk(0, 5, 2, [0, 0, 0, 0, 0], [19, 19, 20, 45, 46]),
        _chunk(5, 8, 1, [0, 0, 0], [47, 48, 50]),
    ]
    plans = mxfp4.plan_prefill_chunks(
        chunks,
        query_start_loc,
        seq_lens,
        context,
        ratio,
        block,
        min_gather_width=45.0,
        query_start_loc_device=torch.tensor(query_start_loc, dtype=torch.int32),
    )
    first, second = plans
    assert first.requests == [(0, 3, 0), (3, 5, 1)]
    assert first.first_request == 0 and first.width == 50
    torch.testing.assert_close(
        first.query_start_loc, torch.tensor([0, 3, 5], dtype=torch.int32)
    )
    torch.testing.assert_close(
        first.row_ends, torch.tensor([19, 19, 20, 45, 46], dtype=torch.int32)
    )
    torch.testing.assert_close(first.block_ends, (first.row_ends + block - 1) // block)
    assert first.use_gather  # 50 compressed positions >= 45
    assert second.requests == [(0, 3, 0)] and second.first_request == 1
    torch.testing.assert_close(
        second.query_start_loc, torch.tensor([0, 3], dtype=torch.int32)
    )
    torch.testing.assert_close(second.context_lens, context[1:2])


def test_plan_prefill_chunks_without_candidates():
    """A single-level indexer (DeepSeek V4) has no candidate blocks to bound."""
    plans = mxfp4.plan_prefill_chunks(
        [_chunk(0, 2, 1, [0, 0], [3, 4])],
        [0, 2],
        torch.tensor([8]),
        torch.tensor([2]),
        2,
        0,
        None,
        torch.tensor([0, 2], dtype=torch.int32),
    )
    assert plans[0].block_ends is None and not plans[0].use_gather


def test_plan_gather_launches_rejoins_cut_requests():
    query_start_loc = [0, 3, 8]
    seq_lens = torch.tensor([40, 100])
    chunks = [
        _chunk(0, 5, 2, [0] * 5, [19, 19, 20, 45, 46]),
        _chunk(5, 8, 1, [0] * 3, [47, 48, 50]),
    ]
    plans = mxfp4.plan_prefill_chunks(
        chunks,
        query_start_loc,
        seq_lens,
        seq_lens // 2,
        2,
        8,
        min_gather_width=0.0,
        query_start_loc_device=torch.tensor(query_start_loc, dtype=torch.int32),
    )
    launches = mxfp4.plan_gather_launches(chunks, plans, max_rows=4)
    spans = [(launch.token_start, launch.token_end) for launch in launches]
    # Request 0's 3 rows; request 1's 5 rows rejoined across both chunks,
    # then cut at 4 rows a launch.
    assert spans == [(0, 3), (3, 7), (7, 8)]
    torch.testing.assert_close(
        torch.cat([launch.row_ends for launch in launches]),
        torch.tensor([19, 19, 20, 45, 46, 47, 48, 50], dtype=torch.int32),
    )
    assert all(launch.pool is None for launch in launches)


def test_plan_gather_launches_skips_dense_chunks():
    chunks = [_chunk(0, 2, 1, [0, 0], [3, 4])]
    plans = mxfp4.plan_prefill_chunks(
        chunks,
        [0, 2],
        torch.tensor([8]),
        torch.tensor([4]),
        2,
        8,
        min_gather_width=1e9,
        query_start_loc_device=torch.tensor([0, 2], dtype=torch.int32),
    )
    assert mxfp4.plan_gather_launches(chunks, plans, max_rows=4) == []
