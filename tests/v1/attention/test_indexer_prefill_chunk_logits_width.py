# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Planner test for the fixed-width sparse indexer prefill logits.

With ``logits_width`` the sub-chunk launch shapes of a long prefill are
identical at every step (so the caching allocator can reuse the same blocks);
with ``logits_width=0`` the planner is unchanged.
"""

import pytest
import torch

from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder

split = DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks

BUDGET = 512 * 1024 * 1024
MAX_MODEL_LEN = 409_600
COMPRESS_RATIO = 4
WIDTH = -(-MAX_MODEL_LEN // COMPRESS_RATIO)
WORKSPACE = MAX_MODEL_LEN * 40


def _launches(specs):
    return tuple(sorted(q.stop - q.start for _, q in specs))


@pytest.mark.parametrize("chunk_tokens", [4608, 8192])
def test_fixed_width_gives_constant_launch_shapes(chunk_tokens):
    shapes = set()
    for ctx in range(chunk_tokens, MAX_MODEL_LEN + 1, chunk_tokens):
        n = ctx // COMPRESS_RATIO
        specs = split(
            torch.tensor([n]),
            torch.tensor([chunk_tokens]),
            WORKSPACE,
            BUDGET,
            logits_width=WIDTH,
        )
        rows = _launches(specs)
        assert sum(rows) == chunk_tokens
        assert all(r * WIDTH * 4 <= BUDGET for r in rows)
        shapes.add(rows)
    # the same launch shapes at every step of the prefill
    assert len(shapes) == 1


def test_zero_width_keeps_the_legacy_plan():
    for ctx in (4608, 65_536, 200_704):
        n = ctx // COMPRESS_RATIO
        args = (torch.tensor([n]), torch.tensor([4608]), WORKSPACE, BUDGET)
        assert split(*args, logits_width=0) == split(*args)
        # legacy: the logits are budgeted at the chunk's own kv length
        for _, q in split(*args):
            assert (q.stop - q.start) * n * 4 <= BUDGET


def test_width_is_never_below_the_kv_length():
    n = 3 * WIDTH  # a chunk longer than the configured width
    specs = split(
        torch.tensor([n]), torch.tensor([4608]), WORKSPACE, BUDGET, logits_width=WIDTH
    )
    for _, q in specs:
        assert (q.stop - q.start) * n * 4 <= BUDGET


def test_packed_requests_are_budgeted_at_the_fixed_width():
    # Two short requests: at their own kv length they pack into one launch;
    # at the fixed width the rows must be sub-chunked to stay under budget.
    seq_lens = torch.tensor([100, 100])
    query_lens = torch.tensor([1000, 1000])
    legacy = split(seq_lens, query_lens, WORKSPACE, BUDGET)
    assert len(legacy) == 1
    fixed = split(seq_lens, query_lens, WORKSPACE, BUDGET, logits_width=WIDTH)
    max_rows = BUDGET // 4 // WIDTH
    assert all(q.stop - q.start <= max_rows for _, q in fixed)
    assert sum(q.stop - q.start for _, q in fixed) == 2000
