# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm.v1.attention.backends.utils import fill_mm_prefix_query_ranges

# Stands in for the builder's persistent (max_num_batched_tokens, 2) buffer.
STAGING_CAPACITY = 64


def _query_ranges(mm_ranges, query_lens, seq_lens):
    """Fill a staging buffer and return the written rows, or None if empty."""
    query_start_loc = torch.tensor(
        [0, *torch.tensor(query_lens).cumsum(0).tolist()], dtype=torch.int32
    )
    # Poison the buffer so a stale row surviving the fill is visible.
    out = np.full((STAGING_CAPACITY, 2), 12345, dtype=np.int32)
    num_tokens = fill_mm_prefix_query_ranges(
        out,
        mm_ranges,
        query_start_loc,
        torch.tensor(seq_lens, dtype=torch.int32),
    )
    if num_tokens == 0:
        return None
    return torch.from_numpy(out[:num_tokens])


def test_matches_range_scan_semantics_with_context_offset():
    """Pin the equivalence the O(1) lookup rests on.

    The kernel checks ``r_start <= kv_idx <= r_end`` for the range containing
    the query token, which is only equal to the old scan's
    ``any(q in r and kv in r)`` because ranges never overlap.  Request 1 carries
    a context offset so the local-to-absolute query mapping is exercised too.
    """
    mm_ranges = {0: [(1, 3), (5, 7)], 1: [(2, 4), (9, 12)]}
    query_lens = [8, 6]
    seq_lens = [8, 13]

    query_ranges = _query_ranges(mm_ranges, query_lens, seq_lens)
    assert query_ranges is not None

    token_start = 0
    for req_idx, query_len in enumerate(query_lens):
        context_len = seq_lens[req_idx] - query_len
        for q_local in range(query_len):
            q_abs = context_len + q_local
            r_start, r_end = query_ranges[token_start + q_local].tolist()
            for kv_idx in range(seq_lens[req_idx]):
                old_scan_keep = any(
                    start < end and start <= q_abs <= end and start <= kv_idx <= end
                    for start, end in mm_ranges[req_idx]
                )
                new_keep = r_start <= kv_idx <= r_end
                assert new_keep == old_scan_keep, (req_idx, q_abs, kv_idx)
        token_start += query_len


def test_ranges_beyond_scheduled_chunk_are_clipped():
    """Chunked prefill must not error or produce a wrong mask.

    ``disable_chunked_mm_input`` keeps a single mm item intact but still splits
    a prompt across steps, so a request's ranges routinely sit entirely past the
    tokens scheduled so far.  Sizing by query token makes the out-of-chunk part
    a no-op rather than an out-of-bounds condition.
    """
    # Step 1 of "TTTT IIIIII": only the 4 text tokens are scheduled.
    query_ranges = _query_ranges({0: [(4, 9)]}, query_lens=[4], seq_lens=[4])
    assert query_ranges is None

    # Step 2: the image tokens are scheduled with 4 tokens of context.
    query_ranges = _query_ranges({0: [(4, 9)]}, query_lens=[6], seq_lens=[10])
    assert query_ranges is not None
    expected = torch.tensor([[4, 9]] * 6, dtype=torch.int32)
    torch.testing.assert_close(query_ranges, expected)

    # A range that starts mid-chunk and runs past its end keeps its absolute
    # bounds, so the bidirectional block still reaches the range's later keys
    # once they are cached. Query tokens here are absolute positions 2..5.
    query_ranges = _query_ranges({0: [(4, 9)]}, query_lens=[4], seq_lens=[6])
    assert query_ranges is not None
    expected = torch.tensor([[-1, -1], [-1, -1], [4, 9], [4, 9]], dtype=torch.int32)
    torch.testing.assert_close(query_ranges, expected)


def test_buffer_reuse_does_not_leak_previous_rows():
    """The staging buffer persists across steps, so every reported row must be
    rewritten. Guards against dropping the ``-1`` fill as an optimization: a
    step whose ranges shrink would otherwise expose the prior step's bounds and
    silently widen the bidirectional mask.
    """
    out = np.zeros((STAGING_CAPACITY, 2), dtype=np.int32)
    query_start_loc = torch.tensor([0, 8], dtype=torch.int32)
    seq_lens = torch.tensor([8], dtype=torch.int32)

    num_tokens = fill_mm_prefix_query_ranges(
        out, {0: [(0, 7)]}, query_start_loc, seq_lens
    )
    assert num_tokens == 8
    np.testing.assert_array_equal(out[:8], np.tile([0, 7], (8, 1)))

    # Same rows scheduled, but now only tokens 2..3 sit inside a range.
    num_tokens = fill_mm_prefix_query_ranges(
        out, {0: [(2, 3)]}, query_start_loc, seq_lens
    )
    assert num_tokens == 8
    expected = np.full((8, 2), -1, dtype=np.int32)
    expected[2:4] = (2, 3)
    np.testing.assert_array_equal(out[:8], expected)


def test_returns_none_when_no_range_covers_a_query_token():
    """No aux tensor means forward() skips the mask_mod entirely.

    Returning None rather than an all-``-1`` tensor is what keeps text-only and
    decode-only batches off the allocation path.
    """
    assert _query_ranges(None, query_lens=[4], seq_lens=[4]) is None
    assert _query_ranges({0: [], 1: []}, query_lens=[2, 2], seq_lens=[2, 2]) is None
    # Degenerate single-token ranges are skipped, matching the Triton path's
    # `start < end` validity check.
    assert _query_ranges({0: [(1, 1)]}, query_lens=[4], seq_lens=[4]) is None
    # Decode rows: the query token is generated, so it is in no range.
    assert _query_ranges({0: [(1, 3)]}, query_lens=[1], seq_lens=[9]) is None
