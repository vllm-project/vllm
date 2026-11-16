# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for splitting PrefixLM ranges into causal and bidirectional rows."""

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.attention.backends.composite import requires_mm_prefix
from vllm.v1.attention.backends.prefix_lm_rows import (
    PrefixLMRowsBuilder,
    compute_prefix_lm_rows,
)
from vllm.v1.attention.backends.utils import fill_mm_prefix_query_ranges


def _qsl(query_lens):
    return np.concatenate([[0], np.cumsum(query_lens)]).astype(np.int32)


def _mask_horizons(ranges, qsl, seq_lens):
    """Reference: the key horizon of every scheduled token under the mask the
    FA4 mask_mod applies, from the query ranges the builder hands it."""
    out = np.empty((int(qsl[-1]), 2), dtype=np.int32)
    n = fill_mm_prefix_query_ranges(
        out, ranges, torch.from_numpy(qsl), torch.from_numpy(seq_lens)
    )
    horizons = []
    for req in range(len(qsl) - 1):
        query_len = int(qsl[req + 1] - qsl[req])
        for i in range(query_len):
            pos = int(seq_lens[req]) - query_len + i
            token = int(qsl[req]) + i
            end = int(out[token, 1]) if n else -1
            # A token attends causally, and to every key of its range.
            horizons.append(max(pos + 1, end + 1))
    return horizons


def _row_horizons(rows, qsl, seq_lens):
    """The key horizon of every scheduled token under the rows."""
    out = np.zeros(int(qsl[-1]), dtype=np.int64)
    for start, length, req, k, bidir in zip(
        rows.row_starts,
        rows.row_lens,
        rows.row_req,
        rows.row_seqused_k,
        rows.row_bidirectional,
    ):
        k = int(seq_lens[req]) if k < 0 else int(k)
        for i in range(int(length)):
            # A causal row is bottom-right aligned: token i of a row of length
            # n over k keys sees the keys below k - n + i + 1.
            out[start + i] = k if bidir else k - length + i + 1
    return out.tolist()


def _check(ranges, query_lens, seq_lens):
    qsl = _qsl(query_lens)
    seq_lens = np.asarray(seq_lens, dtype=np.int32)
    rows = compute_prefix_lm_rows(ranges, qsl, seq_lens)
    assert rows is not None
    # The rows tile the scheduled tokens in order.
    assert rows.row_starts[0] == 0
    assert (rows.row_starts[1:] == rows.row_starts[:-1] + rows.row_lens[:-1]).all()
    assert rows.row_lens.sum() == qsl[-1]
    # A bidirectional row always holds a query that sees a later key.
    assert (rows.row_lens[rows.row_bidirectional] >= 2).all()
    _check_rows_stay_in_their_request(rows, qsl)
    assert _row_horizons(rows, qsl, seq_lens) == _mask_horizons(ranges, qsl, seq_lens)
    return rows


def _check_rows_stay_in_their_request(rows, qsl):
    """Each row's tokens lie inside the request whose KV the row reads."""
    for start, length, req in zip(rows.row_starts, rows.row_lens, rows.row_req):
        assert qsl[req] <= start and start + length <= qsl[req + 1]


# Inclusive ranges, as CommonAttentionMetadata.mm_req_doc_ranges carries them.
@pytest.mark.parametrize(
    "ranges,context_len,seq_len",
    [
        ([(0, 3), (4, 7), (8, 9)], 0, 14),  # spans, then a causal tail
        ([(0, 3), (4, 7), (8, 9)], 4, 14),  # the first span cached
        ([(0, 3), (4, 7), (8, 9)], 6, 14),  # a cache hit ends inside a span
        ([(0, 3), (4, 7), (8, 9)], 3, 14),  # only a span's last token is left
        ([(2, 4), (7, 8)], 0, 12),  # causal runs before, between and after
        ([(0, 9)], 0, 10),  # the prompt is one span
        ([(0, 0), (1, 1), (3, 5)], 0, 8),  # one-token ranges
        ([(7, 8), (2, 4)], 0, 12),  # unsorted
        ([(5, 3), (0, 3)], 0, 8),  # a degenerate range
    ],
)
def test_rows_reproduce_the_mask(ranges, context_len, seq_len):
    _check({0: ranges}, [seq_len - context_len], [seq_len])


def test_requests_without_ranges_keep_one_row_read_on_device():
    rows = _check({1: [(0, 3)], 2: []}, [1, 6, 1, 3], [40, 6, 17, 9])
    assert rows.row_req.tolist() == [0, 1, 1, 2, 3]
    assert rows.row_seqused_k.tolist() == [-1, 4, 6, -1, -1]
    assert rows.row_bidirectional.tolist() == [False, True, False, False, False]
    assert rows.row_starts.tolist() == [0, 1, 5, 7, 8]


@pytest.mark.parametrize(
    "ranges,context_len",
    [
        ([(0, 3)], 4),  # the only range is cached
        ([(0, 3)], 3),  # only its last token is scheduled, which is causal
        ([(20, 23)], 0),  # the range is scheduled in a later step
        ([(0, 0), (5, 5)], 0),  # one-token ranges attend causally
        ([], 0),
    ],
)
def test_no_rows_when_no_query_sees_a_later_key(ranges, context_len):
    rows = compute_prefix_lm_rows(
        {0: ranges}, _qsl([10 - context_len]), np.array([10], np.int32)
    )
    assert rows is None


def test_partly_scheduled_range_keeps_the_mask_path():
    # Tokens 0 to 9 are scheduled and the range runs to 11: its tokens need
    # keys that are not computed yet, which rows cannot express (a chunked
    # multimodal item).
    for ranges in ([(8, 11)], [(0, 3), (8, 11)]):
        rows = compute_prefix_lm_rows({0: ranges}, _qsl([10]), np.array([10], np.int32))
        assert rows is None


def test_rows_only_for_steps_routed_to_the_prefix_lm_child():
    """Every step with rows needs the PrefixLM child of the composite backend,
    so the composite's causal child is never handed metadata with rows."""
    rng = random.Random(0)
    for _ in range(2000):
        num_reqs = rng.choice([1, 1, 1, 2, 3, 4])
        query_lens, seq_lens, ranges = [], [], {}
        if num_reqs == 1 and rng.random() < 0.5:
            # A lone new question about a cached state: the uncached tail of
            # the state's last range, then the question as a range.
            question, span = rng.randint(2, 40), rng.randint(2, 60)
            seq_len = rng.randint(4 * (question + span), 400)
            state = seq_len - question
            query_lens.append(rng.randint(0, span - 1) + question)
            seq_lens.append(seq_len)
            ranges[0] = [(state - span, state - 1), (state, seq_len - 1)]
        for req in range(len(query_lens), num_reqs):
            seq_len = rng.randint(1, 40)
            query_len = rng.randint(1, seq_len)
            query_lens.append(query_len)
            seq_lens.append(seq_len)
            cursor, req_ranges = 0, []
            while cursor < seq_len + 5 and rng.random() < 0.7:
                start = cursor + rng.randint(0, 4)
                end = start + rng.randint(0, 8)
                req_ranges.append((start, end))
                cursor = end + 1
            ranges[req] = req_ranges
        qsl = _qsl(query_lens)
        seq = np.asarray(seq_lens, dtype=np.int32)
        rows = compute_prefix_lm_rows(ranges, qsl, seq)
        if rows is None:
            continue
        _check_rows_stay_in_their_request(rows, qsl)
        assert _row_horizons(rows, qsl, seq) == _mask_horizons(ranges, qsl, seq)
        metadata = SimpleNamespace(
            mm_req_doc_ranges=ranges,
            max_query_len=max(query_lens),
            seq_lens_cpu_upper_bound=torch.from_numpy(seq),
            query_start_loc_cpu=torch.from_numpy(qsl),
        )
        assert requires_mm_prefix(metadata)


def test_builder_device_tensors():
    qsl = torch.from_numpy(_qsl([1, 10, 2]))
    seq_lens = torch.tensor([30, 10, 50], dtype=torch.int32)
    # The host copy is only an upper bound for the decodes; their rows read
    # the exact device lengths.
    seq_lens_cpu = torch.tensor([33, 10, 52], dtype=torch.int32)
    block_table = torch.arange(3 * 4, dtype=torch.int32).reshape(3, 4)
    builder = PrefixLMRowsBuilder(torch.device("cpu"))
    rows = builder.build(
        {1: [(0, 2), (3, 5)]}, qsl, seq_lens_cpu, seq_lens, block_table, 50
    )
    assert rows is not None
    assert rows.num_rows == 5  # decode, span, span, question run, decode
    assert rows.cu_seqlens_q.tolist() == [0, 1, 4, 7, 11, 13]
    assert rows.seqused_k.tolist() == [30, 3, 6, 10, 50]
    assert rows.row_causal.tolist() == [1, 0, 0, 1, 1]
    assert rows.row_req.tolist() == [0, 1, 1, 1, 2]
    assert rows.block_table[:, 0].tolist() == [0, 4, 4, 4, 8]
    assert (rows.max_seqlen_q, rows.max_seqlen_k) == (4, 50)
    assert builder.build({}, qsl, seq_lens_cpu, seq_lens, block_table, 50) is None
