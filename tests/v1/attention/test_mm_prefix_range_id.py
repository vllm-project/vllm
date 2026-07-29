# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.backends.utils import (
    compute_mm_prefix_range_id_tensor,
)


def test_compute_mm_prefix_range_id_tensor_uses_per_request_ids():
    range_ids = compute_mm_prefix_range_id_tensor(
        {
            0: [(2, 4), (7, 8)],
            1: [(0, 0), (1, 2), (8, 9)],
            2: [],
        },
        num_seqs=3,
        max_seq_len=10,
        device=torch.device("cpu"),
    )

    assert range_ids is not None
    expected = torch.tensor(
        [
            [-1, -1, 0, 0, 0, -1, -1, 1, 1, -1],
            [-1, 0, 0, -1, -1, -1, -1, -1, 1, 1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        ],
        dtype=torch.int32,
    )
    torch.testing.assert_close(range_ids, expected)


def test_compute_mm_prefix_range_id_tensor_returns_none_without_valid_ranges():
    assert (
        compute_mm_prefix_range_id_tensor(
            {0: [], 1: [(3, 3)]},
            num_seqs=2,
            max_seq_len=8,
            device=torch.device("cpu"),
        )
        is None
    )

    assert (
        compute_mm_prefix_range_id_tensor(
            None,
            num_seqs=1,
            max_seq_len=8,
            device=torch.device("cpu"),
        )
        is None
    )

    assert (
        compute_mm_prefix_range_id_tensor(
            {0: [(1, 2)]},
            num_seqs=1,
            max_seq_len=0,
            device=torch.device("cpu"),
        )
        is None
    )


def test_compute_mm_prefix_range_id_tensor_rejects_out_of_bounds():
    with pytest.raises(ValueError, match="Invalid mm_prefix range"):
        compute_mm_prefix_range_id_tensor(
            {0: [(-1, 1)]},
            num_seqs=1,
            max_seq_len=8,
            device=torch.device("cpu"),
        )

    with pytest.raises(ValueError, match="Invalid mm_prefix range"):
        compute_mm_prefix_range_id_tensor(
            {0: [(4, 8)]},
            num_seqs=1,
            max_seq_len=8,
            device=torch.device("cpu"),
        )


def test_compute_mm_prefix_range_id_tensor_rejects_overlapping_ranges():
    with pytest.raises(ValueError, match="Overlapping mm_prefix ranges"):
        compute_mm_prefix_range_id_tensor(
            {0: [(1, 4), (3, 6)]},
            num_seqs=1,
            max_seq_len=8,
            device=torch.device("cpu"),
        )


def test_compute_mm_prefix_range_id_tensor_matches_range_scan_semantics():
    mm_ranges = {
        0: [(1, 3), (5, 7), (9, 9)],
        1: [(0, 2), (4, 6)],
    }
    max_seq_len = 8
    range_ids = compute_mm_prefix_range_id_tensor(
        mm_ranges,
        num_seqs=2,
        max_seq_len=max_seq_len,
        device=torch.device("cpu"),
    )

    assert range_ids is not None
    for req_idx, req_ranges in mm_ranges.items():
        for q_idx in range(max_seq_len):
            for kv_idx in range(max_seq_len):
                old_scan_keep = any(
                    start < end and start <= q_idx <= end and start <= kv_idx <= end
                    for start, end in req_ranges
                )
                q_range_id = range_ids[req_idx, q_idx].item()
                kv_range_id = range_ids[req_idx, kv_idx].item()
                new_range_id_keep = q_range_id >= 0 and q_range_id == kv_range_id
                assert new_range_id_keep == old_scan_keep
