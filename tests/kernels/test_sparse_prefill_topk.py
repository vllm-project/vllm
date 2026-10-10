# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract for deterministic DSA prefill top-k.

Guards equal-score membership: the same valid range must keep the same
compressed-context indices, including when a row is batched with others.
The CUDA top_k_per_row_prefill kernel is not exercised here.
"""

import pytest
import torch

from vllm.model_executor.layers.sparse_prefill_topk import (
    stable_prefill_topk_from_valid_range,
)


def _run(
    logits: torch.Tensor,
    ks: list[int],
    ke: list[int],
    topk_tokens: int,
) -> list[list[int]]:
    idx = torch.full(
        (logits.shape[0], topk_tokens),
        -1,
        dtype=torch.int32,
    )
    stable_prefill_topk_from_valid_range(
        logits,
        torch.tensor(ks, dtype=torch.int32),
        torch.tensor(ke, dtype=torch.int32),
        idx,
        topk_tokens,
    )
    return idx.tolist()


def test_equal_finite_scores_keep_smaller_index() -> None:
    logits = torch.tensor([[1.0, 5.0, 5.0, 0.0, 9.0]])
    assert _run(logits, [0], [5], 3) == [[4, 1, 2]]


def test_valid_range_excludes_columns_outside_ks_ke() -> None:
    logits = torch.tensor([[9.0, 1.0, 3.0, 3.0, 0.0]])
    assert _run(logits, [1], [5], 2) == [[2, 3]]


def test_empty_range_writes_padding() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    assert _run(logits, [2], [2], 2) == [[-1, -1]]


def test_k_wider_than_valid_range_pads() -> None:
    logits = torch.tensor([[4.0, 1.0, 2.0]])
    assert _run(logits, [0], [2], 4) == [[0, 1, -1, -1]]


def test_valid_neg_inf_is_not_replaced_by_invalid_column() -> None:
    logits = torch.tensor([[100.0, float("-inf"), 1.0, 50.0]])
    assert _run(logits, [1], [3], 2) == [[2, 1]]


def test_two_row_ties_keep_lower_index_per_row() -> None:
    logits = torch.tensor([[1.0, 1.0, 0.0], [0.0, 5.0, 5.0]])
    assert _run(logits, [0, 1], [3, 3], 2) == [[0, 1], [1, 2]]


def test_boundary_tie_prefers_lower_compressed_index() -> None:
    """Equal finite scores at 244 and 640; k=1 must keep 244."""
    cols = 641
    logits = torch.full((1, cols), -2.0)
    logits[0, 244] = 1.0
    logits[0, 640] = 1.0
    assert _run(logits, [0], [cols], 1) == [[244]]
    assert _run(logits, [0], [cols], 2) == [[244, 640]]


def test_batched_rows_match_single_row_under_ties() -> None:
    """Same logits must not change selection when the row batch changes."""
    cols = 32
    k = 4
    row0 = torch.zeros(cols)
    # Four-way tie at the cut, plus two unique higher scores.
    row0[3] = 9.0
    row0[11] = 8.0
    row0[5] = 1.0
    row0[7] = 1.0
    row0[13] = 1.0
    row0[19] = 1.0
    row1 = torch.arange(cols, dtype=torch.float32)
    solo = _run(row0.unsqueeze(0), [0], [cols], k)
    batched = _run(torch.stack([row0, row1]), [0, 0], [cols, cols], k)
    assert solo[0] == batched[0]
    assert solo[0] == [3, 11, 5, 7]


def test_selection_is_repeatable() -> None:
    logits = torch.tensor([[1.0, 5.0, 5.0, 0.0, 9.0], [0.0, 5.0, 5.0, 5.0, 1.0]])
    first = _run(logits, [0, 1], [5, 5], 3)
    for _ in range(19):
        assert _run(logits, [0, 1], [5, 5], 3) == first
    assert first == [[4, 1, 2], [1, 2, 3]]


def test_cpu_rejects_inverted_bounds() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    idx = torch.full((1, 2), -1, dtype=torch.int32)
    with pytest.raises(ValueError, match="ks/ke"):
        stable_prefill_topk_from_valid_range(
            logits,
            torch.tensor([2], dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
            idx,
            2,
        )
