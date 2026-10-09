# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU coverage for mixed-causal native FlashInfer planning."""

import pytest
import torch

from vllm.v1.attention.backends.flashinfer_causal import (
    causal_group_indices,
    symmetric_window_mask,
)


@pytest.mark.parametrize(
    "flags", [[True, False, True], [False, True, False], [True] * 3, [False] * 3]
)
def test_partition_preserves_query_and_page_order(flags):
    # KV indptr intentionally starts at a nonzero offset, as sliced metadata
    # does; page gathering must index the full batch's page list.
    query_indptr = torch.tensor([0, 1, 4, 6], dtype=torch.int32)
    kv_indptr = torch.tensor([5, 7, 10, 11], dtype=torch.int32)
    groups = causal_group_indices(torch.tensor(flags), query_indptr, kv_indptr)
    all_tokens = []
    all_pages = []
    for mode, requests, tokens, pages in groups:
        assert requests.tolist() == [i for i, flag in enumerate(flags) if flag == mode]
        expected_tokens = []
        expected_pages = []
        for i in requests.tolist():
            expected_tokens.extend(range(query_indptr[i], query_indptr[i + 1]))
            expected_pages.extend(range(kv_indptr[i], kv_indptr[i + 1]))
        assert tokens.tolist() == expected_tokens
        assert pages.tolist() == expected_pages
        all_tokens.extend(tokens.tolist())
        all_pages.extend(pages.tolist())
    assert sorted(all_tokens) == list(range(6))
    assert sorted(all_pages) == list(range(5, 11))


@pytest.mark.parametrize("window_left", [0, 1, 2, 10])
def test_symmetric_window_mask_matches_absolute_positions(window_left):
    query_lens = [4, 2, 1]
    kv_lens = [4, 7, 9]
    actual = symmetric_window_mask(
        torch.tensor(query_lens),
        torch.tensor(kv_lens),
        window_left,
        torch.device("cpu"),
    )
    expected = [
        abs(query_pos - key_pos) <= window_left
        for query_len, kv_len in zip(query_lens, kv_lens)
        for query_pos in range(kv_len - query_len, kv_len)
        for key_pos in range(kv_len)
    ]
    assert actual.dtype == torch.bool
    assert actual.tolist() == expected


def test_bidirectional_window_excludes_distant_future():
    mask = symmetric_window_mask(
        torch.tensor([5]), torch.tensor([5]), 1, torch.device("cpu")
    ).reshape(5, 5)
    assert mask[0].tolist() == [True, True, False, False, False]
    assert mask[2].tolist() == [False, True, True, True, False]


@pytest.mark.parametrize("flags", [[True], [[True, False]], [1, 0]])
def test_invalid_causal_flags_rejected(flags):
    with pytest.raises(ValueError):
        causal_group_indices(
            torch.tensor(flags), torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])
        )
