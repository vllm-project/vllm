# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.backends.flash_attn import _fa2_row_view_indices


def _reference(query_lens: list[int], seq_lens: list[int], num_rows: int):
    row_req, row_seq_lens = [], []
    for req, (query_len, seq_len) in enumerate(zip(query_lens, seq_lens)):
        for j in range(query_len):
            row_req.append(req)
            # Causal attention: row j sees every key up to and including itself.
            row_seq_lens.append(seq_len - (query_len - 1 - j))
    num_padding = num_rows - len(row_req)
    return row_req, row_seq_lens + [0] * num_padding


@pytest.mark.parametrize(
    "query_lens,seq_lens,num_padding_reqs,num_padding_rows",
    [
        ([4], [100], 0, 0),
        ([4, 4, 4], [100, 4, 77], 0, 0),
        ([1, 4, 2, 3], [9, 50, 2, 1000], 0, 0),
        ([3, 0, 3], [10, 0, 20], 0, 0),
        ([4, 4], [100, 40], 2, 8),
        ([2, 3], [5, 7], 1, 3),
    ],
)
def test_fa2_row_view_indices(query_lens, seq_lens, num_padding_reqs, num_padding_rows):
    num_tokens = sum(query_lens)
    num_rows = num_tokens + num_padding_rows
    # Padded requests have an empty query and a KV length of 0.
    padded_query_lens = query_lens + [0] * num_padding_reqs
    query_start_loc = torch.tensor(
        [0] + torch.tensor(padded_query_lens).cumsum(0).tolist(), dtype=torch.int32
    )
    padded_seq_lens = torch.tensor(seq_lens + [0] * num_padding_reqs, dtype=torch.int32)
    rows = torch.arange(num_rows, dtype=torch.int32)

    row_req, row_seq_lens = _fa2_row_view_indices(
        query_start_loc, padded_seq_lens, rows
    )

    ref_row_req, ref_row_seq_lens = _reference(query_lens, seq_lens, num_rows)
    assert row_req[:num_tokens].tolist() == ref_row_req
    assert row_seq_lens.tolist() == ref_row_seq_lens
    assert row_seq_lens.dtype == torch.int32
    # Padding rows must still index a valid block table row.
    assert int(row_req.max()) < len(padded_query_lens)
