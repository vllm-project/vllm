# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank-local candidate rows for replicated-cache PCP prefill."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.v1.attention.backend import CommonAttentionMetadata


def prepare_pcp_candidate_rows(
    common: "CommonAttentionMetadata",
    compress_ratio: int,
    row_lens: torch.Tensor,
    row_indices: torch.Tensor,
) -> bool:
    """Map absolute query positions to local segment tables.

    Return False without modifying outputs if positions are unavailable.
    Slot mappings cannot identify padding: prefix replay also masks valid KV
    writes. Query boundaries identify padding and preserve duplicate requests.
    """
    positions = common.positions
    rows = row_lens.numel()
    requests = common.num_reqs
    if (
        positions is None
        or positions.ndim != 1
        or positions.numel() < rows
        or positions.dtype not in (torch.int32, torch.int64)
        or positions.device != row_lens.device
        or common.query_start_loc.device != row_lens.device
        or common.query_start_loc.numel() < requests + 1
        or common.block_table_tensor.shape[0] < requests
        or requests == 0
    ):
        return False
    assert compress_ratio in (1, 2)
    assert row_lens.dtype == row_indices.dtype == torch.int32
    assert row_lens.shape == row_indices.shape
    token = torch.arange(rows, device=row_lens.device, dtype=torch.int32)
    segment = torch.searchsorted(
        common.query_start_loc[1 : requests + 1],
        token,
        right=True,
        out_int32=True,
    )
    valid = (
        (token < common.query_start_loc[requests])
        & (segment < requests)
        & (positions[:rows] >= 0)
    )
    lengths = torch.div(
        positions[:rows].to(torch.int64) + 1,
        compress_ratio,
        rounding_mode="floor",
    ).to(torch.int32)
    row_lens.copy_(torch.where(valid, lengths, 0))
    row_indices.copy_(torch.where(valid, segment, 0))
    return True
