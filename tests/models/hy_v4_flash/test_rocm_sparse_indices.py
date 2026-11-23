# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HYV4 Flash ROCm row contract: AITER sparse MLA reads the first
`paged_kv_indptr` delta entries of each row and maps -1 to slot 0, so builder
counts must equal the indexers' live entries and rows must be front-packed.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("HYV4 Flash ROCm sparse indices", allow_module_level=True)

from vllm.models.hy_v4_flash.amd import swa
from vllm.models.hy_v4_flash.amd.indexpool_ops import expand_groups, select_groups

DEVICE = "cuda"
# (seq_len, query_len) per request: decode, short and long extend, an empty
# padded request, a full prefill, and a decode past the compressed budget.
REQUESTS = [(1, 1), (300, 3), (0, 0), (64, 64), (2000, 1), (150, 64)]


def _logical_positions() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_lens = torch.tensor([s for s, _ in REQUESTS], dtype=torch.int32)
    query_lens = torch.tensor([q for _, q in REQUESTS], dtype=torch.int32)
    query_start_loc = torch.zeros(len(REQUESTS) + 1, dtype=torch.int32)
    torch.cumsum(query_lens, 0, out=query_start_loc[1:])
    positions = torch.cat(
        [torch.arange(s - q, s, dtype=torch.int64) for s, q in REQUESTS]
    )
    return (
        seq_lens.to(DEVICE),
        query_start_loc.to(DEVICE),
        positions.to(DEVICE),
    )


def _assert_front_packed(rows: torch.Tensor, counts: torch.Tensor) -> None:
    columns = torch.arange(rows.shape[1], device=rows.device)
    head = columns < counts[:, None].long()
    assert torch.all((rows >= 0) == head)


@pytest.mark.parametrize("window", [0, 128], ids=["compressed", "swa"])
def test_live_counts_match_indexer_rows(window):
    ratio, budget, width = 16, 7, 128
    seq_lens, query_start_loc, positions = _logical_positions()
    num_tokens = positions.numel()
    if window:
        rows = swa.window_indices(positions, window, width)
    else:
        upper = (positions // ratio).int()
        torch.manual_seed(0)
        logits = torch.randn(num_tokens, int(upper.max()) + 1, device=DEVICE)
        rows = expand_groups(select_groups(logits, upper, budget), positions, ratio)

    counts = swa.live_counts(
        seq_lens,
        query_start_loc,
        num_tokens,
        max(q for _, q in REQUESTS),
        window,
        ratio,
        budget,
    )

    torch.testing.assert_close(counts, (rows >= 0).sum(-1).int(), rtol=0, atol=0)
    _assert_front_packed(rows, counts)
