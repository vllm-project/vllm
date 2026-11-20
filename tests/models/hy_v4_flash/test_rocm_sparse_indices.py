# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HYV4 Flash ROCm row contract: AITER sparse MLA reads the first
`paged_kv_indptr` delta entries of each row and maps -1 to slot 0, so builder
counts must equal the indexers' live entries and rows must be front-packed.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("HYV4 Flash ROCm sparse indices", allow_module_level=True)

from vllm.models.hy_v4_flash.amd import swa
from vllm.models.hy_v4_flash.amd.indexpool_ops import expand_groups, select_groups
from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
    ROCMAiterMLASparseImpl,
)

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


def _stable_pack_reference(rows: torch.Tensor) -> torch.Tensor:
    packed = torch.full_like(rows, -1)
    for i, row in enumerate(rows):
        live = row[row >= 0]
        packed[i, : live.numel()] = live
    return packed


@pytest.mark.parametrize("width", [2176, 300])
def test_pack_live_indices_is_stable_compaction(width):
    torch.manual_seed(0)
    storage = torch.randint(0, 1 << 20, (9, width + 32), device=DEVICE).int()
    storage.masked_fill_(torch.rand_like(storage, dtype=torch.float) < 0.6, -1)
    storage[3] = -1
    storage[5] = torch.arange(width + 32, device=DEVICE)
    original = storage.clone()
    rows = storage[:, :width]

    counts = swa.pack_live_indices_(rows)

    expected = _stable_pack_reference(original[:, :width])
    torch.testing.assert_close(rows, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        counts, (original[:, :width] >= 0).sum(-1).int(), rtol=0, atol=0
    )
    torch.testing.assert_close(storage[:, width:], original[:, width:], rtol=0, atol=0)


def test_packed_impl_rebuilds_indptr_from_live_counts(monkeypatch):
    monkeypatch.setattr(
        ROCMAiterMLASparseImpl, "forward_mqa", lambda *args: "forwarded"
    )
    torch.manual_seed(0)
    buffer = torch.randint(0, 4096, (8, 256), device=DEVICE).int()
    buffer.masked_fill_(torch.rand_like(buffer, dtype=torch.float) < 0.5, -1)
    buffer[6:] = 7
    original = buffer.clone()
    num_tokens = 5
    indptr = torch.full((10,), -3, dtype=torch.int32, device=DEVICE)
    indptr[0] = 0
    impl = object.__new__(swa.FlashPackedSparseImpl)
    impl.topk_indices_buffer = buffer
    q = torch.empty(num_tokens, 16, 576, device=DEVICE)

    result = impl.forward_mqa(q, None, SimpleNamespace(paged_kv_indptr=indptr), None)

    assert result == "forwarded"
    counts = (original[:num_tokens] >= 0).sum(-1).int()
    expected = torch.zeros(10, dtype=torch.int32, device=DEVICE)
    expected[1 : num_tokens + 1] = torch.cumsum(counts, 0)
    expected[num_tokens + 1 :] = expected[num_tokens]
    torch.testing.assert_close(indptr, expected, rtol=0, atol=0)
    _assert_front_packed(buffer[:num_tokens], counts)
    torch.testing.assert_close(buffer[num_tokens:], original[num_tokens:])
