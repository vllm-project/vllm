# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HYV4 Flash ROCm top-k rows against the AITER sparse MLA contract.

AITER reads the first `paged_kv_indptr[i+1] - paged_kv_indptr[i]` entries of
each row, so every row must hold its live indices at the front, and the
builder's per-row count must equal the number of live entries.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm() or not HAS_TRITON or not torch.cuda.is_available(),
    reason="Requires ROCm, Triton, and a GPU",
)

DEVICE = "cuda"


@pytest.fixture(autouse=True)
def _reset_random_seed() -> None:
    set_random_seed(0)


def _batch(seq_lens: list[int], query_lens: list[int]):
    seq_lens_t = torch.tensor(seq_lens, device=DEVICE, dtype=torch.int32)
    query_lens_t = torch.tensor(query_lens, device=DEVICE, dtype=torch.int32)
    query_start_loc = torch.zeros(len(seq_lens) + 1, device=DEVICE, dtype=torch.int32)
    query_start_loc[1:] = query_lens_t.cumsum(0)
    num_tokens = sum(query_lens)
    requests = torch.repeat_interleave(
        torch.arange(len(seq_lens), device=DEVICE), query_lens_t.long()
    )
    positions = (
        (seq_lens_t - query_lens_t)[requests]
        + torch.arange(num_tokens, device=DEVICE)
        - query_start_loc[requests]
    ).long()
    return seq_lens_t, query_start_loc, num_tokens, max(query_lens), positions


def _assert_front_packed(indices: torch.Tensor, counts: torch.Tensor) -> None:
    columns = torch.arange(indices.shape[1], device=DEVICE)
    front = columns[None] < counts[:, None]
    assert torch.equal(indices >= 0, front)


def test_window_rows_match_live_counts():
    from vllm.models.hy_v4_flash.amd.swa import live_counts, window_indices

    window = 512
    seq_lens, query_start_loc, num_tokens, max_query_len, positions = _batch(
        [7, 600, 513, 2000], [7, 3, 1, 6]
    )
    indices = window_indices(positions, window, window)
    counts = live_counts(
        seq_lens, query_start_loc, num_tokens, max_query_len, window, 4, 511
    )
    _assert_front_packed(indices, counts)
    for row, position in enumerate(positions.tolist()):
        expected = torch.arange(
            max(0, position - window + 1), position + 1, device=DEVICE
        )
        assert torch.equal(indices[row, : int(counts[row])].long(), expected)


def test_compressed_rows_match_live_counts():
    from vllm.models.hy_v4_flash.amd.indexpool_ops import (
        expand_groups,
        select_groups,
    )
    from vllm.models.hy_v4_flash.amd.swa import live_counts

    ratio, budget = 4, 511
    seq_lens, query_start_loc, num_tokens, max_query_len, positions = _batch(
        [5, 2047, 2048, 2049, 9000, 70000], [5, 1, 1, 1, 6, 2]
    )
    upper = (positions // ratio).int()
    logits = torch.randn(num_tokens, int(upper.max()) + 64, device=DEVICE)
    indices = expand_groups(select_groups(logits, upper, budget), positions, ratio)
    counts = live_counts(
        seq_lens, query_start_loc, num_tokens, max_query_len, 0, ratio, budget
    )
    _assert_front_packed(indices, counts)
    assert bool((indices <= positions[:, None]).all())


def test_pack_live_indices_is_stable_and_in_place():
    from vllm.models.hy_v4_flash.amd.swa import pack_live_indices_

    rows, width = 37, 2176
    values = torch.randint(0, 100000, (rows, width), device=DEVICE, dtype=torch.int32)
    holes = torch.rand(rows, width, device=DEVICE) < 0.6
    holes[0] = True
    holes[1] = False
    indices = torch.where(holes, -1, values)
    expected = [row[row >= 0] for row in indices.clone()]
    counts = pack_live_indices_(indices)
    _assert_front_packed(indices, counts)
    for row in range(rows):
        assert torch.equal(indices[row, : int(counts[row])], expected[row])
