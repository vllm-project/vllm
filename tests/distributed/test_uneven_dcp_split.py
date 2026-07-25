# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for the uneven-DCP token-axis split helpers."""

import pytest
import torch

from vllm.distributed.utils import (
    cp_rank_ratio_prefix,
    cp_token_split_factor,
    set_cp_token_ratios,
    set_tp_partition_ratios,
    uneven_cp_ratios,
)
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens


@pytest.fixture(autouse=True)
def clean_ratios():
    set_tp_partition_ratios(None)
    set_cp_token_ratios(None)
    yield
    set_tp_partition_ratios(None)
    set_cp_token_ratios(None)


def test_even_split_is_default():
    assert uneven_cp_ratios(3) is None
    assert cp_token_split_factor(3) == 3
    assert cp_rank_ratio_prefix(3, 1) == (1, 1, 3)


def test_uniform_ratios_degenerate_to_even():
    set_cp_token_ratios([2, 2, 2])
    assert uneven_cp_ratios(3) is None
    assert cp_token_split_factor(3) == 3


def test_ratio_vector_size_mismatch_is_even():
    set_cp_token_ratios([2, 1, 1])
    assert uneven_cp_ratios(2) is None
    assert uneven_cp_ratios(4) is None


def test_uneven_split_parameters():
    set_cp_token_ratios([2, 1, 1])
    assert uneven_cp_ratios(3) == [2, 1, 1]
    assert cp_token_split_factor(3) == 4
    # (ratio, prefix, sum) per rank.
    assert cp_rank_ratio_prefix(3, 0) == (2, 0, 4)
    assert cp_rank_ratio_prefix(3, 1) == (1, 2, 4)
    assert cp_rank_ratio_prefix(3, 2) == (1, 3, 4)


def test_tp_ratio_fallback_for_token_vector():
    # Without an explicit token vector, the --rank-tp-ratio weights apply.
    set_tp_partition_ratios([3, 1])
    assert uneven_cp_ratios(2) == [3, 1]
    assert cp_token_split_factor(2) == 4


@pytest.mark.parametrize("interleave", [1, 16])
@pytest.mark.parametrize(
    "seq_lens",
    [[0], [1], [7], [64], [65], [255], [256], [1000], [4096], [63, 129, 4097]],
)
def test_local_seq_lens_even_partition_sums(seq_lens, interleave):
    dcp_size = 4
    lens = torch.tensor(seq_lens, dtype=torch.int32)
    local = get_dcp_local_seq_lens(
        lens, dcp_size=dcp_size, cp_kv_cache_interleave_size=interleave
    )
    assert local.shape == (len(seq_lens), dcp_size)
    # Every token is owned by exactly one rank.
    assert torch.equal(local.sum(dim=-1), lens.to(local.dtype))


@pytest.mark.parametrize("interleave", [1, 16])
@pytest.mark.parametrize(
    "seq_lens",
    [[0], [1], [7], [64], [65], [255], [256], [1000], [4096], [63, 129, 4097]],
)
def test_local_seq_lens_uneven_partition_sums(seq_lens, interleave):
    dcp_size = 3
    ratios = [2, 1, 1]
    set_cp_token_ratios(ratios)
    lens = torch.tensor(seq_lens, dtype=torch.int32)
    local = get_dcp_local_seq_lens(
        lens, dcp_size=dcp_size, cp_kv_cache_interleave_size=interleave
    )
    assert local.shape == (len(seq_lens), dcp_size)
    # Every token is owned by exactly one rank.
    assert torch.equal(local.sum(dim=-1), lens.to(local.dtype))
    # Per-rank result matches the all-ranks column.
    for rank in range(dcp_size):
        per_rank = get_dcp_local_seq_lens(
            lens,
            dcp_size=dcp_size,
            dcp_rank=rank,
            cp_kv_cache_interleave_size=interleave,
        )
        assert torch.equal(per_rank, local[:, rank])


def test_local_seq_lens_uneven_proportionality():
    # For sequence lengths that are a multiple of a full round
    # (sum(ratios) * interleave), the split is exactly proportional.
    ratios = [2, 1, 1]
    set_cp_token_ratios(ratios)
    interleave = 16
    round_size = sum(ratios) * interleave  # 64
    lens = torch.tensor([10 * round_size], dtype=torch.int32)
    local = get_dcp_local_seq_lens(
        lens, dcp_size=3, cp_kv_cache_interleave_size=interleave
    )
    assert local.tolist() == [[10 * ratios[r] * interleave for r in range(3)]]
