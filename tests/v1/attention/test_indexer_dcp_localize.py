# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens


def _local_count(length: int, rank: int, world: int, interleave: int) -> int:
    return sum(1 for pos in range(length) if (pos // interleave) % world == rank)


@pytest.mark.parametrize("world", [1, 2, 4])
@pytest.mark.parametrize("interleave", [1, 2, 4])
def test_get_dcp_local_seq_lens_matches_naive(world: int, interleave: int):
    seq_lens = torch.arange(0, 33, dtype=torch.int32)

    for rank in range(world):
        actual = get_dcp_local_seq_lens(seq_lens, world, rank, interleave)
        expected = torch.tensor(
            [
                _local_count(int(seq_len), rank, world, interleave)
                for seq_len in seq_lens
            ],
            dtype=torch.int32,
        )
        torch.testing.assert_close(actual, expected)


def test_get_dcp_local_seq_lens_can_localize_per_token_bounds():
    seq_lens = torch.tensor([0, 1, 2, 3, 4, 7, 8, 17], dtype=torch.int32)
    world = 4
    interleave = 2

    for rank in range(world):
        actual = get_dcp_local_seq_lens(seq_lens, world, rank, interleave)
        expected = torch.tensor(
            [
                _local_count(int(seq_len), rank, world, interleave)
                for seq_len in seq_lens
            ],
            dtype=torch.int32,
        )
        torch.testing.assert_close(actual, expected)


def test_get_dcp_local_seq_lens_must_run_after_decode_expansion():
    world = 2
    rank = 1
    interleave = 1
    expanded_bounds = torch.tensor([8, 9, 10], dtype=torch.int32)

    localized_after_expansion = get_dcp_local_seq_lens(
        expanded_bounds, world, rank, interleave
    )
    localized_request_len_minus_offsets = get_dcp_local_seq_lens(
        torch.tensor([10], dtype=torch.int32), world, rank
    ) - torch.tensor([2, 1, 0], dtype=torch.int32)

    assert not torch.equal(
        localized_after_expansion, localized_request_len_minus_offsets
    )
    torch.testing.assert_close(
        localized_after_expansion, torch.tensor([4, 4, 5], dtype=torch.int32)
    )
