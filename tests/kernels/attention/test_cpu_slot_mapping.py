# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

current_platform.import_kernels()

BLOCK_SIZE = 4
SENTINEL = -7


def _block_table() -> torch.Tensor:
    # 3 request rows x 4 block columns, so a request maps positions [0, 16).
    return torch.arange(12, dtype=torch.int32).view(3, 4)


def _slot_mapping(numel: int) -> torch.Tensor:
    return torch.full((numel,), SENTINEL, dtype=torch.int64)


def _compute(
    query_start_loc: list[int],
    positions: list[int],
    slot_mapping: torch.Tensor,
    block_size: int = BLOCK_SIZE,
) -> None:
    torch.ops._C.compute_slot_mapping_kernel_impl(
        torch.tensor(query_start_loc, dtype=torch.int32),
        torch.tensor(positions, dtype=torch.int64),
        _block_table(),
        slot_mapping,
        block_size,
    )


def test_compute_slot_mapping_kernel_impl_maps_positions() -> None:
    slot_mapping = _slot_mapping(3)
    _compute([0, 2, 3], [0, 5, 15], slot_mapping)
    # request 0: block ids [0, 1, 2, 3]; request 1: block ids [4, 5, 6, 7]
    assert slot_mapping.tolist() == [0 * 4 + 0, 1 * 4 + 1, 7 * 4 + 3]


def test_compute_slot_mapping_kernel_impl_maps_last_row_and_column() -> None:
    slot_mapping = _slot_mapping(1)
    _compute([0, 0, 0, 1], [15], slot_mapping)
    assert slot_mapping.tolist() == [11 * 4 + 3]


def test_compute_slot_mapping_kernel_impl_accepts_longer_buffers() -> None:
    slot_mapping = _slot_mapping(8)
    _compute([0, 2, 3], [0, 5, 15, 0, 0, 0, 0, 0], slot_mapping)
    assert slot_mapping.tolist() == [0, 5, 31] + [SENTINEL] * 5


def test_compute_slot_mapping_kernel_impl_accepts_empty_requests() -> None:
    slot_mapping = _slot_mapping(2)
    _compute([0, 0, 0, 0], [], slot_mapping)
    assert slot_mapping.tolist() == [SENTINEL] * 2


def test_compute_slot_mapping_kernel_impl_accepts_padding_requests() -> None:
    # Trailing requests without tokens never read a block table row, so they
    # may outnumber the rows.
    slot_mapping = _slot_mapping(1)
    _compute([0, 1, 1, 1, 1, 1], [5], slot_mapping)
    assert slot_mapping.tolist() == [1 * 4 + 1]


@pytest.mark.parametrize(
    ("query_start_loc", "positions", "slot_numel", "block_size", "error"),
    [
        ([0, 1], [0], 1, 0, r"block_size \(0\) must be positive"),
        ([0, 1], [0], 1, -4, r"block_size \(-4\) must be positive"),
        ([], [], 1, BLOCK_SIZE, r"query_start_loc must be a non-empty 1-D tensor"),
        (
            [0, 1, 2, 3, 4],
            [0, 0, 0, 0],
            4,
            BLOCK_SIZE,
            r"request 3 has tokens but block_table has only 3 rows",
        ),
        ([1, 2, 3], [0, 0, 0], 3, BLOCK_SIZE, r"query_start_loc\[0\] \(1\) must be 0"),
        (
            [0, 3, 2, 4],
            [0, 0, 0, 0],
            4,
            BLOCK_SIZE,
            r"query_start_loc\[2\] = 2 is out of range \[3, 4\]",
        ),
        (
            [0, 2, 5],
            [0, 0, 0, 0],
            5,
            BLOCK_SIZE,
            r"query_start_loc\[2\] = 5 is out of range \[2, 4\]",
        ),
        (
            [0, 2, 3],
            [0, 0, 0],
            2,
            BLOCK_SIZE,
            r"query_start_loc\[2\] = 3 exceeds slot_mapping.numel\(\) \(2\)",
        ),
        ([0, 2, 3], [0, -1, 0], 3, BLOCK_SIZE, r"positions\[1\] = -1 is negative"),
        (
            [0, 2, 3],
            [0, 0, 16],
            3,
            BLOCK_SIZE,
            r"positions\[2\] = 16 maps to block 4, out of range \[0, 4\)",
        ),
    ],
    ids=[
        "zero-block-size",
        "negative-block-size",
        "empty-query-start-loc",
        "more-requests-than-rows",
        "nonzero-first-offset",
        "non-monotonic-query-start-loc",
        "query-tail-past-positions",
        "slot-mapping-too-short",
        "negative-position",
        "block-column-past-table",
    ],
)
def test_compute_slot_mapping_kernel_impl_rejects_malformed_input(
    query_start_loc: list[int],
    positions: list[int],
    slot_numel: int,
    block_size: int,
    error: str,
) -> None:
    slot_mapping = _slot_mapping(slot_numel)
    with pytest.raises(RuntimeError, match=error):
        _compute(query_start_loc, positions, slot_mapping, block_size)
    # A rejected call must not write any slot.
    assert slot_mapping.tolist() == [SENTINEL] * slot_numel
