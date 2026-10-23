# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The decode segment's guards for block-table entries its padding rows index."""

import pytest
import torch

from vllm.models.glm53.nvidia.decode import kernel_block_table, null_padding_block

BLOCK = 64
ROWS = 4
KEY_TILE = 128


def _kernel_entries(position: int, rows: int) -> list[int]:
    """Entries the kernels index for a step at ``position``: each row's block and the
    next.
    """
    last = (position + rows - 1) // BLOCK
    return list(range(position // BLOCK, last + 2))


@pytest.mark.parametrize("offset", [61, 62, 63])
def test_padding_rows_past_the_block_go_to_the_null_block(offset):
    table = torch.arange(1, 9, dtype=torch.int32) * 100
    position = 3 * BLOCK + offset
    null_padding_block(table, position, ROWS, BLOCK)
    assert table[4] == 0
    expected = torch.arange(1, 9, dtype=torch.int32) * 100
    expected[4] = 0
    torch.testing.assert_close(table, expected)


@pytest.mark.parametrize("offset", [0, 1, 60])
def test_padding_rows_inside_the_block_keep_the_table(offset):
    table = torch.arange(1, 9, dtype=torch.int32) * 100
    null_padding_block(table, 3 * BLOCK + offset, ROWS, BLOCK)
    torch.testing.assert_close(table, torch.arange(1, 9, dtype=torch.int32) * 100)


def test_padding_rows_past_the_row_leave_it_alone():
    table = torch.arange(1, 5, dtype=torch.int32)
    null_padding_block(table, 4 * BLOCK - 2, ROWS, BLOCK)
    torch.testing.assert_close(table, torch.arange(1, 5, dtype=torch.int32))


def _vllm_width(max_model_len: int) -> int:
    """Return vLLM's block-table width: cdiv(max_model_len, 64) rounded to 128
    tokens.
    """
    blocks = -(-max_model_len // BLOCK)
    return -(-blocks // 2) * 2


@pytest.mark.parametrize(
    "max_model_len", [202752, 128, 254, 255, 256, 1000, 200000, 524288 + 1024, 688128]
)
@pytest.mark.parametrize("rows", [1, ROWS])
def test_kernel_table_covers_every_entry_up_to_max_model_len(max_model_len, rows):
    width = _vllm_width(max_model_len)
    vllm_row = torch.randint(1, 1 << 20, (width,), dtype=torch.int32)
    padded = kernel_block_table(vllm_row, max_model_len, rows, BLOCK, KEY_TILE)
    bound = vllm_row if padded is None else padded
    assert bound.numel() * BLOCK % KEY_TILE == 0
    if padded is not None:
        padded[:width].copy_(vllm_row)
        assert bool((padded[width:] == 0).all())
    for position in range(max(0, max_model_len - 2 * BLOCK), max_model_len):
        assert max(_kernel_entries(position, rows)) < bound.numel()


def test_kernel_table_pads_at_capacity():
    assert (
        kernel_block_table(
            torch.zeros(_vllm_width(202752), dtype=torch.int32),
            202752,
            ROWS,
            BLOCK,
            KEY_TILE,
        )
        is not None
    )
    assert (
        kernel_block_table(
            torch.zeros(_vllm_width(202752 - 200), dtype=torch.int32),
            202752 - 200,
            ROWS,
            BLOCK,
            KEY_TILE,
        )
        is None
    )
