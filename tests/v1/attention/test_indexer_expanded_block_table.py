# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.block_table import get_block_table_width


@pytest.mark.parametrize(
    ("max_num_blocks", "block_size", "kernel_block_size", "expected_width"),
    [(1875, 64, 64, 1876), (235, 256, 64, 940)],
)
def test_get_block_table_width(
    max_num_blocks: int,
    block_size: int,
    kernel_block_size: int,
    expected_width: int,
):
    assert (
        get_block_table_width(max_num_blocks, block_size, kernel_block_size)
        == expected_width
    )
