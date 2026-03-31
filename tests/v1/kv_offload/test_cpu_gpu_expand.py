# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np

from vllm.v1.kv_offload.worker.cpu_gpu import expand_block_ids


def test_expand_block_ids_full_blocks():
    output = np.empty(12, dtype=np.int64)
    expand_block_ids(
        np.array([0, 1, 3], dtype=np.int64),
        block_size_factor=4,
        output=output,
    )

    np.testing.assert_array_equal(
        output,
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15], dtype=np.int64),
    )


def test_expand_block_ids_partial_ranges():
    output = np.empty(6, dtype=np.int64)
    expand_block_ids(
        np.array([0, 1], dtype=np.int64),
        block_size_factor=8,
        output=output,
        block_offsets=np.array([2, 0], dtype=np.int64),
        block_counts=np.array([3, 3], dtype=np.int64),
    )

    np.testing.assert_array_equal(
        output,
        np.array([2, 3, 4, 8, 9, 10], dtype=np.int64),
    )


def test_expand_block_ids_partial_ranges_can_repeat_same_block():
    output = np.empty(4, dtype=np.int64)
    expand_block_ids(
        np.array([0, 0], dtype=np.int64),
        block_size_factor=8,
        output=output,
        block_offsets=np.array([0, 4], dtype=np.int64),
        block_counts=np.array([2, 2], dtype=np.int64),
    )

    np.testing.assert_array_equal(
        output,
        np.array([0, 1, 4, 5], dtype=np.int64),
    )
