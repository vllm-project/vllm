# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    _get_fp8_marlin_padded_sizes,
)


def _is_valid_fp8_marlin_thread_tile(size_n: int, size_k: int) -> bool:
    return (size_n % 128 == 0 and size_k % 64 == 0) or (
        size_n % 64 == 0 and size_k % 128 == 0
    )


@pytest.mark.parametrize(
    ("size_n", "size_k", "group_size", "expected"),
    [
        (4640, 4096, -1, (4672, 4096)),
        (200, 129, -1, (256, 192)),
        (129, 65, -1, (192, 128)),
        (200, 257, 128, (256, 384)),
    ],
)
def test_fp8_marlin_padding_maps_invalid_shapes_to_valid_thread_tiles(
    size_n: int,
    size_k: int,
    group_size: int,
    expected: tuple[int, int],
) -> None:
    assert not _is_valid_fp8_marlin_thread_tile(size_n, size_k)

    padded_n, padded_k = _get_fp8_marlin_padded_sizes(size_n, size_k, group_size)

    assert (padded_n, padded_k) == expected
    assert _is_valid_fp8_marlin_thread_tile(padded_n, padded_k)
    assert padded_n >= size_n
    assert padded_k >= size_k
    if group_size > 0:
        assert padded_k % group_size == 0
