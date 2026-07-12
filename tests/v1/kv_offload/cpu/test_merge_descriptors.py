# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for descriptor run-coalescing in the CPU offloading worker."""

import numpy as np
import pytest

from vllm.v1.kv_offload.cpu.gpu_worker import merge_contiguous_descriptors

PAGE = 8192


def _arrays(src, dst, sizes, dtype=np.int64):
    return (
        np.array(src, dtype=dtype),
        np.array(dst, dtype=dtype),
        np.array(sizes, dtype=dtype),
    )


def test_fully_contiguous_run_collapses_to_one():
    n = 16
    src, dst, sizes = _arrays(
        [1000 + i * PAGE for i in range(n)],
        [5000 + i * PAGE for i in range(n)],
        [PAGE] * n,
    )
    assert merge_contiguous_descriptors(src, dst, sizes) == 1
    assert src[0] == 1000 and dst[0] == 5000 and sizes[0] == n * PAGE


def test_per_layer_runs_merge_per_layer():
    # Two "layers" at distant bases; blocks contiguous within each layer.
    src, dst, sizes = _arrays(
        [10**6 + i * PAGE for i in range(4)] + [10**9 + i * PAGE for i in range(4)],
        [2 * 10**6 + i * PAGE for i in range(4)]
        + [2 * 10**9 + i * PAGE for i in range(4)],
        [PAGE] * 8,
    )
    assert merge_contiguous_descriptors(src, dst, sizes) == 2
    assert list(sizes[:2]) == [4 * PAGE, 4 * PAGE]
    assert list(src[:2]) == [10**6, 10**9]
    assert list(dst[:2]) == [2 * 10**6, 2 * 10**9]


def test_requires_contiguity_on_both_sides():
    # src contiguous, dst scattered -> no merge.
    src, dst, sizes = _arrays(
        [i * PAGE for i in range(4)],
        [0, 10 * PAGE, 20 * PAGE, 30 * PAGE],
        [PAGE] * 4,
    )
    assert merge_contiguous_descriptors(src, dst, sizes) == 4
    assert list(dst) == [0, 10 * PAGE, 20 * PAGE, 30 * PAGE]


def test_scattered_descriptors_unchanged():
    src, dst, sizes = _arrays(
        [i * 3 * PAGE for i in range(5)],
        [i * 7 * PAGE for i in range(5)],
        [PAGE] * 5,
    )
    assert merge_contiguous_descriptors(src, dst, sizes) == 5


def test_mixed_runs_and_variable_sizes():
    # [a: 2 pages contiguous] [gap] [b: 3 pages contiguous with mixed sizes]
    src, dst, sizes = _arrays(
        [0, PAGE, 100 * PAGE, 101 * PAGE, 101 * PAGE + PAGE // 2],
        [0, PAGE, 200 * PAGE, 201 * PAGE, 201 * PAGE + PAGE // 2],
        [PAGE, PAGE, PAGE, PAGE // 2, PAGE // 2],
    )
    assert merge_contiguous_descriptors(src, dst, sizes) == 2
    assert list(sizes[:2]) == [2 * PAGE, 2 * PAGE]
    assert src[1] == 100 * PAGE and dst[1] == 200 * PAGE


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_single_descriptor_and_dtypes(dtype):
    src, dst, sizes = _arrays([123], [456], [PAGE], dtype=dtype)
    assert merge_contiguous_descriptors(src, dst, sizes) == 1
    src, dst, sizes = _arrays(
        [0, PAGE, 2 * PAGE], [0, PAGE, 2 * PAGE], [PAGE] * 3, dtype=dtype
    )
    assert merge_contiguous_descriptors(src, dst, sizes) == 1
    assert sizes[0] == 3 * PAGE
