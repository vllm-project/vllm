# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The packed hybrid block size must land on a size every backend executes."""

import types

import pytest

from vllm.platforms.interface import _largest_common_kernel_block_size
from vllm.v1.attention.backend import MultipleOf


def _backend(sizes):
    return types.SimpleNamespace(get_supported_kernel_block_sizes=lambda: sizes)


@pytest.mark.parametrize(
    ("backend_sizes", "desired", "expected"),
    [
        # SM12x hybrid: the packed row is sized for the mamba state (256) but the
        # indexer only instantiates 64-token blocks, so the row is clamped.
        ([[MultipleOf(64)], [64]], 256, 64),
        # Only unbounded constraints -> the packed size is kept.
        ([[MultipleOf(64)]], 256, 256),
        ([[MultipleOf(16)]], 544, 544),
        # Intersection of two concrete sets.
        ([[64, 128], [64]], 256, 64),
        ([[64, 128]], 256, 128),
        # A concrete value that must also satisfy a MultipleOf entry.
        ([[64, 128], [MultipleOf(64)]], 128, 128),
        # A smaller concrete value is the only usable size.
        ([[32]], 256, 32),
        # No constraints at all.
        ([[]], 256, 256),
    ],
)
def test_largest_common_kernel_block_size(backend_sizes, desired, expected):
    backends = [_backend(sizes) for sizes in backend_sizes]
    assert _largest_common_kernel_block_size(backends, desired) == expected
