# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm.config import CUDAGraphMode
from vllm.models.kimi_k3.amd.specialized_decode.runtime import (
    DecodeConfig,
    served_widths,
)

pytestmark = pytest.mark.cpu_test


def _config(mode, sizes):
    cc = SimpleNamespace(
        cudagraph_mode=mode, decode_cudagraph_capture_request_sizes=sizes
    )
    return SimpleNamespace(compilation_config=cc)


@pytest.mark.parametrize(
    ("mode", "sizes", "expected"),
    [
        (CUDAGraphMode.FULL_AND_PIECEWISE, [1, 2, 4, 8, 16, 24, 32], (1, 2, 4, 8, 16)),
        (
            CUDAGraphMode.FULL_AND_PIECEWISE,
            list(range(1, 17)) + [24],
            tuple(range(1, 17)),
        ),
        (CUDAGraphMode.FULL_DECODE_ONLY, [4, 8, 64], (4, 8)),
        (CUDAGraphMode.FULL_AND_PIECEWISE, [24, 32], ()),
        (CUDAGraphMode.PIECEWISE, [1, 2, 4], None),
        (CUDAGraphMode.NONE, [1, 2, 4], None),
        (None, [1, 2, 4], None),
        (CUDAGraphMode.FULL_AND_PIECEWISE, None, None),
    ],
)
def test_served_widths(mode, sizes, expected):
    assert served_widths(_config(mode, sizes)) == expected


def test_widths_filtered_by_served():
    assert DecodeConfig(q=1, max_num_seqs=128).widths == tuple(range(1, 17))
    stock = DecodeConfig(q=1, max_num_seqs=128, served=(1, 2, 4, 8, 16))
    assert stock.widths == (1, 2, 4, 8, 16)
    # a small max_num_seqs caps the widths before the filter
    assert DecodeConfig(q=1, max_num_seqs=4, served=(1, 2, 4, 8, 16)).widths == (
        1,
        2,
        4,
    )
