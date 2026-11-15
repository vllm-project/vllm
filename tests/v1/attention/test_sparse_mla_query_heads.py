# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The trtllm-gen sparse MLA kernels only exist for a few query head counts."""

import pytest

from vllm.v1.attention.backends.mla import flashinfer_mla_sparse as fims


@pytest.mark.parametrize(
    ("num_heads", "expected"),
    [
        (1, 8),
        (3, 8),  # 96 heads over TP32
        (8, 8),
        (12, 16),  # 96 heads over TP8
        (16, 16),
        (24, 32),  # 96 heads over TP4
        (32, 32),
        (48, 64),  # 96 heads over TP2
        (64, 64),
        (96, 128),
        (128, 128),
        (256, 256),  # above the widest instantiation: unchanged
    ],
)
def test_query_heads_round_up_to_an_instantiated_width(num_heads, expected):
    assert fims.trtllm_gen_query_heads(num_heads) == expected


def test_query_heads_never_exceed_the_widest_instantiation():
    """Padding must not invent a width the kernels do not have."""
    for num_heads in range(1, 129):
        padded = fims.trtllm_gen_query_heads(num_heads)
        assert padded >= num_heads
        assert padded in (8, 16, 32, 64, 128)
