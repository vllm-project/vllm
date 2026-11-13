# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 specialized decode: the routed experts' in-kernel prefetch table (the MoE
tail's up-projection shard first, then the next attention weights up to the cap)."""

import pytest
import torch

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.experts import (
    PREFETCH_ROWS,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.prefetch import (
    PrefetchController,
)

MIB = 1024**2


def controller():
    prefetch = PrefetchController.__new__(PrefetchController)
    prefetch.device, prefetch.ahead_tables = torch.device("cpu"), {}
    return prefetch


@pytest.mark.parametrize("cap", [4, 8, 64])
def test_ahead_table_puts_the_tail_shard_first(cap):
    up = torch.empty((896, 3584), dtype=torch.bfloat16)  # TP8 up shard, 6.1 MiB
    weights = (
        torch.empty((6288, 7168), dtype=torch.bfloat16),  # in_proj, 86 MiB
        torch.empty((7168, 4096), dtype=torch.bfloat16),  # o_proj
    )
    prefetch = controller()
    prefetch._add_ahead(3, 5, weights, cap, (up,))
    table = prefetch.ahead(3, 5)
    assert table.shape == (PREFETCH_ROWS, 2)
    # The shard whole, then the next weights in order, capped at ``cap`` MiB.
    assert table[0].tolist() == [up.data_ptr(), up.numel() * 2]
    assert table[1, 0] == weights[0].data_ptr()
    assert int(table[1:, 1].sum()) == cap * MIB
    assert not table[2 + (cap > 86) :].any()


def test_ahead_table_without_tail_shard():
    weights = (torch.empty((1024, 1024), dtype=torch.bfloat16),)
    prefetch = controller()
    prefetch._add_ahead(7, 5, weights, 1)
    table = prefetch.ahead(7, 5)
    assert table[0].tolist() == [weights[0].data_ptr(), MIB]
    assert not table[1:].any()
