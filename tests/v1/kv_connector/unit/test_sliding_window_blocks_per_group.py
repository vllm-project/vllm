# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Blocks a KV connector transfers per sliding-window group."""

from dataclasses import replace
from types import SimpleNamespace

import torch

from vllm.distributed.kv_transfer.kv_connector.utils import (
    get_sliding_window_blocks_per_group,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

_FULL = FullAttentionSpec(
    block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float32
)
_WINDOW = SlidingWindowSpec(
    block_size=16,
    num_kv_heads=1,
    head_size=8,
    dtype=torch.float32,
    sliding_window=64,
)


def _config(*specs):
    groups = [SimpleNamespace(kv_cache_spec=spec) for spec in specs]
    return SimpleNamespace(transfer_groups=groups)


def test_full_attention_group_has_no_window_blocks():
    assert get_sliding_window_blocks_per_group(_config(_FULL), 16) == [0]


def test_window_group_counts_window_blocks_plus_boundary_overlap():
    # cdiv(64, 16) + 1
    assert get_sliding_window_blocks_per_group(_config(_FULL, _WINDOW), 16) == [0, 5]


def test_window_group_includes_the_retained_tokens_below_the_window():
    # The widened EAGLE window retains 17 tokens below it: cdiv(64 + 17, 16) + 1.
    retained = replace(_WINDOW, extra_retained_tokens=17)
    assert get_sliding_window_blocks_per_group(_config(retained), 16) == [7]
