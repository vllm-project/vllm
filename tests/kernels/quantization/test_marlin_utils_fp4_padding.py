# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for FP4 Marlin MoE n-dim padding helpers.

Run `pytest tests/kernels/quantization/test_marlin_utils_fp4_padding.py`.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    FP4_MARLIN_TILE_N_SIZE,
    _pad_w2_for_marlin_tile,
    _pad_w13_for_marlin_tile,
)

# (per_shard_unpadded, per_shard_padded) covering per-rank intermediate sizes
# from PR #41947's validation matrix: TP=1 (1856) aligned; TP=2 (928),
# TP=4 (464), TP=8 (232) round up to the next 64-multiple per shard.
TILE_CASES = [(1856, 1856), (928, 960), (464, 512), (232, 256)]


@pytest.mark.parametrize("per_shard,expected_per_shard", TILE_CASES)
def test_pad_w13_non_gated_pads_per_shard(per_shard, expected_per_shard):
    e, half_k, scale_k = 4, 16, 4
    w13 = torch.ones(e, per_shard, half_k)
    scale = torch.ones(e, per_shard, scale_k)
    out_w13, out_scale, padded_n = _pad_w13_for_marlin_tile(
        w13, scale, unpadded_w13_size_n=per_shard, w13_num_shards=1
    )
    assert padded_n == expected_per_shard
    assert padded_n % FP4_MARLIN_TILE_N_SIZE == 0
    assert out_w13.shape == (e, expected_per_shard, half_k)
    assert out_scale.shape == (e, expected_per_shard, scale_k)
    if expected_per_shard == per_shard:
        # In-place caller relies on identity to skip nn.Parameter rewrap.
        assert out_w13 is w13 and out_scale is scale


@pytest.mark.parametrize("per_shard,expected_per_shard", TILE_CASES)
def test_pad_w13_gated_pads_each_shard_independently(per_shard, expected_per_shard):
    """Gated layout must be ``[a, pad_a, b, pad_b]`` so silu_and_mul can
    split halves at the padded boundary."""
    e, half_k, scale_k = 4, 16, 4
    unpadded_total = 2 * per_shard
    expected_total = 2 * expected_per_shard
    a_value, b_value = 3.0, 5.0
    w13 = torch.empty(e, unpadded_total, half_k)
    w13[:, :per_shard].fill_(a_value)
    w13[:, per_shard:].fill_(b_value)
    scale = torch.empty(e, unpadded_total, scale_k)
    scale[:, :per_shard].fill_(a_value)
    scale[:, per_shard:].fill_(b_value)

    out_w13, out_scale, padded_n = _pad_w13_for_marlin_tile(
        w13, scale, unpadded_w13_size_n=unpadded_total, w13_num_shards=2
    )
    assert padded_n == expected_total
    assert padded_n % FP4_MARLIN_TILE_N_SIZE == 0
    assert out_w13.shape == (e, expected_total, half_k)
    assert out_scale.shape == (e, expected_total, scale_k)

    if expected_per_shard == per_shard:
        assert out_w13 is w13 and out_scale is scale
        return

    # First shard: real values at [:per_shard], zeros at [per_shard:padded_per_shard].
    assert torch.equal(
        out_w13[:, :per_shard], torch.full_like(out_w13[:, :per_shard], a_value)
    )
    assert torch.equal(
        out_w13[:, per_shard:expected_per_shard],
        torch.zeros_like(out_w13[:, per_shard:expected_per_shard]),
    )
    # Second shard sits AFTER the first shard's pad, not concatenated to the
    # first shard's real values.
    second_shard_start = expected_per_shard
    second_shard_real_end = expected_per_shard + per_shard
    assert torch.equal(
        out_w13[:, second_shard_start:second_shard_real_end],
        torch.full_like(out_w13[:, second_shard_start:second_shard_real_end], b_value),
    )
    assert torch.equal(
        out_w13[:, second_shard_real_end:expected_total],
        torch.zeros_like(out_w13[:, second_shard_real_end:expected_total]),
    )


@pytest.mark.parametrize("unpadded,expected", TILE_CASES)
def test_pad_w2_for_marlin_tile_matches_design_table(unpadded, expected):
    e, hidden, group_size = 4, 32, 16
    w2 = torch.ones(e, hidden, unpadded // 2)
    scale = torch.ones(e, hidden, unpadded // group_size)
    out_w2, out_scale, padded_k = _pad_w2_for_marlin_tile(
        w2, scale, unpadded_w2_size_k=unpadded, group_size=group_size
    )
    assert padded_k == expected
    assert padded_k % FP4_MARLIN_TILE_N_SIZE == 0
    assert out_w2.shape == (e, hidden, expected // 2)
    assert out_scale.shape == (e, hidden, expected // group_size)
    if expected == unpadded:
        assert out_w2 is w2 and out_scale is scale
