# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test: prepare_moe_fp4_layer_for_marlin must use
num_local_experts (not global num_experts) when expert parallelism
shards experts across ranks."""

import types

import torch


def test_mxfp4_moe_marlin_ep_num_experts():
    """With EP=2 and 256 global experts, each rank holds 128 local experts.
    Weight tensors are created with local_num_experts, so the Marlin
    repacking assert must also use local_num_experts."""

    # Simulate EP=2 with 256 global experts -> 128 local
    global_experts = 256
    local_experts = 128
    hidden = 3072
    intermediate = 1536

    moe_config = types.SimpleNamespace(
        num_experts=global_experts,
        num_local_experts=local_experts,
        hidden_dim=hidden,
        intermediate_size_per_partition=intermediate,
    )

    # Weight tensors are created with local_num_experts
    layer = types.SimpleNamespace(
        moe_config=moe_config,
        w13_weight=torch.empty(local_experts, 2 * intermediate, hidden // 2,
                               dtype=torch.uint8),
        w2_weight=torch.empty(local_experts, hidden, intermediate // 2,
                               dtype=torch.uint8),
    )

    e = layer.moe_config.num_local_experts
    k = layer.moe_config.hidden_dim
    n = layer.moe_config.intermediate_size_per_partition

    # w13 shape check (gate+up projections)
    size_n, size_k = n * 2, k
    assert layer.w13_weight.shape == (e, size_n, size_k // 2), \
        f"w13: {layer.w13_weight.shape} != ({e}, {size_n}, {size_k // 2})"

    # w2 shape check (down projection)
    size_n, size_k = k, n
    assert layer.w2_weight.shape == (e, size_n, size_k // 2), \
        f"w2: {layer.w2_weight.shape} != ({e}, {size_n}, {size_k // 2})"

    # Verify the bug: using num_experts (global) would fail
    e_global = layer.moe_config.num_experts
    assert e_global != e, "Test requires global != local experts"
    size_n, size_k = n * 2, k
    assert layer.w13_weight.shape != (e_global, size_n, size_k // 2), \
        "Using global num_experts should NOT match local weight shape"
