# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-CPU tests for load_per_expert_moe_weight in moe_loading_utils.py.

The helper only routes a per-expert checkpoint key at the right stacked param
and shard; the sharding itself belongs to `RoutedExperts.weight_loader`, so the
loader is recorded here rather than exercised.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.compressed_tensors.moe_loading_utils import (  # noqa: E501
    load_per_expert_moe_weight,
)


def _make_params(e: int, i: int, k: int, layer_id: int = 0, has_bias: bool = True):
    """Synthesize the stacked params a FusedMoE layer would expose, each
    carrying a `weight_loader` that records how it was called."""
    calls: list[tuple] = []

    def weight_loader(param, weight, name, shard_id, expert_id):
        calls.append((name, shard_id, expert_id, weight))

    def make(*shape):
        param = torch.nn.Parameter(torch.zeros(*shape), requires_grad=False)
        param.weight_loader = weight_loader
        return param

    base = f"layers.{layer_id}.mlp.experts"
    params = {
        f"{base}.w13_weight": make(e, 2 * i, k),
        f"{base}.w13_weight_scale": make(e, 2 * i),
        f"{base}.w2_weight": make(e, k, i),
        f"{base}.w2_weight_scale": make(e, k),
    }
    if has_bias:
        params[f"{base}.w13_bias"] = make(e, 2 * i)
        params[f"{base}.w2_bias"] = make(e, k)
    return params, calls


@pytest.mark.parametrize(
    "suffix,expected_param,expected_shard",
    [
        (".w1_weight", "w13_weight", "w1"),
        (".w1_weight_scale", "w13_weight_scale", "w1"),
        (".w1_bias", "w13_bias", "w1"),
        (".w3_weight", "w13_weight", "w3"),
        (".w3_weight_scale", "w13_weight_scale", "w3"),
        (".w3_bias", "w13_bias", "w3"),
        (".w2_weight", "w2_weight", "w2"),
        (".w2_weight_scale", "w2_weight_scale", "w2"),
        (".w2_bias", "w2_bias", "w2"),
    ],
)
def test_per_expert_key_reaches_expected_param_and_shard(
    suffix, expected_param, expected_shard
):
    params, calls = _make_params(4, 16, 32)
    loaded: set[str] = set()
    expert_id = 2

    weight = torch.ones(4)
    ok = load_per_expert_moe_weight(
        f"layers.0.mlp.experts.experts.{expert_id}{suffix}",
        weight,
        params,
        loaded,
        tp_rank=0,
    )

    assert ok
    expected_name = f"layers.0.mlp.experts.{expected_param}"
    assert len(calls) == 1
    name, shard_id, loaded_expert_id, forwarded = calls[0]
    assert (name, shard_id, loaded_expert_id) == (
        expected_name,
        expected_shard,
        expert_id,
    )
    assert forwarded is weight
    assert loaded == {expected_name}


def test_w2_bias_is_dropped_off_rank_zero():
    """w2_bias is replicated, so only rank 0 may feed the all-reduced sum."""
    params, calls = _make_params(4, 16, 32)
    loaded: set[str] = set()
    bias = torch.randn(32)

    load_per_expert_moe_weight(
        "layers.0.mlp.experts.experts.0.w2_bias", bias, params, loaded, tp_rank=1
    )

    ((_, _, _, forwarded),) = calls
    torch.testing.assert_close(forwarded, torch.zeros_like(bias))


@pytest.mark.parametrize(
    "name",
    [
        "layers.0.self_attn.q_proj.weight",
        "embed_tokens.weight",
        "lm_head.weight",
        # Already-stacked name (NOT per-expert) — must fall through.
        "layers.0.mlp.experts.w13_weight",
        # Per-expert but not a projection we stack.
        "layers.0.mlp.experts.experts.0.w1_input_scale",
    ],
)
def test_non_per_expert_names_fall_through(name):
    """Names outside the per-expert layout return False, letting the caller
    dispatch them through its stacked-tensor or default-loader branches."""
    params, calls = _make_params(4, 16, 32)
    loaded: set[str] = set()

    assert not load_per_expert_moe_weight(
        name, torch.zeros(1), params, loaded, tp_rank=0
    )
    assert not calls


def test_missing_param_is_claimed_but_not_loaded():
    """With has_bias=False the bias keys must still be claimed, so the caller
    doesn't fall through and KeyError on a param the layer never allocated."""
    params, calls = _make_params(4, 16, 32, has_bias=False)
    loaded: set[str] = set()

    assert load_per_expert_moe_weight(
        "layers.0.mlp.experts.experts.0.w1_bias",
        torch.zeros(16),
        params,
        loaded,
        tp_rank=0,
    )
    assert not calls
    assert not loaded
