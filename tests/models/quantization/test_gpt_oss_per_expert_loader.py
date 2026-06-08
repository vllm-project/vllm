# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-CPU smoke tests for _load_per_expert_moe_weight in gpt_oss.py:
gate/up w13 packing, w2 TP-slicing, the w2_bias rank-0 rule, and
off-rank (EP) / non-MoE / missing-param handling."""

import pytest
import torch

from vllm.model_executor.models.gpt_oss import _load_per_expert_moe_weight


def _make_params(
    e_local: int,
    i_local: int,
    k: int,
    layer_id: int = 0,
    has_bias: bool = True,
):
    """Synthesize a params_dict that mimics CompressedTensorsW8A8Int8MoEMethod's
    allocated stacked-MoE tensors for one layer."""
    nn_p = lambda t: torch.nn.Parameter(t, requires_grad=False)
    base = f"layers.{layer_id}.mlp.experts"
    out = {
        f"{base}.w13_weight": nn_p(torch.zeros(e_local, 2 * i_local, k)),
        f"{base}.w13_weight_scale": nn_p(torch.zeros(e_local, 2 * i_local)),
        f"{base}.w2_weight": nn_p(torch.zeros(e_local, k, i_local)),
        f"{base}.w2_weight_scale": nn_p(torch.zeros(e_local, k)),
    }
    if has_bias:
        out[f"{base}.w13_bias"] = nn_p(torch.zeros(e_local, 2 * i_local))
        out[f"{base}.w2_bias"] = nn_p(torch.zeros(e_local, k))
    return out


@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 0), (2, 1), (4, 3)])
def test_per_expert_dispatch_tp(tp_size, tp_rank):
    """Per-expert keys land in correct halves of w13 and correctly TP-sliced
    inputs of w2 (and w2_bias follows the rank-0-only semantics)."""
    e, i, k = 4, 16, 32
    assert i % tp_size == 0
    i_local = i // tp_size
    tp_rank_start = tp_rank * i_local
    tp_rank_end = (tp_rank + 1) * i_local

    params = _make_params(e, i_local, k, has_bias=True)
    loaded: set[str] = set()

    torch.manual_seed(0)
    gate = torch.randn(i, k)
    up = torch.randn(i, k)
    down = torch.randn(k, i)
    gate_scale = torch.randn(i)
    up_scale = torch.randn(i)
    down_scale = torch.randn(k)
    gate_bias = torch.randn(i)
    up_bias = torch.randn(i)
    down_bias = torch.randn(k)

    expert_id = 0
    for suffix, tensor in [
        (".w1_weight", gate),
        (".w1_weight_scale", gate_scale),
        (".w1_bias", gate_bias),
        (".w3_weight", up),
        (".w3_weight_scale", up_scale),
        (".w3_bias", up_bias),
        (".w2_weight", down),
        (".w2_weight_scale", down_scale),
        (".w2_bias", down_bias),
    ]:
        name = f"layers.0.mlp.experts.experts.{expert_id}{suffix}"
        ok = _load_per_expert_moe_weight(
            name,
            tensor.clone(),
            params_dict=params,
            loaded_params=loaded,
            use_ep=False,
            ep_rank_start=0,
            ep_rank_end=e,
            tp_rank=tp_rank,
            tp_rank_start=tp_rank_start,
            tp_rank_end=tp_rank_end,
            per_rank_intermediate_size=i_local,
        )
        assert ok, f"dispatch should consume {name}"

    # Gate landed in lower half [0:i_local] of expert 0's w13 slot.
    torch.testing.assert_close(
        params["layers.0.mlp.experts.w13_weight"].data[0, :i_local],
        gate[tp_rank_start:tp_rank_end],
    )
    torch.testing.assert_close(
        params["layers.0.mlp.experts.w13_weight_scale"].data[0, :i_local],
        gate_scale[tp_rank_start:tp_rank_end],
    )
    torch.testing.assert_close(
        params["layers.0.mlp.experts.w13_bias"].data[0, :i_local],
        gate_bias[tp_rank_start:tp_rank_end],
    )
    # Up landed in upper half.
    torch.testing.assert_close(
        params["layers.0.mlp.experts.w13_weight"].data[0, i_local : 2 * i_local],
        up[tp_rank_start:tp_rank_end],
    )
    # w2_weight TP-sliced along its input dim.
    torch.testing.assert_close(
        params["layers.0.mlp.experts.w2_weight"].data[0],
        down[:, tp_rank_start:tp_rank_end],
    )
    # w2_weight_scale: same on all ranks (no slice).
    torch.testing.assert_close(
        params["layers.0.mlp.experts.w2_weight_scale"].data[0],
        down_scale,
    )
    # w2_bias: only rank 0 contributes when tp_size > 1.
    if tp_size > 1 and tp_rank != 0:
        torch.testing.assert_close(
            params["layers.0.mlp.experts.w2_bias"].data[0],
            torch.zeros_like(down_bias),
        )
    else:
        torch.testing.assert_close(
            params["layers.0.mlp.experts.w2_bias"].data[0],
            down_bias,
        )


def test_per_expert_dispatch_ep_skips_off_rank_experts():
    """Under EP, off-rank experts are silently skipped (returns True so caller
    moves on, no write into local params)."""
    e_local, i, k = 2, 16, 32
    ep_rank_start, ep_rank_end = 2, 4  # this rank owns experts [2, 4)
    params = _make_params(e_local, i, k, has_bias=False)
    loaded: set[str] = set()

    torch.manual_seed(0)
    gate_e2 = torch.randn(i, k)
    gate_e1 = torch.randn(i, k)  # NOT owned by this rank

    # Local expert: global id 2 → local id 0 → write to params.data[0].
    ok = _load_per_expert_moe_weight(
        "layers.0.mlp.experts.experts.2.w1_weight",
        gate_e2,
        params_dict=params,
        loaded_params=loaded,
        use_ep=True,
        ep_rank_start=ep_rank_start,
        ep_rank_end=ep_rank_end,
        tp_rank=0,
        tp_rank_start=0,
        tp_rank_end=i,
        per_rank_intermediate_size=i,
    )
    assert ok
    torch.testing.assert_close(
        params["layers.0.mlp.experts.w13_weight"].data[0, :i],
        gate_e2,
    )

    # Off-rank expert: returns True (claimed by the per-expert path) but
    # MUST NOT mutate any local param.
    before = params["layers.0.mlp.experts.w13_weight"].data.clone()
    ok = _load_per_expert_moe_weight(
        "layers.0.mlp.experts.experts.1.w1_weight",
        gate_e1,
        params_dict=params,
        loaded_params=loaded,
        use_ep=True,
        ep_rank_start=ep_rank_start,
        ep_rank_end=ep_rank_end,
        tp_rank=0,
        tp_rank_start=0,
        tp_rank_end=i,
        per_rank_intermediate_size=i,
    )
    assert ok
    assert torch.equal(params["layers.0.mlp.experts.w13_weight"].data, before)


@pytest.mark.parametrize(
    "name",
    [
        "layers.0.self_attn.q_proj.weight",
        "embed_tokens.weight",
        "lm_head.weight",
        # Already-stacked name (NOT per-expert) — must fall through.
        "layers.0.mlp.experts.w13_weight",
    ],
)
def test_per_expert_dispatch_non_moe_returns_false(name):
    """Names outside the experts.experts.N.* pattern return False, letting the
    surrounding loop dispatch them through the existing stacked-tensor or
    default-loader branches."""
    params = _make_params(4, 16, 32)
    loaded: set[str] = set()
    ok = _load_per_expert_moe_weight(
        name,
        torch.zeros(1),
        params_dict=params,
        loaded_params=loaded,
        use_ep=False,
        ep_rank_start=0,
        ep_rank_end=4,
        tp_rank=0,
        tp_rank_start=0,
        tp_rank_end=16,
        per_rank_intermediate_size=16,
    )
    assert not ok, f"{name} should fall through (return False)"


def test_per_expert_dispatch_missing_bias_param_is_handled():
    """If the model was constructed with has_bias=False, the dispatcher must
    still return True for bias keys (so the caller doesn't fall through to
    the default loader and KeyError on the missing param)."""
    params = _make_params(4, 16, 32, has_bias=False)
    loaded: set[str] = set()
    ok = _load_per_expert_moe_weight(
        "layers.0.mlp.experts.experts.0.w1_bias",
        torch.zeros(16),
        params_dict=params,
        loaded_params=loaded,
        use_ep=False,
        ep_rank_start=0,
        ep_rank_end=4,
        tp_rank=0,
        tp_rank_start=0,
        tp_rank_end=16,
        per_rank_intermediate_size=16,
    )
    assert ok
    assert "layers.0.mlp.experts.w13_bias" not in loaded
