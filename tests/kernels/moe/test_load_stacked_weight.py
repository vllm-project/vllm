# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU test for `RoutedExperts.load_weights` consulting the quant method for
stacked checkpoint tensors (gpt-oss MXFP4) while other layouts are unchanged."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.quantization.mxfp4 import GptOssMxfp4MoEMethod

from . import _stacked_cases as cc


def _make_experts(method, shapes):
    experts = torch.nn.Module()
    experts.layer_name = "model.layers.0.mlp.experts"
    experts.shared_expert_online_loader = None
    experts.moe_config = SimpleNamespace(is_act_and_mul=True)
    experts.quant_method = method
    experts.is_fused_checkpoint_transposed = False
    experts._orient_fused_weight = RoutedExperts._orient_fused_weight
    mapping = RoutedExperts.build_expert_params_mapping(
        "gate_proj",
        "down_proj",
        "up_proj",
        num_experts=cc.E,
        routed_experts_prefix="",
        include_fused=True,
    )
    experts.get_expert_mapping = lambda **_: mapping
    for name, shape in shapes.items():
        quantized = shapes is cc.GPT_OSS_PARAM_SHAPES and "bias" not in name
        dtype = torch.uint8 if quantized else torch.float32
        param = torch.nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)
        param.weight_loader = cc.standin_loader
        setattr(experts, name, param)
    return experts


def _gpt_oss_method(tp=1, ep=1):
    method = object.__new__(GptOssMxfp4MoEMethod)
    method.moe = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(tp_size=tp, ep_size=ep)
    )
    return method


def test_default_hook_is_off():
    assert FusedMoEMethodBase.load_stacked_weight(None, None, "x", None) is None


def test_normal_fused_unchanged():
    method = _gpt_oss_method()  # hook present but names don't match
    experts = _make_experts(method, cc.NORMAL_PARAM_SHAPES)
    loaded = list(RoutedExperts.load_weights(experts, iter(cc.NORMAL_FUSED.items())))
    assert len(loaded) == 2 * cc.E + cc.E
    for name, expected in cc.NORMAL_FUSED_EXPECTED.items():
        torch.testing.assert_close(
            getattr(experts, name).data, expected, rtol=0, atol=0
        )


def test_normal_per_expert_unchanged():
    experts = _make_experts(_gpt_oss_method(), cc.NORMAL_PARAM_SHAPES)
    list(RoutedExperts.load_weights(experts, iter(cc.NORMAL_PER_EXPERT.items())))
    torch.testing.assert_close(
        experts.w13_weight.data[3],
        torch.cat(
            [
                cc.NORMAL_PER_EXPERT["3.gate_proj.weight"],
                cc.NORMAL_PER_EXPERT["3.up_proj.weight"],
            ]
        ),
        rtol=0,
        atol=0,
    )


def test_gpt_oss_loaded_whole_in_one_call_each(monkeypatch):
    experts = _make_experts(_gpt_oss_method(), cc.GPT_OSS_PARAM_SHAPES)
    calls = []
    for name in cc.GPT_OSS_PARAM_SHAPES:
        param = getattr(experts, name)
        param.weight_loader = lambda p, w, _n=name, **kw: (
            calls.append((_n, kw.get("shard_id"), kw.get("expert_id"), tuple(w.shape))),
            cc.standin_loader(p, w, **kw),
        )[1]
    loaded = list(RoutedExperts.load_weights(experts, iter(cc.GPT_OSS.items())))
    assert sorted(loaded) == sorted(cc.GPT_OSS_PARAM_SHAPES)
    assert len(calls) == 6 and all(c[1:3] == (None, None) for c in calls)
    for name, expected in cc.GPT_OSS_EXPECTED.items():
        torch.testing.assert_close(
            getattr(experts, name).data, expected, rtol=0, atol=0
        )


@pytest.mark.parametrize(("tp", "ep"), [(2, 1), (1, 2)])
def test_gpt_oss_parallel_unsupported(tp, ep):
    """The whole stacked tensors are loaded, so a rank can't take a shard."""
    experts = _make_experts(_gpt_oss_method(tp, ep), cc.GPT_OSS_PARAM_SHAPES)
    with pytest.raises(NotImplementedError, match="does not support TP or EP"):
        list(RoutedExperts.load_weights(experts, iter(cc.GPT_OSS.items())))
