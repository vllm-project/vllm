# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight layout and routing dispatch for the FlashInfer CuTeDSL MoE backend."""

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts import flashinfer_cutedsl_moe
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_moe import (
    FlashInferCuteDSLExperts,
    FlashInferCuteDSLExpertsMonolithic,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    reorder_w13_to_w31_for_flashinfer_cutedsl,
)

_GATE = torch.tensor([[[1], [2], [3], [4]]])
_UP = torch.tensor([[[10], [20], [30], [40]]])
_EXPECTED = torch.cat([_UP, _GATE], dim=1)


def test_reorder_w13_swigluoai_interleaved():
    """gpt-oss w13 is [gate0, up0, gate1, ...] rather than packed [gate; up]."""
    w13 = torch.empty(1, 8, 1, dtype=_GATE.dtype)
    w13[:, 0::2] = _GATE
    w13[:, 1::2] = _UP

    out, out_scale = reorder_w13_to_w31_for_flashinfer_cutedsl(
        MoEActivation.SWIGLUOAI, w13, w13 + 100
    )

    torch.testing.assert_close(out, _EXPECTED)
    torch.testing.assert_close(out_scale, _EXPECTED + 100)


@pytest.mark.parametrize(
    "activation", [MoEActivation.SILU, MoEActivation.SWIGLUOAI_UNINTERLEAVE]
)
def test_reorder_w13_packed_layouts(activation: MoEActivation):
    w13 = torch.cat([_GATE, _UP], dim=1)

    out, out_scale = reorder_w13_to_w31_for_flashinfer_cutedsl(
        activation, w13, w13 + 100
    )

    torch.testing.assert_close(out, _EXPECTED)
    torch.testing.assert_close(out_scale, _EXPECTED + 100)


def _stub_experts(cls):
    """Build an experts instance with only the fields ``_fused_moe`` reads."""
    obj = object.__new__(cls)
    # quant_dtype/w*_scale/g*_alphas/a2_gscale are properties over quant_config.
    obj.quant_config = SimpleNamespace(
        quant_dtype="mxfp4",
        w1_scale=torch.empty(0),
        w2_scale=torch.empty(0),
        g1_alphas=None,
        g2_alphas=None,
        a2_gscale=None,
    )
    obj.out_dtype = torch.bfloat16
    obj.hidden_dim = 8
    obj.topk = 2
    obj.global_num_experts = 4
    obj.local_num_experts = 4
    obj.local_expert_offset = 0
    obj._w1_bias = obj._w2_bias = None
    obj._weight_interleave = 16
    obj.gemm1_alpha = obj.gemm1_beta = obj.gemm1_clamp_limit = None
    return obj


def test_modular_and_monolithic_select_routing_mode(monkeypatch):
    """Modular passes precomputed topk; monolithic passes router logits.

    The CuTeDSL kernel picks its routing path from these mutually exclusive
    kwargs, so sending the wrong pair silently changes which routing kernel
    runs, or trips the library's validation.
    """
    captured: dict = {}
    monkeypatch.setattr(
        flashinfer_cutedsl_moe, "flashinfer_cute_dsl_fused_moe", captured.update
    )

    x = torch.zeros(2, 8, dtype=torch.bfloat16)
    common = dict(
        hidden_states=x,
        w1=torch.empty(0),
        # [experts, hidden_dim, N]: monolithic sizes its output from w2.size(1).
        w2=torch.empty(4, 8, 8),
        activation=MoEActivation.SILU,
        a1q_scale=torch.empty(0),
    )

    _stub_experts(FlashInferCuteDSLExperts).apply(
        output=torch.zeros_like(x),
        topk_weights=torch.zeros(2, 2, dtype=torch.bfloat16),
        topk_ids=torch.zeros(2, 2, dtype=torch.int64),
        global_num_experts=4,
        expert_map=None,
        a2_scale=None,
        workspace13=None,
        workspace2=None,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
        **common,
    )
    assert captured["router_logits"] is None
    # dtypes the kernel requires, normalized by apply()
    assert captured["token_selected_experts"].dtype == torch.int32
    assert captured["token_final_scales"].dtype == torch.float32

    captured.clear()
    _stub_experts(FlashInferCuteDSLExpertsMonolithic).apply(
        router_logits=torch.zeros(2, 4, dtype=torch.bfloat16),
        global_num_experts=4,
        expert_map=None,
        apply_router_weight_on_input=False,
        **common,
    )
    assert captured["token_selected_experts"] is None
    assert captured["token_final_scales"] is None
    assert captured.pop("router_logits").shape == (2, 4)


def test_monolithic_opts_out_when_fused_routing_cannot_be_used():
    """Fused routing needs every expert local (EP=1) and TopK->softmax routing;
    anything else must fall back to the modular variant."""
    cls = FlashInferCuteDSLExpertsMonolithic

    assert cls._supports_parallel_config(FusedMoEParallelConfig.make_no_parallel())
    ep = dataclasses.replace(
        FusedMoEParallelConfig.make_no_parallel(), ep_size=2, use_ep=True
    )
    assert not cls._supports_parallel_config(ep)

    assert cls._supports_routing_method(RoutingMethodType.Renormalize, None, None)
    # What get_routing_method_type actually returns for softmax models (gpt-oss);
    # same function as Renormalize. Default omits the renormalize, so weights differ.
    assert cls._supports_routing_method(RoutingMethodType.RenormalizeNaive, None, None)
    assert not cls._supports_routing_method(RoutingMethodType.Default, None, None)
    assert not cls._supports_routing_method(RoutingMethodType.DeepSeekV3, None, None)
